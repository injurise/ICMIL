"""MambaMIL baseline.

MambaMIL (Yang et al. 2024, https://arxiv.org/abs/2403.06800,
https://github.com/isyangshu/MambaMIL) — a patch_embed MLP, a stack of
pre-norm residual Sequence Reordering Mamba (SRMamba) blocks, then attention
pooling and a linear head. SRMamba runs one selective state-space scan over the
instances in their given order and a second, separately parameterised scan over
a reordered copy (every ``rate``-th instance grouped together), and sums both.

It is slow and memory-hungry on very long bags: pass ``batch_size`` there.

Same training protocol as
:class:`icmil.baselines.abmil_baseline.ABMILBaseline`, which
this subclasses — only the model differs.

Standalone usage::

    model = MambaMIL(in_dim=1024, num_classes=2)
    logits = model(features)            # features: (B, M, D)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from icmil.baselines.abmil_baseline import ABMILBaseline
from icmil.mil_pooling import GlobalAttention, create_mlp


class _SelectiveScan(nn.Module):
    """One direction of a Mamba block: causal depthwise conv -> selective SSM scan -> gate."""

    def __init__(self, d_inner: int, d_state: int, d_conv: int, dt_rank: int) -> None:
        super().__init__()
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, xz: torch.Tensor) -> torch.Tensor:
        """``xz`` is ``(B, M, 2 * d_inner)``; returns ``(B, M, d_inner)``."""
        x, z = xz.chunk(2, dim=-1)
        m = x.shape[1]
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :m].transpose(1, 2))  # causal: drop the right pad

        dt, b, c = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, M, d_inner)
        d_a = torch.exp(dt.unsqueeze(-1) * -torch.exp(self.A_log))  # (B, M, d_inner, d_state)
        d_bx = (dt * x).unsqueeze(-1) * b.unsqueeze(2)  # (B, M, d_inner, d_state)

        h = x.new_zeros(x.shape[0], x.shape[2], self.d_state)
        hs = []
        for a_t, bx_t in zip(d_a.unbind(1), d_bx.unbind(1), strict=True):
            h = torch.addcmul(bx_t, a_t, h)
            hs.append(h)
        y = torch.einsum("bmdn,bmn->bmd", torch.stack(hs, dim=1), c) + self.D * x
        return y * F.silu(z)


class SRMamba(nn.Module):
    """Sequence Reordering Mamba: a scan in the given order plus one in reordered order."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        d_inner = expand * d_model
        dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.scan = _SelectiveScan(d_inner, d_state, d_conv, dt_rank)
        self.scan_reordered = _SelectiveScan(d_inner, d_state, d_conv, dt_rank)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, h: torch.Tensor, rate: int) -> torch.Tensor:
        """``h`` is ``(B, M, d_model)``; returns the same shape."""
        xz = self.in_proj(h)
        b, m, d = xz.shape
        # Zero-pad M to a multiple of `rate`, then read the (segment, offset)
        # grid column-wise: offset 0 of every segment first, then offset 1, ...
        k = math.ceil(m / rate)
        reordered = F.pad(xz, (0, 0, 0, k * rate - m)).view(b, k, rate, d).transpose(1, 2).reshape(b, k * rate, d)
        y = self.scan_reordered(reordered)
        y = y.view(b, rate, k, -1).transpose(1, 2).reshape(b, k * rate, -1)[:, :m]  # back to the given order
        return self.out_proj(self.scan(xz) + y)


class MambaMIL(nn.Module):
    """MambaMIL: patch-embed MLP -> residual SRMamba blocks -> attention pooling -> classifier."""

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 512,
        n_layers: int = 2,
        rate: int = 10,
        dropout: float = 0.25,
        attn_dim: int = 128,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.rate = rate
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[],
            dropout=dropout,
            out_dim=embed_dim,
            act=nn.GELU(),
            end_with_fc=False,
            end_with_dropout=True,
        )
        self.norms = nn.ModuleList(nn.LayerNorm(embed_dim) for _ in range(n_layers))
        self.mixers = nn.ModuleList(SRMamba(embed_dim) for _ in range(n_layers))
        self.norm = nn.LayerNorm(embed_dim)
        self.global_attn = GlobalAttention(L=embed_dim, D=attn_dim, num_classes=1)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        # As in the reference: every Linear, including those inside the Mamba blocks.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass. ``h`` is ``(B, M, D)``; returns ``(B, num_classes)``."""
        h = self.patch_embed(h)  # (B, M, E)
        for norm, mixer in zip(self.norms, self.mixers, strict=True):
            h = h + mixer(norm(h), self.rate)
        h = self.norm(h)
        a = F.softmax(self.global_attn(h).transpose(-2, -1), dim=-1)  # (B, 1, M)
        return self.classifier(torch.bmm(a, h).squeeze(1))


class MambaMILBaseline(ABMILBaseline):
    """MambaMIL under the :class:`ABMILBaseline` protocol.

    Adam, ``(lr, wd)`` selected on a held-out val split with early stopping —
    no CV, no ensembling. ``gate`` and ``num_fc_layers`` are unused.
    """

    def __init__(self, n_layers: int = 2, rate: int = 5, embed_dim: int = 512, **kwargs) -> None:
        super().__init__(embed_dim=embed_dim, **kwargs)
        self.n_layers = n_layers
        self.rate = rate

    def _make_model(self, in_dim: int, device: torch.device) -> MambaMIL:
        return MambaMIL(
            in_dim=in_dim,
            embed_dim=self.embed_dim,
            n_layers=self.n_layers,
            rate=self.rate,
            dropout=self.dropout,
            attn_dim=self.attn_dim,
            num_classes=self.max_classes,
        ).to(device)
