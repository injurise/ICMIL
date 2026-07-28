"""Attention-based MIL pooling.

An independent implementation of the attention pooling introduced by

    M. Ilse, J. M. Tomczak and M. Welling,
    "Attention-based Deep Multiple Instance Learning", ICML 2018.
    https://arxiv.org/abs/1802.04712

A bag of instance embeddings ``h_1..h_M`` is pooled into a single bag embedding
``z = sum_i a_i h_i`` with attention weights ``a = softmax(s)``. The two scoring
functions are Eq. 8 and Eq. 9 of that paper:

* :class:`GlobalAttention` (Eq. 8)        ``s_i = w^T tanh(V h_i)``
* :class:`GlobalGatedAttention` (Eq. 9)   ``s_i = w^T (tanh(V h_i) * sigm(U h_i))``

The gated variant multiplies the ``tanh`` branch elementwise by a ``sigmoid``
gate, which lets the network suppress instances that ``tanh`` alone would score
highly simply because it is near-linear in its mid-range.

Used by the ABMIL and ACMIL baselines, and by the synthetic prior generator's
``embedding_abmil`` bag-label rule. Layer names, shapes and construction order
match the reference implementation these baselines were originally run against,
so a fixed seed yields identical parameters and identical outputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def create_mlp(
    in_dim: int = 768,
    hid_dims: list[int] | None = None,
    out_dim: int = 512,
    act: nn.Module | None = None,
    dropout: float = 0.0,
    end_with_fc: bool = True,
    end_with_dropout: bool = False,
    bias: bool = True,
) -> nn.Sequential:
    """Build an MLP: ``[Linear, act, Dropout] * len(hid_dims)`` then a final Linear.

    Args:
        in_dim: Input width.
        hid_dims: Hidden widths; ``None`` or ``[]`` gives a single Linear.
        out_dim: Output width.
        act: Activation between hidden layers (default ``nn.ReLU()``).
        dropout: Dropout probability after each hidden activation.
        end_with_fc: If False, append ``act`` after the final Linear.
        end_with_dropout: If True, append a Dropout after everything else.
        bias: Bias for the *hidden* Linears. The final Linear always has a bias.
    """
    hid_dims = [512, 512] if hid_dims is None else hid_dims
    act = nn.ReLU() if act is None else act

    layers: list[nn.Module] = []
    for hid_dim in hid_dims:
        layers += [nn.Linear(in_dim, hid_dim, bias=bias), act, nn.Dropout(dropout)]
        in_dim = hid_dim
    # Deliberately unconditional bias: `bias` governs the hidden layers only.
    layers.append(nn.Linear(in_dim, out_dim))
    if not end_with_fc:
        layers.append(act)
    if end_with_dropout:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class GlobalAttention(nn.Module):
    """Ungated attention scoring, ``s_i = w^T tanh(V h_i)`` (Ilse et al. Eq. 8).

    Args:
        L: Instance embedding width.
        D: Attention hidden width.
        dropout: Dropout after the ``tanh``.
        num_classes: Number of attention heads (1 for plain bag pooling).
    """

    def __init__(self, L: int = 1024, D: int = 256, dropout: float = 0.0, num_classes: int = 1) -> None:
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(D, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score instances: ``(..., M, L) -> (..., M, num_classes)``."""
        return self.module(x)


class GlobalGatedAttention(nn.Module):
    """Gated attention scoring, ``s_i = w^T (tanh(V h_i) * sigm(U h_i))`` (Eq. 9).

    Args:
        L: Instance embedding width.
        D: Attention hidden width.
        dropout: Dropout on both branches.
        num_classes: Number of attention heads (1 for plain bag pooling).
    """

    def __init__(self, L: int = 1024, D: int = 256, dropout: float = 0.0, num_classes: int = 1) -> None:
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(L, D), nn.Tanh(), nn.Dropout(dropout))
        self.attention_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_c = nn.Linear(D, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score instances: ``(..., M, L) -> (..., M, num_classes)``."""
        return self.attention_c(self.attention_a(x).mul(self.attention_b(x)))


class ABMILAggregator(nn.Module):
    """Pool ``(B, NB, M, D)`` instance embeddings into ``(B, NB, D)`` bag embeddings.

    Args:
        embed_dim: Instance embedding width.
        dropout: Dropout inside the attention network.
        attn_dim: Attention hidden width.
        gate: Use :class:`GlobalGatedAttention` when True, else :class:`GlobalAttention`.
    """

    def __init__(self, embed_dim: int = 512, dropout: float = 0.0, attn_dim: int = 384, gate: bool = True) -> None:
        super().__init__()
        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(L=embed_dim, D=attn_dim, dropout=dropout, num_classes=1)

    def forward_attention(
        self,
        h: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_only: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return pre-softmax attention scores for ``h`` of shape ``(B, M, D)``.

        Args:
            h: Instance embeddings ``(B, M, D)``.
            attn_mask: Optional ``(B, M)`` mask; 1 keeps, 0 masks. Masked positions
                are pushed to the dtype minimum so they vanish under the softmax.
            attn_only: Return only the scores; otherwise return ``(h, scores)``.

        Returns:
            Scores ``(B, K, M)``, or ``(h, scores)`` when ``attn_only`` is False.
        """
        A = self.global_attn(h)  # (B, M, K)
        A = torch.transpose(A, -2, -1)  # (B, K, M)
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min
        if attn_only:
            return A
        return h, A

    def forward(
        self,
        h: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_attention: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """Attention-pool every bag.

        Args:
            h: ``(B, NB, M, D)`` instance embeddings.
            attn_mask: Optional ``(B, NB, M)`` mask.
            return_attention: Include the attention scores in the returned dict.

        Returns:
            ``(bags, log_dict)`` where ``bags`` is ``(B, NB, D)`` and
            ``log_dict["attention"]`` holds the **pre-softmax** scores
            ``(B, NB, K, M)`` (or ``None`` when not requested).
        """
        B, NB, M, D = h.shape
        # Bags are independent, so fold them into the batch dim and pool in one call.
        h = h.reshape(B * NB, M, D)
        if attn_mask is not None:
            attn_mask = attn_mask.reshape(B * NB, M)

        h, scores = self.forward_attention(h, attn_mask=attn_mask, attn_only=False)
        A = F.softmax(scores, dim=-1)  # over instances
        bags = torch.bmm(A, h).squeeze(dim=1).reshape(B, NB, -1)

        attention = scores.reshape(B, NB, scores.shape[1], M) if return_attention else None
        return bags, {"attention": attention}
