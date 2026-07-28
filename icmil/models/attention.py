"""Attention primitives for the ICMIL model (copied verbatim from the training code)."""

import torch
import torch.nn.functional as F
from torch import nn


class LowerPrecisionLayerNorm(nn.LayerNorm):
    """LayerNorm that stays in fp16/bf16 under autocast instead of upcasting to fp32.

    Safe for embedding dimensions ≤ 512. Halves memory for all normalized activations.
    From TabPFN v2.6 (PriorLabs).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.layer_norm(
                x,
                self.normalized_shape,
                self.weight.to(x.dtype) if self.weight is not None else None,
                self.bias.to(x.dtype) if self.bias is not None else None,
                self.eps,
            )


class MultiheadAttention(nn.Module):
    """Minimal Multi-Head Attention using PyTorch's scaled_dot_product_attention (SDPA).

    On CUDA with supported dtypes and head_dim <= 128 this dispatches to Flash
    Attention; otherwise it falls back to the memory-efficient or math kernel.

    Shapes: B=batch, T=sequence length, E=embed dim, H=heads, D=E/H.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        batch_first: bool = True,
        qkv_bias: bool = False,
        out_proj_bias: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        fw = {"device": device, "dtype": dtype}
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, **fw)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, **fw)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, **fw)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias, **fw)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Compute multi-head attention; returns ``(output, None)`` (no attn weights)."""
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, Tq, _ = query.shape
        Tk = key.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        H, D = self.num_heads, self.head_dim
        q = q.view(B, Tq, H, D).transpose(1, 2)
        k = k.view(B, Tk, H, D).transpose(1, 2)
        v = v.view(B, Tk, H, D).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

        attn = attn.transpose(1, 2).contiguous().view(B, Tq, H * D)
        out = self.out_proj(attn)

        if not self.batch_first:
            out = out.transpose(0, 1)
        return out, None
