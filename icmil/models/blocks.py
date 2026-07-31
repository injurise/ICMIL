"""Shared building blocks for the ICMIL model.

Contains the per-feature embedder, label embedder, decoder, the column/row
TransformerEncoderLayer, and the ``memory_chunking`` helper it uses.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear


from icmil.models.attention import LowerPrecisionLayerNorm, MultiheadAttention


class BagFeatureEmbedder(nn.Module):
    def __init__(self, embedding_size: int | None = None, normalize_only: bool = False, group_size: int = 1) -> None:
        """Creates the linear layer that we will use to embed our features. Optionally only normalize the features of the training data."""
        super().__init__()
        self.normalize_only = normalize_only
        self.group_size = group_size
        if not normalize_only:
            if embedding_size is None:
                raise ValueError("embedding_size is required when normalize_only=False")
            self.linear_layer = nn.Linear(group_size, embedding_size)

    def forward(self, x: torch.Tensor, single_eval_pos: int) -> torch.Tensor:
        """Normalize features over all training instances, clip to [-100, 100], then embed.

        Returns (B, NB, N, G, E) when group_size>1 (or per-feature groups when ==1),
        or the normalized tensor itself when ``normalize_only``.
        """
        B, NB, N, F = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_train_flat = x[:, :single_eval_pos].reshape(B, -1, F).double()
            mean = x_train_flat.mean(dim=1, keepdim=True).unsqueeze(1)
            std = x_train_flat.std(dim=1, keepdim=True).unsqueeze(1) + 1e-6
            x = ((x.double() - mean) / std).clip(-100, 100).float()
        if self.normalize_only:
            return x
        if self.group_size > 1:
            pad = (-F) % self.group_size
            if pad > 0:
                x = torch.nn.functional.pad(x, (0, pad))
            G = (F + pad) // self.group_size
            x = x.reshape(B, NB, N, G, self.group_size)
            return self.linear_layer(x)
        return self.linear_layer(x.unsqueeze(-1))


class BagTargetEmbedder(nn.Module):
    def __init__(self, embedding_size: int | None = None, return_identity: bool = False) -> None:
        """Creates the linear layer that we will use to embed our targets."""
        super().__init__()
        self.return_identity = return_identity
        if return_identity:
            self.embedder = nn.Identity()
        else:
            if embedding_size is None:
                raise ValueError("embedding_size is required when return_identity=False")
            self.embedder = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_bags: int) -> torch.Tensor:
        """Pad y_train up to num_bags using the per-dataset mean, then embed.

        Returns (B, num_bags, 1, E).
        """
        y_train = y_train.float()
        mean = torch.mean(y_train, dim=1, keepdim=True)
        padding = mean.repeat(1, num_bags - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.embedder(y)


class Decoder(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int) -> None:
        """Initializes the linear layers for use in the forward"""
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a 2-layer MLP to embeddings (B, num_rows, E) -> logits (B, num_rows, C)."""
        return self.linear2(F.gelu(self.linear1(x)))


class TransformerEncoderLayer(nn.Module):
    """Self-attention between features and between datapoints, then a 2-layer MLP.

    Modified from torch.nn.modules.transformer (v2.6.0). The datapoint attention
    enforces the in-context split: train bags self-attend; test bags cross-attend
    to train only.
    """

    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )
        self.self_attention_between_features = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )

        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)

        self.norm1 = LowerPrecisionLayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LowerPrecisionLayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LowerPrecisionLayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, single_eval_position: int, num_mem_chunks: int = 1) -> torch.Tensor:
        """(B, rows, cols, E) -> same shape, after feature attn, datapoint attn, MLP."""
        batch_size, rows_size, col_size, embedding_size = src.shape
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)

        @memory_chunking(num_mem_chunks)
        def feature_attention(x):
            return self.self_attention_between_features(x, x, x)[0] + x

        src = feature_attention(src)
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, embedding_size)

        @memory_chunking(num_mem_chunks)
        def datapoint_attention(x):
            x_left = self.self_attention_between_datapoints(
                x[:, :single_eval_position], x[:, :single_eval_position], x[:, :single_eval_position]
            )[0]
            x_right = self.self_attention_between_datapoints(
                x[:, single_eval_position:], x[:, :single_eval_position], x[:, :single_eval_position]
            )[0]
            return torch.cat([x_left, x_right], dim=1) + x

        src = datapoint_attention(src)
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        src = src.reshape(-1, embedding_size)

        @memory_chunking(num_mem_chunks)
        def mlp(x):
            return self.linear2(F.gelu(self.linear1(x))) + x

        src = mlp(src)
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm3(src)
        return src


def memory_chunking(num_mem_chunks: int) -> Callable:
    """Decorator that splits the first dim into chunks and applies the wrapped fn per chunk.

    ``num_mem_chunks<=1`` disables chunking. Chunking is also disabled under
    grad (in-place writes would corrupt gradients); use ``torch.no_grad()``.
    """

    def decorator(func: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
        def wrapper(x: torch.Tensor) -> torch.Tensor:
            if num_mem_chunks <= 1 or x.shape[0] == 0:
                return func(x)
            elif torch.is_grad_enabled():
                warnings.warn(
                    "Memory chunking is disabled since gradient computation is enabled to avoid incorrect gradients. "
                    "Please use `with torch.no_grad():` during inference to enable chunking."
                )
                return func(x)
            chunk_size = max(1, math.ceil(x.shape[0] / num_mem_chunks))
            for x_split in torch.split(x, split_size_or_sections=chunk_size, dim=0):
                x_split[:] = func(x_split)
            return x

        return wrapper

    return decorator
