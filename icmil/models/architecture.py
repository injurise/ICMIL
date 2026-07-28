"""The ICMIL model: latent column-row attention over bags.

Per-feature-group embedding of every instance, a learnable latent matrix per bag,
then ``num_column_row_iterations`` rounds of (a) cross-attention pulling instance
embeddings into the latents and (b) column attention across feature groups plus
row attention across bags. Row attention carries the in-context split: context
bags attend among themselves, query bags attend only to context. The bag label
token is read out of the final latent stack and decoded to class logits.

One detail worth knowing when comparing against the training code: ``final_norm``
is a LayerNorm applied to the bag label token just before the decoder. It is a
trained, non-identity layer present in every released checkpoint
(``final_norm.{weight,bias}``), so it is part of the architecture — the model will
not load a checkpoint strictly without it. Placement is output-equivalent to
normalising the whole latent stack, since only the label token is decoded.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from icmil.models.attention import LowerPrecisionLayerNorm, MultiheadAttention
from icmil.models.blocks import BagFeatureEmbedder, BagTargetEmbedder, Decoder, TransformerEncoderLayer

# SDPA caps the leading batch dim of (batch, heads, T, D) at the CUDA grid-dim-x limit.
_SDPA_MAX_BATCH = 65535


class InstanceAggregationBlock(nn.Module):
    """Cross-attention: latent group tokens ← per-group instance embeddings.

    Chunked over bags to bound memory. Only the G feature-group latents
    participate; the label token is passed through unchanged.
    """

    def __init__(self, embed_dim: int, num_heads: int, bag_chunk_size: int = 32) -> None:
        super().__init__()
        self.bag_chunk_size = bag_chunk_size
        self.cross_attn = MultiheadAttention(embed_dim, num_heads)
        self.norm = LowerPrecisionLayerNorm(embed_dim)

    def forward(
        self,
        latents: torch.Tensor,  # (B, NB, G+1, E)
        instances_grouped: torch.Tensor,  # (B, NB, G, N, E)
    ) -> torch.Tensor:  # (B, NB, G+1, E)
        B, NB, Gp1, E = latents.shape
        G = Gp1 - 1
        N = instances_grouped.shape[3]

        feature_latents = latents[:, :, :G, :]
        label_token = latents[:, :, G:, :]

        chunk = min(self.bag_chunk_size, max(1, _SDPA_MAX_BATCH // max(1, B * G)))

        chunks = []
        for nb0 in range(0, NB, chunk):
            nb1 = min(nb0 + chunk, NB)
            cnb = nb1 - nb0
            q = feature_latents[:, nb0:nb1].reshape(B * cnb * G, 1, E)
            kv = instances_grouped[:, nb0:nb1].reshape(B * cnb * G, N, E)
            out, _ = self.cross_attn(q, kv, kv)
            chunks.append(out.reshape(B, cnb, G, E))

        cross_out = torch.cat(chunks, dim=1)
        feature_latents = self.norm(cross_out + feature_latents)

        return torch.cat([feature_latents, label_token], dim=2)


class InterBagAttentionBlock(nn.Module):
    """Column + row attention with the in-context train/test split.

    Delegates to TransformerEncoderLayer: column attention over G+1 token
    positions, row attention over bags (train self-attn, test cross-attn to train).
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_hidden_size: int) -> None:
        super().__init__()
        self.col_row = TransformerEncoderLayer(embed_dim, num_heads, mlp_hidden_size)

    def forward(self, latents: torch.Tensor, eval_bag_position: int) -> torch.Tensor:
        return self.col_row(latents, single_eval_position=eval_bag_position)


class ICMIL(nn.Module):
    """The ICMIL model: latent column-row attention over bags."""

    def __init__(
        self,
        in_features: int,
        embedding_size: int = 160,
        num_attention_heads: int = 4,
        mlp_hidden_size: int = 512,
        num_column_row_iterations: int = 4,
        num_outputs: int = 2,
        feature_group_size: int = 3,
        num_final_col_row_layers: int = 0,
        gradient_checkpointing: bool = False,
        bag_chunk_size: int = 32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_outputs = num_outputs
        self.feature_group_size = feature_group_size
        self.gradient_checkpointing = gradient_checkpointing
        num_groups = math.ceil(in_features / feature_group_size)
        self.bag_encoder = BagFeatureEmbedder(
            embedding_size=embedding_size, normalize_only=False, group_size=feature_group_size
        )
        self.target_encoder = BagTargetEmbedder(embedding_size=embedding_size, return_identity=False)
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)
        self.bag_latent_init = nn.Parameter(torch.randn(num_groups, embedding_size) * 0.02)
        self.cross_attn_blocks = nn.ModuleList(
            [
                InstanceAggregationBlock(embedding_size, num_attention_heads, bag_chunk_size=bag_chunk_size)
                for _ in range(num_column_row_iterations)
            ]
        )
        self.col_row_blocks = nn.ModuleList(
            [
                InterBagAttentionBlock(embedding_size, num_attention_heads, mlp_hidden_size)
                for _ in range(num_column_row_iterations)
            ]
        )
        self.final_col_row_blocks = nn.ModuleList(
            [
                InterBagAttentionBlock(embedding_size, num_attention_heads, mlp_hidden_size)
                for _ in range(num_final_col_row_layers)
            ]
        )
        # Applied to the bag label token before decoding (present in trained checkpoints).
        self.final_norm = LowerPrecisionLayerNorm(embedding_size)

    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
        eval_bag_position = X_train.shape[1]
        X = torch.cat([X_train, X_test], dim=1)  # (B, NB, N, F)

        # Enforce the trained feature-count contract (in_features). Benchmark tasks
        # may pass wider raw features; truncate or right-pad so F == in_features.
        F = X.shape[-1]
        if self.in_features < F:
            X = X[..., : self.in_features]
        elif self.in_features > F:
            X = torch.nn.functional.pad(X, (0, self.in_features - F))
        B, NB, _, _F = X.shape

        instances = self.bag_encoder(X, eval_bag_position)  # (B, NB, N, G, E)
        G = instances.shape[3]
        instances_grouped = instances.permute(0, 1, 3, 2, 4).contiguous()  # (B, NB, G, N, E)

        label_emb = self.target_encoder(y_train.unsqueeze(-1), NB)  # (B, NB, 1, E)
        label_emb = label_emb.squeeze(2)  # (B, NB, E)

        feature_latents = self.bag_latent_init[:G].unsqueeze(0).unsqueeze(0).expand(B, NB, -1, -1).clone()
        label_token = label_emb.unsqueeze(2)  # (B, NB, 1, E)
        latents = torch.cat([feature_latents, label_token], dim=2)  # (B, NB, G+1, E)

        for cross_block, col_row_block in zip(self.cross_attn_blocks, self.col_row_blocks, strict=False):
            if self.gradient_checkpointing and torch.is_grad_enabled():
                latents = grad_checkpoint(cross_block, latents, instances_grouped, use_reentrant=False)
                latents = grad_checkpoint(col_row_block, latents, eval_bag_position, use_reentrant=False)
            else:
                latents = cross_block(latents, instances_grouped)
                latents = col_row_block(latents, eval_bag_position)

        for col_row_block in self.final_col_row_blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                latents = grad_checkpoint(col_row_block, latents, eval_bag_position, use_reentrant=False)
            else:
                latents = col_row_block(latents, eval_bag_position)

        label_tokens = latents[:, :, -1, :]  # (B, NB, E)
        label_tokens = self.final_norm(label_tokens)
        output = self.decoder(label_tokens)  # (B, NB, C)
        return output[:, eval_bag_position:, :]
