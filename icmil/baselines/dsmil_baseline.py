"""DSMIL baseline (Dual-Stream Multiple Instance Learning).

DSMIL from Li et al. 2021 ("Dual-stream Multiple Instance Learning Network for
Whole Slide Image Classification with Self-supervised Contrastive Learning",
https://arxiv.org/abs/2011.08939, repo https://github.com/binli123/dsmil-wsi),
wrapped as a per-split MIL baseline with the same
``forward(X_train, y_train, X_test) -> logits`` interface as
:mod:`icmil.baselines.abmil_baseline`.

A ``patch_embed`` MLP feeds two streams whose logits are averaged:

1. **Instance stream** — a per-instance linear classifier, max-pooled over the
   instance axis.
2. **Bag stream** — attention of every instance against the "critical" instance
   per class (the one with the highest instance-stream score for that class),
   pooling one bag embedding per class.

HP selection matches :class:`~icmil.baselines.abmil_baseline.ABMILBaseline` and
:class:`~icmil.baselines.acmil_baseline.ACMILBaseline`: every ``(lr, wd)``
combination is trained with :class:`torch.optim.Adam` and early stopping on
validation bag-CE over a stratified held-out split, and the combination reaching
the lowest validation CE is kept, at its best-CE checkpoint. All three rows
therefore differ only in architecture. The spread reported in the benchmark table
comes from running several seeds.

Standalone usage::

    model = DSMIL(in_dim=1024, num_classes=2)
    logits = model(features)            # features: (B, M, D)
"""

from __future__ import annotations

import itertools
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn

from icmil.baselines.abmil_baseline import _strip_trailing_zeros
from icmil.mil_pooling import create_mlp

logger = logging.getLogger(__name__)


class BClassifier(nn.Module):
    """DSMIL bag stream: attention of every instance against the critical instances.

    The critical instance for class ``c`` is the one with the highest
    instance-stream score for ``c``. Queries are compared against those
    critical-instance queries, softmaxed over the instance axis, and used to
    pool values into one bag embedding per class.
    """

    def __init__(self, in_dim: int, attn_dim: int = 384, dropout: float = 0.0) -> None:
        super().__init__()
        self.q = nn.Linear(in_dim, attn_dim)
        self.v = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, in_dim))
        self.norm = nn.LayerNorm(in_dim)

    def forward(
        self,
        h: torch.Tensor,
        c: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``h`` is ``(B, M, E)``, ``c`` is ``(B, M, C)``; returns ``((B, C, E), (B, M, C))``."""
        v = self.v(h)  # (B, M, E)
        q = self.q(h)  # (B, M, A)

        crit_idx = c.argmax(dim=1)  # (B, C) — top-scoring instance per class
        crit_feats = torch.gather(h, 1, crit_idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))  # (B, C, E)
        q_crit = self.q(crit_feats)  # (B, C, A)

        a = torch.bmm(q, q_crit.transpose(1, 2))  # (B, M, C)
        if attn_mask is not None:
            a = a + (1 - attn_mask).unsqueeze(-1) * torch.finfo(a.dtype).min
        a = F.softmax(a / math.sqrt(q.shape[-1]), dim=1)  # over instances

        bag = torch.bmm(a.transpose(1, 2), v)  # (B, C, E)
        return self.norm(bag), a


class DSMIL(nn.Module):
    """Dual-stream MIL: patch-embed MLP -> instance stream + bag stream -> averaged logits."""

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        dropout: float = 0.0,
        attn_dim: int = 384,
        dropout_v: float = 0.0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
        )
        self.i_classifier = nn.Linear(embed_dim, num_classes)
        self.b_classifier = BClassifier(in_dim=embed_dim, attn_dim=attn_dim, dropout=dropout_v)
        # Per-class 1D conv over the embedding axis: one weight vector per class,
        # as in the reference implementation.
        self.classifier = nn.Conv1d(num_classes, num_classes, kernel_size=embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear | nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass. ``h`` is ``(B, M, D)``; returns ``(B, num_classes)``."""
        h = self.patch_embed(h)  # (B, M, E)
        inst_logits = self.i_classifier(h)  # (B, M, C)
        if attn_mask is not None:
            inst_logits = inst_logits + (1 - attn_mask).unsqueeze(-1) * torch.finfo(inst_logits.dtype).min
        bag_feats, _ = self.b_classifier(h, inst_logits, attn_mask=attn_mask)  # (B, C, E)

        bag_logits = self.classifier(bag_feats).squeeze(-1)  # (B, C)
        max_inst_logits = inst_logits.max(dim=1).values  # (B, C)
        return 0.5 * (bag_logits + max_inst_logits)


class DSMILBaseline(nn.Module):
    """DSMIL fitted per split, with (lr, wd) selected on a held-out validation split.

    When ``val_fraction > 0`` every combination in ``lr_grid x wd_grid`` is
    trained with early stopping (``patience``) and the one reaching the lowest
    validation CE is kept. When ``val_fraction == 0`` the first entry of each grid
    is used and the model trains on all context bags for ``epochs``.
    """

    def __init__(
        self,
        max_classes: int = 2,
        embed_dim: int = 512,
        attn_dim: int = 384,
        num_fc_layers: int = 1,
        dropout: float = 0.0,
        dropout_v: float = 0.0,
        lr_grid: tuple[float, ...] = (0.01, 0.005, 0.001, 0.0005, 0.0001),
        wd_grid: tuple[float, ...] = (0.0, 0.0001, 0.0005),
        epochs: int = 200,
        patience: int = 20,
        batch_size: int | None = None,
        seed: int = 0,
        val_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_classes = max_classes
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.num_fc_layers = num_fc_layers
        self.dropout = dropout
        self.dropout_v = dropout_v
        self.lr_grid = lr_grid
        self.wd_grid = wd_grid
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.val_fraction = val_fraction

    def train(self, mode: bool = True) -> DSMILBaseline:
        return super().train(False)

    def _make_model(self, in_dim: int, device: torch.device) -> DSMIL:
        return DSMIL(
            in_dim=in_dim,
            embed_dim=self.embed_dim,
            num_fc_layers=self.num_fc_layers,
            dropout=self.dropout,
            attn_dim=self.attn_dim,
            dropout_v=self.dropout_v,
            num_classes=self.max_classes,
        ).to(device)

    def _train_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        in_dim: int,
        lr: float,
        wd: float,
        X_va: torch.Tensor | None = None,
        y_va: torch.Tensor | None = None,
    ) -> tuple[DSMIL, float]:
        """Train one DSMIL with Adam; early-stop on val CE when val data provided.

        Returns the model at the best val CE checkpoint and that CE value.
        When no val data is given, trains for the full ``epochs`` and returns inf.
        """
        torch.manual_seed(self.seed)
        model = self._make_model(in_dim, X.device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        n = X.shape[0]
        bs = n if self.batch_size is None else min(self.batch_size, n)

        best_val_ce = float("inf")
        best_state: dict = {}
        no_improve = 0

        for epoch in range(self.epochs):
            model.train()
            perm = torch.randperm(n, device=X.device)
            for start in range(0, n, bs):
                idx = perm[start : start + bs]
                loss = F.cross_entropy(model(X[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

            if X_va is not None:
                model.eval()
                with torch.no_grad():
                    val_ce = F.cross_entropy(model(X_va), y_va).item()
                if val_ce < best_val_ce:
                    best_val_ce = val_ce
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self.patience:
                    logger.info("  early stop at epoch %d", epoch + 1)
                    break

        if best_state:
            model.load_state_dict(best_state)
        return model.eval(), best_val_ce

    def _fit_predict(self, X_tr: torch.Tensor, y_tr: torch.Tensor, X_te: torch.Tensor) -> torch.Tensor:
        """Select (lr, wd) on val split (or use grid[0] when val_fraction=0) → log_softmax on test."""
        combined = _strip_trailing_zeros(torch.cat([X_tr, X_te], dim=0))
        in_dim = combined.shape[-1]
        n_tr = X_tr.shape[0]
        X_tr = combined[:n_tr].contiguous()
        X_te = combined[n_tr:].contiguous()

        if self.val_fraction > 0.0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_fraction, random_state=self.seed)
            tr_idx_np, va_idx_np = next(sss.split(np.zeros(len(y_tr)), y_tr.cpu().numpy()))
            tr_idx = torch.as_tensor(tr_idx_np, device=X_tr.device, dtype=torch.long)
            va_idx = torch.as_tensor(va_idx_np, device=X_tr.device, dtype=torch.long)
            X_tr_s, y_tr_s = X_tr[tr_idx], y_tr[tr_idx]
            X_va, y_va = X_tr[va_idx], y_tr[va_idx]

            best_val_ce = float("inf")
            best_model: DSMIL | None = None
            for lr, wd in itertools.product(self.lr_grid, self.wd_grid):
                logger.info("DSMIL: training lr=%g wd=%g", lr, wd)
                m, val_ce = self._train_model(X_tr_s, y_tr_s, in_dim, lr, wd, X_va, y_va)
                if val_ce < best_val_ce:
                    best_val_ce = val_ce
                    best_model = m
        else:
            best_model, _ = self._train_model(X_tr, y_tr, in_dim, self.lr_grid[0], self.wd_grid[0])

        with torch.no_grad():
            return F.log_softmax(best_model(X_te), dim=-1)

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """Per-batch wrapper: run ``_fit_predict`` over the batch axis."""
        return torch.stack(
            [self._fit_predict(X_train[b], y_train[b], X_test[b]) for b in range(X_train.shape[0])],
            dim=0,
        )
