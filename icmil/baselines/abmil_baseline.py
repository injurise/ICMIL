"""ABMIL baseline.

Attention-based MIL (Ilse et al., 2018) — patch-embedding MLP → gated attention
pooling → linear head — wrapped as a per-split baseline that fits from scratch
on each ``(X_train, y_train)`` and predicts on ``X_test``.

:class:`ABMILBaseline` selects ``(lr, wd)`` on a stratified held-out split of the
context bags: every combination is trained with :class:`torch.optim.Adam` and
early stopping on validation bag-CE, and the combination reaching the lowest
validation CE is kept, at its best-CE checkpoint.
:mod:`icmil.baselines.acmil_baseline` and :mod:`icmil.baselines.dsmil_baseline`
fit the same way, so the three rows differ only in architecture.

The model is deterministic given ``seed``; the spread reported in the benchmark
table comes from :mod:`icmil.reproduce` running several seeds.

Standalone usage::

    model = ABMIL(in_dim=1024, num_classes=2)
    logits = model(features)            # features: (B, M, D)
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn

from icmil.mil_pooling import GlobalAttention, GlobalGatedAttention, create_mlp

logger = logging.getLogger(__name__)


def _strip_trailing_zeros(X: torch.Tensor) -> torch.Tensor:
    """Remove trailing all-zero feature columns from a ``(..., F)`` tensor.

    The eval harness right-pads features to ``max_features``; stripping the
    pad keeps the patch-embed MLP from wasting capacity on zeros.
    """
    active = X.abs().sum(dim=tuple(range(X.ndim - 1))) > 0
    if not active.any():
        return X[..., :1]
    last = active.nonzero(as_tuple=True)[0].max().item()
    return X[..., : last + 1]


class ABMIL(nn.Module):
    """Attention-based MIL: patch-embed MLP -> gated attention pooling -> classifier."""

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        dropout: float = 0.25,
        attn_dim: int = 384,
        gate: bool = True,
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
        attn = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn(L=embed_dim, D=attn_dim, dropout=dropout, num_classes=1)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass. ``h`` is ``(B, M, D)``; returns ``(B, num_classes)``."""
        h = self.patch_embed(h)  # (B, M, E)
        a = self.global_attn(h)  # (B, M, 1)
        a = a.transpose(-2, -1)  # (B, 1, M)
        if attn_mask is not None:
            a = a + (1 - attn_mask).unsqueeze(1) * torch.finfo(a.dtype).min
        a = F.softmax(a, dim=-1)
        bag = torch.bmm(a, h).squeeze(1)  # (B, E)
        return self.classifier(bag)


class ABMILBaseline(nn.Module):
    """ABMIL fitted per split, with (lr, wd) selected on a held-out validation split.

    Training loop follows Ilse et al. 2018 (https://arxiv.org/abs/1802.04712).
    When ``val_fraction > 0`` every combination in ``lr_grid x wd_grid`` is
    trained with early stopping (``patience``) and the one reaching the lowest
    validation CE is kept. When ``val_fraction == 0`` the first entry of each grid
    is used and the model trains on all context bags for ``epochs``.

    With ``refit=True`` the selected early-stopped checkpoint is discarded and one
    model is retrained from scratch on all context bags (train + validation) at
    the winning ``(lr, wd)`` for the winner's best epoch, with no early stopping.
    """

    def __init__(
        self,
        max_classes: int = 2,
        embed_dim: int = 500,
        attn_dim: int = 128,
        num_fc_layers: int = 1,
        dropout: float = 0.0,
        gate: bool = True,
        lr_grid: tuple[float, ...] = (0.01, 0.005, 0.001, 0.0005, 0.0001),
        wd_grid: tuple[float, ...] = (0.0, 0.0001, 0.0005),
        epochs: int = 200,
        patience: int = 20,
        batch_size: int | None = None,
        seed: int = 0,
        val_fraction: float = 0.1,
        refit: bool = False,
    ) -> None:
        super().__init__()
        self.max_classes = max_classes
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.num_fc_layers = num_fc_layers
        self.dropout = dropout
        self.gate = gate
        self.lr_grid = lr_grid
        self.wd_grid = wd_grid
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.val_fraction = val_fraction
        self.refit = refit

    def train(self, mode: bool = True) -> ABMILBaseline:
        return super().train(False)

    def _make_model(self, in_dim: int, device: torch.device) -> ABMIL:
        return ABMIL(
            in_dim=in_dim,
            embed_dim=self.embed_dim,
            num_fc_layers=self.num_fc_layers,
            dropout=self.dropout,
            attn_dim=self.attn_dim,
            gate=self.gate,
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
        max_epochs: int | None = None,
    ) -> tuple[ABMIL, float, int]:
        """Train one ABMIL with Adam; early-stop on val CE when val data provided.

        Returns the model at the best val CE checkpoint, that CE value and the
        1-indexed epoch of that checkpoint. When no val data is given, trains for
        ``max_epochs`` (default ``epochs``) and returns inf and the epoch budget.
        """
        budget = self.epochs if max_epochs is None else max_epochs
        torch.manual_seed(self.seed)
        model = self._make_model(in_dim, X.device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        n = X.shape[0]
        bs = n if self.batch_size is None else min(self.batch_size, n)

        best_val_ce = float("inf")
        best_state: dict = {}
        best_epoch = budget
        no_improve = 0

        for epoch in range(budget):
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
                    best_epoch = epoch + 1
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self.patience:
                    logger.info("  early stop at epoch %d", epoch + 1)
                    break

        if best_state:
            model.load_state_dict(best_state)
        return model.eval(), best_val_ce, best_epoch

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
            best_model: ABMIL | None = None
            best_hp: tuple[float, float, int] | None = None
            for lr, wd in itertools.product(self.lr_grid, self.wd_grid):
                logger.info("ABMIL: training lr=%g wd=%g", lr, wd)
                m, val_ce, best_epoch = self._train_model(X_tr_s, y_tr_s, in_dim, lr, wd, X_va, y_va)
                if val_ce < best_val_ce:
                    best_val_ce = val_ce
                    best_model = m
                    best_hp = (lr, wd, best_epoch)
            if self.refit and best_hp is not None:
                lr, wd, best_epoch = best_hp
                logger.info("ABMIL refit: lr=%g wd=%g for %d epochs on all context bags", lr, wd, best_epoch)
                best_model, _, _ = self._train_model(X_tr, y_tr, in_dim, lr, wd, max_epochs=best_epoch)
        else:
            best_model, _, _ = self._train_model(X_tr, y_tr, in_dim, self.lr_grid[0], self.wd_grid[0])

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
