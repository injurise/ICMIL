"""ABMIL baseline.

Attention-based MIL (Ilse et al., 2018) — patch-embedding MLP → gated attention
pooling → linear head — wrapped as a per-split baseline that fits from scratch
on each ``(X_train, y_train)`` and predicts on ``X_test``.

:class:`ABMILRefitBaseline` selects the hyper-parameters ``(lr, wd, dropout)`` by
mean cross-fold validation cross-entropy under K-fold CV with per-fold early
stopping (best-val-CE checkpoint, governed by ``patience`` and ``min_delta``).
It then discards the fold models and refits a single model on the full
``X_train`` at the winning combination for ``ē = ceil(mean(best_epochs_per_fold))``
epochs without early stopping. Note ``self.epochs`` caps the per-fold search, not
that final refit.

The model is deterministic given ``seed``; the spread reported in the benchmark
table comes from :mod:`icmil.reproduce` running several seeds.

Cross-validation is bag-level: the inner split operates on the bag axis of
``y_train`` via :func:`icmil.baselines.tabpfn_baselines._bag_level_stratified_folds`,
so no instance leaks between folds.

Standalone usage::

    model = ABMIL(in_dim=1024, num_classes=2)
    logits = model(features)            # features: (B, M, D)
"""

from __future__ import annotations

import copy
import itertools
import logging
import math
from typing import Literal

import numpy as np
import schedulefree
import torch
import torch.nn.functional as F
from torch import nn

from icmil.baselines.tabpfn_baselines import _bag_level_stratified_folds
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


HPCombo = tuple[float, float, float]


class _ABMILBaselineBase(nn.Module):
    """Shared internals for ABMIL MIL baselines.

    Defaults: gated attention, ``attn_dim=128``, mini-batch ``batch_size=32``,
    ``epochs=200`` (max budget — per-fold early stopping with ``patience=20``
    typically stops earlier), 5-fold CV over
    ``lr_grid x wd_grid x dropout_grid = (1e-3, 1e-4) x (0, 0.01) x (0, 0.1)``.

    See the module docstring for the pipeline.
    """

    mode: Literal["refit", "ensemble"]

    def __init__(
        self,
        max_classes: int = 4,
        embed_dim: int = 256,
        attn_dim: int = 128,
        num_fc_layers: int = 1,
        gate: bool = True,
        epochs: int = 200,
        batch_size: int = 32,
        warmup_steps: int = 20,
        seed: int = 0,
        lr_grid: tuple[float, ...] = (1e-3, 1e-4),
        wd_grid: tuple[float, ...] = (0.0, 0.01),
        dropout_grid: tuple[float, ...] = (0.0, 0.1),
        n_cv_splits: int = 5,
        patience: int = 20,
        min_delta: float = 1e-4,
    ) -> None:
        super().__init__()
        self.max_classes = max_classes
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.num_fc_layers = num_fc_layers
        self.gate = gate
        self.epochs = epochs
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.seed = seed
        self.lr_grid = lr_grid
        self.wd_grid = wd_grid
        self.dropout_grid = dropout_grid
        self.n_cv_splits = n_cv_splits
        self.patience = patience
        self.min_delta = min_delta

    def train(self, mode: bool = True) -> _ABMILBaselineBase:
        return super().train(False)

    def _make_model(self, in_dim: int, device: torch.device, dropout: float) -> ABMIL:
        return ABMIL(
            in_dim=in_dim,
            embed_dim=self.embed_dim,
            num_fc_layers=self.num_fc_layers,
            dropout=dropout,
            attn_dim=self.attn_dim,
            gate=self.gate,
            num_classes=self.max_classes,
        ).to(device)

    def _train_once(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        in_dim: int,
        lr: float,
        wd: float,
        dropout: float,
        seed: int,
        X_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
        max_epochs: int | None = None,
    ) -> tuple[ABMIL, int]:
        """Train one ABMIL; return ``(model, best_epoch)``.

        If both ``X_val`` and ``y_val`` are provided, run early stopping on
        val CE (patience ``self.patience``, ``self.min_delta``) and return
        the best-val-CE checkpoint plus the 1-indexed epoch it was last
        updated at. Otherwise train ``max_epochs`` (default ``self.epochs``)
        with no early stopping and return the final model and ``max_epochs``.
        """
        budget = max_epochs if max_epochs is not None else self.epochs
        torch.manual_seed(seed)
        model = self._make_model(in_dim, X.device, dropout)
        opt = schedulefree.AdamWScheduleFree(
            model.parameters(), lr=lr, weight_decay=wd, warmup_steps=self.warmup_steps
        )
        n = X.shape[0]
        bs = min(self.batch_size, n)

        do_es = X_val is not None and y_val is not None
        best_val = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch_idx = 0  # 0-based epoch index when best_state was last updated
        wait = 0

        model.train()
        opt.train()
        for epoch in range(budget):
            perm = torch.randperm(n, device=X.device)
            for start in range(0, n, bs):
                idx = perm[start : start + bs]
                logits = model(X[idx])
                loss = F.cross_entropy(logits, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

            if do_es:
                model.eval()
                opt.eval()
                with torch.no_grad():
                    val_loss = F.cross_entropy(model(X_val), y_val).item()
                model.train()
                opt.train()
                if val_loss < best_val - self.min_delta:
                    best_val = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch_idx = epoch
                    wait = 0
                else:
                    wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        opt.eval()
        best_epoch = best_epoch_idx + 1 if do_es else budget
        return model.eval(), best_epoch

    @torch.no_grad()
    def _logits(self, model: ABMIL, X: torch.Tensor) -> torch.Tensor:
        return model(X)

    def _make_folds(self, y: torch.Tensor, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        np_folds = _bag_level_stratified_folds(y.cpu().numpy(), self.n_cv_splits, seed=seed)
        return [
            (
                torch.as_tensor(tr, device=y.device, dtype=torch.long),
                torch.as_tensor(va, device=y.device, dtype=torch.long),
            )
            for tr, va in np_folds
        ]

    def _cv_train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        in_dim: int,
        seed: int,
    ) -> tuple[HPCombo, list[ABMIL] | int]:
        """K-fold CV over the (lr, wd, dropout) grid.

        Per-fold training always runs early stopping on the held-out fold
        (best-val-CE checkpoint, ``patience`` / ``min_delta``). The grid is
        scored by mean cross-fold val CE. The second return depends on mode:

        * ``mode == 'ensemble'``: the K early-stopped fold models for the
          winning combo, reused as the test-time ensemble.
        * ``mode == 'refit'``: ``e_bar = ceil(mean(best_epochs[best]))``,
          the per-fold-mean best epoch (rounded up). Caller refits one
          model on the full train set for ``e_bar`` epochs, no ES.

        When stratified CV is infeasible (single class or any class with
        <2 samples), returns the grid-centre combo and an empty list so
        the caller falls back to a full-budget refit.
        """
        folds = self._make_folds(y, seed=seed)
        combos = list(itertools.product(self.lr_grid, self.wd_grid, self.dropout_grid))

        if not folds:
            mid = combos[len(combos) // 2]
            return mid, []

        keep = self.mode == "ensemble"
        n_combos = len(combos)
        per_outer = n_combos * len(folds)
        ces: dict[HPCombo, list[float]] = {c: [] for c in combos}
        best_epochs: dict[HPCombo, list[int]] = {c: [] for c in combos}
        models: dict[HPCombo, list[ABMIL]] = {c: [] for c in combos}

        for combo_idx, (lr, wd, dropout) in enumerate(combos):
            for fold_idx, (tr, va) in enumerate(folds):
                fit_seed = seed * per_outer + combo_idx * len(folds) + fold_idx
                m, best_epoch = self._train_once(
                    X[tr],
                    y[tr],
                    in_dim,
                    lr,
                    wd,
                    dropout,
                    fit_seed,
                    X_val=X[va],
                    y_val=y[va],
                )
                ces[(lr, wd, dropout)].append(F.cross_entropy(self._logits(m, X[va]), y[va]).item())
                best_epochs[(lr, wd, dropout)].append(best_epoch)
                if keep:
                    models[(lr, wd, dropout)].append(m)
        means = {c: float(np.mean(v)) for c, v in ces.items()}
        best = min(means, key=means.get)
        logger.info(
            "ABMIL CV (mode=%s, seed=%d) picked lr=%g wd=%g dropout=%g (mean val CE per combo: %s)",
            self.mode,
            seed,
            *best,
            means,
        )
        if keep:
            return best, models[best]
        e_bar = max(1, math.ceil(float(np.mean(best_epochs[best]))))
        logger.info(
            "ABMIL refit (seed=%d): best_epochs per fold=%s -> e_bar=%d (budget=%d)",
            seed,
            best_epochs[best],
            e_bar,
            self.epochs,
        )
        return best, e_bar

    def _fit_predict(self, X_tr: torch.Tensor, y_tr: torch.Tensor, X_te: torch.Tensor) -> torch.Tensor:
        """Run CV, produce test predictions per the active ``mode``; return ``log(mean softmax)``."""
        combined = _strip_trailing_zeros(torch.cat([X_tr, X_te], dim=0))
        in_dim = combined.shape[-1]
        n_tr = X_tr.shape[0]
        X_tr = combined[:n_tr].contiguous()
        X_te = combined[n_tr:].contiguous()

        best, cv_out = self._cv_train(X_tr, y_tr, in_dim, seed=self.seed)
        lr, wd, dropout = best

        if self.mode == "ensemble" and isinstance(cv_out, list) and cv_out:
            members = cv_out
        else:
            # refit: train one model on the full X_train for e_bar epochs, no ES.
            # Fallback (CV infeasible -> cv_out is []) refits for the full budget.
            refit_epochs = cv_out if isinstance(cv_out, int) else self.epochs
            model, _ = self._train_once(X_tr, y_tr, in_dim, lr, wd, dropout, seed=self.seed, max_epochs=refit_epochs)
            members = [model]

        probs = torch.stack(
            [F.softmax(self._logits(m, X_te), dim=-1) for m in members],
            dim=0,
        ).mean(dim=0)
        return torch.log(probs.clamp_min(1e-12))

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


class ABMILRefitBaseline(_ABMILBaselineBase):
    """ABMIL with K-fold CV for HP scoring, then a single refit on full X_train."""

    mode: Literal["refit", "ensemble"] = "refit"
