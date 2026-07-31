"""ACMIL baseline (Attention-Challenging Multiple Instance Learning).

ACMIL-GA (gated-attention variant) from Zhang et al. 2023
("Attention-Challenging Multiple Instance Learning for Whole Slide Image
Classification", https://arxiv.org/abs/2311.07125, repo
https://github.com/dazhangyu123/ACMIL), wrapped as a per-split MIL baseline
with the same ``forward(X_train, y_train, X_test) -> logits`` interface and
K-fold-CV-then-refit fitting as :mod:`icmil.baselines.abmil_baseline`.

ACMIL extends gated-attention ABMIL with three "attention-challenging" tricks:

1. **Multiple Branch Attention** — ``n_token`` parallel gated-attention branches
   over the instances, each with its own classifier head. The bag prediction
   pools instances by the mean attention across branches.
2. **Stochastic Top-K masking** — during training only, for each (bag, branch)
   the top ``n_masked_patch`` attention instances are found and a random
   ``int(n_masked_patch * mask_drop)`` of them are masked out (attention logit
   set to ``-1e9`` before softmax). This stops the model over-relying on a few
   high-attention instances.
3. **Composite loss** — ``bag_ce + branch_ce + diff_loss`` where ``branch_ce`` is
   cross-entropy on the per-branch logits (only when ``n_token > 1``) and
   ``diff_loss`` is the mean pairwise cosine similarity of the branch attention
   maps (a diversity penalty pushing branches to attend to different instances).

HP selection mirrors :class:`~icmil.baselines.abmil_baseline.ABMILRefitBaseline`
exactly: bag-level 5-fold CV over an ``(lr, wd, dropout)`` grid scored by mean
cross-fold val bag-CE (with per-fold early stopping), then a single refit on the
full ``X_train`` for ``e_bar = ceil(mean(best_epochs_per_fold))`` epochs. The spread
reported in the benchmark table comes from running several seeds.

Standalone usage::

    model = ACMILGatedAttention(in_dim=1024, num_classes=2, n_token=5)
    sub_logits, slide_logits, attn = model(bags)   # bags: (B, M, D)
"""

from __future__ import annotations

import copy
import itertools
import logging
import math

import numpy as np
import schedulefree
import torch
import torch.nn.functional as F
from torch import nn

from icmil.baselines.abmil_baseline import _strip_trailing_zeros
from icmil.baselines.tabpfn_baselines import _bag_level_stratified_folds

logger = logging.getLogger(__name__)

HPCombo = tuple[float, float, float]


class ACMILGatedAttention(nn.Module):
    """Batched ACMIL-GA: dim-reduction -> multi-branch gated attention -> heads.

    Stochastic top-K masking is applied only when ``self.training`` is ``True``.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 256,
        attn_dim: int = 128,
        dropout: float = 0.0,
        n_token: int = 5,
        n_masked_patch: int = 10,
        mask_drop: float = 0.6,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.n_token = n_token
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop

        # DimReduction: Linear(no bias) + ReLU (matches reference `DimReduction`).
        self.dim_reduction = nn.Sequential(
            nn.Linear(in_dim, embed_dim, bias=False),
            nn.ReLU(inplace=True),
        )
        # Gated attention producing `n_token` branch scores per instance.
        self.attention_v = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(attn_dim, n_token)
        # One classifier per branch + one slide-level classifier.
        self.branch_classifiers = nn.ModuleList(
            [_Classifier1fc(embed_dim, num_classes, dropout) for _ in range(n_token)]
        )
        self.slide_classifier = _Classifier1fc(embed_dim, num_classes, dropout)

    def _attention_logits(self, h: torch.Tensor) -> torch.Tensor:
        """``h``: ``(B, M, E)`` -> attention logits ``(B, n_token, M)``."""
        a = self.attention_weights(self.attention_v(h) * self.attention_u(h))  # (B, M, K)
        return a.transpose(-2, -1)  # (B, K, M)

    def _apply_stochastic_mask(self, a: torch.Tensor) -> torch.Tensor:
        """Randomly mask a subset of each branch's top-K attention instances."""
        b, k, m = a.shape
        n_masked = min(self.n_masked_patch, m)
        n_drop = int(n_masked * self.mask_drop)
        if n_drop <= 0:
            return a
        _, top_idx = torch.topk(a, n_masked, dim=-1)  # (B, K, n_masked)
        rand = torch.rand(b, k, n_masked, device=a.device)
        rand_selected = torch.argsort(rand, dim=-1)[..., :n_drop]  # (B, K, n_drop)
        masked_idx = torch.gather(top_idx, -1, rand_selected)  # (B, K, n_drop)
        keep = torch.ones(b, k, m, device=a.device)
        keep.scatter_(-1, masked_idx, 0.0)
        return a.masked_fill(keep == 0, -1e9)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x``: ``(B, M, in_dim)`` mini-batch of bags."""
        h = self.dim_reduction(x)  # (B, M, E)
        a = self._attention_logits(h)  # (B, K, M)
        if self.n_masked_patch > 0 and self.training:
            a = self._apply_stochastic_mask(a)

        attn = F.softmax(a, dim=-1)  # (B, K, M)
        branch_feat = torch.bmm(attn, h)  # (B, K, E)
        sub_logits = torch.stack(
            [head(branch_feat[:, i]) for i, head in enumerate(self.branch_classifiers)],
            dim=1,
        )  # (B, K, C)

        bag_attn = attn.mean(dim=1, keepdim=True)  # (B, 1, M)
        bag_feat = torch.bmm(bag_attn, h).squeeze(1)  # (B, E)
        slide_logits = self.slide_classifier(bag_feat)  # (B, C)
        return sub_logits, slide_logits, attn


class _Classifier1fc(nn.Module):
    """Optional dropout + single linear layer (reference ``Classifier_1fc``)."""

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc(x)


def acmil_composite_loss(
    sub_logits: torch.Tensor,
    slide_logits: torch.Tensor,
    attn: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """ACMIL training loss: ``bag_ce + branch_ce + diff_loss``."""
    b, k, c = sub_logits.shape
    loss_bag = F.cross_entropy(slide_logits, y)
    if k <= 1:
        return loss_bag

    loss_branch = F.cross_entropy(sub_logits.reshape(b * k, c), y.repeat_interleave(k))
    diff = slide_logits.new_zeros(())
    n_pairs = 0
    for i in range(k):
        for j in range(i + 1, k):
            diff = diff + F.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean()
            n_pairs += 1
    diff_loss = diff / max(n_pairs, 1)
    return loss_bag + loss_branch + diff_loss


class ACMILRefitBaseline(nn.Module):
    """ACMIL-GA with K-fold CV for HP scoring, then a single refit on full X_train."""

    def __init__(
        self,
        max_classes: int = 4,
        embed_dim: int = 256,
        attn_dim: int = 128,
        n_token: int = 5,
        n_masked_patch: int = 10,
        mask_drop: float = 0.6,
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
        self.n_token = n_token
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop
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

    def train(self, mode: bool = True) -> ACMILRefitBaseline:
        return super().train(False)

    def _make_model(self, in_dim: int, device: torch.device, dropout: float) -> ACMILGatedAttention:
        return ACMILGatedAttention(
            in_dim=in_dim,
            embed_dim=self.embed_dim,
            attn_dim=self.attn_dim,
            dropout=dropout,
            n_token=self.n_token,
            n_masked_patch=self.n_masked_patch,
            mask_drop=self.mask_drop,
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
    ) -> tuple[ACMILGatedAttention, int]:
        """Train one ACMIL model; return ``(model, best_epoch)``.

        With ``X_val``/``y_val`` runs early stopping on val bag-CE (masking off
        for the val forward) and returns the best-val checkpoint plus its
        1-indexed epoch. Otherwise trains ``max_epochs`` (default ``self.epochs``)
        with no early stopping.
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
        best_epoch_idx = 0
        wait = 0

        model.train()
        opt.train()
        for epoch in range(budget):
            perm = torch.randperm(n, device=X.device)
            for start in range(0, n, bs):
                idx = perm[start : start + bs]
                sub_logits, slide_logits, attn = model(X[idx])
                loss = acmil_composite_loss(sub_logits, slide_logits, attn, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

            if do_es:
                model.eval()
                opt.eval()
                with torch.no_grad():
                    val_loss = F.cross_entropy(model(X_val)[1], y_val).item()
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
    def _slide_logits(self, model: ACMILGatedAttention, X: torch.Tensor) -> torch.Tensor:
        """Bag-level logits with masking off (``model`` is in eval mode)."""
        return model(X)[1]

    def _make_folds(self, y: torch.Tensor, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        np_folds = _bag_level_stratified_folds(y.cpu().numpy(), self.n_cv_splits, seed=seed)
        return [
            (
                torch.as_tensor(tr, device=y.device, dtype=torch.long),
                torch.as_tensor(va, device=y.device, dtype=torch.long),
            )
            for tr, va in np_folds
        ]

    def _cv_train(self, X: torch.Tensor, y: torch.Tensor, in_dim: int, seed: int) -> tuple[HPCombo, int]:
        """Bag-level K-fold CV over the (lr, wd, dropout) grid, scored by mean val bag-CE.

        Returns the winning combo and ``e_bar = ceil(mean(best_epochs))`` for the
        refit pass. When stratified CV is infeasible (single class or a class with
        <2 samples) returns the grid-centre combo and the full epoch budget.
        """
        folds = self._make_folds(y, seed=seed)
        combos = list(itertools.product(self.lr_grid, self.wd_grid, self.dropout_grid))
        if not folds:
            return combos[len(combos) // 2], self.epochs

        per_outer = len(combos) * len(folds)
        ces: dict[HPCombo, list[float]] = {c: [] for c in combos}
        best_epochs: dict[HPCombo, list[int]] = {c: [] for c in combos}

        for combo_idx, (lr, wd, dropout) in enumerate(combos):
            for fold_idx, (tr, va) in enumerate(folds):
                fit_seed = seed * per_outer + combo_idx * len(folds) + fold_idx
                m, best_epoch = self._train_once(
                    X[tr], y[tr], in_dim, lr, wd, dropout, fit_seed, X_val=X[va], y_val=y[va]
                )
                ces[(lr, wd, dropout)].append(F.cross_entropy(self._slide_logits(m, X[va]), y[va]).item())
                best_epochs[(lr, wd, dropout)].append(best_epoch)

        means = {c: float(np.mean(v)) for c, v in ces.items()}
        best = min(means, key=means.get)
        e_bar = max(1, math.ceil(float(np.mean(best_epochs[best]))))
        logger.info(
            "ACMIL CV (seed=%d) picked lr=%g wd=%g dropout=%g -> e_bar=%d (mean val CE per combo: %s)",
            seed,
            *best,
            e_bar,
            means,
        )
        return best, e_bar

    def _fit_predict(self, X_tr: torch.Tensor, y_tr: torch.Tensor, X_te: torch.Tensor) -> torch.Tensor:
        """Run CV, refit on full train, return ``log_softmax(slide_logits)`` for X_test."""
        combined = _strip_trailing_zeros(torch.cat([X_tr, X_te], dim=0))
        in_dim = combined.shape[-1]
        n_tr = X_tr.shape[0]
        X_tr = combined[:n_tr].contiguous()
        X_te = combined[n_tr:].contiguous()

        (lr, wd, dropout), e_bar = self._cv_train(X_tr, y_tr, in_dim, seed=self.seed)
        model, _ = self._train_once(X_tr, y_tr, in_dim, lr, wd, dropout, seed=self.seed, max_epochs=e_bar)
        return F.log_softmax(self._slide_logits(model, X_te), dim=-1)

    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
        """Per-batch wrapper: run ``_fit_predict`` over the batch axis."""
        return torch.stack(
            [self._fit_predict(X_train[b], y_train[b], X_test[b]) for b in range(X_train.shape[0])],
            dim=0,
        )
