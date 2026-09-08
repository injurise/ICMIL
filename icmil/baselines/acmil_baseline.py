"""ACMIL baseline (Attention-Challenging Multiple Instance Learning).

ACMIL-GA (gated-attention variant) from Zhang et al. 2023
("Attention-Challenging Multiple Instance Learning for Whole Slide Image
Classification", https://arxiv.org/abs/2311.07125, repo
https://github.com/dazhangyu123/ACMIL), wrapped as a per-split MIL baseline
with the same ``forward(X_train, y_train, X_test) -> logits`` interface as
:mod:`icmil.baselines.abmil_baseline`.

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

HP selection matches :class:`~icmil.baselines.abmil_baseline.ABMILBaseline` and
:class:`~icmil.baselines.dsmil_baseline.DSMILBaseline`: ``(lr, wd, dropout)``
picked on a stratified held-out validation split with :class:`torch.optim.Adam`
and early stopping, so the three rows differ only in architecture. The spread
reported in the benchmark table comes from running several seeds.

Standalone usage::

    model = ACMILGatedAttention(in_dim=1024, num_classes=2, n_token=5)
    sub_logits, slide_logits, attn = model(bags)   # bags: (B, M, D)
"""

from __future__ import annotations

import copy
import itertools
import logging

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn

from icmil.baselines.abmil_baseline import _strip_trailing_zeros

logger = logging.getLogger(__name__)


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


class ACMILBaseline(nn.Module):
    """ACMIL-GA fitted per split, with (lr, wd, dropout) selected on a held-out split.

    When ``val_fraction > 0`` every combination in
    ``lr_grid x wd_grid x dropout_grid`` is trained with early stopping
    (``patience``) on one stratified validation split and the one reaching the
    lowest validation CE is kept, at its best-CE checkpoint. When
    ``val_fraction == 0`` the first entry of each grid is used and the model
    trains on all context bags for ``epochs``.

    The search space and optimizer match
    :class:`~icmil.baselines.abmil_baseline.ABMILBaseline` and
    :class:`~icmil.baselines.dsmil_baseline.DSMILBaseline` — the same
    ``lr_grid``/``wd_grid``, dropout fixed at 0 by default (``dropout_grid=(0.0,)``
    — pass more values to sweep it), and :class:`torch.optim.Adam` with L2-style
    weight decay and no schedule — so ``acmil`` vs ``abmil`` vs ``dsmil`` isolates
    architecture only.
    """

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
        seed: int = 0,
        lr_grid: tuple[float, ...] = (0.01, 0.005, 0.001, 0.0005, 0.0001),
        wd_grid: tuple[float, ...] = (0.0, 0.0001, 0.0005),
        dropout_grid: tuple[float, ...] = (0.0,),
        patience: int = 20,
        min_delta: float = 1e-4,
        val_fraction: float = 0.1,
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
        self.seed = seed
        self.lr_grid = lr_grid
        self.wd_grid = wd_grid
        self.dropout_grid = dropout_grid
        self.patience = patience
        self.min_delta = min_delta
        self.val_fraction = val_fraction

    def train(self, mode: bool = True) -> ACMILBaseline:
        """No-op so the eval harness cannot flip the wrapper into train mode.

        The *inner* :class:`ACMILGatedAttention` still toggles train/eval
        normally, which is what enables and disables stochastic masking.
        """
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
    ) -> ACMILGatedAttention:
        """Train one ACMIL model on the composite loss.

        With ``X_val``/``y_val`` runs early stopping on val bag-CE (masking off
        for the val forward) and returns the best-val checkpoint. Otherwise
        trains ``self.epochs`` with no early stopping.
        """
        torch.manual_seed(seed)
        model = self._make_model(in_dim, X.device, dropout)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        n = X.shape[0]
        bs = min(self.batch_size, n)

        do_es = X_val is not None and y_val is not None
        best_val = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        wait = 0

        model.train()
        for _epoch in range(self.epochs):
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
                with torch.no_grad():
                    val_loss = F.cross_entropy(model(X_val)[1], y_val).item()
                model.train()
                if val_loss < best_val - self.min_delta:
                    best_val = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    wait = 0
                else:
                    wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model.eval()

    @torch.no_grad()
    def _slide_logits(self, model: ACMILGatedAttention, X: torch.Tensor) -> torch.Tensor:
        """Bag-level logits with masking off (``model`` is in eval mode)."""
        return model(X)[1]

    def _fit_predict(self, X_tr: torch.Tensor, y_tr: torch.Tensor, X_te: torch.Tensor) -> torch.Tensor:
        """Select the combo on a val split (or use grid[0]s when val_fraction=0) → log_softmax on test."""
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
            best_model: ACMILGatedAttention | None = None
            for lr, wd, dropout in itertools.product(self.lr_grid, self.wd_grid, self.dropout_grid):
                logger.info("ACMIL: training lr=%g wd=%g dropout=%g", lr, wd, dropout)
                m = self._train_once(X_tr_s, y_tr_s, in_dim, lr, wd, dropout, seed=self.seed, X_val=X_va, y_val=y_va)
                val_ce = F.cross_entropy(self._slide_logits(m, X_va), y_va).item()
                if val_ce < best_val_ce:
                    best_val_ce = val_ce
                    best_model = m
        else:
            best_model = self._train_once(
                X_tr, y_tr, in_dim, self.lr_grid[0], self.wd_grid[0], self.dropout_grid[0], seed=self.seed
            )

        return F.log_softmax(self._slide_logits(best_model, X_te), dim=-1)

    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
        """Per-batch wrapper: run ``_fit_predict`` over the batch axis."""
        return torch.stack(
            [self._fit_predict(X_train[b], y_train[b], X_test[b]) for b in range(X_train.shape[0])],
            dim=0,
        )
