"""TabPFN-based MIL baselines + a mean-pool logistic regression baseline.

Baseline 1 — TabPFNConcatBaseline / TabPFNSubsampleBaseline:
    Frozen TabPFN v2 backbone over flattened bags. Two sibling classes that
    share machinery via a private base, exposing only the kwargs that apply
    to each variant so the table renderer can key rows by class name:

    - ``TabPFNConcatBaseline`` flattens each bag in original instance order
      and truncates the flat vector at ``max_tabpfn_features``. Single view.
    - ``TabPFNSubsampleBaseline`` draws, per view, one random subset of
      ``n_keep`` instance positions (shared across bags) and flattens just
      those, with ``n_keep`` sized so ``n_keep * F`` fits TabPFN's feature
      budget. Runs ``n_views`` views and aggregates logits across them.

Baseline 2 — MeanLogRegBaseline:
    Mean-pools instances per bag and fits a scikit-learn ``LogisticRegressionCV``
    (or plain ``LogisticRegression`` when stratified CV is infeasible).

Baseline 3 — SVMSummBaseline:
    Aggregates each bag with a fixed set of summary statistics
    (sum, mean, median, min, max, stdev) along the instance axis and
    classifies with a scikit-learn ``SVC`` (RBF kernel, Platt-scaled
    probabilities). ``C`` is selected by stratified K-fold ``GridSearchCV``.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from torch import nn

from icmil.baselines._tabpfn_backbone import FrozenTabPFNBackbone

logger = logging.getLogger(__name__)


def _bag_level_stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Bag-level stratified K-fold indices, clamped to smallest class count.

    ``y`` carries one label per bag; the returned ``(train_idx, val_idx)``
    pairs index the bag axis. ``n_splits`` is clamped by the smallest class
    count (sklearn's hard requirement) with a floor of 2. Returns ``[]``
    when stratified CV is infeasible (single class, or any class with <2
    samples).
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        return []
    k = max(2, min(int(n_splits), int(counts.min())))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(tr.copy(), va.copy()) for tr, va in skf.split(np.zeros(len(y)), y)]


def _strip_trailing_zero_features(X: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Remove trailing all-zero feature columns from per-instance features.

    Args:
        X: Bag features, shape (B, NB, N, F).

    Returns:
        Tuple of (trimmed tensor, number of active feature columns).
    """
    active_mask = X.abs().sum(dim=(0, 1, 2)) > 0  # (F,)
    if not active_mask.any():
        return X[..., :1], 1
    last_active = int(active_mask.nonzero(as_tuple=True)[0].max().item())
    f_active = last_active + 1
    return X[..., :f_active], f_active


def _compute_n_keep(
    bag_size: int,
    f_active: int,
    max_features: int,
    features_per_group: int,
) -> int:
    """Pick ``n_keep`` so the flattened ``n_keep * f_active`` fits the budget.

    Rounds down so ``n_keep * f_active`` is a multiple of ``features_per_group``
    when possible — TabPFN groups input features and a non-multiple wastes a
    slot. Bumps to 1 if it would round to zero.
    """
    raw = max_features // f_active
    if raw <= 0:
        logger.warning(
            "Per-instance feature dim %d exceeds max_tabpfn_features %d; falling back to n_keep=1.",
            f_active,
            max_features,
        )
        return 1
    flat = (raw * f_active // features_per_group) * features_per_group
    n_keep = flat // f_active if f_active > 0 else 1
    if n_keep == 0:
        n_keep = 1
    return min(n_keep, bag_size)


def _compute_summary_stats(X: torch.Tensor, stats: tuple[str, ...]) -> torch.Tensor:
    """Concatenate per-bag summary stats along the feature axis.

    Args:
        X: Instance features, shape (B, n_bags, bag_size, n_features).
        stats: Names of statistics to compute. Supported: ``"sum"``, ``"mean"``,
            ``"median"``, ``"min"``, ``"max"``, ``"std"``.

    Returns:
        Summary features, shape (B, n_bags, len(stats) * n_features).
    """
    parts: list[torch.Tensor] = []
    for stat in stats:
        if stat == "sum":
            parts.append(X.sum(dim=2))
        elif stat == "mean":
            parts.append(X.mean(dim=2))
        elif stat == "median":
            parts.append(X.median(dim=2).values)
        elif stat == "min":
            parts.append(X.amin(dim=2))
        elif stat == "max":
            parts.append(X.amax(dim=2))
        elif stat == "std":
            # unbiased=False so single-instance bags don't NaN
            parts.append(X.std(dim=2, unbiased=False))
        else:
            raise ValueError(f"Unknown stat: {stat!r}")
    return torch.cat(parts, dim=-1)


def _run_tabpfn_per_batch(
    tabpfn: nn.Module,
    flat_train: torch.Tensor,
    y_train: torch.Tensor,
    flat_test: torch.Tensor,
) -> torch.Tensor:
    """Run frozen TabPFN over each batch element, stacking logits.

    Args:
        tabpfn: Frozen TabPFN backbone callable.
        flat_train: (B, n_train, F_flat).
        y_train: (B, n_train).
        flat_test: (B, n_test, F_flat).

    Returns:
        Logits, shape (B, n_test, max_classes).
    """
    B = flat_train.shape[0]
    all_logits = []
    for b in range(B):
        logits_b = tabpfn(flat_train[b], y_train[b], flat_test[b])
        all_logits.append(logits_b)
    return torch.stack(all_logits, dim=0)


class _TabPFNBaselineBase(nn.Module):
    """Shared internals for the TabPFN MIL baselines."""

    strategy: Literal["concat", "subsample"]

    def __init__(
        self,
        max_classes: int,
        model_path: str | None,
        features_per_group: int,
        max_tabpfn_features: int,
        n_views: int,
        n_keep: int | None,
        aggregation: Literal["mean_logits", "mean_probs"],
        seed: int,
    ) -> None:
        super().__init__()
        if aggregation not in ("mean_logits", "mean_probs"):
            raise ValueError(f"Unknown aggregation: {aggregation!r}. Choose 'mean_logits' or 'mean_probs'.")
        if self.strategy == "subsample" and n_views < 1:
            raise ValueError(f"n_views must be >= 1, got {n_views}.")
        if n_keep is not None and n_keep < 1:
            raise ValueError(f"n_keep must be >= 1 if provided, got {n_keep}.")

        self.max_classes = max_classes
        self.features_per_group = features_per_group
        self.max_tabpfn_features = max_tabpfn_features
        self.n_views = n_views if self.strategy == "subsample" else 1
        self.n_keep = n_keep
        self.aggregation = aggregation
        self.seed = seed
        self.tabpfn = FrozenTabPFNBackbone(
            max_classes=max_classes,
            model_path=model_path,
            features_per_group=features_per_group,
        )

    def train(self, mode: bool = True) -> _TabPFNBaselineBase:
        """Override to keep TabPFN always in eval mode."""
        return super().train(False)

    def _resolve_n_keep(self, bag_size: int, f_active: int) -> int:
        """Resolve n_keep from override or auto-compute, asserting fit."""
        if self.n_keep is None:
            return _compute_n_keep(bag_size, f_active, self.max_tabpfn_features, self.features_per_group)
        if self.n_keep * f_active > self.max_tabpfn_features:
            raise ValueError(
                f"n_keep={self.n_keep} with f_active={f_active} yields flat dim "
                f"{self.n_keep * f_active} > max_tabpfn_features={self.max_tabpfn_features}."
            )
        return min(self.n_keep, bag_size)

    def _forward_concat(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        f_active: int,
    ) -> torch.Tensor:
        """Single-view concat: flatten in original order, truncate flat dim."""
        B, n_train, bag_size, _ = X_train.shape
        n_test = X_test.shape[1]
        flat_train = X_train.reshape(B, n_train, bag_size * f_active)
        flat_test = X_test.reshape(B, n_test, bag_size * f_active)
        flat_dim = flat_train.shape[-1]
        if flat_dim > self.max_tabpfn_features:
            logger.warning(
                "Truncating %d concatenated features to %d (TabPFN limit).",
                flat_dim,
                self.max_tabpfn_features,
            )
            flat_train = flat_train[..., : self.max_tabpfn_features]
            flat_test = flat_test[..., : self.max_tabpfn_features]
        return _run_tabpfn_per_batch(self.tabpfn, flat_train, y_train, flat_test)

    def _forward_subsample(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        f_active: int,
    ) -> torch.Tensor:
        """K-view subsample: sample n_keep instances per view, flatten, aggregate."""
        bag_size = X_train.shape[2]
        n_keep = self._resolve_n_keep(bag_size, f_active)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        all_logits: list[torch.Tensor] = []
        for _ in range(self.n_views):
            idx = torch.randperm(bag_size, generator=generator)[:n_keep].to(X_train.device)
            view_train = X_train.index_select(2, idx)  # (B, n_train, n_keep, F)
            view_test = X_test.index_select(2, idx)
            B, n_train, _, _ = view_train.shape
            n_test = view_test.shape[1]
            flat_train = view_train.reshape(B, n_train, n_keep * f_active)
            flat_test = view_test.reshape(B, n_test, n_keep * f_active)
            logits = _run_tabpfn_per_batch(self.tabpfn, flat_train, y_train, flat_test)
            all_logits.append(logits)

        stacked = torch.stack(all_logits, dim=0)  # (K, B, n_test, max_classes)
        if self.n_views == 1:
            return stacked.squeeze(0)
        if self.aggregation == "mean_probs":
            probs = torch.softmax(stacked, dim=-1).mean(dim=0)
            return torch.log(probs + 1e-8)
        return stacked.mean(dim=0)

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            X_train: Training bag features, shape (B, n_train, bag_size, n_features).
            y_train: Training bag labels, shape (B, n_train).
            X_test: Test bag features, shape (B, n_test, bag_size, n_features).

        Returns:
            Logits for test bags, shape (B, n_test, max_classes).
        """
        all_X = torch.cat([X_train, X_test], dim=1)
        _, f_active = _strip_trailing_zero_features(all_X)
        X_train = X_train[..., :f_active]
        X_test = X_test[..., :f_active]

        if self.strategy == "concat":
            return self._forward_concat(X_train, y_train, X_test, f_active)
        return self._forward_subsample(X_train, y_train, X_test, f_active)


class TabPFNConcatBaseline(_TabPFNBaselineBase):
    """TabPFN MIL baseline — concat strategy.

    Flattens each bag in original instance order and truncates the resulting
    flat vector at ``max_tabpfn_features``.
    """

    strategy: Literal["concat", "subsample"] = "concat"

    def __init__(
        self,
        max_classes: int = 4,
        model_path: str | None = None,
        features_per_group: int = 2,
        max_tabpfn_features: int = 500,
        seed: int = 42,
    ) -> None:
        super().__init__(
            max_classes=max_classes,
            model_path=model_path,
            features_per_group=features_per_group,
            max_tabpfn_features=max_tabpfn_features,
            n_views=1,
            n_keep=None,
            aggregation="mean_logits",
            seed=seed,
        )


class TabPFNSubsampleBaseline(_TabPFNBaselineBase):
    """TabPFN MIL baseline — subsample strategy.

    Per view, draws one random subset of ``n_keep`` instance positions
    (shared across bags, without replacement) and flattens just those.
    ``n_keep`` is auto-sized so ``n_keep * f_active`` fits
    ``max_tabpfn_features``. Runs ``n_views`` views and aggregates logits
    per ``aggregation``.
    """

    strategy: Literal["concat", "subsample"] = "subsample"

    def __init__(
        self,
        max_classes: int = 4,
        model_path: str | None = None,
        features_per_group: int = 2,
        max_tabpfn_features: int = 500,
        n_views: int = 10,
        n_keep: int | None = None,
        aggregation: Literal["mean_logits", "mean_probs"] = "mean_logits",
        seed: int = 42,
    ) -> None:
        super().__init__(
            max_classes=max_classes,
            model_path=model_path,
            features_per_group=features_per_group,
            max_tabpfn_features=max_tabpfn_features,
            n_views=n_views,
            n_keep=n_keep,
            aggregation=aggregation,
            seed=seed,
        )


class MeanLogRegBaseline(nn.Module):
    """Mean-pooling + logistic regression MIL baseline.

    Mean-pools instances per bag to get a single feature vector, then fits
    ``LogisticRegressionCV`` (or plain ``LogisticRegression`` when a class
    has fewer than two samples) on the training bags and predicts on test bags.
    CV fold count is capped by the smallest per-class count so stratified splits
    stay valid on small MIL prompts.
    """

    def __init__(self, max_classes: int = 2, max_iter: int = 1000, seed: int = 0) -> None:
        super().__init__()
        self.max_classes = max_classes
        self.max_iter = max_iter
        self.seed = seed
        self.logreg_cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
        self.logreg_cv = 5

    def train(self, mode: bool = True) -> MeanLogRegBaseline:
        """Override to keep always in eval mode."""
        return super().train(False)

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """Fit logistic regression on mean-pooled bags and predict.

        Args:
            X_train: Training bag features, shape (B, n_train, bag_size, n_features).
            y_train: Training bag labels, shape (B, n_train).
            X_test: Test bag features, shape (B, n_test, bag_size, n_features).

        Returns:
            Log-probabilities for test bags, shape (B, n_test, max_classes).
        """
        X_train_mean = X_train.mean(dim=2)
        X_test_mean = X_test.mean(dim=2)

        B = X_train.shape[0]
        all_logits = []
        for b in range(B):
            x_tr = X_train_mean[b].cpu().numpy()
            y_tr = y_train[b].cpu().numpy()
            x_te = X_test_mean[b].cpu().numpy()

            unique_classes = np.unique(y_tr)
            full_probs = np.full((x_te.shape[0], self.max_classes), 1e-8, dtype=np.float32)

            if len(unique_classes) < 2:
                full_probs[:, int(unique_classes[0])] = 1.0
            else:
                folds = _bag_level_stratified_folds(y_tr, n_splits=self.logreg_cv, seed=self.seed)
                if not folds:
                    clf = LogisticRegression(max_iter=self.max_iter, random_state=self.seed)
                else:
                    clf = LogisticRegressionCV(
                        Cs=list(self.logreg_cs),
                        cv=folds,
                        max_iter=self.max_iter,
                    )
                clf.fit(x_tr, y_tr)
                probs = clf.predict_proba(x_te)
                for i, cls in enumerate(clf.classes_):
                    if int(cls) < self.max_classes:
                        full_probs[:, int(cls)] = probs[:, i]

            all_logits.append(torch.tensor(np.log(np.clip(full_probs, 1e-8, 1.0)), device=X_train.device))

        return torch.stack(all_logits, dim=0)


class SVMSummBaseline(nn.Module):
    """Summary-statistics MIL baseline: per-bag stats + CV-tuned SVC.

    Per-bag summary statistics (sum, mean, median, min, max, stdev),
    classified by a scikit-learn ``SVC(kernel='rbf', probability=True)``
    whose ``C`` is selected by stratified K-fold ``GridSearchCV`` per
    forward call.

    Two evaluation variants, selected by ``mode``:

    * ``mode="refit"`` — ``GridSearchCV`` picks the best ``C`` and refits
      a single SVC on the full ``X_train`` at that ``C``. Standard sklearn
      flow; default.
    * ``mode="ensemble"`` — ``GridSearchCV`` picks the best ``C``; one
      fresh ``SVC(C=best_C)`` is then fit on each of the K folds' train
      indices, and per-fold ``predict_proba`` outputs are averaged on
      ``X_test``.

    When stratified CV is infeasible, the search is skipped and a default 
    ``C=1`` SVC is fit on the full ``X_train``.
    """

    STATS: tuple[str, ...] = ("sum", "mean", "median", "min", "max", "std")
    KERNEL: str = "rbf"
    GAMMA: str = "scale"
    SVM_CS: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    SVM_CV: int = 5

    def __init__(
        self,
        max_classes: int = 2,
        seed: int = 0,
        mode: Literal["refit", "ensemble"] = "refit",
    ) -> None:
        super().__init__()
        if mode not in ("refit", "ensemble"):
            raise ValueError(f"mode must be 'refit' or 'ensemble', got {mode!r}")
        self.max_classes = max_classes
        self.seed = seed
        self.mode = mode

    def train(self, mode: bool = True) -> SVMSummBaseline:
        """Override to keep always in eval mode."""
        return super().train(False)

    def _summarize(self, X: torch.Tensor) -> torch.Tensor:
        """Concatenate per-bag summary stats along the feature axis."""
        return _compute_summary_stats(X, self.STATS)

    def _make_svc(self, C: float = 1.0) -> SVC:
        return SVC(C=C, kernel=self.KERNEL, gamma=self.GAMMA, probability=True, random_state=self.seed)

    def _scatter_probs(self, full_probs: np.ndarray, classes_: np.ndarray, probs: np.ndarray) -> None:
        for i, cls in enumerate(classes_):
            if int(cls) < self.max_classes:
                full_probs[:, int(cls)] = probs[:, i]

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """Fit a CV-tuned SVC on summary-stat bags and predict.

        Args:
            X_train: Training bag features, shape (1, n_train, bag_size, n_features).
            y_train: Training bag labels, shape (1, n_train).
            X_test: Test bag features, shape (1, n_test, bag_size, n_features).

        Returns:
            Log-probabilities for test bags, shape (1, n_test, max_classes).
        """
        assert X_train.shape[0] == 1, "Baselines support only B=1; the eval harness always batches a single episode."
        x_tr = self._summarize(X_train)[0].cpu().numpy()
        y_tr = y_train[0].cpu().numpy()
        x_te = self._summarize(X_test)[0].cpu().numpy()

        full_probs = np.full((x_te.shape[0], self.max_classes), 1e-8, dtype=np.float32)
        unique_classes = np.unique(y_tr)

        if len(unique_classes) < 2:
            full_probs[:, int(unique_classes[0])] = 1.0
        else:
            folds = _bag_level_stratified_folds(y_tr, n_splits=self.SVM_CV, seed=self.seed)
            if not folds:
                clf = self._make_svc(C=1.0)
                clf.fit(x_tr, y_tr)
                self._scatter_probs(full_probs, clf.classes_, clf.predict_proba(x_te))
            else:
                search = GridSearchCV(
                    self._make_svc(),
                    param_grid={"C": list(self.SVM_CS)},
                    cv=folds,
                    n_jobs=1,
                )
                search.fit(x_tr, y_tr)
                if self.mode == "refit":
                    self._scatter_probs(full_probs, search.classes_, search.predict_proba(x_te))
                else:
                    best_C = float(search.best_params_["C"])
                    fold_probs: list[np.ndarray] = []
                    for tr_idx, _ in folds:
                        clf = self._make_svc(C=best_C)
                        clf.fit(x_tr[tr_idx], y_tr[tr_idx])
                        per_fold = np.full_like(full_probs, 1e-8)
                        self._scatter_probs(per_fold, clf.classes_, clf.predict_proba(x_te))
                        fold_probs.append(per_fold)
                    full_probs = np.mean(np.stack(fold_probs, axis=0), axis=0)

        log_probs = torch.tensor(np.log(np.clip(full_probs, 1e-8, 1.0)), device=X_train.device)
        return log_probs.unsqueeze(0)  # (1, n_test, max_classes)


class ClusterTabPFNBaseline(nn.Module):
    """K-means selective pooling + ensemble of TabPFN predictors for MIL.

    Implements the KMeans variant from "Utilizing TabPFN for Multi-Instance
    Data with Scarce Labels" (Kopp et al., NeurIPS 2025 AITD Workshop),
    adapted from regression to classification.

    All training instances are clustered into K groups via K-means. For each
    cluster k a separate TabPFN support-query pair is constructed: each bag is
    represented by the (I_n/|I_{n,k}|)-scaled sum of its cluster-k instances
    (equivalent to I_n x cluster-k mean). TabPFN is called K times; the K
    per-cluster logit sets are averaged to give the final prediction.
    """

    def __init__(
        self,
        max_classes: int = 4,
        n_clusters: int = 8,
        n_pca_components: int | None = None,
        max_tabpfn_features: int = 500,
        seed: int = 42,
        model_path: str | None = None,
        features_per_group: int = 2,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.n_pca_components = n_pca_components
        self.max_tabpfn_features = max_tabpfn_features
        self.seed = seed
        self.tabpfn = FrozenTabPFNBackbone(
            max_classes=max_classes,
            model_path=model_path,
            features_per_group=features_per_group,
        )

    def train(self, mode: bool = True) -> ClusterTabPFNBaseline:
        """Override to keep TabPFN always in eval mode."""
        return super().train(False)

    def _selective_pool(
        self,
        X_bags: np.ndarray,
        mask: np.ndarray,
        kmeans: KMeans,
        cluster_idx: int,
        pca: PCA | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the selective pool for one cluster across all bags (Eq. 1).

        For each bag, gathers valid instances, assigns them to clusters, and
        computes ``(I_n / |I_{n,k}|) · Σ_{i∈I_{n,k}} x_{n,i}`` — equivalent
        to I_n x cluster-k mean.

        Args:
            X_bags: (n_bags, bag_size, n_dims) float32.
            mask: (n_bags, bag_size) bool, True for non-padded instances.
            kmeans: Fitted KMeans.
            cluster_idx: Which cluster to pool.
            pca: Fitted PCA or None.

        Returns:
            pools: (n_bags, n_dims) — zero rows for bags absent from this cluster.
            has_cluster: (n_bags,) bool — True iff bag has ≥1 instance in this cluster.
        """
        n_bags, _, n_dims = X_bags.shape
        pools = np.zeros((n_bags, n_dims), dtype=np.float32)
        has_cluster = np.zeros(n_bags, dtype=bool)

        for bag_idx in range(n_bags):
            valid = X_bags[bag_idx][mask[bag_idx]]  # (n_valid, n_dims)
            if len(valid) == 0:
                continue
            if pca is not None:
                valid = pca.transform(valid)
            assignments = kmeans.predict(valid)
            c_mask = assignments == cluster_idx
            n_c = int(c_mask.sum())
            if n_c == 0:
                continue
            has_cluster[bag_idx] = True
            # (I_n / |I_{n,k}|) * sum == I_n * cluster_mean  (Eq. 1)
            pools[bag_idx] = (len(valid) / n_c) * valid[c_mask].sum(axis=0)

        return pools, has_cluster

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """Run K selective-pool TabPFN calls and average logits.

        Args:
            X_train: Training bag features, shape (1, n_train, bag_size, n_features).
            y_train: Training bag labels, shape (1, n_train).
            X_test: Test bag features, shape (1, n_test, bag_size, n_features).

        Returns:
            Logits for test bags, shape (1, n_test, max_classes).
        """
        assert X_train.shape[0] == 1, "Baselines support only B=1; the eval harness always batches a single episode."

        X_tr_t = X_train[0]  # (n_train, bag_size, n_features)
        X_te_t = X_test[0]  # (n_test,  bag_size, n_features)

        mask_tr = (X_tr_t.abs().sum(-1) > 0).cpu().numpy()  # (n_train, bag_size)
        mask_te = (X_te_t.abs().sum(-1) > 0).cpu().numpy()  # (n_test,  bag_size)

        X_tr_np = X_tr_t.cpu().numpy()
        X_te_np = X_te_t.cpu().numpy()

        # Strip trailing zero-padded feature columns using training instances
        all_tr_raw = X_tr_np[mask_tr]  # (N_valid, n_features)
        active_cols = np.abs(all_tr_raw).sum(axis=0) > 0
        n_active = int(np.where(active_cols)[0].max()) + 1 if active_cols.any() else 1
        X_tr_np = X_tr_np[:, :, :n_active]
        X_te_np = X_te_np[:, :, :n_active]
        all_tr_instances = X_tr_np[mask_tr]  # (N_valid, n_active)

        # Optional PCA: fit on training instances, transform both splits
        pca: PCA | None = None
        if self.n_pca_components is not None:
            n_comp = min(self.n_pca_components, n_active, len(all_tr_instances))
            pca = PCA(n_components=n_comp, random_state=self.seed)
            all_tr_instances = pca.fit_transform(all_tr_instances)

        # K-means on all valid training instances
        k = min(self.n_clusters, len(all_tr_instances))
        if k < self.n_clusters:
            logger.warning(
                "Only %d valid training instances; reducing n_clusters from %d to %d.",
                len(all_tr_instances),
                self.n_clusters,
                k,
            )
        kmeans = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
        kmeans.fit(all_tr_instances)

        n_test = X_te_np.shape[0]
        max_classes = self.tabpfn.max_classes
        logits_sum = np.zeros((n_test, max_classes), dtype=np.float32)
        logits_count = np.zeros(n_test, dtype=np.float32)
        y_tr_np = y_train[0].cpu().numpy()

        for c in range(k):
            pool_tr, has_tr = self._selective_pool(X_tr_np, mask_tr, kmeans, c, pca)
            pool_te, has_te = self._selective_pool(X_te_np, mask_te, kmeans, c, pca)

            train_idx = np.where(has_tr)[0]
            test_idx = np.where(has_te)[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            y_c = y_tr_np[train_idx]
            if len(np.unique(y_c)) < 2:
                continue

            x_tr_c = pool_tr[train_idx]  # (n_tr_c, n_dims)
            x_te_c = pool_te[test_idx]  # (n_te_c, n_dims)

            x_tr_t = torch.from_numpy(x_tr_c).float().to(X_train.device)
            x_te_t = torch.from_numpy(x_te_c).float().to(X_train.device)
            if x_tr_t.shape[-1] > self.max_tabpfn_features:
                logger.warning(
                    "Truncating %d cluster features to %d (TabPFN limit).",
                    x_tr_t.shape[-1],
                    self.max_tabpfn_features,
                )
                x_tr_t = x_tr_t[..., : self.max_tabpfn_features]
                x_te_t = x_te_t[..., : self.max_tabpfn_features]
            y_tr_t = torch.from_numpy(y_c).to(X_train.device)

            logits_c = self.tabpfn(x_tr_t, y_tr_t, x_te_t).detach().cpu().numpy()
            logits_sum[test_idx] += logits_c
            logits_count[test_idx] += 1

        # Average logits across clusters; zero logits (→ uniform softmax) for
        # non-participating bags (should not occur in practice).
        participated = logits_count > 0
        logits_sum[participated] /= logits_count[participated, np.newaxis]

        return torch.from_numpy(logits_sum).float().to(X_train.device).unsqueeze(0)
