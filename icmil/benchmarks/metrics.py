"""Benchmark task protocol and shared metrics for MIL in-context evaluation."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from typing import Protocol

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class BenchmarkTask(Protocol):
    """Protocol for benchmark tasks that yield independent dataset draws."""

    @property
    def name(self) -> str:
        """Unique task name for logging (e.g. 'tcga_fixed', 'rsna_ich')."""
        ...

    def sample_datasets(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (X_train, y_train, X_test, y_test) tuples, one per dataset draw.

        Depending on the task, draws may be freshly sampled (synthetic tasks)
        or read from pre-generated storage (fixed-split tasks).

        X shapes: (n_bags, bag_size, n_features). y shapes: (n_bags,).
        """
        ...


def _pad_features(X: np.ndarray, max_features: int | None) -> np.ndarray:
    """Pad last dimension of X to ``max_features`` with zeros.

    ``max_features=None`` is a no-op used for baselines whose models have
    no fixed input width (TabPFN / sklearn / ABMIL baselines).
    """
    if max_features is None:
        return X
    n_features = X.shape[-1]
    if n_features >= max_features:
        return X
    pad_width = ((0, 0),) * (X.ndim - 1) + ((0, max_features - n_features),)
    return np.pad(X, pad_width, mode="constant", constant_values=0.0)


def _compute_roc_auc(y_true: np.ndarray, probs: np.ndarray) -> float | None:
    """Compute ROC AUC, returning None when the metric is undefined.

    Handles both binary and multiclass cases. Returns None when fewer than
    two classes are present in y_true or when the number of present classes
    doesn't match the number of probability columns.
    """
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return None
    if len(unique_classes) == 2:
        # Binary: use raw probability of the higher class (ranking preserved)
        return float(roc_auc_score(y_true, probs[:, unique_classes[1]]))
    # Multiclass: only valid when all probability columns have corresponding targets
    if len(unique_classes) != probs.shape[1]:
        return None
    label_map = {cls: i for i, cls in enumerate(unique_classes)}
    y_mapped = np.array([label_map[y] for y in y_true])
    return float(roc_auc_score(y_mapped, probs, multi_class="ovr"))


def compute_split_metrics(
    y_test: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics for a single train/test split.

    Args:
        y_test: Ground-truth labels for test set, shape (n_test,).
        preds: Predicted class labels, shape (n_test,).
        probs: Predicted probabilities, shape (n_test, n_classes).
        y_train: Training labels (used for majority/random baselines).

    Returns:
        Dict with accuracy, balanced_accuracy, macro_f1, roc_auc (if defined),
        and majority/random baseline metrics.
    """
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_test, preds),
        "balanced_accuracy": balanced_accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro", zero_division=0),
    }
    roc_auc = _compute_roc_auc(y_test, probs)
    if roc_auc is not None:
        metrics["roc_auc"] = roc_auc

    # Majority baseline
    maj_cls = int(np.bincount(y_train).argmax())
    maj_preds = np.full_like(y_test, maj_cls)
    metrics["majority_accuracy"] = accuracy_score(y_test, maj_preds)
    metrics["majority_balanced_accuracy"] = balanced_accuracy_score(y_test, maj_preds)
    metrics["majority_macro_f1"] = f1_score(y_test, maj_preds, average="macro", zero_division=0)

    # Random baseline
    n_classes = len(np.unique(y_train))
    metrics["random_accuracy"] = 1.0 / n_classes
    metrics["random_balanced_accuracy"] = 1.0 / n_classes

    return metrics


_BOOTSTRAP_N = 2000
_BOOTSTRAP_ALPHA = 0.05
_BOOTSTRAP_SEED = 0


def _bootstrap_ci_mean(values: np.ndarray) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of ``values`` (95% by default).

    Caller must ensure ``values.size >= 2``. Seeded for reproducibility so a
    fixed set of per-split metrics always yields the same CI bounds.
    """
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    resamples = rng.choice(values, size=(_BOOTSTRAP_N, values.size), replace=True)
    means = resamples.mean(axis=1)
    lo = float(np.quantile(means, _BOOTSTRAP_ALPHA / 2))
    hi = float(np.quantile(means, 1 - _BOOTSTRAP_ALPHA / 2))
    return lo, hi


def _summarize_timings(timings: list[dict[str, float | int]]) -> dict[str, float | int | list]:
    """Reduce a list of per-split timing dicts to summary fields.

    Each input dict carries ``elapsed_s`` plus shape stats (``n_train_bags``,
    ``n_test_bags``, ``bag_size``, ``n_features``). Returns aggregate timing
    keys, the mean shape stats, and the raw list under ``per_split_timings`` 
    for post-hoc analysis.
    """
    out: dict[str, float | int | list] = {}
    if not timings:
        return out
    elapsed = np.asarray([t["elapsed_s"] for t in timings], dtype=np.float64)
    test_bags = np.asarray([t["n_test_bags"] for t in timings], dtype=np.float64)
    train_bags = np.asarray([t["n_train_bags"] for t in timings], dtype=np.float64)
    bag_sizes = np.asarray([t["bag_size"] for t in timings], dtype=np.float64)
    n_features = np.asarray([t["n_features"] for t in timings], dtype=np.float64)
    total = float(elapsed.sum())
    out["time_total_s"] = total
    out["time_per_split_s_mean"] = float(elapsed.mean())
    out["time_per_split_s_std"] = float(elapsed.std(ddof=1)) if elapsed.size >= 2 else float("nan")
    total_test = float(test_bags.sum())
    out["time_per_test_bag_s"] = total / total_test if total_test > 0 else float("nan")
    out["n_train_bags_mean"] = float(train_bags.mean())
    out["n_test_bags_mean"] = float(test_bags.mean())
    out["bag_size"] = float(bag_sizes.mean())
    out["n_features"] = float(n_features.mean())
    out["per_split_timings"] = list(timings)
    return out


def aggregate_split_metrics(
    per_split: list[dict[str, float]],
    samples: list[tuple[int, int]],
    timings: list[dict[str, float | int]] | None = None,
) -> dict[str, float | int | list]:
    """Summarize per-split metrics with mean/std/SEM/95% CI and attach samples.

    For each metric key present in at least one split, the result contains:
    ``{key}`` (mean), ``{key}_std`` (unbiased std; NaN when only one split),
    ``{key}_sem`` (standard error of the mean = std / sqrt(n); NaN when only
    one split), ``{key}_ci_low`` / ``{key}_ci_high`` (percentile bootstrap CI
    of the mean; equal to the mean when only one split), ``{key}_per_split``
    (raw list of per-split values, for post-hoc box plots / scatter overlays).

    Also emits ``n_splits`` and ``samples``. When ``timings`` is provided,
    also emits aggregate timing fields and the raw per-split timings list.
    """
    results: dict[str, float | int | list] = {}
    all_keys = {k for m in per_split for k in m}
    for key in all_keys:
        values = np.asarray([m[key] for m in per_split if key in m], dtype=np.float64)
        if values.size == 0:
            continue
        mean = float(values.mean())
        results[key] = mean
        # Raw per-split values: lets post-hoc analysis (box plots, scatter
        # overlays of the 20 bias-variance subsamples) skip a re-run.
        results[f"{key}_per_split"] = [float(v) for v in values]
        if values.size >= 2:
            std = float(values.std(ddof=1))
            results[f"{key}_std"] = std
            results[f"{key}_sem"] = std / float(np.sqrt(values.size))
            lo, hi = _bootstrap_ci_mean(values)
        else:
            results[f"{key}_std"] = float("nan")
            results[f"{key}_sem"] = float("nan")
            lo, hi = mean, mean
        results[f"{key}_ci_low"] = lo
        results[f"{key}_ci_high"] = hi
    results["n_splits"] = len(per_split)
    results["samples"] = samples
    if timings is not None:
        results.update(_summarize_timings(timings))
    return results


def aggregate_outer_metrics(
    per_outer_per_split: list[list[dict[str, float]]],
    samples: list[tuple[int, int]],
    per_outer_timings: list[list[dict[str, float | int]]] | None = None,
) -> dict[str, float | int | list]:
    """Aggregate when each outer (e.g. model-seed) instance has its own per-split list.

    For each metric: per-outer value = mean of that outer's split metrics.
    Reported ``mean`` is the mean across outer values; ``std`` / ``sem`` /
    bootstrap CI are over the outer values (n = number of outer instances).

    Use this when you want std to reflect outer-instance variance (e.g. model
    seeds) rather than the per-split or pooled (outer x split) variance.

    Emits ``n_outer`` and ``n_splits`` (per outer; assumed constant across
    outer instances). When ``per_outer_timings`` is provided, also emits
    timing aggregates pooled across all outer instances and splits, plus the
    raw per-split timings of the first outer (representative; per_split count
    is constant).
    """
    results: dict[str, float | int | list] = {}
    if not per_outer_per_split:
        return {"n_outer": 0, "n_splits": 0, "samples": samples}
    all_keys = {k for outer in per_outer_per_split for m in outer for k in m}
    for key in all_keys:
        per_outer = []
        for outer in per_outer_per_split:
            outer_values = [m[key] for m in outer if key in m]
            if outer_values:
                per_outer.append(float(np.mean(outer_values)))
        if not per_outer:
            continue
        values = np.asarray(per_outer, dtype=np.float64)
        mean = float(values.mean())
        results[key] = mean
        # Per-outer means (e.g. one value per model seed). Distinct from the
        # per-split list saved by aggregate_split_metrics — the per-split
        # granularity is collapsed inside each outer mean, so we can't recover
        # it here.
        results[f"{key}_per_outer"] = [float(v) for v in values]
        if values.size >= 2:
            std = float(values.std(ddof=1))
            results[f"{key}_std"] = std
            results[f"{key}_sem"] = std / float(np.sqrt(values.size))
            lo, hi = _bootstrap_ci_mean(values)
        else:
            results[f"{key}_std"] = float("nan")
            results[f"{key}_sem"] = float("nan")
            lo, hi = mean, mean
        results[f"{key}_ci_low"] = lo
        results[f"{key}_ci_high"] = hi
    results["n_outer"] = len(per_outer_per_split)
    results["n_splits"] = len(per_outer_per_split[0]) if per_outer_per_split else 0
    results["samples"] = samples
    if per_outer_timings:
        pooled = [t for outer in per_outer_timings for t in outer]
        results.update(_summarize_timings(pooled))
        # Replace the pooled per_split_timings with just the first outer's
        # splits — the post-hoc plot only needs one bag-size sample per task.
        results["per_split_timings"] = list(per_outer_timings[0])
    return results


def per_split_metrics(
    model: torch.nn.Module | Callable[[int], torch.nn.Module],
    task: BenchmarkTask,
    device: torch.device,
    max_features: int | None,
) -> tuple[list[dict[str, float]], list[tuple[int, int]], list[dict[str, float | int]]]:
    """Run ``model`` on every split of ``task``; return per-split metrics, sample predictions, and per-split timings.

    No aggregation. Use :func:`aggregate_split_metrics` afterward — this
    split lets callers pool measurements across multiple model instances
    (e.g. ABMIL with different seeds) before computing mean/std.

    When ``model`` is a callable factory (``Callable[[int], nn.Module]``) it is
    invoked with the split index before each forward, so each resample sees a
    freshly built model whose seed depends on the split. This is the
    bias-variance sweep's per-resample seed mode; the factory's construction
    cost is included in ``elapsed_s``, which is the right behavior for refit
    baselines (the existing nn.Module path also pays init cost inside
    ``forward`` because they refit per call).

    The third return value is a list of timing dicts (one per split) with
    keys ``elapsed_s`` (wall time of the model forward call, with CUDA sync
    when applicable), ``n_train_bags``, ``n_test_bags``, ``bag_size``,
    ``n_features``. For baselines whose ``forward`` does train+infer per
    split (ABMIL, mean-logreg) ``elapsed_s`` covers both stages; for
    inference-only models it covers inference only.
    """
    per_split: list[dict[str, float]] = []
    samples: list[tuple[int, int]] = []
    timings: list[dict[str, float | int]] = []
    is_cuda = device.type == "cuda"
    is_factory = not isinstance(model, torch.nn.Module)
    # Per-split progress: lets you read per-subsample wall time off the job's
    # stdout and extrapolate cell duration during long sweeps. ``total`` is
    # opportunistic — concrete tasks can expose ``n_splits`` (MNISTXAITask
    # does); when absent tqdm just shows a counter without ETA.
    n_splits_hint = getattr(task, "n_splits", None)
    pbar = tqdm(
        enumerate(task.sample_datasets()),
        total=n_splits_hint,
        desc=task.name,
        unit="split",
        leave=False,
    )
    for split_idx, (X_train, y_train, X_test, y_test) in pbar:
        X_train_pad = _pad_features(X_train, max_features)
        X_test_pad = _pad_features(X_test, max_features)
        # (1, n_bags, bag_size, features)
        X_train_t = torch.tensor(X_train_pad, dtype=torch.float32).unsqueeze(0).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).unsqueeze(0).to(device)
        X_test_t = torch.tensor(X_test_pad, dtype=torch.float32).unsqueeze(0).to(device)

        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        # No outer no_grad — some baselines (e.g. ABMIL) train internally
        # and need gradients. Each model manages its own context.
        active = model(split_idx) if is_factory else model
        logits = active(X_train_t, y_train_t, X_test_t)
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        with torch.no_grad():
            logits_2d = logits.squeeze(0)
            preds = logits_2d.argmax(dim=-1).cpu().numpy()
            probs = torch.softmax(logits_2d, dim=-1).cpu().numpy()

        per_split.append(compute_split_metrics(y_test, preds, probs, y_train))
        timings.append(
            {
                "elapsed_s": float(elapsed),
                "n_train_bags": int(X_train.shape[0]),
                "n_test_bags": int(X_test.shape[0]),
                "bag_size": int(X_train.shape[1]),
                "n_features": int(X_train.shape[2]),
            }
        )

        if not samples:
            samples = [(int(p), int(t)) for p, t in zip(preds, y_test, strict=False)]
    return per_split, samples, timings


def evaluate_benchmark_task(
    model: torch.nn.Module,
    task: BenchmarkTask,
    device: torch.device,
    max_features: int | None,
) -> dict[str, float | int | list]:
    """Run model on all splits from a task; return summary metrics, sample predictions, and timings.

    Pads task features to max_features when needed so the model receives
    the expected input dimension.

    Returns:
        Dict with per-metric mean / std / CI entries (see
        :func:`aggregate_split_metrics`), ``n_splits``, ``samples``, and
        timing aggregates (``time_total_s``, ``time_per_split_s_mean``,
        ``time_per_split_s_std``, ``time_per_test_bag_s``,
        ``n_train_bags_mean``, ``n_test_bags_mean``, ``bag_size``,
        ``n_features``, ``per_split_timings``).
    """
    per_split, samples, timings = per_split_metrics(model, task, device, max_features)
    return aggregate_split_metrics(per_split, samples, timings)
