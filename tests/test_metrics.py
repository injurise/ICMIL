"""The metric and aggregation code behind every cell of the benchmark table.

The subtle contracts here are about *absence*: a metric that is undefined on a split
must be missing rather than NaN, and a quantity with nothing to average over must be
NaN rather than zero. Both are easy to "fix" into silently wrong table cells.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from icmil.benchmarks.metrics import (
    _bootstrap_ci_mean,
    _pad_features,
    aggregate_outer_metrics,
    aggregate_split_metrics,
    compute_split_metrics,
)


def _split(seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {k: float(rng.uniform(0.3, 0.95)) for k in ("accuracy", "roc_auc", "macro_f1")}


# --------------------------------------------------------------------------- padding


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_pad_widens_to_max_features(ndim: int) -> None:
    X = np.ones((2, 3, 4, 5)[-ndim:], dtype=np.float32)
    out = _pad_features(X, 8)
    assert out.shape[-1] == 8
    assert (out[..., 5:] == 0).all(), "padding must be zeros"
    assert (out[..., :5] == 1).all(), "original features must be untouched"


def test_pad_never_truncates() -> None:
    """Models declare a width; the harness widens to it but must not cut features."""
    X = np.ones((4, 3, 10), dtype=np.float32)
    assert _pad_features(X, 4).shape[-1] == 10


def test_pad_is_a_noop_without_a_declared_width() -> None:
    """Baselines accept any width, and must see the raw features, not padded zeros."""
    X = np.ones((4, 3, 10), dtype=np.float32)
    assert _pad_features(X, None) is X


# --------------------------------------------------------------------------- per split


def test_undefined_auroc_is_absent_not_nan() -> None:
    """One class in the test set means AUROC is undefined.

    The key must be missing: the aggregators average over the splits that *have* a
    key, so a NaN would poison the task's mean instead of being skipped.
    """
    y_test = np.zeros(4, dtype=np.int64)
    probs = np.full((4, 2), 0.5)
    metrics = compute_split_metrics(y_test, np.zeros(4, dtype=np.int64), probs, np.array([0, 1, 0, 1]))
    assert "roc_auc" not in metrics
    assert "accuracy" in metrics


def test_majority_and_random_baselines_are_reported() -> None:
    y_train = np.array([0, 0, 0, 1])
    metrics = compute_split_metrics(np.array([0, 1]), np.array([0, 0]), np.full((2, 2), 0.5), y_train)
    assert metrics["majority_accuracy"] == pytest.approx(0.5)
    assert metrics["random_accuracy"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- aggregation


def test_single_split_has_no_spread() -> None:
    """With one observation std/SEM are undefined; reporting 0.0 would claim certainty."""
    out = aggregate_split_metrics([_split(0)], [])
    assert math.isnan(out["roc_auc_std"])
    assert math.isnan(out["roc_auc_sem"])
    assert out["roc_auc_ci_low"] == out["roc_auc"] == out["roc_auc_ci_high"]


def test_sem_is_std_over_sqrt_n() -> None:
    per_split = [_split(i) for i in range(5)]
    out = aggregate_split_metrics(per_split, [])
    values = np.array([s["roc_auc"] for s in per_split])
    assert out["roc_auc"] == pytest.approx(values.mean())
    assert out["roc_auc_std"] == pytest.approx(values.std(ddof=1))
    assert out["roc_auc_sem"] == pytest.approx(out["roc_auc_std"] / np.sqrt(5))
    assert out["n_splits"] == 5


def test_missing_metric_averages_over_the_splits_that_have_it() -> None:
    per_split = [_split(0), {k: v for k, v in _split(1).items() if k != "roc_auc"}, _split(2)]
    out = aggregate_split_metrics(per_split, [])
    assert out["roc_auc"] == pytest.approx(np.mean([per_split[0]["roc_auc"], per_split[2]["roc_auc"]]))


def test_outer_aggregation_measures_spread_across_runs() -> None:
    """The table's ± for a multi-run row is the spread across runs, not across splits.

    Each run is collapsed to its own mean first; the std is then over those means.
    """
    per_outer = [[_split(10 * o + s) for s in range(4)] for o in range(3)]
    out = aggregate_outer_metrics(per_outer, [])
    run_means = [np.mean([s["roc_auc"] for s in run]) for run in per_outer]
    assert out["roc_auc"] == pytest.approx(np.mean(run_means))
    assert out["roc_auc_std"] == pytest.approx(np.std(run_means, ddof=1))
    assert out["n_outer"] == 3
    assert out["roc_auc_per_outer"] == pytest.approx(run_means)


def test_bootstrap_ci_is_seeded() -> None:
    """A reported CI must not move between runs on the same numbers."""
    values = np.array([0.61, 0.72, 0.88, 0.9, 0.95])
    assert _bootstrap_ci_mean(values) == _bootstrap_ci_mean(values)
