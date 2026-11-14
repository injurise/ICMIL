"""Tests for pooled-bag TabPFN baselines."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from icmil.baselines.pooled_tabpfn_baselines import (
    MeanTabPFN3Baseline,
    MeanTabPFN25Baseline,
    MeanTabPFN35Baseline,
    MeanTabPFNBaseline,
    SummTabPFN3Baseline,
    SummTabPFN25Baseline,
    SummTabPFN35Baseline,
    SummTabPFNBaseline,
    _PooledTabPFNBaseline,
)

BASELINES = (
    MeanTabPFNBaseline,
    SummTabPFNBaseline,
    MeanTabPFN25Baseline,
    SummTabPFN25Baseline,
    MeanTabPFN3Baseline,
    SummTabPFN3Baseline,
    MeanTabPFN35Baseline,
    SummTabPFN35Baseline,
)


class RecordingClassifier:
    """Small TabPFN stand-in that records the rows it was fit on."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> RecordingClassifier:
        self.fit_X = X
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X), len(self.classes_)), 1.0 / len(self.classes_), dtype=np.float32)


@pytest.fixture
def train_test_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    X_train = torch.randn(1, 12, 4, 8, generator=generator)
    y_train = (torch.arange(12) % 3).unsqueeze(0)
    X_test = torch.randn(1, 3, 4, 8, generator=generator)
    return X_train, y_train, X_test


@pytest.mark.parametrize("baseline", BASELINES)
def test_output_shape_and_pooling_width(
    baseline: type[_PooledTabPFNBaseline],
    train_test_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    X_train, y_train, X_test = train_test_tensors
    X_train[..., 5:] = 0
    X_test[..., 5:] = 0
    classifier = RecordingClassifier()

    with patch.object(_PooledTabPFNBaseline, "_make_classifier", return_value=classifier):
        output = baseline(max_classes=3)(X_train, y_train, X_test)

    assert output.shape == (1, 3, 3)
    assert torch.allclose(output.exp().sum(dim=-1), torch.ones(1, 3))
    assert classifier.fit_X.shape == (12, len(baseline.stats) * 5)


def test_classifier_uses_package_defaults_for_the_model_version() -> None:
    try:
        from tabpfn.constants import ModelVersion
    except ImportError:
        pytest.skip("needs a tabpfn release with per-version defaults")

    with patch("icmil.baselines.pooled_tabpfn_baselines.TabPFNClassifier") as classifier:
        SummTabPFN35Baseline(seed=7)._make_classifier("cpu")

    classifier.create_default_for_version.assert_called_once_with(
        ModelVersion.V3_5,
        device="cpu",
        random_state=7,
        ignore_pretraining_limits=True,
    )


def test_single_class_context_does_not_construct_classifier(
    train_test_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    X_train, y_train, X_test = train_test_tensors
    with patch.object(_PooledTabPFNBaseline, "_make_classifier") as make_classifier:
        output = MeanTabPFNBaseline(max_classes=3)(X_train, torch.zeros_like(y_train), X_test)

    make_classifier.assert_not_called()
    assert torch.allclose(output.exp()[..., 0], torch.ones(1, 3))
    assert (output.exp()[..., 1:] < 1e-6).all()


@pytest.mark.parametrize("baseline", BASELINES)
def test_baselines_are_always_in_eval_mode(baseline: type[_PooledTabPFNBaseline]) -> None:
    model = baseline()
    model.train()
    assert not model.training


def test_every_baseline_spec_has_a_class() -> None:
    from icmil.reproduce import BASELINE_SPECS, _baseline_classes

    classes = _baseline_classes()
    assert set(BASELINE_SPECS) == set(classes)
    assert set(BASELINES) <= set(classes.values())
