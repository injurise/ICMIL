"""MIL baselines that pool each bag into one row and classify the rows with TabPFN.

Every baseline uses the TabPFN package defaults for its model version. The
weights are downloaded from Hugging Face on first use; TabPFN v2.5 is gated and
requires accepting its license there.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch
from tabpfn import TabPFNClassifier
from torch import nn

from icmil.baselines.tabpfn_baselines import (
    SVMSummBaseline,
    _compute_summary_stats,
    _strip_trailing_zero_features,
)


class _PooledTabPFNBaseline(nn.Module):
    """Shared implementation; subclasses pick the pooling statistics and model version."""

    stats: ClassVar[tuple[str, ...]]
    version: ClassVar[str]

    def __init__(self, max_classes: int = 4, seed: int = 0) -> None:
        super().__init__()
        self.max_classes = max_classes
        self.seed = seed

    def train(self, mode: bool = True) -> _PooledTabPFNBaseline:
        """Override to keep always in eval mode."""
        return super().train(False)

    def _make_classifier(self, device: str) -> TabPFNClassifier:
        from tabpfn.constants import ModelVersion

        return TabPFNClassifier.create_default_for_version(
            ModelVersion(self.version),
            device=device,
            random_state=self.seed,
            ignore_pretraining_limits=True,
        )

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """Fit TabPFN on pooled training bags and predict the test bags.

        Args:
            X_train: Training bag features, shape (1, n_train, bag_size, n_features).
            y_train: Training bag labels, shape (1, n_train).
            X_test: Test bag features, shape (1, n_test, bag_size, n_features).

        Returns:
            Log-probabilities for test bags, shape (1, n_test, max_classes).
        """
        assert X_train.shape[0] == 1, "Baselines support only B=1; the eval harness always batches a single episode."
        n_train = X_train.shape[1]
        bags, _ = _strip_trailing_zero_features(torch.cat((X_train, X_test), dim=1))
        pooled = _compute_summary_stats(bags, self.stats)[0].float().cpu().numpy()
        x_tr, x_te = pooled[:n_train], pooled[n_train:]
        y_tr = y_train[0].cpu().numpy()

        full_probs = np.full((x_te.shape[0], self.max_classes), 1e-8, dtype=np.float32)
        unique_classes = np.unique(y_tr)

        if len(unique_classes) < 2:
            full_probs[:, int(unique_classes[0])] = 1.0
        else:
            clf = self._make_classifier(str(X_train.device))
            clf.fit(x_tr, y_tr)
            probs = clf.predict_proba(x_te)
            for i, cls in enumerate(clf.classes_):
                if int(cls) < self.max_classes:
                    full_probs[:, int(cls)] = probs[:, i]

        log_probs = torch.tensor(np.log(np.clip(full_probs, 1e-8, 1.0)), device=X_train.device)
        return log_probs.unsqueeze(0)  # (1, n_test, max_classes)


class MeanTabPFNBaseline(_PooledTabPFNBaseline):
    """Mean pooling followed by TabPFN v2."""

    stats = ("mean",)
    version = "v2"


class SummTabPFNBaseline(_PooledTabPFNBaseline):
    """Six summary statistics followed by TabPFN v2."""

    stats = SVMSummBaseline.STATS
    version = "v2"


class MeanTabPFN25Baseline(_PooledTabPFNBaseline):
    """Mean pooling followed by TabPFN v2.5."""

    stats = ("mean",)
    version = "v2.5"


class SummTabPFN25Baseline(_PooledTabPFNBaseline):
    """Six summary statistics followed by TabPFN v2.5."""

    stats = SVMSummBaseline.STATS
    version = "v2.5"


class MeanTabPFN3Baseline(_PooledTabPFNBaseline):
    """Mean pooling followed by TabPFN v3."""

    stats = ("mean",)
    version = "v3"


class SummTabPFN3Baseline(_PooledTabPFNBaseline):
    """Six summary statistics followed by TabPFN v3."""

    stats = SVMSummBaseline.STATS
    version = "v3"


class MeanTabPFN35Baseline(_PooledTabPFNBaseline):
    """Mean pooling followed by TabPFN v3.5."""

    stats = ("mean",)
    version = "v3.5"


class SummTabPFN35Baseline(_PooledTabPFNBaseline):
    """Six summary statistics followed by TabPFN v3.5."""

    stats = SVMSummBaseline.STATS
    version = "v3.5"
