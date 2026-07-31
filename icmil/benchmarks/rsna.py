"""RSNA intracranial-haemorrhage MIL benchmark task: reads frozen draws.

Each CT scan is a bag and its slices are the instances; a bag is positive when
any slice shows a haemorrhage. Instances are ResNet-50 features. Each ``draw_i``
group of the shipped ``rsna_ich_resnet50_draws_100bags.h5`` is a (train, test)
pair drawn at preparation time; draws differ in which scans form the context set.

Source dataset: https://huggingface.co/datasets/torchmil/RSNA_ICH_MIL
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np


class RSNAMILDrawsTask:
    """Iterate over the frozen draws of the RSNA-ICH benchmark."""

    def __init__(self, h5_path: str | Path, default_features: str = "resnet50", name_prefix: str = "rsna_ich") -> None:
        self._h5_path = Path(h5_path)
        if not self._h5_path.exists():
            raise FileNotFoundError(f"RSNA draws H5 not found: {self._h5_path}")
        with h5py.File(self._h5_path, "r") as h5:
            features = h5.attrs.get("features", default_features)
            if isinstance(features, bytes):
                features = features.decode()
            n_train_bags = int(h5.attrs.get("n_train_bags", 0))
            draw_keys = [k for k in h5 if k.startswith("draw_")]
            if not draw_keys:
                raise ValueError(f"{h5_path} contains no draw_* groups")
            self._n_splits = len(draw_keys)
            self._n_features = int(h5[next(iter(sorted(draw_keys)))]["X_train"].shape[-1])
        self._name = f"{name_prefix}_{features}_draws_{n_train_bags}bags"

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def n_splits(self) -> int:
        return self._n_splits

    def sample_datasets(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (X_train, y_train, X_test, y_test) for every ``draw_*`` group."""
        with h5py.File(self._h5_path, "r") as h5:
            for key in sorted(h5):
                if not key.startswith("draw_"):
                    continue
                sg = h5[key]
                yield (
                    sg["X_train"][:].astype(np.float32),
                    sg["y_train"][:].astype(np.int64),
                    sg["X_test"][:].astype(np.float32),
                    sg["y_test"][:].astype(np.int64),
                )
