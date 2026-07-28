"""TCGA slide-level benchmark task (LUAD vs LUSC): reads frozen splits.

Bags are whole-slide images, instances are patch embeddings from a pathology
foundation model. The H5 is self-identifying via its ``variant`` attribute, so
this one class serves any fixed TCGA benchmark built the same way.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np


class TCGAFixedTask:
    """Iterate over the frozen splits of a fixed TCGA benchmark.

    H5 layout::

        attrs: {variant, patch_fm, projects, n_patients_per_project, bag_size,
                test_fraction, n_splits, n_pca_components, seed}
        split_0/{X_train, y_train, X_test, y_test}
        split_1/...
    """

    def __init__(self, h5_path: str | Path) -> None:
        self._h5_path = Path(h5_path)
        if not self._h5_path.exists():
            raise FileNotFoundError(f"TCGA benchmark H5 not found: {self._h5_path}")
        with h5py.File(self._h5_path, "r") as h5:
            variant = h5.attrs["variant"]
            if isinstance(variant, bytes):
                variant = variant.decode()
            self._name = f"tcga_{variant}"
            split_keys = [k for k in h5 if k.startswith("split_")]
            if not split_keys:
                raise ValueError(f"{h5_path} contains no split_* groups")
            self._n_splits = len(split_keys)
            self._n_features = int(h5["split_0"]["X_train"].shape[-1])

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
        """Yield (X_train, y_train, X_test, y_test) for every ``split_*`` group."""
        with h5py.File(self._h5_path, "r") as h5:
            for key in sorted(k for k in h5 if k.startswith("split_")):
                sg = h5[key]
                yield (
                    sg["X_train"][:].astype(np.float32),
                    sg["y_train"][:].astype(np.int64),
                    sg["X_test"][:].astype(np.float32),
                    sg["y_test"][:].astype(np.int64),
                )
