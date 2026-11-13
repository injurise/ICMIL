"""Shared reader for benchmark files that hold frozen ``split_*`` groups at their root."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np


class FixedSplitTask:
    """Iterate over the frozen splits of a benchmark file.

    H5 layout::

        split_0/{X_train, y_train, X_test, y_test}
        split_1/...
    """

    def __init__(self, h5_path: str | Path, name: str) -> None:
        self._h5_path = Path(h5_path)
        self._name = name
        if not self._h5_path.exists():
            raise FileNotFoundError(f"Benchmark H5 not found: {self._h5_path}")
        with h5py.File(self._h5_path, "r") as h5:
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
