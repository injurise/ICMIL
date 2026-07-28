"""Andrews image MIL benchmark tasks (Fox / Tiger / Elephant): reads frozen splits.

The three classic image-bag datasets of Andrews et al. (2002). Each ``split_i``
group of the shipped ``andrews_mil_benchmark.h5`` is one fold of a stratified
5-fold CV over bags, so every bag is tested exactly once.
"""

from __future__ import annotations

from collections.abc import Iterator

import h5py
import numpy as np

ANDREWS_VARIANTS = ("fox", "tiger", "elephant")


class AndrewsImageTask:
    """Iterate over the frozen CV folds of a single Andrews variant."""

    VARIANTS = ANDREWS_VARIANTS

    def __init__(self, variant: str, h5_path: str) -> None:
        if variant not in ANDREWS_VARIANTS:
            raise ValueError(f"Unknown Andrews variant {variant!r}. Valid: {ANDREWS_VARIANTS}")
        self._variant = variant
        self._h5_path = h5_path
        with h5py.File(h5_path, "r") as h5:
            if variant not in h5:
                raise ValueError(f"{h5_path} has no group {variant!r}; found {sorted(h5.keys())}")
            grp = h5[variant]
            split_keys = [k for k in grp if k.startswith("split_")]
            if not split_keys:
                raise ValueError(f"{h5_path}[{variant}] contains no split_* groups")
            self._n_splits = len(split_keys)
            self._n_features = int(grp[next(iter(sorted(split_keys)))]["X_train"].shape[-1])

    @property
    def name(self) -> str:
        return f"andrews_{self._variant}"

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def n_splits(self) -> int:
        return self._n_splits

    def sample_datasets(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (X_train, y_train, X_test, y_test) for every ``split_*`` group."""
        with h5py.File(self._h5_path, "r") as h5:
            grp = h5[self._variant]
            for key in sorted(grp.keys()):
                if not key.startswith("split_"):
                    continue
                sg = grp[key]
                yield (
                    sg["X_train"][:].astype(np.float32),
                    sg["y_train"][:].astype(np.int64),
                    sg["X_test"][:].astype(np.float32),
                    sg["y_test"][:].astype(np.int64),
                )
