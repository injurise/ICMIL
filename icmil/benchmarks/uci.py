"""UCI MIL benchmark tasks (Musk1 / Musk2 / Letters / HEPMASS): reads frozen splits.

Each ``split_i`` group of the shipped ``uci_benchmark.h5`` is one independent
(train, test) pair. Musk1/Musk2 are natural MIL problems and their splits are
the folds of a stratified 5-fold CV over molecules, so every bag is tested
exactly once; Letters and HEPMASS are constructed-bag problems whose splits are
random stratified holdouts. Both were frozen at preparation time.
"""

from __future__ import annotations

from collections.abc import Iterator

import h5py
import numpy as np

UCI_VARIANTS = ("musk1", "musk2", "letters", "hepmass")


class UCIMILTask:
    """Iterate over the frozen splits of a single UCI variant."""

    VARIANTS = UCI_VARIANTS

    def __init__(self, variant: str, h5_path: str) -> None:
        if variant not in UCI_VARIANTS:
            raise ValueError(f"Unknown UCI variant {variant!r}. Valid: {UCI_VARIANTS}")
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
        return f"uci_{self._variant}"

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
