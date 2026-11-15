"""Train-set subsamples of a frozen split, for studies over the number of training bags.

Reads ``split_0`` of a fixed-split ``.h5`` and yields class-stratified subsamples of
its training bags against the full, unchanged test set. The subsamples are drawn at
load time from a seed, so one file serves every bag count.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np


def stratified_subsample(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick ``n`` sorted indices into ``y`` that keep the class proportions.

    Each class contributes ``floor(n * share)`` bags; the remainder goes to the
    classes with the largest fractional parts.
    """
    classes, counts = np.unique(y, return_counts=True)
    exact = n * counts / counts.sum()
    take = np.floor(exact).astype(int)
    for c in np.argsort(-(exact - take), kind="stable")[: n - take.sum()]:
        take[c] += 1
    picked = [rng.choice(np.flatnonzero(y == cls), size=k, replace=False) for cls, k in zip(classes, take)]
    return np.sort(np.concatenate(picked))


class SubsampledSplitTask:
    """``n_draws`` class-stratified subsamples of ``n_train_bags`` training bags.

    Asking for the whole training set yields a single draw, since every draw
    would be identical.
    """

    def __init__(self, h5_path: str | Path, n_train_bags: int, name: str, n_draws: int = 5, seed: int = 42) -> None:
        self._h5_path = Path(h5_path)
        if not self._h5_path.exists():
            raise FileNotFoundError(f"Fixed-split H5 not found: {self._h5_path}")
        with h5py.File(self._h5_path, "r") as h5:
            n_total, _, self._n_features = h5["split_0"]["X_train"].shape
        if not 0 < n_train_bags <= n_total:
            raise ValueError(f"n_train_bags must be in [1, {n_total}], got {n_train_bags}.")
        self._n_train_bags = n_train_bags
        self._n_splits = 1 if n_train_bags == n_total else n_draws
        self._seed = seed
        self._name = f"{name}_{n_train_bags}bags"

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_features(self) -> int:
        return int(self._n_features)

    @property
    def n_splits(self) -> int:
        return self._n_splits

    def sample_datasets(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (X_train, y_train, X_test, y_test) for every subsample."""
        with h5py.File(self._h5_path, "r") as h5:
            split = h5["split_0"]
            X_train, y_train = split["X_train"][:].astype(np.float32), split["y_train"][:].astype(np.int64)
            X_test, y_test = split["X_test"][:].astype(np.float32), split["y_test"][:].astype(np.int64)
        for draw in range(self._n_splits):
            rng = np.random.default_rng([self._seed, self._n_train_bags, draw])
            idx = stratified_subsample(y_train, self._n_train_bags, rng)
            yield X_train[idx], y_train[idx], X_test, y_test
