"""MNIST-XAI MIL benchmark tasks: reads frozen draws.

Synthetic MIL problems built from MNIST digits, each defined by a different rule
linking the instances of a bag to its label:

* ``smil``            — a single witness digit determines the label,
* ``four_bags``       — a four-way variant of the witness rule,
* ``pos_neg``         — positive and negative evidence digits must be weighed,
* ``adjacent_pairs``  — the label depends on a *pair* of digits occurring
  together, so no single instance is decisive.

Each ``draw_i`` group of the shipped ``mnist_xai_benchmark_100bags.h5`` is an
independent (train, test) pair with a freshly sampled rule.
"""

from __future__ import annotations

from collections.abc import Iterator

import h5py
import numpy as np

VARIANTS = ("smil", "four_bags", "pos_neg", "adjacent_pairs")


class MNISTXAITask:
    """Iterate over the frozen draws of a single MNIST-XAI variant."""

    VARIANTS = VARIANTS

    def __init__(self, variant: str, h5_path: str) -> None:
        self._variant = variant
        self._h5_path = h5_path
        with h5py.File(h5_path, "r") as h5:
            # Validate against the file rather than the constant: the two could
            # otherwise disagree silently if the benchmark is ever rebuilt.
            if variant not in h5:
                raise ValueError(f"{h5_path} has no group {variant!r}; found {sorted(h5.keys())}")
            raw = h5.attrs.get("num_bags", 0)
            self._num_bags = int(raw) if raw is not None else 0
            grp = h5[variant]
            draw_keys = [k for k in grp if k.startswith("draw_")]
            if not draw_keys:
                raise ValueError(f"{h5_path}[{variant}] contains no draw_* groups")
            self._n_splits = len(draw_keys)
            self._n_features = int(grp[next(iter(sorted(draw_keys)))]["X_train"].shape[-1])

    @property
    def name(self) -> str:
        if self._num_bags > 0:
            return f"mnist_xai_{self._variant}_{self._num_bags}bags"
        return f"mnist_xai_{self._variant}"

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def n_splits(self) -> int:
        return self._n_splits

    def sample_datasets(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (X_train, y_train, X_test, y_test) for every ``draw_*`` group."""
        with h5py.File(self._h5_path, "r") as h5:
            grp = h5[self._variant]
            for key in sorted(grp.keys()):
                if not key.startswith("draw_"):
                    continue
                sg = grp[key]
                yield (
                    sg["X_train"][:].astype(np.float32),
                    sg["y_train"][:].astype(np.int64),
                    sg["X_test"][:].astype(np.float32),
                    sg["y_test"][:].astype(np.int64),
                )
