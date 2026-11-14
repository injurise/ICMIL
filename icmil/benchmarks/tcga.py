"""TCGA slide-level benchmark task (LUAD vs LUSC): reads frozen splits.

Bags are whole-slide images, instances are patch embeddings from a pathology
foundation model.
"""

from __future__ import annotations

from pathlib import Path

import h5py

from icmil.benchmarks.fixed_split import FixedSplitTask


class TCGAFixedTask(FixedSplitTask):
    """Iterate over the frozen splits of a fixed TCGA benchmark.

    H5 layout::

        attrs: {variant, patch_fm, projects, n_patients_per_project, bag_size,
                test_fraction, n_splits, n_pca_components, seed}
        split_0/{X_train, y_train, X_test, y_test}
        split_1/...
    """

    def __init__(self, h5_path: str | Path) -> None:
        super().__init__(h5_path, name="tcga")
        with h5py.File(self._h5_path, "r") as h5:
            variant = h5.attrs["variant"]
        if isinstance(variant, bytes):
            variant = variant.decode()
        self._name = f"tcga_{variant}"
