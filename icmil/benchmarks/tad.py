"""TAD traffic-anomaly benchmark tasks (PCA-25 / PCA-50): reads the frozen split.

Each video is a bag and its frames are the instances; a bag is positive when the
video contains an anomaly. Instances are ResNet-50 features reduced by PCA to 25
or 50 dimensions, one shipped file per width. Each file holds the dataset's own
train/test partition as a single ``split_0`` group.

Source dataset: https://huggingface.co/datasets/torchmil/TAD_MIL
"""

from __future__ import annotations

from pathlib import Path

from icmil.benchmarks.fixed_split import FixedSplitTask

TAD_VARIANTS = ("pca25", "pca50")


class TADTask(FixedSplitTask):
    """Iterate over the frozen split of a single TAD variant."""

    VARIANTS = TAD_VARIANTS

    def __init__(self, variant: str, h5_path: str | Path) -> None:
        if variant not in TAD_VARIANTS:
            raise ValueError(f"Unknown TAD variant {variant!r}. Valid: {TAD_VARIANTS}")
        super().__init__(h5_path, name=f"tad_{variant}")
