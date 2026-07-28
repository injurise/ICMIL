"""Shared fixtures.

Most tests run against a tiny synthetic ``.h5`` built here rather than the shipped
benchmarks, so the suite works on a clean checkout with no data and finishes in
seconds. Tests that genuinely need the real artifacts are marked ``artifacts`` and
skip when they are absent.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = REPO_ROOT / "datasets"
CHECKPOINTS = REPO_ROOT / "checkpoints"

requires_artifacts = pytest.mark.skipif(
    not (DATASETS / "uci_benchmark.h5").exists() or not (CHECKPOINTS / "icmil-c5trd795.pt").exists(),
    reason="needs the shipped datasets/ and checkpoints/",
)


def _write_splits(
    group: h5py.Group, prefix: str, n_splits: int, n_bags: int, bag_size: int, n_features: int, seed: int
) -> None:
    """Write ``n_splits`` train/test pairs of the shape the task readers expect."""
    rng = np.random.default_rng(seed)
    for i in range(n_splits):
        sub = group.create_group(f"{prefix}_{i}")
        n_test = max(2, n_bags // 5)
        for name, count in (("train", n_bags), ("test", n_test)):
            sub.create_dataset(f"X_{name}", data=rng.normal(size=(count, bag_size, n_features)).astype(np.float32))
            # Both classes always present, so AUROC is defined on every split.
            labels = np.array([j % 2 for j in range(count)], dtype=np.int64)
            sub.create_dataset(f"y_{name}", data=rng.permutation(labels))


@pytest.fixture(scope="session")
def synthetic_datasets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory of miniature stand-ins for the five shipped benchmark files."""
    out = tmp_path_factory.mktemp("datasets")

    with h5py.File(out / "uci_benchmark.h5", "w") as h5:
        for i, variant in enumerate(("musk1", "musk2", "letters", "hepmass")):
            _write_splits(h5.create_group(variant), "split", 3, 12, 4, 25, seed=i)

    with h5py.File(out / "andrews_mil_benchmark.h5", "w") as h5:
        for i, variant in enumerate(("fox", "tiger", "elephant")):
            _write_splits(h5.create_group(variant), "split", 3, 12, 4, 25, seed=10 + i)

    with h5py.File(out / "mnist_xai_benchmark_100bags.h5", "w") as h5:
        h5.attrs["num_bags"] = 100
        for i, variant in enumerate(("smil", "four_bags", "pos_neg", "adjacent_pairs")):
            _write_splits(h5.create_group(variant), "draw", 3, 12, 4, 25, seed=20 + i)

    with h5py.File(out / "tcga_uni2_luad_vs_lusc.h5", "w") as h5:
        h5.attrs["variant"] = "luad_vs_lusc"
        _write_splits(h5, "split", 3, 12, 4, 25, seed=30)

    with h5py.File(out / "rsna_ich_resnet50_draws_100bags.h5", "w") as h5:
        h5.attrs["features"] = "resnet50"
        h5.attrs["n_train_bags"] = 100
        _write_splits(h5, "draw", 3, 12, 4, 25, seed=40)

    return out


@pytest.fixture(scope="session")
def synthetic_dataset_kwargs(synthetic_datasets: Path) -> dict[str, str]:
    """``create_benchmark_tasks`` keyword arguments pointing at the synthetic files."""
    return {
        "uci_h5_path": str(synthetic_datasets / "uci_benchmark.h5"),
        "andrews_h5_path": str(synthetic_datasets / "andrews_mil_benchmark.h5"),
        "mnist_xai_h5_path": str(synthetic_datasets / "mnist_xai_benchmark_100bags.h5"),
        "tcga_fixed_h5_path": str(synthetic_datasets / "tcga_uni2_luad_vs_lusc.h5"),
        "rsna_draws_h5_path": str(synthetic_datasets / "rsna_ich_resnet50_draws_100bags.h5"),
    }
