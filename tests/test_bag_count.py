"""The bag-count study: load-time subsampling, the table, and the registered files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from icmil.bag_count import BENCHMARKS, format_table
from icmil.benchmarks.subsampled import SubsampledSplitTask, stratified_subsample
from icmil.model import build_icmil


@pytest.fixture
def fixed_split(tmp_path: Path) -> Path:
    path = tmp_path / "fixed.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as h5:
        split = h5.create_group("split_0")
        split.create_dataset("X_train", data=rng.normal(size=(40, 3, 5)).astype(np.float32))
        split.create_dataset("y_train", data=np.array([0] * 30 + [1] * 10))
        split.create_dataset("X_test", data=rng.normal(size=(8, 3, 5)).astype(np.float32))
        split.create_dataset("y_test", data=np.array([0, 1] * 4))
    return path


def test_subsample_keeps_class_proportions() -> None:
    y = np.array([0] * 30 + [1] * 10)
    idx = stratified_subsample(y, 20, np.random.default_rng(0))
    assert len(set(idx)) == 20
    assert (y[idx] == 1).sum() == 5


def test_subsampled_task_is_deterministic_and_leaves_the_test_set_alone(fixed_split: Path) -> None:
    task = SubsampledSplitTask(fixed_split, n_train_bags=20, name="toy", n_draws=3)
    first, second = list(task.sample_datasets()), list(task.sample_datasets())
    assert task.name == "toy_20bags"
    assert len(first) == task.n_splits == 3
    for (Xa, ya, Xta, _), (Xb, yb, _, _) in zip(first, second):
        assert Xa.shape == (20, 3, 5) and Xta.shape == (8, 3, 5)
        assert np.array_equal(Xa, Xb) and np.array_equal(ya, yb)
    assert not np.array_equal(first[0][0], first[1][0]), "draws differ from one another"


def test_the_full_training_set_is_a_single_draw(fixed_split: Path) -> None:
    assert SubsampledSplitTask(fixed_split, n_train_bags=40, name="toy").n_splits == 1
    with pytest.raises(ValueError, match="n_train_bags"):
        SubsampledSplitTask(fixed_split, n_train_bags=41, name="toy")


def test_table_columns_are_training_bags_in_order() -> None:
    def cell(n_train: float, auroc: float) -> dict:
        return {"task": {"roc_auc": auroc, "roc_auc_sem": 0.01, "n_train_bags_mean": n_train}}

    cells = {
        200: {"abmil": cell(180.0, 0.8), "ICMIL": cell(180.0, 0.9)},
        100: {"ICMIL": cell(90.0, 0.85)},
    }
    lines = format_table(cells).splitlines()
    assert lines[0] == "| Model | 90 | 180 |"
    assert "| ICMIL | 85.0 ±1.0 | 90.0 ±1.0 |" in lines
    assert "| abmil | — | 80.0 ±1.0 |" in lines, "a cell that was not run is left empty"


def test_the_paper_cells_read_the_paper_files() -> None:
    assert BENCHMARKS["mnist_pos_neg"][2][100] == "mnist_xai_benchmark_100bags.h5"
    assert BENCHMARKS["rsna_ich"][2][100] == "rsna_ich_resnet50_draws_100bags.h5"


def test_final_norm_is_on_unless_a_checkpoint_says_otherwise() -> None:
    arch = dict(in_features=5, embedding_size=16, mlp_hidden_size=32, num_column_row_iterations=1)
    assert "final_norm.weight" in build_icmil(arch).state_dict()
    assert "final_norm.weight" not in build_icmil(arch | {"final_norm": False}).state_dict()
