"""The single-cell benchmarks: building the files, the folds, PCA and pooled scoring."""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn

from icmil import prepare_scrna
from icmil.benchmarks import create_benchmark_tasks, per_split_metrics
from icmil.benchmarks.scrna import ScRNATask
from icmil.reproduce import SCRNA_TASKS, TASKS, _parse_selection

N_PATIENTS, N_FEATURES, N_CELLS = 30, 40, 12


def _write_h5ad(path: Path, seed: int = 0) -> None:
    """A tiny ``.h5ad`` in AnnData's on-disk layout: dense X, categorical ``patient``, plain ``label``."""
    rng = np.random.default_rng(seed)
    patient_label = np.arange(N_PATIENTS) % 3 == 0  # 10 positive, 20 negative
    sizes = rng.integers(5, 20, size=N_PATIENTS)  # some patients have fewer cells than N_CELLS
    patient_of_cell = rng.permutation(np.repeat(np.arange(N_PATIENTS), sizes))  # cells not contiguous
    cells = rng.normal(size=(len(patient_of_cell), N_FEATURES)) + 2.0 * patient_label[patient_of_cell, None]
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=cells.astype(np.float32))
        patient = f.create_group("obs/patient")
        patient.create_dataset("categories", data=np.array([f"P{i:02d}" for i in range(N_PATIENTS)], dtype="S"))
        patient.create_dataset("codes", data=patient_of_cell.astype(np.int8))
        f["obs"].create_dataset("label", data=patient_label[patient_of_cell].astype(np.int64))


@pytest.fixture
def scrna_h5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A benchmark file built by ``prepare_scrna`` from the synthetic ``.h5ad``."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_h5ad(raw / "covid.h5ad")
    digest = hashlib.sha256((raw / "covid.h5ad").read_bytes()).hexdigest()
    monkeypatch.setitem(prepare_scrna.SOURCES, "covid", ("unused", digest, N_CELLS))
    return prepare_scrna.build("covid", raw, tmp_path / "datasets")


def test_build_writes_bags_labels_and_folds(scrna_h5: Path) -> None:
    with h5py.File(scrna_h5, "r") as h5:
        bags, labels, folds = h5["bags"][:], h5["labels"][:], h5["folds"][:]
    assert bags.shape == (N_PATIENTS, N_CELLS, N_FEATURES)
    assert labels.sum() == 10
    assert folds.shape == (prepare_scrna.N_REPEATS, N_PATIENTS)
    for repeat, fold_of_patient in enumerate(folds):  # the paper's folds, patients in file order
        expected = StratifiedKFold(prepare_scrna.N_FOLDS, shuffle=True, random_state=repeat).split(labels, labels)
        for fold, (_, test) in enumerate(expected):
            assert set(np.flatnonzero(fold_of_patient == fold)) == set(test)


def test_build_rejects_a_changed_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_h5ad(tmp_path / "covid.h5ad")
    monkeypatch.setitem(prepare_scrna.SOURCES, "covid", ("unused", "0" * 64, N_CELLS))
    with pytest.raises(ValueError, match="sha256"):
        prepare_scrna.build("covid", tmp_path, tmp_path / "datasets")


def test_patients_in_order_of_first_appearance(tmp_path: Path) -> None:
    _write_h5ad(tmp_path / "covid.h5ad")
    bags, labels = prepare_scrna.read_patients(tmp_path / "covid.h5ad")
    with h5py.File(tmp_path / "covid.h5ad", "r") as f:
        cells, codes = f["X"][:], f["obs/patient/codes"][:]
    np.testing.assert_array_equal(bags[0], cells[codes == codes[0]])  # first patient = patient of the first cell
    assert sum(len(b) for b in bags) == len(cells)


@pytest.mark.parametrize("n_pca", [5, 10])
def test_task_yields_every_fold_with_pca(scrna_h5: Path, n_pca: int) -> None:
    task = ScRNATask("covid", n_pca, scrna_h5)
    assert (task.name, task.n_features, task.n_splits, task.splits_per_repeat) == (
        f"scrna_covid_pca{n_pca}",
        n_pca,
        100,
        10,
    )
    splits = list(task.sample_datasets())
    assert len(splits) == 100
    for repeat in range(10):
        tested = sum(len(y_test) for _, _, _, y_test in splits[10 * repeat : 10 * repeat + 10])
        assert tested == N_PATIENTS  # every patient tests once per repetition
    X_train, y_train, X_test, _ = splits[0]
    assert X_train.shape[1:] == X_test.shape[1:] == (N_CELLS, n_pca)
    np.testing.assert_allclose(X_train.reshape(-1, n_pca).mean(axis=0), 0, atol=1e-4)  # PCA fitted on train cells


class _BagMean(nn.Module):
    """Scores a bag by the mean of its first feature."""

    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
        score = X_test[..., 0].mean(dim=-1)
        return torch.stack([-score, score], dim=-1)


def test_predictions_are_pooled_per_repetition(scrna_h5: Path) -> None:
    task = ScRNATask("covid", 5, scrna_h5)
    per_repeat, _, timings = per_split_metrics(_BagMean(), task, torch.device("cpu"), None)
    assert len(per_repeat) == 10 and len(timings) == 100
    for repeat, metrics in enumerate(per_repeat):
        splits = list(task.sample_datasets())[10 * repeat : 10 * repeat + 10]
        y = np.concatenate([y_test for _, _, _, y_test in splits])
        score = np.concatenate([X_test[..., 0].mean(axis=-1) for _, _, X_test, _ in splits])
        assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y, score))


def test_scrna_tasks_are_opt_in(scrna_h5: Path) -> None:
    assert not set(SCRNA_TASKS) & set(TASKS)
    assert _parse_selection("scrna_covid_pca50", TASKS, "task", extra=tuple(SCRNA_TASKS)) == ["scrna_covid_pca50"]
    (task,) = create_benchmark_tasks(["scrna_covid_pca50"], scrna_covid_h5_path=str(scrna_h5))
    assert task.name == "scrna_covid_pca50"
