"""Single-cell patient classification (COVID, ICB) with the protocol of Do & Lähdesmäki (2025).

A bag is a patient, an instance a cell. Splits are the 10 folds of 10 cross-validation
repetitions; the harness pools the predictions of each repetition. PCA is fitted per fold on
the training cells. Build the files with ``python -m icmil.prepare_scrna``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA


class ScRNATask:
    def __init__(self, dataset: str, n_pca: int, h5_path: str | Path) -> None:
        self.name = f"scrna_{dataset}_pca{n_pca}"
        self.n_features = n_pca
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as h5:
            n_repeats, n_folds = h5["folds"].shape[0], int(h5["folds"][:].max()) + 1
        self.splits_per_repeat = n_folds
        self.n_splits = n_repeats * n_folds

    def sample_datasets(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        with h5py.File(self.h5_path, "r") as h5:
            bags, labels, folds = h5["bags"][:], h5["labels"][:], h5["folds"][:]
        for fold_of_patient in folds:
            for fold in range(self.splits_per_repeat):
                train, test = bags[fold_of_patient != fold], bags[fold_of_patient == fold]
                pca = PCA(self.n_features, svd_solver="full").fit(train.reshape(-1, bags.shape[-1]))
                reduce = lambda X: pca.transform(X.reshape(-1, X.shape[-1])).reshape(*X.shape[:2], -1)  # noqa: E731
                yield reduce(train), labels[fold_of_patient != fold], reduce(test), labels[fold_of_patient == fold]
