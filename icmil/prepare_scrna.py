"""Build ``datasets/scrna_{covid,icb}.h5`` from the preprocessed files of Do & Lähdesmäki (2025).

Download ``covid.h5ad`` and ``icb.h5ad`` from the Google Drive folder linked in
https://github.com/minhchaudo/hier-mil into ``datasets/raw/`` first. This checks their sha256,
samples a fixed number of cells per patient and freezes the paper's folds.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold

from icmil.artifacts import DEFAULT_DATA_DIR

N_REPEATS = 10
N_FOLDS = 10
SAMPLE_SEED = 42

DRIVE_FOLDER = "https://drive.google.com/drive/folders/1x3zQX3RKMuYqNLNPwaSV82LAH97REiU-"

# dataset -> (sha256 of the .h5ad, cells per patient)
SOURCES: dict[str, tuple[str, int]] = {
    "covid": ("6232f1e55d07c8111619863f46981b380fc94d870ca841d6918dca90f75c59f3", 1000),
    "icb": ("330bf34c39f66e3c6c33f9a5c850ba712f1a93300c1cacff25fed4343982fe95", 242),
}


def verified_source(dataset: str, raw_dir: Path) -> Path:
    """``<dataset>.h5ad`` in ``raw_dir``, checked against its sha256."""
    sha256, _ = SOURCES[dataset]
    path = raw_dir / f"{dataset}.h5ad"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; download {path.name} from {DRIVE_FOLDER}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"{path} has sha256 {digest}, expected {sha256}")
    return path


def read_patients(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    """Each patient's cells and label, patients in order of first appearance in the file."""
    with h5py.File(path, "r") as f:
        patient_of_cell = f["obs/patient/categories"][:][f["obs/patient/codes"][:]]
        label_of_cell = f["obs/label"][:]
        cells = f["X"][:].astype(np.float32)
    _, first = np.unique(patient_of_cell, return_index=True)
    rows = [patient_of_cell == patient for patient in patient_of_cell[np.sort(first)]]
    return [cells[r] for r in rows], np.array([label_of_cell[r][0] for r in rows])


def sample_cells(bags: list[np.ndarray], n_cells: int) -> np.ndarray:
    """``n_cells`` cells per patient, with replacement only for smaller patients."""
    out = []
    for i, bag in enumerate(bags):
        rng = np.random.default_rng([SAMPLE_SEED, i])
        out.append(bag[np.sort(rng.choice(len(bag), size=n_cells, replace=len(bag) < n_cells))])
    return np.stack(out)


def make_folds(labels: np.ndarray) -> np.ndarray:
    """Fold of every patient per repetition, ``StratifiedKFold(10, shuffle=True, random_state=r)``."""
    folds = np.empty((N_REPEATS, len(labels)), dtype=np.int8)
    for repeat in range(N_REPEATS):
        splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=repeat)
        for fold, (_, test) in enumerate(splitter.split(labels, labels)):
            folds[repeat, test] = fold
    return folds


def build(dataset: str, raw_dir: Path, out_dir: Path) -> Path:
    """Write ``scrna_<dataset>.h5`` with ``bags``, ``labels`` and ``folds``."""
    bags, labels = read_patients(verified_source(dataset, raw_dir))
    out = out_dir / f"scrna_{dataset}.h5"
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as h5:
        h5.create_dataset("bags", data=sample_cells(bags, SOURCES[dataset][1]), compression="gzip")
        h5.create_dataset("labels", data=labels)
        h5.create_dataset("folds", data=make_folds(labels))
    print(f"{out}: {len(labels)} patients, {labels.sum()} positive")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(DEFAULT_DATA_DIR / "raw"), help="Directory with covid.h5ad, icb.h5ad")
    parser.add_argument("--out", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    for dataset in SOURCES:
        build(dataset, Path(args.raw_dir), Path(args.out))


if __name__ == "__main__":
    main()
