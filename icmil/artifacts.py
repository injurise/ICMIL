"""Locate the benchmark datasets and model checkpoints.

Resolution order for each kind of artifact:

1. an explicit ``source`` directory passed by the caller (``--data`` / ``--models``),
2. the ``ICMIL_DATA_DIR`` / ``ICMIL_CKPT_DIR`` environment variables,
3. the in-repo defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root: <root>/icmil/artifacts.py -> <root>
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = _REPO_ROOT / "datasets"
DEFAULT_CKPT_DIR = _REPO_ROOT / "checkpoints"

# Benchmark files, keyed by short name.
DATASET_FILES: dict[str, str] = {
    "uci": "uci_benchmark.h5",
    "mnist_xai": "mnist_xai_benchmark_100bags.h5",
    "andrews": "andrews_mil_benchmark.h5",
    "tcga": "tcga_uni2_luad_vs_lusc.h5",
    "rsna": "rsna_ich_resnet50_draws_100bags.h5",
}

# The three trained model seeds. The names are opaque run labels.
CHECKPOINT_FILES: dict[str, str] = {
    "c5trd795": "icmil-c5trd795.pt",
    "ggwsqibd": "icmil-ggwsqibd.pt",
    "k337zhz1": "icmil-k337zhz1.pt",
}
SEEDS: tuple[str, ...] = tuple(CHECKPOINT_FILES)


def _resolve_dir(source: str | Path | None, env_var: str, default: Path) -> Path:
    """Pick the artifact directory, in the documented precedence order."""
    if source is not None:
        return Path(source).expanduser()
    from_env = os.environ.get(env_var)
    if from_env:
        return Path(from_env).expanduser()
    return default


def _fetch(directory: Path, filename: str, kind: str, env_var: str) -> Path:
    """Return ``directory/filename``, or explain how to point at the right place."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{kind} directory {directory} does not exist. Pass an explicit directory or set {env_var}."
        )
    path = directory / filename
    if not path.is_file():
        present = sorted(p.name for p in directory.iterdir() if p.is_file() and not p.name.startswith("."))
        raise FileNotFoundError(f"{kind} {filename} not found in {directory}. Files present: {present or '(none)'}")
    return path


def resolve_dataset(name: str, source: str | Path | None = None) -> Path:
    """Return the local path to a benchmark ``.h5``.

    Args:
        name: Short dataset key, one of :data:`DATASET_FILES`.
        source: Directory holding the ``.h5`` files. Defaults to ``ICMIL_DATA_DIR``
            or the in-repo ``datasets/``.
    """
    if name not in DATASET_FILES:
        raise KeyError(f"Unknown dataset {name!r}; expected one of {sorted(DATASET_FILES)}")
    directory = _resolve_dir(source, "ICMIL_DATA_DIR", DEFAULT_DATA_DIR)
    return _fetch(directory, DATASET_FILES[name], "Dataset", "ICMIL_DATA_DIR")


def resolve_checkpoint(seed: str, source: str | Path | None = None) -> Path:
    """Return the local path to a trained ICMIL checkpoint.

    Args:
        seed: One of :data:`SEEDS`.
        source: Directory holding the ``.pt`` files. Defaults to ``ICMIL_CKPT_DIR``
            or the in-repo ``checkpoints/``.
    """
    if seed not in CHECKPOINT_FILES:
        raise KeyError(f"Unknown seed {seed!r}; expected one of {sorted(CHECKPOINT_FILES)}")
    directory = _resolve_dir(source, "ICMIL_CKPT_DIR", DEFAULT_CKPT_DIR)
    return _fetch(directory, CHECKPOINT_FILES[seed], "Checkpoint", "ICMIL_CKPT_DIR")
