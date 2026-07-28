"""The reproduction pipeline: task construction, model shape, CLI, and the table.

Runs against the synthetic fixtures, so this exercises the whole wiring — including
the end-to-end path — without the shipped artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from icmil.benchmarks import create_benchmark_tasks
from icmil.benchmarks.registry import TASK_NAMES
from icmil.reproduce import BASELINE_SPECS, TASKS, _parse_selection, _write_table, evaluate


# --------------------------------------------------------------------------- tasks


def test_every_paper_task_is_registered() -> None:
    assert set(TASKS) <= set(TASK_NAMES)
    assert len(TASKS) == 12, "the paper reports twelve tasks"


def test_tasks_build_and_yield_the_documented_shapes(synthetic_dataset_kwargs: dict) -> None:
    tasks = create_benchmark_tasks(list(TASKS), **synthetic_dataset_kwargs)
    assert len(tasks) == len(TASKS)
    for task in tasks:
        X_train, y_train, X_test, y_test = next(iter(task.sample_datasets()))
        assert X_train.ndim == 3, "X is (n_bags, bag_size, n_features)"
        assert y_train.ndim == 1 and len(y_train) == len(X_train)
        assert X_test.shape[1:] == X_train.shape[1:]
        assert len(y_test) == len(X_test)


def test_mnist_task_names_carry_their_bag_count(synthetic_dataset_kwargs: dict) -> None:
    """The column header states the size the task was measured at.

    The suffix comes from the file's own ``num_bags`` attribute, so a table column can
    never claim a bag count the data does not have.
    """
    (task,) = create_benchmark_tasks(["mnist_xai_smil"], **synthetic_dataset_kwargs)
    assert task.name == "mnist_xai_smil_100bags"


def test_unknown_task_is_rejected_rather_than_skipped(synthetic_dataset_kwargs: dict) -> None:
    """A typo used to drop a column silently; now it fails loudly."""
    with pytest.raises(ValueError, match="Unknown task"):
        create_benchmark_tasks(["uci_musk3"], **synthetic_dataset_kwargs)


def test_missing_dataset_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises((ValueError, FileNotFoundError)):
        create_benchmark_tasks(["uci_musk1"], uci_h5_path=str(tmp_path / "absent.h5"))


# --------------------------------------------------------------------------- CLI


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("none", []),
        ("all", list(BASELINE_SPECS)),
        ("mean_logreg", ["mean_logreg"]),
        ("mean_logreg,svm_summ", ["mean_logreg", "svm_summ"]),
    ],
)
def test_selection_parsing(raw: str, expected: list[str]) -> None:
    assert _parse_selection(raw, list(BASELINE_SPECS), "baseline") == expected


def test_unknown_selection_lists_the_valid_options() -> None:
    with pytest.raises(SystemExit, match="Unknown baseline"):
        _parse_selection("no_such_baseline", list(BASELINE_SPECS), "baseline")


# --------------------------------------------------------------------------- table


def _results(mean: float, sem: float | None) -> dict:
    metrics = {"roc_auc": mean}
    if sem is not None:
        metrics["roc_auc_sem"] = sem
    return {"m": {"uci_musk1": metrics}}


def test_table_renders_mean_and_sem_as_percentages(tmp_path: Path) -> None:
    _write_table(_results(0.9271, 0.0046), tmp_path, {"m": 3})
    assert "| m | 92.7% ±0.5 |" in (tmp_path / "benchmark_table.md").read_text()


def test_table_omits_sem_when_there_is_none(tmp_path: Path) -> None:
    _write_table(_results(0.9271, float("nan")), tmp_path, {"m": 3})
    table = (tmp_path / "benchmark_table.md").read_text()
    assert "92.7%" in table and "±" not in table, "a NaN SEM must not render as '±nan'"


def test_single_run_rows_are_flagged(tmp_path: Path) -> None:
    """A single run's ± is split-level, not cross-run: a different quantity.

    Printing both in one column without saying so invites a false comparison.
    """
    _write_table(_results(0.9255, 0.0046), tmp_path, {"m": 1})
    table = (tmp_path / "benchmark_table.md").read_text()
    assert "| m* |" in table
    assert "single run" in table


def test_missing_cell_renders_as_a_dash(tmp_path: Path) -> None:
    _write_table({"m": {"uci_musk1": {}}}, tmp_path, {"m": 3})
    assert "| m | — |" in (tmp_path / "benchmark_table.md").read_text()


def test_column_headers_are_shortened(tmp_path: Path) -> None:
    _write_table(
        {"m": {"mnist_xai_smil_100bags": {"roc_auc": 0.5}, "uci_musk1": {"roc_auc": 0.5}}}, tmp_path, {"m": 3}
    )
    header = (tmp_path / "benchmark_table.md").read_text().splitlines()[0]
    assert "xai_smil_100bags" in header and "musk1" in header


# --------------------------------------------------------------------------- end to end


def test_end_to_end_on_synthetic_data(synthetic_datasets: Path, tmp_path: Path) -> None:
    """The full pipeline: build tasks, run baselines, aggregate, write all three outputs."""
    results = evaluate(
        data_source=synthetic_datasets,
        model_source=None,
        baselines=["mean_logreg"],
        icmil_seeds=[],
        tasks=["uci_musk1", "andrews_fox"],
        n_seeds=2,
        device="cpu",
        output_dir=tmp_path,
    )
    assert set(results) == {"mean_logreg"}, "rows are keyed by the CLI name, not the class name"
    assert set(results["mean_logreg"]) == {"uci_musk1", "andrews_fox"}
    for metrics in results["mean_logreg"].values():
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert metrics["n_outer"] == 2

    for name in ("benchmark_table.md", "results.json", "run_meta.json"):
        assert (tmp_path / name).exists(), f"{name} was not written"

    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert meta["device"] == "cpu"
    assert meta["n_seeds"] == 2
    assert set(meta["artifact_sha256"]) == {"uci_benchmark.h5", "andrews_mil_benchmark.h5"}


def test_icmil_runs_on_synthetic_data(synthetic_dataset_kwargs: dict) -> None:
    """An untrained model still has to satisfy the harness's interface contract."""
    from icmil.benchmarks import per_split_metrics
    from icmil.model import ICMILInference, build_icmil

    model = ICMILInference(build_icmil({**{k: v for k, v in _tiny_arch().items()}}))
    (task,) = create_benchmark_tasks(["uci_musk1"], **synthetic_dataset_kwargs)
    per_split, samples, timings = per_split_metrics(model, task, torch.device("cpu"), model.model.in_features)
    assert len(per_split) == 3
    assert samples and len(timings) == 3


def _tiny_arch() -> dict:
    from icmil.model import ICMIL_ARCH

    return {**ICMIL_ARCH, "embedding_size": 16, "mlp_hidden_size": 32, "num_column_row_iterations": 1}
