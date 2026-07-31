"""Reproduce the ICMIL benchmark table (baselines + ICMIL) with a single command.

Usage:
    python -m icmil.reproduce                                   # full table
    python -m icmil.reproduce --tasks uci_musk1                 # one task
    python -m icmil.reproduce --baselines mean_logreg,abmil_refit
    python -m icmil.reproduce --baselines none --icmil-seeds c5trd795

Rows are models, columns are tasks, cells are AUROC as ``mean ± SEM``. Each row
aggregates several independent runs and the ± is the spread **across** them: for
ICMIL, across the three trained seeds; for a baseline, across ``--n-seeds``
random seeds. Within a run the score is the mean over the task's frozen splits.

Alongside the table this writes ``results.json`` (every metric at full
precision) and ``run_meta.json`` (versions, device, artifact hashes).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from icmil.artifacts import SEEDS, resolve_dataset
from icmil.benchmarks import (
    aggregate_outer_metrics,
    aggregate_split_metrics,
    create_benchmark_tasks,
    per_split_metrics,
)
from icmil.model import load_icmil

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("icmil.reproduce")

# The twelve tasks of the paper table, in table-column order.
TASKS: list[str] = [
    "tcga_fixed",
    "rsna_ich_draws",
    "mnist_xai_smil",
    "mnist_xai_pos_neg",
    "mnist_xai_adjacent_pairs",
    "uci_musk1",
    "uci_musk2",
    "uci_letters",
    "uci_hepmass",
    "andrews_fox",
    "andrews_tiger",
    "andrews_elephant",
]

# Baseline hyper-parameters as reported. ``seed`` is set per run (0..n_seeds-1).
BASELINE_SPECS: dict[str, dict] = {
    "mean_logreg": dict(max_classes=4),
    "svm_summ": dict(max_classes=4, mode="refit"),
    "abmil_refit": dict(
        max_classes=4,
        embed_dim=256,
        attn_dim=128,
        num_fc_layers=1,
        gate=True,
        epochs=200,
        batch_size=32,
        warmup_steps=20,
        n_cv_splits=5,
        patience=20,
        min_delta=1e-4,
    ),
    "acmil": dict(
        max_classes=4,
        embed_dim=256,
        attn_dim=128,
        n_token=5,
        n_masked_patch=10,
        mask_drop=0.6,
        epochs=200,
        batch_size=32,
        warmup_steps=20,
        n_cv_splits=5,
        patience=20,
        min_delta=1e-4,
    ),
    "tabpfn_concat": dict(max_classes=4, max_tabpfn_features=500, features_per_group=2),
    "tabpfn_subsample": dict(
        max_classes=4,
        max_tabpfn_features=500,
        features_per_group=2,
        n_views=10,
        n_keep=None,
        aggregation="mean_logits",
    ),
    "cluster_tabpfn": dict(max_classes=4, n_clusters=5, n_pca_components=None, max_tabpfn_features=500),
}

# Which .h5 each task family needs, as the keyword `create_benchmark_tasks` expects.
_DATASET_FOR_PREFIX: list[tuple[str, str, str]] = [
    ("tcga_fixed", "tcga", "tcga_fixed_h5_path"),
    ("rsna_ich_draws", "rsna", "rsna_draws_h5_path"),
    ("uci_", "uci", "uci_h5_path"),
    ("andrews_", "andrews", "andrews_h5_path"),
    ("mnist_xai_", "mnist_xai", "mnist_xai_h5_path"),
]


def _baseline_classes() -> dict[str, type]:
    """Import the baseline classes lazily — none of this is needed for ICMIL alone."""
    from icmil.baselines.abmil_baseline import ABMILRefitBaseline
    from icmil.baselines.acmil_baseline import ACMILRefitBaseline
    from icmil.baselines.tabpfn_baselines import (
        ClusterTabPFNBaseline,
        MeanLogRegBaseline,
        SVMSummBaseline,
        TabPFNConcatBaseline,
        TabPFNSubsampleBaseline,
    )

    return {
        "mean_logreg": MeanLogRegBaseline,
        "svm_summ": SVMSummBaseline,
        "abmil_refit": ABMILRefitBaseline,
        "acmil": ACMILRefitBaseline,
        "tabpfn_concat": TabPFNConcatBaseline,
        "tabpfn_subsample": TabPFNSubsampleBaseline,
        "cluster_tabpfn": ClusterTabPFNBaseline,
    }


def _dataset_kwargs(data_source: str | Path | None, tasks: list[str]) -> dict[str, str]:
    """Resolve only the .h5 files the requested tasks actually need."""
    kwargs: dict[str, str] = {}
    for prefix, dataset_key, kwarg in _DATASET_FOR_PREFIX:
        if any(t == prefix or t.startswith(prefix) for t in tasks):
            kwargs[kwarg] = str(resolve_dataset(dataset_key, source=data_source))
    return kwargs


def evaluate(
    data_source: str | Path | None,
    model_source: str | Path | None,
    baselines: list[str],
    icmil_seeds: list[str],
    tasks: list[str],
    n_seeds: int,
    device: str,
    output_dir: Path,
) -> dict:
    """Run the requested models over the requested tasks and write the outputs."""
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", dev)

    benchmark_tasks = create_benchmark_tasks(tasks, **_dataset_kwargs(data_source, tasks))

    # display label -> (runs, max_features)
    models: dict[str, tuple[list, int | None]] = {}
    classes = _baseline_classes() if baselines else {}
    for name in baselines:
        cls = classes[name]
        # Baselines accept any feature width, so they see the raw features:
        # harness-side zero padding would distort their summary statistics.
        models[name] = ([cls(seed=s, **BASELINE_SPECS[name]).to(dev) for s in range(n_seeds)], None)
    if icmil_seeds:
        runs = [load_icmil(source=model_source, seed=s, device=dev) for s in icmil_seeds]
        label = "ICMIL" if len(icmil_seeds) > 1 else f"ICMIL (seed {icmil_seeds[0]})"
        models[label] = (runs, runs[0].model.in_features)

    all_results: dict[str, dict] = {}
    timings: dict[str, dict[str, float]] = {}
    for label, (runs, max_features) in models.items():
        logger.info("=== %s (%d run%s) ===", label, len(runs), "s" if len(runs) > 1 else "")
        task_results: dict[str, dict] = {}
        for task in benchmark_tasks:
            started = time.perf_counter()
            per_outer, per_outer_timings, samples = [], [], []
            for run in runs:
                splits, sample, split_timings = per_split_metrics(run, task, dev, max_features)
                per_outer.append(splits)
                per_outer_timings.append(split_timings)
                samples = samples or sample
            if len(per_outer) > 1:
                metrics = aggregate_outer_metrics(per_outer, samples, per_outer_timings)
            else:
                metrics = aggregate_split_metrics(per_outer[0], samples, per_outer_timings[0])
            task_results[task.name] = metrics
            timings.setdefault(label, {})[task.name] = time.perf_counter() - started
            auroc = metrics.get("roc_auc")
            logger.info("  %s: AUROC %.4f", task.name, auroc if auroc is not None else float("nan"))
        all_results[label] = task_results

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_table(all_results, output_dir, {label: len(runs) for label, (runs, _) in models.items()})
    (output_dir / "results.json").write_text(json.dumps(all_results, indent=2, default=float))
    _write_run_meta(output_dir, dev, models, tasks, n_seeds, timings, _dataset_kwargs(data_source, tasks))
    logger.info("Wrote %s", ", ".join(sorted(p.name for p in output_dir.glob("*"))))
    return all_results


def _write_table(all_results: dict, output_dir: Path, n_runs: dict[str, int]) -> None:
    """Write the AUROC markdown table (mean ± SEM).

    For a row with several runs the ± is the SEM **across runs**. A row with 
    a single run has no cross-run spread, so its ± is the SEM across the 
    task's splits instead.
    """
    metric = "roc_auc"
    task_names: list[str] = []
    for per_task in all_results.values():
        for task in per_task:
            if task not in task_names:
                task_names.append(task)

    def short(task: str) -> str:
        return task.replace("mnist_xai_", "xai_").replace("uci_", "")

    rows = [
        "| Model | " + " | ".join(short(t) for t in task_names) + " |",
        "|---|" + "|".join("---:" for _ in task_names) + "|",
    ]
    single_run = sorted(label for label, count in n_runs.items() if count < 2)
    for label, per_task in all_results.items():
        cells = []
        for task in task_names:
            metrics = per_task.get(task, {})
            value = metrics.get(metric)
            if value is None:
                cells.append("—")
                continue
            sem = metrics.get(f"{metric}_sem")
            # `sem != sem` is the NaN case: nothing to average over, so no ±.
            has_sem = sem is not None and sem == sem
            cells.append(f"{value * 100:.1f}% ±{sem * 100:.1f}" if has_sem else f"{value * 100:.1f}%")
        marker = "*" if label in single_run else ""
        rows.append(f"| {label}{marker} | " + " | ".join(cells) + " |")

    if single_run:
        rows += [
            "",
            "\\* single run — the ± is the spread across the task's splits, not across runs, "
            "and is not comparable to the other rows.",
        ]

    table = "\n".join(rows)
    print("\n" + table)
    (output_dir / "benchmark_table.md").write_text(table + "\n")


def _write_run_meta(
    output_dir: Path,
    device: torch.device,
    models: dict,
    tasks: list[str],
    n_seeds: int,
    timings: dict,
    dataset_paths: dict[str, str],
) -> None:
    """Record metadata."""
    import hashlib
    import platform
    from importlib.metadata import PackageNotFoundError, version

    def pkg(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    def sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    meta = {
        "models": sorted(models),
        "tasks": tasks,
        "n_seeds": n_seeds,
        "baseline_specs": {k: v for k, v in BASELINE_SPECS.items() if k in models},
        "device": str(device),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {n: pkg(n) for n in ("torch", "numpy", "scikit-learn", "h5py", "tabpfn", "schedulefree")},
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        # TF32 silently changes matmul precision on Ampere+ and is the most common
        # cause of a GPU run not matching a recorded one. Always record it.
        "allow_tf32": {
            "matmul": torch.backends.cuda.matmul.allow_tf32,
            "cudnn": torch.backends.cudnn.allow_tf32,
        },
        "artifact_sha256": {Path(p).name: sha256(p) for p in sorted(dataset_paths.values())},
        "seconds_per_cell": timings,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))


def _parse_selection(raw: str, all_values: list[str], what: str) -> list[str]:
    """Resolve an ``all`` / ``none`` / comma-list argument, rejecting unknown names."""
    if raw == "none":
        return []
    if raw == "all":
        return list(all_values)
    chosen = [v.strip() for v in raw.split(",") if v.strip()]
    unknown = [v for v in chosen if v not in all_values]
    if unknown:
        raise SystemExit(f"Unknown {what}: {sorted(unknown)}. Valid: {sorted(all_values)}")
    return chosen


def main() -> None:
    p = argparse.ArgumentParser(description="Reproduce the ICMIL benchmark table.")
    p.add_argument("--data", default=None, help="Directory of benchmark .h5 files (default: ./datasets)")
    p.add_argument("--models", default=None, help="Directory of ICMIL .pt checkpoints (default: ./checkpoints)")
    p.add_argument("--baselines", default="all", help="'all', 'none', or a comma list")
    p.add_argument("--icmil-seeds", default="all", help="'all', 'none', or a comma list of seeds")
    p.add_argument("--tasks", default="all", help="'all' or a comma list of task names")
    p.add_argument("--n-seeds", type=int, default=3, help="Random seeds per baseline")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--output", default="results", help="Output directory")
    args = p.parse_args()

    evaluate(
        data_source=args.data,
        model_source=args.models,
        baselines=_parse_selection(args.baselines, list(BASELINE_SPECS), "baseline"),
        icmil_seeds=_parse_selection(args.icmil_seeds, list(SEEDS), "seed"),
        tasks=_parse_selection(args.tasks, TASKS, "task"),
        n_seeds=args.n_seeds,
        device=args.device,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
