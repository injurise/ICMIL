"""ICMIL against ABMIL and ACMIL as the number of labeled training bags grows.

Usage:
    python -m icmil.bag_count                                    # all three benchmarks
    python -m icmil.bag_count --benchmarks rsna_ich --bag-counts 25,100
    python -m icmil.bag_count --baselines none                   # ICMIL only

Each benchmark keeps its paper configuration and only the number of bags changes;
the 100-bag cells of MNIST Pos/Neg and RSNA-ICH read the files behind the paper
table. One table per benchmark is written to ``<output>/bag_count_tables.md``:
rows are models, columns the number of training bags, cells AUROC as
``mean ± SEM`` over runs (see ``icmil.reproduce``). Every cell also keeps its own
``results.json`` under ``<output>/<benchmark>/<N>bags/`` and is skipped on a rerun.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from icmil.artifacts import DEFAULT_DATA_DIR, SEEDS, SEEDS_50F
from icmil.benchmarks.mnist_xai import MNISTXAITask
from icmil.benchmarks.rsna import RSNAMILDrawsTask
from icmil.benchmarks.subsampled import SubsampledSplitTask
from icmil.reproduce import _parse_selection, evaluate

BASELINES = ["abmil", "acmil"]

# benchmark -> (title, ICMIL seeds, {bag count: .h5 relative to the data directory}).
# MNIST counts are total bags, split 90:10 into train and test; the others are
# training bags. TAD has 50 features, so it needs the 50-feature models.
BENCHMARKS: dict[str, tuple[str, tuple[str, ...], dict[int, str]]] = {
    "mnist_pos_neg": (
        "MNIST Pos/Neg",
        SEEDS,
        {100: "mnist_xai_benchmark_100bags.h5"}
        | {n: f"bag_count/mnist_xai_benchmark_{n}bags.h5" for n in (50, 200, 400, 800)},
    ),
    "rsna_ich": (
        "RSNA-ICH",
        SEEDS,
        {100: "rsna_ich_resnet50_draws_100bags.h5"}
        | {n: f"bag_count/rsna_ich_resnet50_draws_{n}bags.h5" for n in (25, 50, 200, 400)},
    ),
    "tad": (
        "TAD",
        SEEDS_50F,
        {n: "tad_mil_resnet50_pca50.h5" for n in (25, 50, 100, 150, 200, 300, 408)},
    ),
}


def _build_task(benchmark: str, n_bags: int, h5_path: Path):
    if benchmark == "mnist_pos_neg":
        return MNISTXAITask(variant="pos_neg", h5_path=str(h5_path))
    if benchmark == "rsna_ich":
        return RSNAMILDrawsTask(h5_path=h5_path)
    return SubsampledSplitTask(h5_path=h5_path, n_train_bags=n_bags, name="tad_mil_resnet50_pca50")


def format_table(cells: dict[int, dict]) -> str:
    """One benchmark's table from ``{bag count: results.json content}``."""
    columns = {n: next(iter(next(iter(res.values())).values()))["n_train_bags_mean"] for n, res in cells.items()}
    order = sorted(cells, key=columns.get)
    models = list(dict.fromkeys(m for n in order for m in cells[n]))
    rows = [
        "| Model | " + " | ".join(str(round(columns[n])) for n in order) + " |",
        "|---|" + "|".join("---:" for _ in order) + "|",
    ]
    for model in models:
        row = []
        for n in order:
            metrics = next(iter(cells[n].get(model, {}).values()), None)
            row.append(f"{metrics['roc_auc'] * 100:.1f} ±{metrics['roc_auc_sem'] * 100:.1f}" if metrics else "—")
        rows.append(f"| {model} | " + " | ".join(row) + " |")
    return "\n".join(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="ICMIL vs. ABMIL / ACMIL by number of training bags.")
    p.add_argument("--benchmarks", default="all", help="'all' or a comma list of " + ", ".join(BENCHMARKS))
    p.add_argument("--bag-counts", default="all", help="'all' or a comma list; counts a benchmark lacks are skipped")
    p.add_argument("--baselines", default="all", help="'all', 'none', or a comma list of " + ", ".join(BASELINES))
    p.add_argument("--icmil", default="all", choices=["all", "none"], help="Run the ICMIL row")
    p.add_argument("--data", default=None, help="Directory of benchmark .h5 files (default: ./datasets)")
    p.add_argument("--models", default=None, help="Directory of ICMIL .pt checkpoints (default: ./checkpoints)")
    p.add_argument("--n-seeds", type=int, default=3, help="Random seeds per baseline")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--output", default="results/bag_count", help="Output directory")
    args = p.parse_args()

    data_dir = Path(args.data).expanduser() if args.data else DEFAULT_DATA_DIR
    output = Path(args.output)
    baselines = _parse_selection(args.baselines, BASELINES, "baseline")
    wanted = None if args.bag_counts == "all" else {int(n) for n in args.bag_counts.split(",")}

    sections = []
    for benchmark in _parse_selection(args.benchmarks, list(BENCHMARKS), "benchmark"):
        title, seeds, files = BENCHMARKS[benchmark]
        cells: dict[int, dict] = {}
        for n_bags, filename in sorted(files.items()):
            if wanted is not None and n_bags not in wanted:
                continue
            cell_dir = output / benchmark / f"{n_bags}bags"
            if (cell_dir / "results.json").exists():
                cells[n_bags] = json.loads((cell_dir / "results.json").read_text())
                continue
            h5_path = data_dir / filename
            task = _build_task(benchmark, n_bags, h5_path)
            cells[n_bags] = evaluate(
                data_source=None,
                model_source=args.models,
                baselines=baselines,
                icmil_seeds=list(seeds) if args.icmil == "all" else [],
                tasks=[task.name],
                n_seeds=args.n_seeds,
                device=args.device,
                output_dir=cell_dir,
                benchmark_tasks=[task],
                dataset_paths={"h5_path": str(h5_path)},
            )
        if cells:
            sections.append(f"### {title}\n\n{format_table(cells)}")

    text = "\n\n".join(sections) + "\n"
    print("\n" + text)
    output.mkdir(parents=True, exist_ok=True)
    (output / "bag_count_tables.md").write_text(text)


if __name__ == "__main__":
    main()
