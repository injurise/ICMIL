"""Build benchmark tasks by name.

Task-class imports are deferred into each branch of :func:`create_benchmark_tasks`
so that importing this module does not eagerly pull every task's dependencies —
only the families actually requested get imported.

Unlike the research harness this was extracted from, a requested task that cannot
be built is an **error**, not a warning: every task in this repository has a
shipped ``.h5``, so a missing file or a mistyped name means the resulting table
would silently be missing a column.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from icmil.benchmarks.metrics import BenchmarkTask

MNIST_XAI_VARIANTS = ("smil", "four_bags", "pos_neg", "adjacent_pairs")
UCI_VARIANTS = ("musk1", "musk2", "letters", "hepmass")
ANDREWS_VARIANTS = ("fox", "tiger", "elephant")

# Every task this repository can evaluate, and the keyword naming the .h5 it needs.
TASK_H5_KWARG: dict[str, str] = {
    "tcga_fixed": "tcga_fixed_h5_path",
    "rsna_ich_draws": "rsna_draws_h5_path",
    **{f"uci_{v}": "uci_h5_path" for v in UCI_VARIANTS},
    **{f"mnist_xai_{v}": "mnist_xai_h5_path" for v in MNIST_XAI_VARIANTS},
    **{f"andrews_{v}": "andrews_h5_path" for v in ANDREWS_VARIANTS},
}
TASK_NAMES: tuple[str, ...] = tuple(TASK_H5_KWARG)


def _require_h5(name: str, kwargs: dict, task: str) -> str:
    """Return the .h5 path for ``task``, or explain precisely what is missing."""
    path = kwargs.get(name)
    if path is None:
        raise ValueError(f"Task {task!r} needs {name}=<path to .h5>, which was not provided.")
    if not Path(path).is_file():
        raise FileNotFoundError(f"Task {task!r} needs {name}, but {path} does not exist.")
    return str(path)


def create_benchmark_tasks(names: list[str], **kwargs: str | None) -> list[BenchmarkTask]:
    """Build the requested benchmark tasks, preserving the order of ``names``.

    Args:
        names: Task names, each one of :data:`TASK_NAMES`.
        **kwargs: Paths to the shipped benchmark files — ``uci_h5_path``,
            ``andrews_h5_path``, ``mnist_xai_h5_path``, ``tcga_fixed_h5_path``,
            ``rsna_draws_h5_path``. Only those needed by ``names`` are read.

    Returns:
        One task per requested name, in the same order.

    Raises:
        ValueError: An unknown task name, or a needed path was not supplied.
        FileNotFoundError: A needed path was supplied but does not exist.
    """
    unknown = [n for n in names if n not in TASK_H5_KWARG]
    if unknown:
        raise ValueError(f"Unknown task(s) {sorted(unknown)}. Valid tasks: {sorted(TASK_NAMES)}")
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"Duplicate task(s) requested: {duplicates}")

    tasks: list[BenchmarkTask] = []
    for name in names:
        h5_kwarg = TASK_H5_KWARG[name]
        h5_path = _require_h5(h5_kwarg, kwargs, name)

        if name.startswith("uci_"):
            from icmil.benchmarks.uci import UCIMILTask

            tasks.append(UCIMILTask(variant=name.removeprefix("uci_"), h5_path=h5_path))

        elif name.startswith("andrews_"):
            from icmil.benchmarks.andrews import AndrewsImageTask

            tasks.append(AndrewsImageTask(variant=name.removeprefix("andrews_"), h5_path=h5_path))

        elif name.startswith("mnist_xai_"):
            from icmil.benchmarks.mnist_xai import MNISTXAITask

            tasks.append(MNISTXAITask(variant=name.removeprefix("mnist_xai_"), h5_path=h5_path))

        elif name == "tcga_fixed":
            from icmil.benchmarks.tcga import TCGAFixedTask

            tasks.append(TCGAFixedTask(h5_path=h5_path))

        elif name == "rsna_ich_draws":
            from icmil.benchmarks.rsna import RSNAMILDrawsTask

            tasks.append(RSNAMILDrawsTask(h5_path=h5_path))

        else:  # pragma: no cover — TASK_H5_KWARG and the branches above are kept in sync
            raise AssertionError(f"Task {name!r} is registered but has no constructor branch")

    return tasks
