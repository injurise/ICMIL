"""Benchmark tasks and the shared evaluation harness."""

from icmil.benchmarks.metrics import (
    BenchmarkTask,
    aggregate_outer_metrics,
    aggregate_split_metrics,
    compute_split_metrics,
    evaluate_benchmark_task,
    per_split_metrics,
)
from icmil.benchmarks.registry import TASK_NAMES, create_benchmark_tasks

__all__ = [
    "TASK_NAMES",
    "BenchmarkTask",
    "aggregate_outer_metrics",
    "aggregate_split_metrics",
    "compute_split_metrics",
    "create_benchmark_tasks",
    "evaluate_benchmark_task",
    "per_split_metrics",
]
