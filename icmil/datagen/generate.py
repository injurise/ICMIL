"""Generate the synthetic prior corpus ICMIL is pre-trained on.

One H5 file per prior arm, each a flat sequence of ``batch_i`` groups. Every batch
is an independent synthetic MIL dataset: fresh shape, fresh latent function, its own
train/test split of bags.

    python -m icmil.datagen.generate --out-dir workdir/priors
    python -m icmil.datagen.generate --arm joint_mlp_long_curr --num-batches 100 --num-workers 8
    python -m icmil.datagen.generate --dry-run          # print the resolved config
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import multiprocessing
import os
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from icmil.datagen.config import CONFIGS, PAPER_TRAIN_WEIGHTS, CurriculumStage, GenConfig, PriorArm, ShapeConfig
from icmil.datagen.h5_writer import BaggedPriorH5Writer

logger = logging.getLogger("icmil.datagen")


# ------------------------------------------------------------------ curriculum


def _stage_for(curriculum: list[CurriculumStage], batch_idx: int) -> CurriculumStage:
    """Return the stage governing ``batch_idx`` (the last stage also covers overflow)."""
    for stage in curriculum:
        if batch_idx < stage.until_batch:
            return stage
    return curriculum[-1]


def _curriculum_overrides(curriculum: list[CurriculumStage] | None, batch_idx: int) -> tuple[int | None, int | None]:
    """Sample this batch's ``(bag_size, max_instance_classes)`` from the curriculum.

    Returns ``(None, None)`` when there is no curriculum, in which case the prior's
    own sampler decides. Consumes NumPy RNG exactly once per bound, which is part of
    the generation's reproducibility contract.
    """
    if curriculum is None:
        return None, None
    stage = _stage_for(curriculum, batch_idx)
    lo, hi = stage.min_bag_size, stage.max_bag_size
    bag_size = int(np.random.randint(lo, hi + 1)) if lo < hi else lo

    classes = None
    if stage.max_instance_classes is not None:
        c_lo = stage.min_instance_classes if stage.min_instance_classes is not None else 2
        c_hi = stage.max_instance_classes
        classes = int(np.random.randint(c_lo, c_hi + 1)) if c_lo < c_hi else c_lo
    return bag_size, classes


def _widen_for_curriculum(shape: ShapeConfig, curriculum: list[CurriculumStage] | None) -> dict[str, int]:
    """H5 padding dimensions must cover the widest bag any stage can request."""
    max_bag_size = shape.max_bag_size
    max_classes = shape.max_classes
    if curriculum:
        max_bag_size = max(max_bag_size, *(s.max_bag_size for s in curriculum))
        max_classes = max(max_classes, *(s.max_instance_classes or 0 for s in curriculum))
    return {
        "max_n_bags": shape.max_n_bags,
        "max_bag_size": max_bag_size,
        "max_features": shape.max_features,
        "max_classes": max_classes,
    }


# ------------------------------------------------------------------ generation


def _prior_kwargs(shape: ShapeConfig, arm: PriorArm, device: str) -> dict[str, Any]:
    """Merge the shared shape settings with the arm's own prior settings."""
    return {**asdict(shape), "device": device, **arm.prior_kwargs}


def _generate_batch(
    prior_kwargs: dict[str, Any],
    n_bags: int,
    bag_size: int,
    seed: int,
    min_train_size: float,
    max_train_size: float,
    max_instance_classes: int | None,
) -> tuple[np.ndarray, ...]:
    """Generate one batch. Runs in a worker process, so it takes only picklable args."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    from icmil.datagen.priors.dataset import BaggedPriorDataset

    dataset = BaggedPriorDataset(**prior_kwargs)
    params = dataset.sample_params()
    params.n_bags = n_bags
    params.bag_size = bag_size
    if max_instance_classes is not None:
        params.max_instance_classes = max_instance_classes

    train_ratio = np.random.uniform(min_train_size, max_train_size)
    params.n_train = max(1, min(int(n_bags * train_ratio), n_bags - 1))

    X, y, d, n_bags_t, bag_sizes, n_train = dataset.get_batch(params=params)
    return (
        X.cpu().numpy(),
        y.cpu().numpy(),
        d.cpu().numpy(),
        n_bags_t.cpu().numpy(),
        bag_sizes.cpu().numpy(),
        n_train.cpu().numpy(),
    )


def generate_arm(cfg: GenConfig, arm: PriorArm, out_path: Path) -> None:
    """Generate one prior arm into ``out_path``."""
    from icmil.datagen.priors.dataset import BaggedPriorDataset

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    prior_kwargs = _prior_kwargs(cfg.shape, arm, cfg.device)
    dataset = BaggedPriorDataset(**prior_kwargs)
    sampler = dataset.sampler
    dims = _widen_for_curriculum(cfg.shape, cfg.curriculum)

    with BaggedPriorH5Writer(
        str(out_path),
        cfg.shape.batch_size,
        dims["max_n_bags"],
        dims["max_bag_size"],
        dims["max_features"],
        dims["max_classes"],
    ) as writer:
        metadata: dict[str, Any] = {
            "prior_configs_json": [{"name": arm.name, "prior_kwargs": arm.prior_kwargs}],
            "common_kwargs_json": {**asdict(cfg.shape), "device": cfg.device},
        }
        if cfg.curriculum:
            # The trainer keys off this: a curriculum means batch order is meaningful,
            # so the loader must not shuffle.
            metadata["curriculum_schedule"] = json.dumps([asdict(s) for s in cfg.curriculum])
        writer.write_metadata(metadata)

        if cfg.num_workers == 0:
            _generate_sequential(cfg, dataset, sampler, writer)
        else:
            _generate_parallel(cfg, dataset, sampler, writer, prior_kwargs)
        writer.finalize()


def _generate_sequential(cfg: GenConfig, dataset, sampler, writer: BaggedPriorH5Writer) -> None:
    """Generate in-process. Deterministic and easy to debug; the default."""
    for batch_idx in tqdm(range(cfg.num_batches), desc="batches"):
        params = dataset.sample_params()
        bag_size, classes = _curriculum_overrides(cfg.curriculum, batch_idx)
        if bag_size is not None:
            params.bag_size = bag_size
        if classes is not None:
            params.max_instance_classes = classes

        train_ratio = np.random.uniform(sampler.min_train_size, sampler.max_train_size)
        params.n_train = max(1, min(int(params.n_bags * train_ratio), params.n_bags - 1))

        X, y, d, n_bags, bag_sizes, n_train = dataset.get_batch(params=params)
        writer.start_batch(batch_idx, n_bags=params.n_bags, bag_size=params.bag_size)
        writer.append_batch(X, y, d, n_bags, bag_sizes, n_train)


def _generate_parallel(cfg: GenConfig, dataset, sampler, writer: BaggedPriorH5Writer, prior_kwargs: dict) -> None:
    """Generate in worker processes."""
    assignments: list[tuple[int, int, int | None]] = []
    for batch_idx in range(cfg.num_batches):
        params = dataset.sample_params()
        bag_size, classes = _curriculum_overrides(cfg.curriculum, batch_idx)
        assignments.append((params.n_bags, bag_size if bag_size is not None else params.bag_size, classes))

    base_seed = int(np.random.randint(0, 2**31))
    args = [
        (
            prior_kwargs,
            n_bags,
            bag_size,
            (base_seed + i) % (2**31),
            sampler.min_train_size,
            sampler.max_train_size,
            mic,
        )
        for i, (n_bags, bag_size, mic) in enumerate(assignments)
    ]

    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.num_workers, mp_context=ctx) as pool:
        results = pool.map(_generate_batch, *zip(*args, strict=True), chunksize=1)
        for batch_idx, arrays in enumerate(tqdm(results, total=cfg.num_batches, desc="batches")):
            n_bags, bag_size, _ = assignments[batch_idx]
            writer.start_batch(batch_idx, n_bags=n_bags, bag_size=bag_size)
            writer.append_batch_numpy(*arrays)


def _pin_thread_counts() -> None:
    """Keep BLAS single-threaded."""
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, "1")
    torch.set_num_threads(1)


# ------------------------------------------------------------------ CLI


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="paper", help=f"Named config {sorted(CONFIGS)} or a path to a JSON config")
    p.add_argument("--arm", action="append", help="Generate only this arm (repeatable). Default: all arms.")
    p.add_argument("--out-dir", type=Path, default=Path("workdir/priors"), help="Where to write <arm>.h5")
    p.add_argument("--num-batches", type=int, help="Override the config's batch count")
    p.add_argument("--num-workers", type=int, help="Worker processes; 0 runs in-process")
    p.add_argument("--seed", type=int, help="Override the config's seed")
    p.add_argument("--device", help="Override the config's device")
    p.add_argument("--no-curriculum", action="store_true", help="Disable the curriculum")
    p.add_argument("--dry-run", action="store_true", help="Print the resolved config as JSON and exit")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _pin_thread_counts()

    if args.config in CONFIGS:
        cfg = CONFIGS[args.config]
    else:
        cfg = GenConfig.from_json(Path(args.config).read_text())

    cfg = cfg.with_overrides(
        num_batches=args.num_batches, num_workers=args.num_workers, seed=args.seed, device=args.device
    )
    if args.no_curriculum:
        cfg = replace(cfg, curriculum=None)

    if args.arm:
        known = {a.name for a in cfg.arms}
        unknown = sorted(set(args.arm) - known)
        if unknown:
            raise SystemExit(f"Unknown arm(s) {unknown}. Available: {sorted(known)}")
        cfg = replace(cfg, arms=[a for a in cfg.arms if a.name in set(args.arm)])

    if args.dry_run:
        print(cfg.to_json())
        print("\n# training mix for the full recipe:")
        print("#   --prior-weights " + ",".join(f"{k}={v}" for k, v in PAPER_TRAIN_WEIGHTS.items()))
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for arm in cfg.arms:
        out_path = args.out_dir / f"{arm.name}.h5"
        logger.info(
            "Generating %s (%d batches, %d workers) -> %s", arm.name, cfg.num_batches, cfg.num_workers, out_path
        )
        generate_arm(cfg, arm, out_path)
        logger.info("Wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
