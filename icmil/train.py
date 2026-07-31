"""Pre-train ICMIL on the synthetic prior corpus.

    python -m icmil.train --data-dir workdir/priors --out checkpoints/my-icmil.pt

With no flags this builds the released architecture and trains it with settings in
the range that produced the released checkpoints.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np
import schedulefree
import torch
from torch import nn

from icmil.datagen.h5_dataset import MultiPriorH5Dataset
from icmil.model import ICMIL_ARCH, build_icmil

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("icmil.train")

# Arguments that name locations on the training machine; never recorded in a checkpoint.
_PATH_ARGS = frozenset({"data_dir", "out", "resume"})


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch together.

    Note this makes runs seed-reproducible, not bitwise-deterministic on GPU:
    ``torch.use_deterministic_algorithms`` is deliberately *not* set, because it
    would change the numerics relative to the runs that produced the released
    checkpoints.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, model: nn.Module, optimizer, epoch: int, arch: dict, train_args: dict) -> None:
    """Write a checkpoint that :func:`icmil.model.load_icmil` can read.

    ``model.eval()`` / ``optimizer.eval()`` first is not cosmetic: schedulefree keeps
    an interpolated iterate and only exposes the evaluation weights in eval mode, so
    saving in train mode stores the wrong parameters.

    ``arch`` travels with the weights, so a model trained here loads through the same
    public API as the released ones. ``train_args`` records the hyper-parameters but
    deliberately not the paths — a checkpoint is a thing people share, and local
    directory layouts are both useless to the recipient and needlessly revealing.
    """
    model.eval()
    optimizer.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "arch": dict(arch),
            "train_args": {k: v for k, v in train_args.items() if k not in _PATH_ARGS},
        },
        path,
    )


def train(args: argparse.Namespace) -> list[float]:
    """Run training; return the mean loss of each epoch."""
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(args.seed)

    weights = _parse_weights(args.prior_weights)
    dataset = MultiPriorH5Dataset(prior_dir=args.data_dir, weights=weights)
    logger.info("Priors: %s", dataset)

    arch = {k: getattr(args, k) for k in ICMIL_ARCH}
    model = build_icmil(arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        eps=args.adam_eps,
    )

    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state.get("epoch", 0))
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    epoch_losses: list[float] = []
    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        model.train()
        optimizer.train()
        total_loss = torch.tensor(0.0, device=device)

        # The loader — not a plain DataLoader — implements the per-arm weighted mixing
        # and, when the priors carry a curriculum, preserves batch order.
        loader = dataset.get_loader_for_epoch(
            epoch=epoch,
            steps_per_epoch=args.steps_per_epoch,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            pin_memory=device.type == "cuda",
        )

        for X, y, _d, _n_bags, _bag_sizes, n_train_bags in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            n_train = int(n_train_bags[0])
            X_train, X_test = X[:, :n_train], X[:, n_train:]
            y_train, y_test = y[:, :n_train], y[:, n_train:]

            # Gradient accumulation over the batch axis: the effective batch is the
            # whole group, split into `micro` chunks to bound activation memory.
            batch = X.shape[0]
            micro = args.micro_batch_size or batch
            if batch % micro != 0:
                raise ValueError(f"--micro-batch-size {micro} must divide the prior's batch size {batch}")
            num_micros = batch // micro

            step_loss = torch.tensor(0.0, device=device)
            for start in range(0, batch, micro):
                end = start + micro
                # bf16 autocast is what dispatches attention to the fused kernel the
                # released model was trained under; it is part of the recipe.
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=not args.no_autocast):
                    output = model(X_train[start:end], y_train[start:end], X_test[start:end])
                    loss = criterion(output.reshape(-1, args.num_outputs), y_test[start:end].reshape(-1)) / num_micros
                loss.backward()
                step_loss = step_loss + loss.detach()
            total_loss = total_loss + step_loss

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # Skip the update rather than poison every parameter with NaN. Rare, but
            # a single bad batch would otherwise end the run.
            if torch.isfinite(grad_norm):
                optimizer.step()
            optimizer.zero_grad()

        model.eval()
        optimizer.eval()
        mean_loss = (total_loss / args.steps_per_epoch).item()
        epoch_losses.append(mean_loss)
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            logger.info("epoch %d/%d  loss %.4f  (%.1fs)", epoch + 1, args.epochs, mean_loss, time.time() - started)

    if args.out:
        save_checkpoint(Path(args.out), model, optimizer, args.epochs, arch, vars(args))
        logger.info("Wrote %s", args.out)
    return epoch_losses


def _parse_weights(raw: str | None) -> dict[str, float] | None:
    """Parse ``name=w,name=w`` into a dict; ``None`` weights all priors equally."""
    if not raw:
        return None
    out: dict[str, float] = {}
    for part in raw.split(","):
        if "=" not in part:
            raise SystemExit(f"--prior-weights entries must look like name=weight; got {part!r}")
        name, _, weight = part.partition("=")
        out[name.strip()] = float(weight)
    return out


def build_parser() -> argparse.ArgumentParser:
    from icmil.datagen.config import PAPER_TRAIN_WEIGHTS

    default_weights = ",".join(f"{k}={v}" for k, v in PAPER_TRAIN_WEIGHTS.items())
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--data-dir", required=True, help="Directory of per-arm prior .h5 files")
    p.add_argument("--prior-weights", default=default_weights, help="Mixing weights, name=w,name=w")
    p.add_argument("--out", default="checkpoints/icmil-new.pt", help="Where to write the checkpoint")
    p.add_argument("--resume", help="Resume from a checkpoint")

    opt = p.add_argument_group("optimization")
    opt.add_argument("--epochs", type=int, default=8000)
    opt.add_argument("--steps-per-epoch", type=int, default=5)
    opt.add_argument("--lr", type=float, default=6e-4)
    opt.add_argument("--weight-decay", type=float, default=0.0)
    opt.add_argument("--adam-eps", type=float, default=1e-6)
    opt.add_argument("--warmup-steps", type=int, default=2500)
    opt.add_argument("--micro-batch-size", type=int, default=8, help="Gradient-accumulation chunk")
    opt.add_argument("--max-grad-norm", type=float, default=1.0)
    opt.add_argument("--seed", type=int, default=0)

    run = p.add_argument_group("runtime")
    run.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    run.add_argument("--no-autocast", action="store_true", help="Disable bf16 autocast (needed on CPU)")
    run.add_argument("--num-workers", type=int, default=0)
    run.add_argument("--prefetch-factor", type=int, default=2)
    run.add_argument("--log-every", type=int, default=50)

    arch = p.add_argument_group("architecture (defaults are the released ICMIL)")
    for key, value in ICMIL_ARCH.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            arch.add_argument(flag, dest=key, action="store_true", default=value)
        else:
            arch.add_argument(flag, dest=key, type=int, default=value)
    return p


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
