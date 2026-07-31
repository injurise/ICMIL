"""The trainer: it learns, and its checkpoints are loadable and clean."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from icmil.model import ICMIL_ARCH, load_icmil
from icmil.train import build_parser, save_checkpoint, seed_everything, train


@pytest.fixture(scope="module")
def prior_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A miniature prior file, written by the real writer so the format is guaranteed.

    The *content* is random rather than drawn from a real prior — the trainer does not
    care, and generating a real one would dominate the runtime of this suite.
    """
    from icmil.datagen.h5_writer import BaggedPriorH5Writer

    out = tmp_path_factory.mktemp("priors")
    n_batches, batch, n_bags, bag_size, n_features = 3, 4, 8, 3, 25
    rng = np.random.default_rng(0)

    with BaggedPriorH5Writer(str(out / "tiny_prior.h5"), batch, n_bags, bag_size, n_features, 2) as writer:
        writer.write_metadata({"prior_configs_json": [{"name": "tiny_prior", "prior_kwargs": {}}]})
        for i in range(n_batches):
            writer.start_batch(i, n_bags=n_bags, bag_size=bag_size)
            writer.append_batch_numpy(
                rng.normal(size=(batch, n_bags, bag_size, n_features)).astype(np.float32),
                rng.integers(0, 2, size=(batch, n_bags)).astype(np.int64),
                np.full(batch, n_features, dtype=np.int64),
                np.full(batch, n_bags, dtype=np.int64),
                np.full(batch, bag_size, dtype=np.int64),
                np.full(batch, 6, dtype=np.int64),
            )
        writer.finalize()
    return out


def _args(prior_dir: Path, out: Path, **overrides):
    argv = [
        "--data-dir",
        str(prior_dir),
        "--prior-weights",
        "tiny_prior=1.0",
        "--out",
        str(out),
        "--device",
        "cpu",
        "--no-autocast",
        "--micro-batch-size",
        "4",
        "--epochs",
        "20",
        "--steps-per-epoch",
        "2",
        "--lr",
        "3e-3",
        "--warmup-steps",
        "2",
        "--embedding-size",
        "16",
        "--mlp-hidden-size",
        "32",
        "--num-column-row-iterations",
        "1",
        "--log-every",
        "1000",
    ]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return build_parser().parse_args(argv)


def test_loss_decreases_and_stays_finite(prior_dir: Path, tmp_path: Path) -> None:
    losses = train(_args(prior_dir, tmp_path / "m.pt"))
    assert len(losses) == 20
    assert all(np.isfinite(losses)), "training produced a non-finite loss"
    assert min(losses[-5:]) < losses[0], f"loss never improved: {losses[0]:.4f} -> {min(losses[-5:]):.4f}"


def test_checkpoint_loads_through_the_public_api(prior_dir: Path, tmp_path: Path) -> None:
    """A model you train is loadable exactly like a released one."""
    out = tmp_path / "m.pt"
    train(_args(prior_dir, out))
    model = load_icmil(source=out, seed="m", device="cpu")
    assert model.model.in_features == ICMIL_ARCH["in_features"]
    with torch.no_grad():
        logits = model(torch.randn(1, 4, 3, 25), torch.randint(0, 2, (1, 4)), torch.randn(1, 2, 3, 25))
    assert logits.shape == (1, 2, 2)


def test_checkpoint_records_no_local_paths(prior_dir: Path, tmp_path: Path) -> None:
    """Checkpoints get shared; the trainer's ``--data-dir`` should not travel with them."""
    out = tmp_path / "m.pt"
    train(_args(prior_dir, out))
    state = torch.load(out, map_location="cpu", weights_only=True)
    assert set(state["train_args"]) & {"data_dir", "out", "resume"} == set()
    assert str(prior_dir).encode() not in out.read_bytes()
    assert b"git_commit" not in out.read_bytes()


def test_checkpoint_round_trip_is_exact(prior_dir: Path, tmp_path: Path) -> None:
    """Saving must capture the weights the model actually has.

    schedulefree keeps an interpolated iterate and only exposes the evaluation weights
    in eval mode, so a trainer that saved in train mode would silently store different
    parameters from the ones it just evaluated.
    """
    import schedulefree

    from icmil.model import build_icmil

    seed_everything(0)
    arch = {**ICMIL_ARCH, "embedding_size": 16, "mlp_hidden_size": 32, "num_column_row_iterations": 1}
    model = build_icmil(arch)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-3)

    out = tmp_path / "round_trip.pt"
    save_checkpoint(out, model, optimizer, epoch=1, arch=arch, train_args={"lr": 1e-3})
    reloaded = load_icmil(source=out, seed="round_trip", device="cpu").model

    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, reloaded.state_dict()[name]), f"{name} changed across save/load"


def test_seeding_is_reproducible(prior_dir: Path, tmp_path: Path) -> None:
    first = train(_args(prior_dir, tmp_path / "a.pt", seed=7))
    second = train(_args(prior_dir, tmp_path / "b.pt", seed=7))
    assert first == second, "the same seed produced a different loss trajectory"
