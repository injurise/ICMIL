"""The ICMIL architecture, its loader, and the shipped checkpoints.

The architecture is frozen against a committed name->shape manifest: the released
weights only load into exactly this model, so an accidental change to the layer stack
should fail here rather than at load time on someone else's machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from icmil.artifacts import SEEDS
from icmil.model import ICMIL_ARCH, ICMILInference, build_icmil, load_icmil
from tests.conftest import CHECKPOINTS, requires_artifacts

MANIFEST = Path(__file__).parent / "fixtures" / "state_dict_manifest.json"


def _tiny(**overrides) -> dict:
    """A fast stand-in architecture for tests that only care about wiring."""
    return {**ICMIL_ARCH, "embedding_size": 16, "mlp_hidden_size": 32, "num_column_row_iterations": 1, **overrides}


# --------------------------------------------------------------------------- architecture


def test_architecture_matches_the_committed_manifest() -> None:
    actual = {name: list(t.shape) for name, t in build_icmil().state_dict().items()}
    if not MANIFEST.exists():
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(actual, indent=2, sort_keys=True))
        pytest.skip(f"wrote the initial manifest to {MANIFEST}; commit it")
    assert actual == json.loads(MANIFEST.read_text()), (
        "the architecture changed — the released checkpoints will no longer load. "
        "If this was intended, update the manifest deliberately."
    )


def test_final_norm_is_part_of_the_architecture() -> None:
    """``final_norm`` is trained and present in every released checkpoint.

    The layer is easy to mistake for a redundant cleanup target; deleting it
    silently changes every prediction.
    """
    assert "final_norm.weight" in build_icmil().state_dict()


@pytest.mark.parametrize(("n_train", "n_test", "bag_size"), [(6, 2, 4), (2, 1, 1), (20, 5, 9)])
def test_forward_shapes(n_train: int, n_test: int, bag_size: int) -> None:
    model = build_icmil(_tiny()).eval()
    gen = torch.Generator().manual_seed(0)
    X_train = torch.randn(1, n_train, bag_size, 25, generator=gen)
    y_train = torch.randint(0, 2, (1, n_train), generator=gen)
    X_test = torch.randn(1, n_test, bag_size, 25, generator=gen)
    with torch.no_grad():
        out = model(X_train, y_train, X_test)
    assert out.shape == (1, n_test, 2), "one row of class logits per query bag"
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("n_features", [10, 25, 2048])
def test_feature_width_is_coerced_to_the_trained_width(n_features: int) -> None:
    """Wider inputs are truncated, narrower ones zero-padded.

    A property of the released weights, not of the method.
    """
    model = build_icmil(_tiny()).eval()
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        out = model(
            torch.randn(1, 4, 3, n_features, generator=gen),
            torch.randint(0, 2, (1, 4), generator=gen),
            torch.randn(1, 2, 3, n_features, generator=gen),
        )
    assert out.shape == (1, 2, 2)


def test_query_bags_cannot_see_each_other() -> None:
    """In-context inference must treat query bags independently.

    If a query bag's prediction changed when a *different* query bag changed, the model
    would be leaking test information between predictions.
    """
    model = build_icmil(_tiny()).eval()
    gen = torch.Generator().manual_seed(0)
    X_train = torch.randn(1, 6, 4, 25, generator=gen)
    y_train = torch.randint(0, 2, (1, 6), generator=gen)
    X_test = torch.randn(1, 3, 4, 25, generator=gen)

    perturbed = X_test.clone()
    perturbed[:, 2] = torch.randn(1, 4, 25, generator=gen)
    with torch.no_grad():
        base, other = model(X_train, y_train, X_test), model(X_train, y_train, perturbed)
    torch.testing.assert_close(base[:, :2], other[:, :2], rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- loading


@requires_artifacts
@pytest.mark.parametrize("seed", SEEDS)
def test_released_checkpoints_load_strictly(seed: str) -> None:
    model = load_icmil(seed=seed, device="cpu")
    assert isinstance(model, ICMILInference)
    assert not model.model.training, "the loader must return a model in eval mode"
    assert model.model.in_features == 25


@requires_artifacts
def test_seeds_are_actually_different_models() -> None:
    """Guards against every seed resolving to the same file.

    That failure mode is invisible in the table except as an implausibly small ±.
    """
    weights = [load_icmil(seed=s).model.bag_latent_init for s in SEEDS]
    for i in range(1, len(weights)):
        assert not torch.equal(weights[0], weights[i]), "two seeds loaded identical weights"


def test_a_single_file_cannot_answer_for_every_seed(tmp_path: Path) -> None:
    """Requesting three seeds from one file would report a spread of exactly zero."""
    fake = tmp_path / "icmil-c5trd795.pt"
    fake.write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="single checkpoint file"):
        load_icmil(source=fake, seed="ggwsqibd")


def test_unknown_seed_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown seed"):
        load_icmil(seed="not_a_seed")


def test_missing_checkpoint_directory_explains_the_override(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ICMIL_CKPT_DIR"):
        load_icmil(source=tmp_path / "nope", seed="c5trd795")


@requires_artifacts
def test_checkpoints_are_self_describing_and_clean() -> None:
    """What ships is the weights, the epoch and the architecture — nothing else."""
    for seed in SEEDS:
        state = torch.load(CHECKPOINTS / f"icmil-{seed}.pt", map_location="cpu", weights_only=True)
        assert set(state) == {"model_state_dict", "epoch", "arch"}, sorted(state)
        assert state["arch"] == ICMIL_ARCH
