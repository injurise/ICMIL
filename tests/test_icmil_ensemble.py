"""Tests for the ICMIL view ensemble (a small stand-in model, no checkpoint needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from icmil.baselines.icmil_ensemble import EnsembleTransform, ICMILEnsemble

IN_FEATURES = 6
N_TRAIN, N_TEST, N_INSTANCES = 40, 7, 5


class NearestClassMean(nn.Module):
    """A tiny in-context MIL model: logits are negative distances to the class means of bag means.

    Records the inputs of every call so the tests can inspect the views.
    """

    def __init__(self, in_features: int = IN_FEATURES) -> None:
        super().__init__()
        self.in_features = in_features
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
        self.calls.append((X_train, y_train, X_test))
        train, test = X_train.mean(dim=2), X_test.mean(dim=2)  # (B, n, F)
        means = torch.stack([train[0][y_train[0] == c].mean(dim=0) for c in (0, 1)])
        return -torch.cdist(test[0], means)[None]


def _dataset(
    n_features: int = IN_FEATURES, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    y_train = torch.arange(N_TRAIN) % 2
    y_test = torch.arange(N_TEST) % 2
    X_train = torch.randn(N_TRAIN, N_INSTANCES, n_features, generator=gen) + 2.0 * y_train[:, None, None]
    X_test = torch.randn(N_TEST, N_INSTANCES, n_features, generator=gen) + 2.0 * y_test[:, None, None]
    return X_train[None], y_train[None], X_test[None], y_test


def test_returns_log_probabilities() -> None:
    X_train, y_train, X_test, y_test = _dataset()
    out = ICMILEnsemble(NearestClassMean())(X_train, y_train, X_test)
    assert out.shape == (1, N_TEST, 2)
    torch.testing.assert_close(out.exp().sum(dim=-1), torch.ones(1, N_TEST))
    assert (out[0].argmax(dim=-1) == y_test).all()  # the separable toy problem is solved


def test_members_alternate_transforms_on_stratified_bag_subsamples() -> None:
    X_train, y_train, X_test, _ = _dataset()
    model = NearestClassMean()
    ICMILEnsemble(model, n_members=16)(X_train, y_train, X_test)
    assert len(model.calls) == 16
    context_sizes = set()
    for member, (X_context, y_context, X_query) in enumerate(model.calls):
        n_context = X_context.shape[1]
        context_sizes.add(n_context)
        assert 0.8 * N_TRAIN <= n_context < N_TRAIN  # about 90 % of the bags
        assert abs(int(y_context.sum()) - n_context / 2) <= 1  # stratified
        assert X_query.shape == (1, N_TEST, N_INSTANCES, IN_FEATURES)  # every test bag, every instance
        flat = X_context.reshape(-1, IN_FEATURES)
        torch.testing.assert_close(flat.mean(dim=0), torch.zeros(IN_FEATURES), atol=1e-4, rtol=0)
        covariance = torch.cov(flat.T)
        off_diagonal = covariance - torch.diag(torch.diag(covariance))
        raw_correlation = torch.corrcoef(X_train[0].reshape(-1, IN_FEATURES).T)
        raw_off_diagonal = raw_correlation - torch.diag(torch.diag(raw_correlation))
        if member % 2 == 0:  # power transform: per-feature, so the correlation structure stays close to the raw one
            assert (off_diagonal - raw_off_diagonal).abs().max() < 0.2
        else:  # rotation: features are mixed
            assert (off_diagonal - raw_off_diagonal).abs().max() > 0.2
    assert len(context_sizes) > 1  # the share varies between members


def test_members_cycle_through_the_given_transforms() -> None:
    X_train, y_train, X_test, _ = _dataset(n_features=3)  # padded to 6 columns
    model = NearestClassMean()
    ICMILEnsemble(model, n_members=6, transforms=("none", "rotation", "rotation"))(X_train, y_train, X_test)
    padding_untouched = [bool((X_context[..., 3:] == 0).all()) for X_context, _, _ in model.calls]
    assert padding_untouched == [True, False, False, True, False, False]  # only rotations fill the padding
    X_query = model.calls[0][2]
    torch.testing.assert_close(X_query[0, ..., :3], X_test[0])  # "none" passes the features through

    default = ICMILEnsemble(NearestClassMean(), seed=3)(X_train, y_train, X_test)
    explicit = ICMILEnsemble(NearestClassMean(), seed=3, transforms=("power", "rotation"))(X_train, y_train, X_test)
    torch.testing.assert_close(default, explicit, atol=0, rtol=0)


def test_seed_and_successive_calls() -> None:
    X_train, y_train, X_test, _ = _dataset()
    first = ICMILEnsemble(NearestClassMean(), seed=0)
    a = first(X_train, y_train, X_test)
    b = ICMILEnsemble(NearestClassMean(), seed=0)(X_train, y_train, X_test)
    c = ICMILEnsemble(NearestClassMean(), seed=1)(X_train, y_train, X_test)
    torch.testing.assert_close(a, b, atol=0, rtol=0)  # same seed, same views
    assert not torch.equal(a, c)
    assert not torch.equal(a, first(X_train, y_train, X_test))  # a second call draws fresh views


def test_batch_elements_are_independent_datasets() -> None:
    X_train, y_train, X_test, _ = _dataset()
    X_other, y_other, X_other_test, _ = _dataset(seed=1)
    batched = ICMILEnsemble(NearestClassMean(), seed=0)(
        torch.cat([X_train, X_other]), torch.cat([y_train, y_other]), torch.cat([X_test, X_other_test])
    )
    single = ICMILEnsemble(NearestClassMean(), seed=0)
    torch.testing.assert_close(batched[0], single(X_train, y_train, X_test)[0], atol=0, rtol=0)
    torch.testing.assert_close(batched[1], single(X_other, y_other, X_other_test)[0], atol=0, rtol=0)


@pytest.mark.parametrize("n_features", [3, IN_FEATURES, 9])
def test_features_are_padded_or_truncated_to_the_model_width(n_features: int) -> None:
    X_train, y_train, X_test, _ = _dataset(n_features=n_features)
    model = NearestClassMean()
    out = ICMILEnsemble(model, n_members=2)(X_train, y_train, X_test)
    assert torch.isfinite(out).all()
    for X_context, _, X_query in model.calls:
        assert X_context.shape[-1] == X_query.shape[-1] == IN_FEATURES
    if n_features < IN_FEATURES:
        power_context, rotation_context = model.calls[0][0], model.calls[1][0]
        assert (power_context[..., n_features:] == 0).all()  # the power transform leaves the padding at zero
        assert (rotation_context[..., n_features:] != 0).any()  # the rotation spreads features into it


def test_rotation_is_an_isometry_of_the_standardised_features() -> None:
    gen = torch.Generator().manual_seed(0)
    context = torch.randn(200, 5, generator=gen) * torch.tensor([1.0, 10.0, 0.1, 3.0, 5.0]) + 4.0
    transform = EnsembleTransform("rotation", context, gen)
    standardised = (context - context.mean(dim=0)) / context.std(dim=0)
    rotated = transform(context)
    torch.testing.assert_close(
        rotated @ rotated.T, standardised @ standardised.T, atol=1e-3, rtol=0
    )  # inner products kept
    assert not torch.allclose(rotated, standardised, atol=1e-2)
    other = EnsembleTransform("rotation", context, gen)(context)
    assert not torch.allclose(rotated, other, atol=1e-2)  # every member draws its own rotation


def test_none_transform_is_the_identity() -> None:
    context = torch.randn(20, 4, generator=torch.Generator().manual_seed(0))
    assert EnsembleTransform("none", context, torch.Generator())(context) is context


def test_power_transform_standardises_and_keeps_shape() -> None:
    gen = torch.Generator().manual_seed(0)
    context = torch.randn(500, 4, generator=gen).exp()  # skewed
    transform = EnsembleTransform("power", context, gen)
    out = transform(context)
    torch.testing.assert_close(out.mean(dim=0), torch.zeros(4), atol=1e-4, rtol=0)
    torch.testing.assert_close(out.std(dim=0), torch.ones(4), atol=1e-2, rtol=0)
    skew = lambda x: (((x - x.mean(dim=0)) / x.std(dim=0)) ** 3).mean(dim=0).abs()  # noqa: E731
    assert (skew(out) < skew(context)).all()
    bags = torch.randn(3, 7, 4, generator=gen).exp()
    assert transform(bags).shape == bags.shape


def test_power_transform_stays_finite_outside_the_fitted_range() -> None:
    """Values far outside the context's range overflow Yeo-Johnson; they take the context's extremes."""
    gen = torch.Generator().manual_seed(0)
    # Left-skewed features in [0, 1] give a large fitted exponent, so huge values overflow to
    # infinity; scikit-learn's own ``transform`` raises a ValueError on such a query.
    context = torch.tensor(np.random.default_rng(0).beta(8, 1, size=(400, 2)), dtype=torch.float64)
    transform = EnsembleTransform("power", context, gen)
    fitted = transform(context)
    query = torch.tensor([[1e60, -1e60], [1e200, 0.5]], dtype=torch.float64)
    out = transform(query)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:, 0], fitted[:, 0].max().expand(2))  # overflowed: the context's maximum
    assert out[0, 1] < fitted[:, 1].min()  # large but finite results are kept as they are


def test_reproduce_row_wraps_the_trained_seeds(tmp_path) -> None:
    """``--baselines icmil_ensemble`` loads one checkpoint per run and reports a row like any baseline."""
    import h5py

    from icmil.artifacts import SEEDS
    from icmil.model import ICMIL_ARCH, build_icmil
    from icmil.reproduce import evaluate

    arch = {**ICMIL_ARCH, "embedding_size": 16, "mlp_hidden_size": 32, "num_column_row_iterations": 1}
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for seed in SEEDS:
        torch.save(
            {"model_state_dict": build_icmil(arch).state_dict(), "arch": arch}, checkpoints / f"icmil-{seed}.pt"
        )

    datasets = tmp_path / "datasets"
    datasets.mkdir()
    with h5py.File(datasets / "uci_benchmark.h5", "w") as h5:
        for split in range(2):
            group = h5.create_group(f"musk1/split_{split}")
            for name, n_bags in (("train", N_TRAIN), ("test", N_TEST + 1)):
                group.create_dataset(f"X_{name}", data=np.random.default_rng(split).normal(size=(n_bags, 4, 25)))
                group.create_dataset(f"y_{name}", data=np.arange(n_bags) % 2)

    results = evaluate(
        data_source=datasets,
        model_source=checkpoints,
        baselines=["icmil_ensemble"],
        icmil_seeds=[],
        tasks=["uci_musk1"],
        n_seeds=2,
        device="cpu",
        output_dir=tmp_path / "out",
    )
    metrics = results["icmil_ensemble"]["uci_musk1"]
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["n_outer"] == 2
