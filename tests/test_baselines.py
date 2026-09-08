"""Tests for the MIL baselines: ABMIL, ACMIL-GA and DSMIL.

All three fit the same way, so they share the contract asserted here:
``(X_train, y_train, X_test) -> (B, n_test, max_classes)`` log-probabilities,
always-eval mode, and seed determinism.
"""

from __future__ import annotations

import pytest
import torch

from icmil.baselines.abmil_baseline import ABMIL, ABMILBaseline
from icmil.baselines.acmil_baseline import (
    ACMILBaseline,
    ACMILGatedAttention,
    acmil_composite_loss,
)
from icmil.baselines.dsmil_baseline import DSMIL, DSMILBaseline


@pytest.fixture
def train_test_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One batch of 12 train bags over 3 classes plus 2 test bags."""
    # n_train=12 keeps 2-fold stratified CV feasible for 3 classes.
    gen = torch.Generator().manual_seed(42)
    X_train = torch.randn(1, 12, 4, 8, generator=gen)
    y_train = (torch.arange(12) % 3).unsqueeze(0)
    X_test = torch.randn(1, 2, 4, 8, generator=gen)
    return X_train, y_train, X_test


# ---------------------------------------------------------------------------
# ACMILGatedAttention module
# ---------------------------------------------------------------------------


class TestACMILGatedAttention:
    def test_output_shapes(self) -> None:
        model = ACMILGatedAttention(in_dim=8, embed_dim=16, attn_dim=16, n_token=4, num_classes=3)
        x = torch.randn(5, 7, 8)  # (B bags, M instances, F)
        sub_logits, slide_logits, attn = model(x)
        assert sub_logits.shape == (5, 4, 3)
        assert slide_logits.shape == (5, 3)
        assert attn.shape == (5, 4, 7)
        assert torch.isfinite(sub_logits).all()
        assert torch.isfinite(slide_logits).all()

    def test_attention_rows_are_distributions(self) -> None:
        model = ACMILGatedAttention(in_dim=8, embed_dim=16, n_token=3, num_classes=2).eval()
        _, _, attn = model(torch.randn(2, 6, 8))
        assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 3), atol=1e-5)
        assert (attn >= 0).all()

    def test_masking_only_when_training(self) -> None:
        """Masking is stochastic in train mode (outputs vary) and off in eval (stable)."""
        torch.manual_seed(0)
        model = ACMILGatedAttention(in_dim=8, embed_dim=16, n_token=3, n_masked_patch=4, mask_drop=0.6, num_classes=2)
        x = torch.randn(2, 10, 8)

        model.eval()
        assert torch.allclose(model(x)[1], model(x)[1]), "eval forward must be deterministic (no masking)"

        model.train()
        outs = [model(x)[1] for _ in range(5)]
        assert any(not torch.allclose(outs[0], o) for o in outs[1:]), (
            "train forward should vary under stochastic masking"
        )


# ---------------------------------------------------------------------------
# ACMIL composite loss
# ---------------------------------------------------------------------------


class TestCompositeLoss:
    def test_single_token_is_bag_ce_only(self) -> None:
        torch.manual_seed(0)
        sub_logits = torch.randn(4, 1, 3)
        slide_logits = torch.randn(4, 3)
        attn = torch.rand(4, 1, 6)
        y = torch.tensor([0, 1, 2, 1])
        loss = acmil_composite_loss(sub_logits, slide_logits, attn, y)
        assert torch.allclose(loss, torch.nn.functional.cross_entropy(slide_logits, y))

    def test_multi_token_adds_branch_and_diversity(self) -> None:
        torch.manual_seed(0)
        sub_logits = torch.randn(4, 3, 3)
        slide_logits = torch.randn(4, 3)
        attn = torch.softmax(torch.randn(4, 3, 6), dim=-1)
        y = torch.tensor([0, 1, 2, 1])
        loss = acmil_composite_loss(sub_logits, slide_logits, attn, y)
        assert loss.item() != torch.nn.functional.cross_entropy(slide_logits, y).item()
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# ABMIL module
# ---------------------------------------------------------------------------


class TestABMILModule:
    def test_output_shape(self) -> None:
        model = ABMIL(in_dim=8, embed_dim=16, attn_dim=8, num_classes=3)
        assert model(torch.randn(4, 5, 8)).shape == (4, 3)

    def test_attn_mask_ignores_masked_instances(self) -> None:
        # Padding instances with a mask must give the same logits as passing
        # only the real instances.
        torch.manual_seed(0)
        model = ABMIL(in_dim=8, embed_dim=16, attn_dim=8, dropout=0.0, num_classes=2).eval()
        real = torch.randn(1, 4, 8)
        padded = torch.cat([real, torch.randn(1, 3, 8) * 10], dim=1)
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
        with torch.no_grad():
            assert torch.allclose(model(padded, attn_mask=mask), model(real), atol=1e-5)


# ---------------------------------------------------------------------------
# ABMILBaseline
# ---------------------------------------------------------------------------


def _tiny_abmil(max_classes: int = 3, **overrides) -> ABMILBaseline:
    """Fast ABMIL fixture: 1 (lr, wd) combo, 3 epochs, tiny widths."""
    kwargs: dict = {
        "max_classes": max_classes,
        "embed_dim": 16,
        "attn_dim": 8,
        "lr_grid": (1e-3,),
        "wd_grid": (0.0,),
        "epochs": 3,
        "patience": 2,
        # n_train=12 with 3 classes -> 3 val bags, enough for stratification.
        "val_fraction": 0.25,
    }
    kwargs.update(overrides)
    return ABMILBaseline(**kwargs)


class TestABMILBaseline:
    def test_output_shape(self, train_test_tensors) -> None:
        X_train, y_train, X_test = train_test_tensors
        out = _tiny_abmil()(X_train, y_train, X_test)
        assert out.shape == (X_train.shape[0], X_test.shape[1], 3)

    def test_output_is_log_probabilities(self, train_test_tensors) -> None:
        X_train, y_train, X_test = train_test_tensors
        out = _tiny_abmil()(X_train, y_train, X_test)
        assert (out <= 0).all()
        assert torch.allclose(out.logsumexp(dim=-1), torch.zeros_like(out[..., 0]), atol=1e-5)

    def test_always_eval_mode(self) -> None:
        model = ABMILBaseline()
        model.train(True)
        assert not model.training

    def test_feature_truncation(self, train_test_tensors) -> None:
        # Right-padded feature columns (added by the eval harness) are stripped.
        X_train, y_train, X_test = train_test_tensors
        pad = (0, 5)
        out_padded = _tiny_abmil()(
            torch.nn.functional.pad(X_train, pad),
            y_train,
            torch.nn.functional.pad(X_test, pad),
        )
        assert torch.allclose(out_padded, _tiny_abmil()(X_train, y_train, X_test))

    def test_same_seed_reproducible(self, train_test_tensors) -> None:
        X_train, y_train, X_test = train_test_tensors
        out_a = _tiny_abmil(seed=7)(X_train, y_train, X_test)
        out_b = _tiny_abmil(seed=7)(X_train, y_train, X_test)
        assert torch.allclose(out_a, out_b)

    def test_different_seeds_differ(self, train_test_tensors) -> None:
        X_train, y_train, X_test = train_test_tensors
        out_a = _tiny_abmil(seed=0)(X_train, y_train, X_test)
        out_b = _tiny_abmil(seed=1)(X_train, y_train, X_test)
        assert not torch.allclose(out_a, out_b)

    def test_no_val_split_uses_first_grid_entry(self, train_test_tensors) -> None:
        # val_fraction=0 trains on all bags for the full budget, no selection.
        X_train, y_train, X_test = train_test_tensors
        model = _tiny_abmil(val_fraction=0.0, lr_grid=(1e-3, 1e-2), wd_grid=(0.0, 0.1))
        out = model(X_train, y_train, X_test)
        assert out.shape == (1, 2, 3)
        assert (out <= 0).all()


# ---------------------------------------------------------------------------
# ACMILBaseline
# ---------------------------------------------------------------------------


def _tiny_acmil(max_classes: int = 3, **overrides) -> ACMILBaseline:
    """Fast ACMIL fixture: 1 combo, 3 epochs, tiny widths."""
    kwargs: dict = {
        "max_classes": max_classes,
        "embed_dim": 16,
        "attn_dim": 16,
        "n_token": 3,
        "n_masked_patch": 2,
        "mask_drop": 0.5,
        "lr_grid": (1e-3,),
        "wd_grid": (1e-4,),
        "dropout_grid": (0.0,),
        "epochs": 3,
        "patience": 2,
        # n_train=12 with 3 classes -> 3 val bags, enough for stratification.
        "val_fraction": 0.25,
    }
    kwargs.update(overrides)
    return ACMILBaseline(**kwargs)


class TestACMILBaseline:
    def test_output_shape(self, train_test_tensors) -> None:
        X_train, y_train, X_test = train_test_tensors
        out = _tiny_acmil()(X_train, y_train, X_test)
        assert out.shape == (1, 2, 3)

    def test_output_is_log_probabilities(self, train_test_tensors) -> None:
        X_train, y_train, X_test = train_test_tensors
        out = _tiny_acmil()(X_train, y_train, X_test)
        assert (out <= 0).all()
        assert torch.allclose(out.exp().sum(dim=-1), torch.ones(1, 2), atol=1e-5)

    def test_always_eval_mode(self) -> None:
        model = ACMILBaseline()
        model.train(True)
        assert not model.training

    def test_same_seed_reproducible(self, train_test_tensors) -> None:
        X_train, y_train, X_test = train_test_tensors
        out_a = _tiny_acmil(seed=7)(X_train, y_train, X_test)
        out_b = _tiny_acmil(seed=7)(X_train, y_train, X_test)
        assert torch.allclose(out_a, out_b)

    def test_different_seeds_differ(self, train_test_tensors) -> None:
        X_train, y_train, X_test = train_test_tensors
        out_a = _tiny_acmil(seed=0)(X_train, y_train, X_test)
        out_b = _tiny_acmil(seed=1)(X_train, y_train, X_test)
        assert not torch.allclose(out_a, out_b)

    def test_no_val_split_uses_first_grid_entry(self, train_test_tensors) -> None:
        # val_fraction=0 trains on all bags for the full budget, no selection.
        X_train, y_train, X_test = train_test_tensors
        model = _tiny_acmil(val_fraction=0.0, lr_grid=(1e-3, 1e-2), wd_grid=(1e-4, 0.1))
        out = model(X_train, y_train, X_test)
        assert out.shape == (1, 2, 3)
        assert (out <= 0).all()


# ---------------------------------------------------------------------------
# DSMIL module
# ---------------------------------------------------------------------------


class TestDSMILModule:
    def test_output_shape(self) -> None:
        model = DSMIL(in_dim=8, embed_dim=16, attn_dim=8, num_classes=3)
        out = model(torch.randn(4, 5, 8))
        assert out.shape == (4, 3)

    def test_attn_mask_ignores_masked_instances(self) -> None:
        # Padding instances with a mask must give the same logits as passing
        # only the real instances: both streams have to honour the mask.
        torch.manual_seed(0)
        model = DSMIL(in_dim=8, embed_dim=16, attn_dim=8, num_classes=2).eval()
        real = torch.randn(1, 4, 8)
        padded = torch.cat([real, torch.randn(1, 3, 8) * 10], dim=1)
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
        with torch.no_grad():
            out_masked = model(padded, attn_mask=mask)
            out_real = model(real)
        assert torch.allclose(out_masked, out_real, atol=1e-5)


# ---------------------------------------------------------------------------
# DSMILBaseline
# ---------------------------------------------------------------------------


def _tiny_dsmil(max_classes: int = 3, **overrides) -> DSMILBaseline:
    """Fast DSMIL baseline fixture: 1 (lr, wd) combo, 3 epochs, tiny widths."""
    kwargs: dict = {
        "max_classes": max_classes,
        "embed_dim": 16,
        "attn_dim": 8,
        "lr_grid": (1e-3,),
        "wd_grid": (0.0,),
        "epochs": 3,
        "patience": 2,
        # n_train=12 with 3 classes -> 3 val bags, enough for stratification.
        "val_fraction": 0.25,
    }
    kwargs.update(overrides)
    return DSMILBaseline(**kwargs)


class TestDSMILBaseline:
    def test_output_shape(self, train_test_tensors) -> None:
        X_train, y_train, X_test = train_test_tensors
        out = _tiny_dsmil()(X_train, y_train, X_test)
        assert out.shape == (X_train.shape[0], X_test.shape[1], 3)

    def test_always_eval_mode(self) -> None:
        model = DSMILBaseline()
        model.train(True)
        assert not model.training

    def test_output_is_log_probabilities(self, train_test_tensors) -> None:
        X_train, y_train, X_test = train_test_tensors
        out = _tiny_dsmil()(X_train, y_train, X_test)
        assert (out <= 0).all()
        assert torch.allclose(out.logsumexp(dim=-1), torch.zeros_like(out[..., 0]), atol=1e-5)

    def test_feature_truncation(self, train_test_tensors) -> None:
        # Right-padded feature columns (added by the eval harness) are stripped,
        # so padded inputs match unpadded ones exactly.
        X_train, y_train, X_test = train_test_tensors
        pad = (0, 5)
        out_padded = _tiny_dsmil()(
            torch.nn.functional.pad(X_train, pad),
            y_train,
            torch.nn.functional.pad(X_test, pad),
        )
        out_plain = _tiny_dsmil()(X_train, y_train, X_test)
        assert torch.allclose(out_padded, out_plain)

    def test_same_seed_reproducible(self, train_test_tensors) -> None:
        X_train, y_train, X_test = train_test_tensors
        out_a = _tiny_dsmil(seed=3)(X_train, y_train, X_test)
        out_b = _tiny_dsmil(seed=3)(X_train, y_train, X_test)
        assert torch.allclose(out_a, out_b)

    def test_different_seeds_differ(self, train_test_tensors) -> None:
        X_train, y_train, X_test = train_test_tensors
        out_a = _tiny_dsmil(seed=0)(X_train, y_train, X_test)
        out_b = _tiny_dsmil(seed=1)(X_train, y_train, X_test)
        assert not torch.allclose(out_a, out_b)

    def test_no_val_split_uses_first_grid_entry(self, train_test_tensors) -> None:
        # val_fraction=0 trains on all bags for the full budget, no selection.
        X_train, y_train, X_test = train_test_tensors
        model = _tiny_dsmil(val_fraction=0.0, lr_grid=(1e-3, 1e-2), wd_grid=(0.0, 0.1))
        out = model(X_train, y_train, X_test)
        assert out.shape == (1, 2, 3)
        assert (out <= 0).all()
