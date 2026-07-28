"""Load a trained ICMIL model for in-context inference.

The three released checkpoints were all trained with the hyper-parameters in
:data:`ICMIL_ARCH`, which is therefore the fallback architecture when a
checkpoint does not describe itself. Checkpoints written by :mod:`icmil.train`
carry their own ``arch`` dict and are loaded with that instead, so a model you
train yourself goes through the same public API.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from icmil.artifacts import resolve_checkpoint

# Architecture of the three released seeds. Only `embedding_size` and
# `mlp_hidden_size` differ from the model defaults; the rest are the defaults.
ICMIL_ARCH: dict[str, int | bool] = {
    "in_features": 25,
    "embedding_size": 256,
    "mlp_hidden_size": 1054,
    "num_attention_heads": 4,
    "num_column_row_iterations": 6,
    "num_final_col_row_layers": 0,
    "num_outputs": 2,
    "feature_group_size": 1,
    "bag_chunk_size": 32,
    "gradient_checkpointing": False,
}


class ICMILInference(nn.Module):
    """Inference-only wrapper exposing the in-context interface.

    Given labelled context bags ``(X_train, y_train)`` and query bags ``X_test``,
    returns query logits in a single forward pass — no per-dataset training.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.eval()

    @torch.no_grad()
    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
        return self.model(X_train, y_train, X_test)


def build_icmil(arch: dict | None = None) -> nn.Module:
    """Construct an untrained ICMIL model (defaults to the released architecture)."""
    from icmil.models.architecture import ICMIL

    return ICMIL(**(ICMIL_ARCH if arch is None else arch))


def load_icmil(
    source: str | Path | None = None,
    seed: str = "c5trd795",
    device: str | torch.device = "cpu",
) -> ICMILInference:
    """Load a trained ICMIL seed for inference.

    Args:
        source: Directory of ``.pt`` checkpoints, or a path to one ``.pt`` file.
            Defaults to ``ICMIL_CKPT_DIR`` or the in-repo ``checkpoints/``.
            A direct file path is only meaningful for a single seed — see below.
        seed: Which released seed to load (one of ``icmil.artifacts.SEEDS``).
        device: Device to place the model on.

    Returns:
        An :class:`ICMILInference` ready for ``model(X_train, y_train, X_test)``.
    """
    src = Path(source) if source is not None else None
    if src is not None and src.is_file():
        # A single file cannot stand in for a specific seed: silently returning it
        # for every requested seed would collapse the cross-seed spread to zero.
        if seed not in (None, "", src.stem.removeprefix("icmil-")):
            raise ValueError(
                f"source={src} is a single checkpoint file but seed={seed!r} was requested. "
                f"Pass a directory to select seeds by name, or request only the seed this file holds."
            )
        ckpt_path = src
    else:
        ckpt_path = resolve_checkpoint(seed, source=source)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "model_state_dict" not in state:
        raise KeyError(f"{ckpt_path} has no 'model_state_dict' key; found {sorted(state)}")

    model = build_icmil(state.get("arch")).to(device)
    # Strict on purpose: the released weights include `final_norm`, and a silent
    # partial load would produce plausible-looking but wrong numbers.
    model.load_state_dict(state["model_state_dict"], strict=True)
    return ICMILInference(model).to(device)
