"""Helpers shared by the prior generators.

* ``StrategySpec`` / ``_sample_strategy`` — a "strategy" is either a fixed name, a
  list to choose uniformly from, or a name->weight dict. Sampling one per dataset
  is what makes the prior a *distribution over* MIL problems rather than a single one.
* ``ACTIVATIONS`` — the activation functions an SCM may be built from.
"""

from __future__ import annotations

import numpy as np
import torch


# Type for strategy specifications: fixed string, uniform list, or weighted dict
StrategySpec = str | list[str] | dict[str, float]


def _sample_strategy(strategy: StrategySpec) -> str:
    """Sample a strategy from str, list (uniform), or dict (weighted)."""
    if isinstance(strategy, str):
        return strategy
    if isinstance(strategy, list):
        return str(np.random.choice(strategy))
    # dict: keys are strategies, values are weights
    keys = list(strategy.keys())
    weights = np.array([strategy[k] for k in keys], dtype=float)
    weights /= weights.sum()
    return str(np.random.choice(keys, p=weights))


# Available activation functions (shared by SCM and Hierarchical generators)
ACTIVATIONS = {
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,
    "leaky_relu": torch.nn.LeakyReLU,
}
