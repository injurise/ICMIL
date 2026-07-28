"""Parameter sampling for bagged prior data generation.

This module provides:
- GenerationParams: Dataclass holding all parameters for a single batch generation
- SCMConfig / SCMParams: Configuration and sampled parameters for SCM-based label generation
- ParameterSampler: Samples generation parameters from configured ranges
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import loguniform
from tabicl.prior.hp_sampling import HpSamplerList


@dataclass
class GenerationParams:
    """Parameters for a single batch generation.

    Attributes:
    ----------
    n_bags : int
        Number of bags in the batch
    bag_size : int
        Number of instances per bag
    n_features : int
        Number of features per instance
    n_train : int
        Number of training bags (rest are test)
    num_classes : int
        Number of classes
    """

    n_bags: int
    bag_size: int
    n_features: int
    n_train: int
    num_classes: int
    max_instance_classes: int = 15


@dataclass
class SCMConfig:
    """Configuration for SCM-based bag label generation.

    Uses HpSamplerList format directly for flexible hyperparameter sampling.
    Pass hyperparameter configs in the HpSamplerList format.

    Parameters
    ----------
    scm_type : str
        "mlp" for MLPSCM, "tree" for TreeSCM, or "random" to randomly choose
    hp_config : dict
        Hyperparameter config in HpSamplerList format. Keys are parameter names,
        values are dicts with 'distribution' and distribution-specific params.

        Example for MLP:
        {
            "mlp_num_layers": {"distribution": "uniform_int", "min": 2, "max": 10},
            "mlp_hidden_dim": {"distribution": "uniform_int", "min": 16, "max": 128},
            "mlp_activation": {"distribution": "meta_choice", "choice_values": ["tanh", "relu"]},
            ...
        }

        See HpSamplerList for supported distributions:
        - "uniform": min, max
        - "uniform_int": min, max
        - "meta_choice": choice_values
        - "meta_trunc_norm_log_scaled": min_mean, max_mean, lower_bound, round
        - etc.
    """

    scm_type: str = "mlp"
    hp_config: dict = field(default_factory=dict)

    def __post_init__(self):
        """Set default hp_config if not provided."""
        if not self.hp_config:
            self.hp_config = get_default_scm_hp_config(self.scm_type)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization (e.g., HDF5 metadata)."""
        return {
            "scm_type": self.scm_type,
            "hp_config": self.hp_config,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SCMConfig:
        """Create SCMConfig from a dictionary."""
        return cls(
            scm_type=d.get("scm_type", "mlp"),
            hp_config=d.get("hp_config", {}),
        )


def get_default_scm_hp_config(scm_type: str) -> dict:
    """Get default HpSamplerList config for the given SCM type.

    Parameters
    ----------
    scm_type : str
        "mlp" or "tree"

    Returns:
    -------
    dict
        Default hyperparameter config
    """
    # Common parameters
    hp_config = {
        "noise_std": {
            "distribution": "meta_trunc_norm_log_scaled",
            "min_mean": 0.001,
            "max_mean": 0.1,
            "lower_bound": 0.0,
            "round": False,
        },
        "pre_sample_noise_std": {
            "distribution": "meta_choice",
            "choice_values": [True, False],
        },
    }

    if scm_type == "mlp":
        hp_config.update(
            {
                "mlp_num_layers": {"distribution": "uniform_int", "min": 2, "max": 10},
                "mlp_hidden_dim": {"distribution": "uniform_int", "min": 16, "max": 128},
                "mlp_activation": {
                    "distribution": "meta_choice",
                    "choice_values": ["tanh", "relu", "gelu", "silu", "leaky_relu"],
                },
                "mlp_init_std": {"distribution": "uniform", "min": 0.1, "max": 2.0},
                "mlp_block_wise_dropout": {"distribution": "meta_choice", "choice_values": [True, False]},
                "mlp_dropout_prob": {"distribution": "uniform", "min": 0.0, "max": 0.5},
                "num_causes": {
                    "distribution": "meta_trunc_norm_log_scaled",
                    "max_mean": 12,
                    "min_mean": 1,
                    "round": True,
                    "lower_bound": 1,
                },
            }
        )
    elif scm_type == "tree":
        hp_config.update(
            {
                "tree_model": {
                    "distribution": "meta_choice",
                    "choice_values": ["decision_tree", "extra_trees", "random_forest", "xgboost"],
                },
                "tree_num_layers": {"distribution": "uniform_int", "min": 1, "max": 3},
                "tree_hidden_dim": {"distribution": "uniform_int", "min": 3, "max": 15},
                "tree_max_depth_lambda": {"distribution": "uniform", "min": 0.2, "max": 1.0},
                "tree_n_estimators_lambda": {"distribution": "uniform", "min": 0.2, "max": 1.0},
            }
        )

    return hp_config


@dataclass
class SCMParams:
    """Sampled SCM parameters for a single batch generation."""

    scm_type: str  # "mlp" or "tree"

    # MLP params (used when scm_type="mlp")
    mlp_num_layers: int | None = None
    mlp_hidden_dim: int | None = None
    mlp_activation: str | None = None
    mlp_init_std: float | None = None
    mlp_block_wise_dropout: bool | None = None
    mlp_dropout_prob: float | None = None
    # Only consumed by the causal-f=mlp branch of the hierarchical prior;
    # tabicl overrides num_causes to num_features whenever is_causal=False.
    num_causes: int | None = None

    # Tree params (used when scm_type="tree")
    tree_model: str | None = None
    tree_num_layers: int | None = None
    tree_hidden_dim: int | None = None
    tree_max_depth_lambda: float | None = None
    tree_n_estimators_lambda: float | None = None

    # Common params
    noise_std: float = 0.01
    pre_sample_noise_std: bool = False


def _resolve_sampled_value(value, default=None):
    """Resolve a sampled value from HpSamplerList.

    Meta distributions in HpSamplerList return sampler functions that need
    to be called again to get the actual value.
    """
    if value is None:
        return default
    # Meta distributions return callables that need to be invoked
    while callable(value):
        value = value()
    return value


def _sample_scm_params(config: SCMConfig, device: str = "cpu") -> SCMParams:
    """Sample SCM hyperparameters using HpSamplerList from tabicl.

    Parameters
    ----------
    config : SCMConfig
        Configuration with hp_config in HpSamplerList format
    device : str
        Device for HpSamplerList (default: "cpu")

    Returns:
    -------
    SCMParams
        Sampled parameters for SCM label generation
    """
    # Determine SCM type
    scm_type = config.scm_type
    if scm_type == "random":
        scm_type = np.random.choice(["mlp", "tree"])
        # Get appropriate default config for the sampled type
        hp_config = get_default_scm_hp_config(scm_type)
    else:
        hp_config = config.hp_config

    # Sample using HpSamplerList
    sampler = HpSamplerList(hp_config, device=device)
    sampled_raw = sampler.sample()

    # Resolve all sampled values (meta distributions return callables)
    sampled = {k: _resolve_sampled_value(v) for k, v in sampled_raw.items()}

    # Convert sampled values to appropriate types and build SCMParams
    if scm_type == "mlp":
        return SCMParams(
            scm_type="mlp",
            mlp_num_layers=int(sampled.get("mlp_num_layers", 5)),
            mlp_hidden_dim=int(sampled.get("mlp_hidden_dim", 64)),
            mlp_activation=sampled.get("mlp_activation", "tanh"),
            mlp_init_std=float(sampled.get("mlp_init_std", 1.0)),
            mlp_block_wise_dropout=sampled.get("mlp_block_wise_dropout", True),
            mlp_dropout_prob=float(sampled.get("mlp_dropout_prob", 0.1)),
            num_causes=int(sampled["num_causes"]) if "num_causes" in sampled else None,
            noise_std=float(sampled.get("noise_std", 0.01)),
            pre_sample_noise_std=sampled.get("pre_sample_noise_std", False),
        )
    else:  # tree
        return SCMParams(
            scm_type="tree",
            tree_model=sampled.get("tree_model", "xgboost"),
            tree_num_layers=int(sampled.get("tree_num_layers", 2)),
            tree_hidden_dim=int(sampled.get("tree_hidden_dim", 10)),
            tree_max_depth_lambda=float(sampled.get("tree_max_depth_lambda", 0.5)),
            tree_n_estimators_lambda=float(sampled.get("tree_n_estimators_lambda", 0.5)),
            noise_std=float(sampled.get("noise_std", 0.01)),
            pre_sample_noise_std=sampled.get("pre_sample_noise_std", False),
        )


class ParameterSampler:
    """Samples generation parameters from configured ranges.

    This class is responsible for all randomness in choosing dimensions
    and prior-specific hyperparameters. The generator classes receive
    fixed parameters and produce deterministic (given the random seed)
    data of those exact dimensions.

    Parameters
    ----------
    min_features : int
        Minimum number of features per dataset
    max_features : int
        Maximum number of features per dataset
    min_bag_size : int
        Minimum number of instances per bag
    max_bag_size : int
        Maximum number of instances per bag
    min_n_bags : int, optional
        Minimum number of bags. If None, uses max_n_bags (fixed).
    max_n_bags : int
        Maximum number of bags
    log_n_bags : bool
        If True, sample n_bags from log-uniform distribution
    min_train_size : int | float
        Minimum train size (absolute if int, ratio if float)
    max_train_size : int | float
        Maximum train size (absolute if int, ratio if float)
    min_classes : int
        Minimum number of classes
    max_classes : int
        Maximum number of classes
    min_instance_classes : int
        Minimum max_instance_classes to sample
    max_instance_classes : int
        Maximum max_instance_classes to sample
    replay_small : bool
        If True, occasionally sample smaller n_bags for robustness
    prior_type : str, optional
        Type of prior ("hierarchical" or "joint").
    """

    def __init__(
        self,
        min_features: int = 2,
        max_features: int = 100,
        min_bag_size: int = 10,
        max_bag_size: int = 100,
        min_n_bags: int | None = None,
        max_n_bags: int = 50,
        log_n_bags: bool = False,
        min_train_size: int | float = 0.1,
        max_train_size: int | float = 0.9,
        min_classes: int = 2,
        max_classes: int = 10,
        min_instance_classes: int = 2,
        max_instance_classes: int = 15,
        replay_small: bool = False,
        prior_type: str | None = None,
    ) -> None:
        assert min_features <= max_features, "Invalid feature range"
        assert min_bag_size <= max_bag_size, "Invalid bag size range"
        self._validate_train_size_range(min_train_size, max_train_size)

        self.min_features = min_features
        self.max_features = max_features
        self.min_bag_size = min_bag_size
        self.max_bag_size = max_bag_size
        self.min_n_bags = min_n_bags
        self.max_n_bags = max_n_bags
        self.log_n_bags = log_n_bags
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.min_classes = min_classes
        self.max_classes = max_classes
        self.min_instance_classes = min_instance_classes
        self.max_instance_classes = max_instance_classes
        self.replay_small = replay_small
        self.prior_type = prior_type

    @staticmethod
    def _validate_train_size_range(min_train_size: int | float, max_train_size: int | float) -> None:
        """Validates that training size range is valid."""
        if not isinstance(min_train_size, int | float) or not isinstance(max_train_size, int | float):
            raise TypeError("Training sizes must be int or float")

        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            assert 0 < min_train_size <= max_train_size, "0 < min_train_size <= max_train_size"
        elif isinstance(min_train_size, float) and isinstance(max_train_size, float):
            assert 0 < min_train_size <= max_train_size <= 1, "0 < min_train_size <= max_train_size <= 1"
        else:
            raise ValueError("Both training sizes must be of the same type (int or float)")

    def sample(self) -> GenerationParams:
        """Sample a complete set of generation parameters.

        Returns:
        -------
        GenerationParams
            Dataclass containing all parameters needed for batch generation.
        """
        n_bags = self._sample_n_bags()
        return GenerationParams(
            n_bags=n_bags,
            bag_size=self._sample_bag_size(),
            n_features=self._sample_n_features(),
            n_train=self._sample_train_size(n_bags),
            num_classes=self._sample_num_classes(),
            max_instance_classes=self._sample_max_instance_classes(),
        )

    def _sample_n_bags(self) -> int:
        """Sample number of bags."""
        if self.min_n_bags is None:
            return self.max_n_bags

        if self.min_n_bags >= self.max_n_bags:
            return self.max_n_bags

        if self.log_n_bags:
            n_bags = int(loguniform.rvs(self.min_n_bags, self.max_n_bags))
        else:
            n_bags = np.random.randint(self.min_n_bags, self.max_n_bags)

        if self.replay_small:
            p = np.random.random()
            if p < 0.05:
                return np.random.randint(25, 35)
            elif p < 0.2:
                return int(loguniform.rvs(15, 25))

        return n_bags

    def _sample_bag_size(self) -> int:
        """Sample bag size (instances per bag)."""
        if self.min_bag_size == self.max_bag_size:
            return self.min_bag_size
        return np.random.randint(self.min_bag_size, self.max_bag_size + 1)

    def _sample_n_features(self) -> int:
        """Sample number of features."""
        if self.min_features == self.max_features:
            return self.min_features
        return np.random.randint(self.min_features, self.max_features + 1)

    def _sample_train_size(self, n_bags: int) -> int:
        """Sample train/test split position."""
        if isinstance(self.min_train_size, int) and isinstance(self.max_train_size, int):
            if self.min_train_size == self.max_train_size:
                n_train = self.min_train_size
            else:
                n_train = np.random.randint(self.min_train_size, min(self.max_train_size + 1, n_bags))
        else:
            train_ratio = np.random.uniform(self.min_train_size, self.max_train_size)
            n_train = int(n_bags * train_ratio)

        # Ensure at least 1 bag for train and 1 for test
        return max(1, min(n_train, n_bags - 1))

    def _sample_num_classes(self) -> int:
        """Sample number of classes."""
        if self.min_classes == self.max_classes:
            return self.min_classes
        return np.random.randint(self.min_classes, self.max_classes + 1)

    def _sample_max_instance_classes(self) -> int:
        """Sample max_instance_classes (K upper bound for histogram/presence)."""
        if self.min_instance_classes == self.max_instance_classes:
            return self.min_instance_classes
        return np.random.randint(self.min_instance_classes, self.max_instance_classes + 1)
