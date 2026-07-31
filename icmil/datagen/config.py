"""Configuration for synthetic prior generation.

:data:`PAPER_CONFIG` is the recipe the released ICMIL checkpoints were trained on:
three prior "arms" generated separately and then mixed during training according
to :data:`PAPER_TRAIN_WEIGHTS`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any

# ---------------------------------------------------------------- SCM hyper-parameters

# Sampling distributions for the MLP structural causal model that generates each
# dataset's latent function. Taken from TabICL's defaults.
TABICL_MLP_SCM_HP: dict[str, Any] = {
    "mlp_num_layers": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 6,
        "min_mean": 1,
        "round": True,
        "lower_bound": 2,
    },
    "mlp_hidden_dim": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 130,
        "min_mean": 5,
        "round": True,
        "lower_bound": 4,
    },
    "mlp_init_std": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 10.0,
        "min_mean": 0.01,
        "round": False,
        "lower_bound": 0.0,
    },
    "mlp_block_wise_dropout": {"distribution": "meta_choice", "choice_values": [True, False]},
    "mlp_dropout_prob": {"distribution": "meta_beta", "scale": 0.9, "min": 0.1, "max": 5.0},
    "mlp_activation": {
        "distribution": "meta_choice",
        "choice_values": ["tanh", "relu", "gelu", "silu", "leaky_relu"],
    },
    "noise_std": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 0.3,
        "min_mean": 0.0001,
        "round": False,
        "lower_bound": 0.0,
    },
    "pre_sample_noise_std": {"distribution": "meta_choice", "choice_values": [True, False]},
    "num_causes": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 12,
        "min_mean": 1,
        "round": True,
        "lower_bound": 1,
    },
}

_MLP_SCM = {"mlp": {"scm_type": "mlp", "hp_config": TABICL_MLP_SCM_HP}}

# How a bag's instance-level signal becomes its label, as a distribution over rules
# resampled per dataset. The continuous arm leans on learned pooling; the discrete
# arm on counting.
CONTINUOUS_AGGREGATION = {"histogram": 0.0, "presence": 0.0, "embedding_mean": 0.2, "embedding_abmil": 0.8}
DISCRETE_AGGREGATION = {"histogram": 0.8, "presence": 0.2}


# ---------------------------------------------------------------- dataclasses


@dataclass(frozen=True)
class CurriculumStage:
    """Bag-shape limits that apply until ``until_batch``.

    Stages are consumed in ascending ``until_batch`` order; batches past the last
    stage keep using it.
    """

    until_batch: int
    min_bag_size: int
    max_bag_size: int
    min_instance_classes: int | None = None
    max_instance_classes: int | None = None


@dataclass(frozen=True)
class ShapeConfig:
    """Per-batch dataset shape, sampled uniformly between the bounds."""

    batch_size: int = 128
    min_features: int = 25
    max_features: int = 25
    min_classes: int = 2
    max_classes: int = 2
    min_bag_size: int = 2
    max_bag_size: int = 20
    min_n_bags: int = 90
    max_n_bags: int = 125
    min_train_size: float = 0.80
    max_train_size: float = 0.81


@dataclass(frozen=True)
class PriorArm:
    """One prior, generated into its own H5 file.

    Args:
        name: Arm name; also the H5 filename stem and the key used by
            ``--prior-weights`` at training time.
        prior_kwargs: Passed straight to the generator (``prior_type`` selects it).
    """

    name: str
    prior_kwargs: dict[str, Any]


@dataclass(frozen=True)
class GenConfig:
    """A complete generation run."""

    arms: list[PriorArm]
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    curriculum: list[CurriculumStage] | None = None
    num_batches: int = 40000
    num_workers: int = 0
    seed: int = 42
    device: str = "cpu"

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> GenConfig:
        raw = json.loads(text)
        return cls(
            arms=[PriorArm(**a) for a in raw["arms"]],
            shape=ShapeConfig(**raw.get("shape", {})),
            curriculum=[CurriculumStage(**s) for s in raw["curriculum"]] if raw.get("curriculum") else None,
            **{k: raw[k] for k in ("num_batches", "num_workers", "seed", "device") if k in raw},
        )

    def with_overrides(self, **kwargs: Any) -> GenConfig:
        """Return a copy with non-``None`` top-level fields replaced."""
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})


# ---------------------------------------------------------------- the paper's recipe

# Two curriculum stages: start with small, simple bags and widen. The presence of a
# curriculum also fixes batch order at training time (see icmil.datagen.h5_dataset).
PAPER_CURRICULUM = [
    CurriculumStage(until_batch=10000, min_bag_size=4, max_bag_size=15, max_instance_classes=12),
    CurriculumStage(until_batch=40000, min_bag_size=6, max_bag_size=20, max_instance_classes=20),
]

PAPER_CONFIG = GenConfig(
    arms=[
        # A single SCM over the whole flattened bag, so within-bag interactions are
        # expressible; instances are shuffled afterwards to force permutation
        # invariance by exposure rather than by construction.
        PriorArm(
            name="joint_mlp_long_curr",
            prior_kwargs={
                "prior_type": "joint",
                "f_type": "mlp",
                "f_scm_configs": _MLP_SCM,
                "use_reg_2_cls": True,
            },
        ),
        # Two-level: an instance-level SCM f, then an aggregation g over the bag.
        # Continuous variant — pooling-based aggregation, MLP-SCM for g.
        PriorArm(
            name="hierarchical_mlp_cont_mlp_long_curr",
            prior_kwargs={
                "prior_type": "hierarchical",
                "bag_label_strategy": CONTINUOUS_AGGREGATION,
                "f_type": "mlp",
                "g_type": "mlp",
                "f_scm_configs": _MLP_SCM,
                "g_scm_configs": _MLP_SCM,
                "use_reg_2_cls": True,
            },
        ),
        # Discrete variant — counting-based aggregation, lookup-table g.
        PriorArm(
            name="hierarchical_mlp_disc_lookup_long_curr",
            prior_kwargs={
                "prior_type": "hierarchical",
                "bag_label_strategy": DISCRETE_AGGREGATION,
                "f_type": "mlp",
                "g_type": "lookup",
                "f_scm_configs": _MLP_SCM,
                "g_scm_configs": {},
                "use_reg_2_cls": True,
            },
        ),
    ],
    curriculum=PAPER_CURRICULUM,
    num_batches=40000,
)

# Mixing weights used at training time, one per arm of PAPER_CONFIG.
# Pass to the trainer as: --prior-weights "joint_mlp_long_curr=0.7,..."
PAPER_TRAIN_WEIGHTS: dict[str, float] = {
    "joint_mlp_long_curr": 0.70,
    "hierarchical_mlp_cont_mlp_long_curr": 0.15,
    "hierarchical_mlp_disc_lookup_long_curr": 0.15,
}

CONFIGS: dict[str, GenConfig] = {"paper": PAPER_CONFIG}
