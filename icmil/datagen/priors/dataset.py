"""Wires a parameter sampler to a prior generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from icmil.datagen.priors.common import StrategySpec
from icmil.datagen.priors.hierarchical import HierarchicalMILPriorGenerator
from icmil.datagen.priors.joint import JointMILPriorGenerator
from icmil.datagen.priors.sampling import GenerationParams, ParameterSampler, SCMConfig


class BaggedPriorDataset:
    """Main dataset class combining parameter sampling and data generation.

    This class provides an interface for generating synthetic bag-structured
    datasets for MIL training. It combines:
    - ParameterSampler: handles randomness in choosing dimensions
    - BaggedPriorGenerator: generates data with fixed dimensions

    Parameters
    ----------
    batch_size : int
        Number of samples per batch
    min_features : int
        Minimum number of features
    max_features : int
        Maximum number of features (also used for padding)
    max_classes : int
        Maximum number of classes
    min_bag_size : int
        Minimum instances per bag
    max_bag_size : int
        Maximum instances per bag
    min_n_bags : int, optional
        Minimum number of bags (None = fixed at max_n_bags)
    max_n_bags : int
        Maximum number of bags
    log_n_bags : bool
        Sample n_bags from log-uniform distribution
    min_train_size : int | float
        Minimum train size
    max_train_size : int | float
        Maximum train size
    prior_type : str
        "hierarchical" or "joint"
    bag_label_strategy : StrategySpec
        Label generation strategy. For "hierarchical": "histogram",
        "embedding_mean", "embedding_abmil", or a list/dict of these.
    g_type : StrategySpec, optional
        Bag-level function type for hierarchical prior: "mlp", "tree", "random",
        "max", "min", or a list/dict of these. Only used when prior_type="hierarchical".
    device : str
        Computation device
    """

    def __init__(
        self,
        batch_size: int = 256,
        min_features: int = 2,
        max_features: int = 100,
        min_classes: int = 2,
        max_classes: int = 10,
        min_bag_size: int = 10,
        max_bag_size: int = 100,
        min_n_bags: int | None = None,
        max_n_bags: int = 50,
        log_n_bags: bool = False,
        min_train_size: int | float = 0.9,
        max_train_size: int | float = 0.9,
        prior_type: str = "dummy",
        bag_label_strategy: StrategySpec = "random",
        g_type: StrategySpec | None = None,
        instance_threshold: float | None = None,
        noise_ratio: float | tuple[float, float] = 0.0,
        noise_scale: float | tuple[float, float] = 1.0,
        noise_mean_range: tuple[float, float] = (-2.0, 2.0),
        max_instance_classes: int = 15,
        f_type: str = "random",
        f_scm_configs: dict[str, SCMConfig] | None = None,
        g_scm_configs: dict[str, SCMConfig] | None = None,
        use_reg_2_cls: bool = False,
        device: str = "cpu",
        feature_source_config: dict | None = None,
    ) -> None:
        f_scm_configs = self._normalize_scm_config_map(f_scm_configs)
        g_scm_configs = self._normalize_scm_config_map(g_scm_configs)
        _, min_features, sampler_max_features = self._build_feature_source(
            feature_source_config=feature_source_config,
            min_features=min_features,
            max_features=max_features,
        )

        self.sampler = ParameterSampler(
            min_features=min_features,
            max_features=sampler_max_features,
            min_bag_size=min_bag_size,
            max_bag_size=max_bag_size,
            min_n_bags=min_n_bags,
            max_n_bags=max_n_bags,
            log_n_bags=log_n_bags,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            min_classes=min_classes,
            max_classes=max_classes,
            min_instance_classes=2,
            max_instance_classes=max_instance_classes,
            prior_type=prior_type,
        )
        self.generator = self._build_generator(
            prior_type=prior_type,
            max_features=max_features,
            device=device,
            bag_label_strategy=bag_label_strategy,
            g_type=g_type,
            instance_threshold=instance_threshold,
            noise_ratio=noise_ratio,
            noise_scale=noise_scale,
            noise_mean_range=noise_mean_range,
            max_instance_classes=max_instance_classes,
            f_type=f_type,
            f_scm_configs=f_scm_configs,
            g_scm_configs=g_scm_configs,
            use_reg_2_cls=use_reg_2_cls,
        )
        self.use_reg_2_cls = use_reg_2_cls
        self._store_config(
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            min_classes=min_classes,
            max_classes=max_classes,
            min_bag_size=min_bag_size,
            max_bag_size=max_bag_size,
            min_n_bags=min_n_bags,
            max_n_bags=max_n_bags,
            log_n_bags=log_n_bags,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            device=device,
            prior_type=prior_type,
            bag_label_strategy=bag_label_strategy,
            instance_threshold=instance_threshold,
            g_type=g_type,
            noise_ratio=noise_ratio,
            noise_scale=noise_scale,
            noise_mean_range=noise_mean_range,
            max_instance_classes=max_instance_classes,
            feature_source_config=feature_source_config,
        )

    @staticmethod
    def _normalize_scm_config_map(
        scm_configs: dict[str, SCMConfig] | None,
    ) -> dict[str, SCMConfig] | None:
        if scm_configs is None:
            return None
        return {
            key: SCMConfig.from_dict(value) if isinstance(value, dict) else value for key, value in scm_configs.items()
        }

    @staticmethod
    def _build_feature_source(
        feature_source_config: dict | None,
        min_features: int,
        max_features: int,
    ) -> tuple[object | None, int, int]:
        if feature_source_config is not None:
            raise ValueError(
                "feature_source_config is not supported: real-feature priors are not part of this release."
            )
        return None, min_features, max_features

    def _build_generator(
        self,
        prior_type: str,
        max_features: int,
        device: str,
        bag_label_strategy: StrategySpec,
        g_type: StrategySpec | None,
        instance_threshold: float | None,
        noise_ratio: float | tuple[float, float],
        noise_scale: float | tuple[float, float],
        noise_mean_range: tuple[float, float],
        max_instance_classes: int,
        f_type: str,
        f_scm_configs: dict[str, SCMConfig] | None,
        g_scm_configs: dict[str, SCMConfig] | None,
        use_reg_2_cls: bool,
    ) -> HierarchicalMILPriorGenerator | JointMILPriorGenerator:
        if prior_type == "hierarchical":
            hier_kwargs = {
                "max_features": max_features,
                "device": device,
                "max_instance_classes": max_instance_classes,
                "f_scm_configs": f_scm_configs or {"mlp": SCMConfig("mlp"), "tree": SCMConfig("tree")},
                "g_scm_configs": g_scm_configs or {"mlp": SCMConfig("mlp"), "tree": SCMConfig("tree")},
                "f_type": f_type,
                "g_type": g_type if g_type is not None else "random",
                "use_reg_2_cls": use_reg_2_cls,
            }
            if bag_label_strategy != "random":
                hier_kwargs["bag_level_summary"] = bag_label_strategy
            return HierarchicalMILPriorGenerator(**hier_kwargs)
        if prior_type == "joint":
            return JointMILPriorGenerator(
                max_features=max_features,
                device=device,
                f_scm_configs=f_scm_configs or {"mlp": SCMConfig("mlp"), "tree": SCMConfig("tree")},
                f_type=f_type,
                use_reg_2_cls=use_reg_2_cls,
            )
        raise ValueError(f"Unknown prior_type: {prior_type}. Use 'hierarchical' or 'joint'.")

    def _store_config(
        self,
        batch_size: int,
        min_features: int,
        max_features: int,
        min_classes: int,
        max_classes: int,
        min_bag_size: int,
        max_bag_size: int,
        min_n_bags: int | None,
        max_n_bags: int,
        log_n_bags: bool,
        min_train_size: int | float,
        max_train_size: int | float,
        device: str,
        prior_type: str,
        bag_label_strategy: StrategySpec,
        instance_threshold: float | None,
        g_type: StrategySpec | None,
        noise_ratio: float | tuple[float, float],
        noise_scale: float | tuple[float, float],
        noise_mean_range: tuple[float, float],
        max_instance_classes: int,
        feature_source_config: dict | None,
    ) -> None:
        self.batch_size = batch_size
        self.min_features = min_features
        self.max_features = max_features
        self.min_classes = min_classes
        self.max_classes = max_classes
        self.min_bag_size = min_bag_size
        self.max_bag_size = max_bag_size
        self.min_n_bags = min_n_bags
        self.max_n_bags = max_n_bags
        self.log_n_bags = log_n_bags
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.device = device
        self.prior_type = prior_type
        self.bag_label_strategy = bag_label_strategy
        self.instance_threshold = instance_threshold
        self.g_type = g_type
        self.noise_ratio = noise_ratio
        self.noise_scale = noise_scale
        self.noise_mean_range = noise_mean_range
        self.max_instance_classes = max_instance_classes
        self.feature_source_config = feature_source_config

    def get_batch(
        self,
        batch_size: int | None = None,
        params: GenerationParams | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Generate a batch of bag-structured datasets.

        Parameters
        ----------
        batch_size : int, optional
            Override batch size for this call
        params : GenerationParams, optional
            If provided, use these fixed parameters instead of sampling.

        Returns:
        -------
        X : Tensor
            Features of shape (batch_size, n_bags, bag_size, max_features)
        y : Tensor
            Labels of shape (batch_size, n_bags)
        d : Tensor
            Active features count of shape (batch_size,)
        n_bags : Tensor
            Number of bags of shape (batch_size,)
        bag_sizes : Tensor
            Bag sizes of shape (batch_size,)
        n_train_bags : Tensor
            Train split position of shape (batch_size,)
        """
        batch_size = batch_size or self.batch_size
        params = params or self.sampler.sample()
        return self.generator.generate(params, batch_size)

    def sample_params(self) -> GenerationParams:
        """Sample generation parameters without generating data.

        Returns:
        -------
        GenerationParams
            Sampled parameters.
        """
        return self.sampler.sample()

    def __iter__(self) -> BaggedPriorDataset:
        """Returns self as an infinite iterator."""
        return self

    def __next__(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Generate and return the next batch."""
        return self.get_batch()

    def __repr__(self) -> str:
        return (
            f"BaggedPriorDataset(\n"
            f"  prior_type: {self.prior_type}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  features: {self.min_features} - {self.max_features}\n"
            f"  max_classes: {self.max_classes}\n"
            f"  bag_size: {self.min_bag_size} - {self.max_bag_size}\n"
            f"  n_bags: {self.min_n_bags or 'None'} - {self.max_n_bags}\n"
            f"  train_size: {self.min_train_size} - {self.max_train_size}\n"
            f"  bag_label_strategy: {self.bag_label_strategy}\n"
            f"  device: {self.device}\n"
            f")"
        )
