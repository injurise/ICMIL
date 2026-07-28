"""Hierarchical two-level MIL prior generator."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch
import torch.nn.functional as F
from tabicl.prior.mlp_scm import MLPSCM
from tabicl.prior.reg2cls import (
    MulticlassAssigner,
    outlier_removing,
    permute_classes,
    standard_scaling,
)
from tabicl.prior.tree_scm import TreeSCM
from torch import Tensor

from icmil.datagen.priors.sampling import (
    GenerationParams,
    SCMConfig,
    _sample_scm_params,
)
from icmil.mil_pooling import ABMILAggregator

from icmil.datagen.priors.common import ACTIVATIONS, StrategySpec, _sample_strategy

logger = logging.getLogger(__name__)


class SimplifiedHierarchicalMILPriorGenerator:
    """Simplified base class for bag-structured data generation.

    A lighter alternative to BaggedPriorGenerator — no noise injection,
    no feature source.  Subclasses implement ``_generate_labels_and_features``
    which returns both the feature tensor and bag-level labels.

    Parameters
    ----------
    max_features : int
        Maximum features for padding (output tensor width).
    device : str
        Computation device.
    """

    _shuffle_instances: bool = True

    def __init__(
        self,
        max_features: int = 100,
        device: str = "cpu",
    ):
        self.max_features = max_features
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        params: GenerationParams,
        batch_size: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Generate a batch of bag-structured data with FIXED parameters.

        Returns:
        -------
        X : (batch_size, n_bags, bag_size, max_features)
        y : (batch_size, n_bags)
        d, n_bags_tensor, bag_sizes_tensor, n_train_tensor : (batch_size,) each
        """
        n_bags = params.n_bags
        bag_size = params.bag_size
        n_features = params.n_features
        n_train = params.n_train
        num_classes = params.num_classes
        max_instance_classes = params.max_instance_classes

        X, y = self._generate_labels_and_features(
            num_classes, batch_size, n_bags, bag_size, n_features, max_instance_classes=max_instance_classes
        )

        # Pad features to max_features
        if n_features < self.max_features:
            X = F.pad(X, (0, self.max_features - n_features), mode="constant", value=0)

        # Shuffle instances within each bag
        if self._shuffle_instances:
            X = self._shuffle_instance_dim(X, batch_size, n_bags, bag_size)

        # Verify and fix train/test split
        success, X, y = self._sanity_check(X, y, n_train, min_classes=min(2, num_classes))
        if not success:
            logger.warning("Sanity check failed: could not find a valid train/test split for all batch elements.")

        # Metadata tensors
        d = torch.full((batch_size,), n_features, device=self.device, dtype=torch.long)
        n_bags_tensor = torch.full((batch_size,), n_bags, device=self.device, dtype=torch.long)
        bag_sizes_tensor = torch.full((batch_size,), bag_size, device=self.device, dtype=torch.long)
        n_train_tensor = torch.full((batch_size,), n_train, device=self.device, dtype=torch.long)

        return X, y, d, n_bags_tensor, bag_sizes_tensor, n_train_tensor

    def _generate_labels_and_features(
        self,
        num_classes: int,
        batch_size: int,
        n_bags: int,
        bag_size: int,
        n_features: int,
        max_instance_classes: int = 15,
    ) -> tuple[Tensor, Tensor]:
        """Generate features and bag-level labels. Override in subclasses.

        Returns (X, y) where X is (batch_size, n_bags, bag_size, n_features)
        and y is (batch_size, n_bags).
        """
        raise NotImplementedError("Subclasses must implement _generate_labels_and_features")

    def _shuffle_instance_dim(self, X: Tensor, batch_size: int, n_bags: int, bag_size: int) -> Tensor:
        """Shuffle instances within each bag independently (permutation on dim=2)."""
        perm = torch.rand(batch_size, n_bags, bag_size, device=self.device).argsort(dim=-1)
        return X.gather(2, perm.unsqueeze(-1).expand_as(X))

    @staticmethod
    def _check_min_counts(labels: Tensor, num_classes: int, min_per_class: int) -> bool:
        """Check that every class has at least ``min_per_class`` occurrences."""
        counts = torch.bincount(labels.view(-1), minlength=num_classes)[:num_classes]
        return counts.min().item() >= min_per_class

    @staticmethod
    def _is_degenerate(y_i: Tensor) -> bool:
        """True if all bag labels in a single dataset are identical."""
        return torch.unique(y_i).numel() <= 1

    @staticmethod
    def _sanity_check(
        X: Tensor,
        y: Tensor,
        n_train_bags: int,
        min_classes: int = 2,
    ) -> tuple[bool, Tensor, Tensor]:
        """Fix train/test splits by swapping bags to ensure both splits contain all classes."""
        for i in range(X.shape[0]):
            yi = y[i]
            n_test = yi.shape[0] - n_train_bags

            for cls in range(min_classes):
                train_cls = (yi[:n_train_bags] == cls).nonzero(as_tuple=True)[0]
                test_cls = (yi[n_train_bags:] == cls).nonzero(as_tuple=True)[0]

                # Need at least 20% representation in each split
                min_train = max(1, int(0.2 * n_train_bags / min_classes))
                min_test = max(1, int(0.2 * n_test / min_classes))

                while len(test_cls) < min_test and len(train_cls) > min_train:
                    # Pick a random train bag of this class
                    swap_idx = train_cls[torch.randint(len(train_cls), (1,))].item()
                    # Pick a random test bag of a different class
                    test_other = (yi[n_train_bags:] != cls).nonzero(as_tuple=True)[0]
                    if len(test_other) == 0:
                        break
                    target_idx = n_train_bags + test_other[torch.randint(len(test_other), (1,))].item()

                    # Swap
                    X[i, swap_idx], X[i, target_idx] = X[i, target_idx].clone(), X[i, swap_idx].clone()
                    y[i, swap_idx], y[i, target_idx] = y[i, target_idx].clone(), y[i, swap_idx].clone()

                    # Recompute indices
                    train_cls = (yi[:n_train_bags] == cls).nonzero(as_tuple=True)[0]
                    test_cls = (yi[n_train_bags:] == cls).nonzero(as_tuple=True)[0]

        return True, X, y

    def _aggregate_histogram(
        self, instance_outputs: Tensor, K: int, batch_size: int, n_bags: int, bag_size: int
    ) -> tuple[Tensor, int]:
        """Convert instance outputs to class assignments, then build normalized histogram.

        Accepts either (seq_len, K) continuous outputs (argmaxed here) or (seq_len,)
        integer labels (already binned upstream, e.g. under use_reg_2_cls).
        """
        instance_classes = instance_outputs if instance_outputs.ndim == 1 else instance_outputs.argmax(dim=-1)
        instance_classes_2d = instance_classes.reshape(batch_size * n_bags, bag_size)
        hist = torch.zeros(batch_size * n_bags, K, device=self.device)
        for k in range(K):
            hist[:, k] = (instance_classes_2d == k).float().sum(dim=-1)
        hist = hist / bag_size  # normalize to [0,1]
        return hist, K

    def _aggregate_embedding(
        self,
        embeddings_4d: Tensor,
        batch_size: int,
        n_bags: int,
        bag_size: int,
        strategy: str,
        embed_dim: int,
    ) -> tuple[Tensor, int]:
        """Pool instance embeddings (mean or ABMIL)."""
        if strategy == "embedding_mean":
            pooled = embeddings_4d.mean(dim=2)
        else:  # embedding_abmil
            attn_dim = int(np.random.randint(16, 64))
            gate = bool(np.random.choice([True, False]))
            dropout = float(np.random.uniform(0.0, 0.2))
            abmil = ABMILAggregator(embed_dim=embed_dim, attn_dim=attn_dim, gate=gate, dropout=dropout)
            abmil.to(self.device)
            abmil.eval()
            pooled, _ = abmil(embeddings_4d)

        pooled_flat = pooled.reshape(batch_size * n_bags, embed_dim)
        return pooled_flat, embed_dim

    @staticmethod
    def _reg2cls_feature_process(X_flat: Tensor) -> Tensor:
        """Apply tabicl's Reg2Cls feature pipeline per dataset.

        Mirrors Reg2Cls defaults: cat_prob=0.2, outlier_threshold=4, standard_scale,
        permute_features. Operates on (seq_len, n_features).
        """
        if (
            random.random() < 0.05
        ):  # Adjusted to 0.05 to reduce the number of categorical features as MIL problems, mainly images
            col_prob = random.random()
            for col in range(X_flat.shape[1]):
                if random.random() < col_prob:
                    num_cats = max(round(random.gammavariate(1, 10)), 2)
                    X_flat[:, col] = MulticlassAssigner(num_cats, mode="rank", ordered_prob=0.3)(
                        X_flat[:, col]
                    ).float()
        X_flat = outlier_removing(X_flat, threshold=4.0)
        X_flat = standard_scaling(X_flat)
        perm = torch.randperm(X_flat.shape[1], device=X_flat.device)
        X_flat = X_flat[:, perm]
        return X_flat

    def _g_lookup_table(
        self,
        bag_summary: Tensor,
        summary_dim: int,
        num_classes: int,
        batch_size: int,
        n_bags: int,
    ) -> Tensor:
        if not ((bag_summary >= 0).all() and bag_summary.max() <= 1.0):
            raise ValueError("Lookup table requires discrete bag summaries (histogram or presence)")

        n_inputs = min(int(np.random.randint(2, min(5, summary_dim + 1))), summary_dim)
        selected = np.sort(np.random.choice(summary_dim, size=n_inputs, replace=False))

        presence = (bag_summary[:, selected] > 0).long()
        powers = (2 ** torch.arange(n_inputs, device=self.device)).long()
        pattern_idx = (presence * powers).sum(dim=-1)

        n_patterns = 2**n_inputs
        lookup = torch.randint(0, num_classes, (n_patterns,), device=self.device)

        y = lookup[pattern_idx]
        return y.reshape(batch_size, n_bags)

    def _run_causal_scm(
        self,
        seq_len: int,
        num_features: int,
        num_outputs: int,
        scm_config: SCMConfig,
    ) -> tuple[Tensor, Tensor]:
        """Create a causal MLPSCM and jointly generate (X, y).

        Returns:
        -------
        X : Tensor of shape (seq_len, num_features)
        y : Tensor of shape (seq_len, num_outputs)
        """
        # num_causes must be provided by the MLP SCM hp_config (tabicl-style meta-sampling).
        scm_params = _sample_scm_params(scm_config, device=str(self.device))
        scm = MLPSCM(
            seq_len=seq_len,
            num_features=num_features,
            num_outputs=num_outputs,
            is_causal=True,
            num_causes=scm_params.num_causes,
            y_is_effect=True,
            in_clique=False,
            sort_features=False,
            num_layers=scm_params.mlp_num_layers,
            hidden_dim=scm_params.mlp_hidden_dim,
            mlp_activations=ACTIVATIONS[scm_params.mlp_activation],
            init_std=scm_params.mlp_init_std,
            block_wise_dropout=scm_params.mlp_block_wise_dropout,
            mlp_dropout_prob=scm_params.mlp_dropout_prob,
            noise_std=scm_params.noise_std,
            pre_sample_noise_std=scm_params.pre_sample_noise_std,
            device=str(self.device),
        )
        scm.to(self.device)
        scm.eval()

        with torch.no_grad():
            X, y = scm()

        if y.dim() == 1:
            y = y.unsqueeze(-1)

        return X, y

    def _run_scm(
        self,
        X_flat: Tensor,
        num_features: int,
        num_outputs: int,
        scm_type: str,
        scm_config: SCMConfig,
    ) -> Tensor:
        """Instantiate a fresh random SCM, feed X_flat through its forward pass, return outputs.

        X_flat is injected as the SCM's causes via xsampler patching so the real forward()
        is used (NaN handling, noise layers, etc.) rather than raw layer iteration.
        Both SCMs use is_causal=False, so forward() returns (causes, last_layer_output);
        we discard causes and return the transformed output.
        """
        seq_len = X_flat.shape[0]
        scm_params = _sample_scm_params(scm_config, device=str(self.device))

        if scm_type == "mlp":
            scm = MLPSCM(
                seq_len=seq_len,
                num_features=num_features,
                num_outputs=num_outputs,
                is_causal=False,
                num_layers=scm_params.mlp_num_layers,
                hidden_dim=scm_params.mlp_hidden_dim,
                mlp_activations=ACTIVATIONS[scm_params.mlp_activation],
                init_std=scm_params.mlp_init_std,
                block_wise_dropout=scm_params.mlp_block_wise_dropout,
                mlp_dropout_prob=scm_params.mlp_dropout_prob,
                noise_std=scm_params.noise_std,
                pre_sample_noise_std=scm_params.pre_sample_noise_std,
                device=str(self.device),
            )
        else:  # tree
            scm = TreeSCM(
                seq_len=seq_len,
                num_features=num_features,
                num_outputs=num_outputs,
                is_causal=False,
                tree_model=scm_params.tree_model,
                num_layers=scm_params.tree_num_layers,
                hidden_dim=scm_params.tree_hidden_dim,
                max_depth_lambda=scm_params.tree_max_depth_lambda,
                n_estimators_lambda=scm_params.tree_n_estimators_lambda,
                noise_std=scm_params.noise_std,
                pre_sample_noise_std=scm_params.pre_sample_noise_std,
                device=str(self.device),
            )

        scm.to(self.device)
        scm.eval()

        # Patch xsampler so forward() uses X_flat as causes instead of sampling new data.
        scm.xsampler.sample = lambda: X_flat

        with torch.no_grad():
            _, outputs = scm()

        return outputs


class HierarchicalMILPriorGenerator(SimplifiedHierarchicalMILPriorGenerator):
    """Two-level hierarchical MIL prior generator using causal SCMs.

    Level 1 — A causal MLPSCM jointly generates instance features (X) and
      instance-level outputs (y_instance). Both are sampled from the same
      causal graph, so features genuinely participate in the mechanism that
      produces labels.
    Aggregation — Bag summary applied to instance-level outputs. Choices:
      permutation-invariant (histogram, presence, mean pool, ABMIL) or the
      non-PI ``concat`` strategy that flattens per-instance embeddings into
      one long bag-level vector.
    Level 2 — Bag function g: maps the bag summary to a bag-level label via
      a random SCM (mlp/tree) or simpler function (max/min).

    Parameters
    ----------
    max_features : int
        Maximum features for padding.
    device : str
        Computation device.
    bag_level_summary : StrategySpec
        One of "histogram", "presence", "embedding_mean", "embedding_abmil",
        "concat", a list of these (uniform sampling), or a dict mapping them
        to weights. ``concat`` is the only non-PI option.
    max_instance_classes : int
        Maximum K for histogram mode (K sampled uniformly in [2, max_instance_classes]).

    Notes:
    -----
    The causal MLPSCM used by the f=mlp branch expects ``num_causes`` to be
    present in the MLP ``SCMConfig.hp_config`` (see
    ``_sample_scm_params``); it is sampled per batch via tabicl's
    meta-distributions. Other structural tabicl knobs (``is_causal``,
    ``y_is_effect``, ``in_clique``, ``sort_features``) are fixed by this
    generator and are intentionally not forwarded.
    """

    # Probability of rejecting a generated dataset whose bag labels are all identical
    # and resampling the per-dataset pipeline. The loop terminates almost surely as
    # long as DEGENERATE_REJECT_PROB < 1.
    DEGENERATE_REJECT_PROB: float = 0.9

    def __init__(
        self,
        f_scm_configs: dict[str, SCMConfig],
        g_scm_configs: dict[str, SCMConfig],
        max_features: int = 100,
        device: str = "cpu",
        bag_level_summary: StrategySpec | None = None,
        max_instance_classes: int = 20,
        f_type: str = "random",
        g_type: StrategySpec = "random",
        use_reg_2_cls: bool = False,
    ):
        super().__init__(
            max_features=max_features,
            device=device,
        )
        self.bag_level_summary = (
            bag_level_summary
            if bag_level_summary is not None
            else ["histogram", "presence", "embedding_mean", "embedding_abmil"]
        )
        self.f_scm_configs = f_scm_configs
        self.g_scm_configs = g_scm_configs
        self.f_type = f_type
        self.g_type = g_type
        self.use_reg_2_cls = use_reg_2_cls
        self._validate_scm_configs()

    def _validate_scm_configs(self) -> None:
        """Check that required SCM configs are present for the chosen f_type/g_type."""
        # f configs
        if self.f_type == "random":
            required_f = {"mlp", "tree"}
        else:
            required_f = {self.f_type}
        missing_f = required_f - set(self.f_scm_configs)
        if missing_f:
            raise ValueError(
                f"f_type={self.f_type!r} requires f_scm_configs keys {required_f}, but missing: {missing_f}"
            )

        # g configs — only "mlp" and "tree" need SCM configs
        scm_types = {"mlp", "tree"}
        if isinstance(self.g_type, str):
            if self.g_type == "random":
                required_g = {"mlp", "tree"}
            elif self.g_type in scm_types:
                required_g = {self.g_type}
            else:
                required_g = set()
        else:
            possible = set(self.g_type) if isinstance(self.g_type, list) else set(self.g_type.keys())
            required_g = possible & scm_types
        missing_g = required_g - set(self.g_scm_configs)
        if missing_g:
            raise ValueError(
                f"g_type={self.g_type!r} requires g_scm_configs keys {required_g}, but missing: {missing_g}"
            )

    def _generate_labels_and_features(
        self,
        num_classes: int,
        batch_size: int,
        n_bags: int,
        bag_size: int,
        n_features: int,
        max_instance_classes: int = 15,
        min_per_class: int = 6,
    ) -> tuple[Tensor, Tensor]:
        """Generate features and labels with independent f, aggregation, and g per dataset."""
        X_parts = []
        y_parts = []
        for _ in range(batch_size):
            while True:
                strategy = _sample_strategy(self.bag_level_summary)
                if strategy in ("histogram", "presence"):
                    is_discrete_strategy = True
                    K = int(np.random.randint(2, max_instance_classes + 1))
                    num_outputs = 1 if self.use_reg_2_cls else K
                else:
                    is_discrete_strategy = False
                    K = None
                    num_outputs = int(np.random.randint(8, 33))

                X_i, y_instance_i = self._apply_f(
                    n_bags=n_bags,
                    bag_size=bag_size,
                    n_features=n_features,
                    num_outputs=num_outputs,
                    is_discrete_strategy=is_discrete_strategy,
                    f_type=self.f_type,
                    K=K,
                )
                bag_summary_i, summary_dim = self._aggregate(
                    y_instance=y_instance_i,
                    strategy=strategy,
                    K=K,
                    batch_size=1,
                    n_bags=n_bags,
                    bag_size=bag_size,
                    embed_dim=num_outputs,
                )
                y_i = self._apply_g(
                    bag_summary=bag_summary_i,
                    summary_dim=summary_dim,
                    num_classes=num_classes,
                    batch_size=1,
                    n_bags=n_bags,
                    is_discrete=is_discrete_strategy,
                    min_per_class=min_per_class,
                    g_type=self.g_type,
                )
                if not (self._is_degenerate(y_i) and random.random() < self.DEGENERATE_REJECT_PROB):
                    break
            X_parts.append(X_i)
            y_parts.append(y_i)

        X = torch.stack(X_parts).reshape(batch_size, n_bags, bag_size, n_features)
        y = torch.cat(y_parts, dim=0)
        return X, y

    def _apply_f(
        self,
        n_bags: int,
        bag_size: int,
        n_features: int,
        num_outputs: int,
        is_discrete_strategy: bool,
        f_type: str,
        K: int | None,
    ) -> tuple[Tensor, Tensor]:
        """Apply instance-level function f for a single dataset.

        For mlp: causal MLPSCM jointly generates features and outputs.
        For tree: random Gaussian features, TreeSCM applied as function.

        Returns (X, y_instance) where X is (n_bags * bag_size, n_features)
        and y_instance is (n_bags * bag_size, num_outputs).
        """
        seq_len = n_bags * bag_size
        if f_type == "random":
            f_type = str(np.random.choice(["mlp", "tree"]))

        if f_type == "mlp":
            X_flat, y_instance = self._run_causal_scm(
                seq_len=seq_len,
                num_features=n_features,
                num_outputs=num_outputs,
                scm_config=self.f_scm_configs["mlp"],
            )
        else:
            X_flat = torch.randn(seq_len, n_features, device=self.device)
            y_instance = self._run_scm(
                X_flat=X_flat,
                num_features=n_features,
                num_outputs=num_outputs,
                scm_type="tree",
                scm_config=self.f_scm_configs["tree"],
            )

        if self.use_reg_2_cls:
            X_flat = self._reg2cls_feature_process(X_flat)
            if is_discrete_strategy:  # only appling to discrete agg strategies that need labels
                y_cont = y_instance.squeeze(-1) if y_instance.ndim == 2 else y_instance
                y_cont = standard_scaling(y_cont.unsqueeze(-1)).squeeze(-1)
                mode = random.choice(["rank", "value"])
                y_instance = MulticlassAssigner(K, mode=mode, ordered_prob=0.2)(y_cont)

        return X_flat, y_instance

    def _aggregate(
        self,
        y_instance: Tensor,
        strategy: str,
        K: int | None,
        batch_size: int,
        n_bags: int,
        bag_size: int,
        embed_dim: int,
    ) -> tuple[Tensor, int]:
        """Aggregate instance outputs into bag-level summaries."""
        if strategy in ("histogram", "presence"):
            bag_summary, summary_dim = self._aggregate_histogram(
                instance_outputs=y_instance,
                K=K,
                batch_size=batch_size,
                n_bags=n_bags,
                bag_size=bag_size,
            )
            if strategy == "presence":
                bag_summary = (bag_summary > 0).float()
            return bag_summary, summary_dim

        if strategy in ("embedding_mean", "embedding_abmil"):
            embeddings_4d = y_instance.reshape(batch_size, n_bags, bag_size, embed_dim)
            return self._aggregate_embedding(
                embeddings_4d=embeddings_4d,
                batch_size=batch_size,
                n_bags=n_bags,
                bag_size=bag_size,
                strategy=strategy,
                embed_dim=embed_dim,
            )

        if strategy == "concat":
            # Non-PI: keep instance order, flatten into one long bag-level vector.
            summary = y_instance.reshape(batch_size * n_bags, bag_size * embed_dim)
            return summary, bag_size * embed_dim

        raise ValueError(f"Unknown bag_level_summary: {strategy!r}")

    def _apply_g(
        self,
        bag_summary: Tensor,
        summary_dim: int,
        num_classes: int,
        batch_size: int,
        n_bags: int,
        is_discrete: bool,
        min_per_class: int = 6,
        max_retries: int = 2,
        g_type="random",
    ) -> Tensor:
        min_per_class = max(1, min(min_per_class, n_bags // num_classes - 1))

        for _attempt in range(max_retries):
            if g_type == "random":
                if is_discrete:
                    g_type = str(np.random.choice(["lookup", "mlp", "tree", "max", "min"]))
                else:
                    g_type = str(np.random.choice(["mlp", "tree", "max", "min"]))

            if g_type == "lookup":
                y = self._g_lookup_table(
                    bag_summary=bag_summary,
                    summary_dim=summary_dim,
                    num_classes=num_classes,
                    batch_size=batch_size,
                    n_bags=n_bags,
                )
            elif g_type in ("mlp", "tree"):
                if self.use_reg_2_cls:
                    cont_y = self._run_scm(
                        X_flat=bag_summary,
                        num_features=summary_dim,
                        num_outputs=1,
                        scm_type=g_type,
                        scm_config=self.g_scm_configs[g_type],
                    ).squeeze(-1)
                    cont_y = standard_scaling(cont_y.unsqueeze(-1)).squeeze(-1)
                    mode = random.choice(["rank", "value"])
                    y = MulticlassAssigner(num_classes, mode=mode, ordered_prob=0.0)(cont_y)
                    y = y.long().reshape(batch_size, n_bags)
                else:
                    bag_logits = self._run_scm(
                        X_flat=bag_summary,
                        num_features=summary_dim,
                        num_outputs=num_classes,
                        scm_type=g_type,
                        scm_config=self.g_scm_configs[g_type],
                    )
                    y = bag_logits.argmax(dim=-1).reshape(batch_size, n_bags)
            elif g_type in ("max", "min"):
                if summary_dim != num_classes:
                    W = torch.randn(summary_dim, num_classes, device=self.device) / (summary_dim**0.5)
                    projected = bag_summary @ W
                else:
                    projected = bag_summary
                if g_type == "max":
                    y = projected.argmax(dim=-1).reshape(batch_size, n_bags)
                else:
                    y = projected.argmin(dim=-1).reshape(batch_size, n_bags)
            else:
                raise ValueError(f"Unknown g_type: {g_type!r}")

            if self.use_reg_2_cls:
                y = permute_classes(y.view(-1)).view(batch_size, n_bags)

            if all(self._check_min_counts(y[i], num_classes, min_per_class) for i in range(batch_size)):
                return y

        logger.warning(
            "_apply_g: exhausted %d retries without balanced bag labels "
            "(n_bags=%d, num_classes=%d, min_per_class=%d).",
            max_retries,
            n_bags,
            num_classes,
            min_per_class,
        )
        return y
