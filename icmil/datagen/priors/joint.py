"""Joint MIL prior generator: y = f_S(x_1, ..., x_N) over the flattened bag."""

from __future__ import annotations

import logging
import random

import torch
from tabicl.prior.reg2cls import (
    MulticlassAssigner,
    permute_classes,
    standard_scaling,
)
from torch import Tensor

from icmil.datagen.priors.sampling import SCMConfig

from icmil.datagen.priors.common import StrategySpec, _sample_strategy
from icmil.datagen.priors.hierarchical import SimplifiedHierarchicalMILPriorGenerator

logger = logging.getLogger(__name__)


class JointMILPriorGenerator(SimplifiedHierarchicalMILPriorGenerator):
    """Joint MIL prior: a single SCM over the flattened bag.

    A single SCM consumes a (bag_size * n_features)-dim vector per bag and
    emits the bag-level label directly — there is no f / aggregation / g
    decomposition. The whole bag is generated jointly from one data-generating
    process, so inter-instance correlations within a bag are expressible
    (unlike the factorized family, where instances are conditionally
    independent given the label).

    Permutation note:
        The joint SCM treats the bag as a fixed-order vector of length
        bag_size * n_features, so the generated function is in general NOT
        permutation-invariant in the instance index. Instances are shuffled
        within each bag *after* generation (parent's `_shuffle_instances=True`),
        which forces the learner to be permutation-invariant by exposing it to
        many orderings of the same bag mapped to one fixed label. The labels
        are therefore effectively the order-averaged target of f_S — this is
        intentional and matches the standard joint-prior design.

    Parameters
    ----------
    f_scm_configs : dict[str, SCMConfig]
        SCM configs keyed by scm_type. Must include all types reachable from
        `f_type` (e.g. {"mlp": ..., "tree": ...} when f_type="random").
    max_features : int
        Maximum features per instance for output padding.
    device : str
        Computation device.
    f_type : StrategySpec
        Which SCM family to use. "mlp" / "tree" / "random" / list / dict
        (passed through `_sample_strategy`).
    use_reg_2_cls : bool
        If True, apply tabicl's Reg2Cls feature pipeline to X and bin a
        single continuous SCM output into classes (mirrors the hierarchical
        generator's reg2cls path).
    """

    # Probability of rejecting a generated dataset whose bag labels are all identical
    # and resampling the per-dataset pipeline. The loop terminates almost surely as
    # long as DEGENERATE_REJECT_PROB < 1.
    DEGENERATE_REJECT_PROB: float = 0.9

    def __init__(
        self,
        f_scm_configs: dict[str, SCMConfig],
        max_features: int = 100,
        device: str = "cpu",
        f_type: StrategySpec = "random",
        use_reg_2_cls: bool = False,
    ):
        super().__init__(max_features=max_features, device=device)
        self.f_scm_configs = f_scm_configs
        self.f_type = f_type
        self.use_reg_2_cls = use_reg_2_cls
        self._validate_scm_configs()

    def _validate_scm_configs(self) -> None:
        """Check that required SCM configs are present for the chosen f_type."""
        scm_types = {"mlp", "tree"}
        if isinstance(self.f_type, str):
            if self.f_type == "random":
                required = {"mlp", "tree"}
            elif self.f_type in scm_types:
                required = {self.f_type}
            else:
                raise ValueError(f"Unknown f_type: {self.f_type!r}. Use 'mlp', 'tree', or 'random'.")
        else:
            possible = set(self.f_type) if isinstance(self.f_type, list) else set(self.f_type.keys())
            unknown = possible - scm_types
            if unknown:
                raise ValueError(f"Unknown f_type entries: {unknown}. Use 'mlp' or 'tree'.")
            required = possible & scm_types

        missing = required - set(self.f_scm_configs)
        if missing:
            raise ValueError(f"f_type={self.f_type!r} requires f_scm_configs keys {required}, but missing: {missing}")

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
        """Generate features and bag labels from a joint SCM, one dataset at a time."""
        bag_dim = bag_size * n_features
        min_per_class_clamped = max(1, min(min_per_class, n_bags // max(num_classes, 1) - 1))

        X_parts: list[Tensor] = []
        y_parts: list[Tensor] = []
        for _ in range(batch_size):
            f_type = _sample_strategy(self.f_type)
            num_outputs = 1 if self.use_reg_2_cls else num_classes

            X_flat: Tensor | None = None
            y_i: Tensor | None = None
            while True:
                X_flat, y_cont = self._joint_sample(
                    n_bags=n_bags,
                    bag_dim=bag_dim,
                    num_outputs=num_outputs,
                    f_type=f_type,
                )
                y_i = self._labels_from_outputs(y_cont, num_classes=num_classes)
                if not (self._is_degenerate(y_i) and random.random() < self.DEGENERATE_REJECT_PROB):
                    break

            if num_classes > 1 and not self._check_min_counts(y_i, num_classes, min_per_class_clamped):
                logger.warning(
                    "JointMILPriorGenerator: bag labels are unbalanced "
                    "(n_bags=%d, num_classes=%d, min_per_class=%d, f_type=%s).",
                    n_bags,
                    num_classes,
                    min_per_class_clamped,
                    f_type,
                )

            if self.use_reg_2_cls:
                X_flat = self._reg2cls_features_per_instance(X_flat, bag_size, n_features)

            X_i = X_flat.reshape(n_bags, bag_size, n_features)
            X_parts.append(X_i)
            y_parts.append(y_i.reshape(1, n_bags))

        X = torch.stack(X_parts, dim=0)
        y = torch.cat(y_parts, dim=0)
        return X, y

    def _joint_sample(
        self,
        n_bags: int,
        bag_dim: int,
        num_outputs: int,
        f_type: str,
    ) -> tuple[Tensor, Tensor]:
        """Run one SCM on the flattened bag, returning (X_flat, y_cont).

        X_flat shape: (n_bags, bag_dim).  y_cont shape: (n_bags, num_outputs).
        """
        if f_type == "mlp":
            X_flat, y_cont = self._run_causal_scm(
                seq_len=n_bags,
                num_features=bag_dim,
                num_outputs=num_outputs,
                scm_config=self.f_scm_configs["mlp"],
            )
        elif f_type == "tree":
            X_flat = torch.randn(n_bags, bag_dim, device=self.device)
            y_cont = self._run_scm(
                X_flat=X_flat,
                num_features=bag_dim,
                num_outputs=num_outputs,
                scm_type="tree",
                scm_config=self.f_scm_configs["tree"],
            )
        else:
            raise ValueError(f"Unknown f_type: {f_type!r}")

        if y_cont.dim() == 1:
            y_cont = y_cont.unsqueeze(-1)
        return X_flat, y_cont

    def _labels_from_outputs(self, y_cont: Tensor, num_classes: int) -> Tensor:
        """Convert SCM continuous outputs to discrete bag labels of shape (n_bags,)."""
        if self.use_reg_2_cls:
            scalar = y_cont.squeeze(-1) if y_cont.ndim == 2 else y_cont
            scalar = standard_scaling(scalar.unsqueeze(-1)).squeeze(-1)
            mode = random.choice(["rank", "value"])
            y = MulticlassAssigner(num_classes, mode=mode, ordered_prob=0.0)(scalar)
            y = y.long().view(-1)
            y = permute_classes(y)
            return y
        return y_cont.argmax(dim=-1).view(-1)

    def _reg2cls_features_per_instance(
        self,
        X_flat: Tensor,
        bag_size: int,
        n_features: int,
    ) -> Tensor:
        """Apply the reg2cls feature pipeline per-instance, then re-flatten.

        Reshapes (n_bags, bag_size * n_features) -> (n_bags * bag_size, n_features)
        so each instance is a row, applies the standard reg2cls processing
        (categorical-ization, outlier removal, standard scaling, column
        permutation — same column permutation across all instances), then
        reshapes back.
        """
        n_bags = X_flat.shape[0]
        per_instance = X_flat.reshape(n_bags * bag_size, n_features)
        processed = self._reg2cls_feature_process(per_instance)
        return processed.reshape(n_bags, bag_size * n_features)
