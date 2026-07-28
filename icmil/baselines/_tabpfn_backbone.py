"""Frozen TabPFN v2 backbone used by the TabPFN-based MIL baselines.

Extracted from the training code (only the frozen in-context classifier is
needed for the baselines; the trainable aggregator variants are omitted).
"""

import torch
from tabpfn.model.loading import load_model_criterion_config
from torch import nn


class FrozenTabPFNBackbone(nn.Module):
    """Frozen TabPFN v2 wrapper that maintains gradient flow to inputs.

    Loads the TabPFN v2 classifier, freezes all weights, and wraps the forward
    pass to accept ``(x_train, y_train, x_test)``.
    """

    def __init__(
        self,
        max_classes: int = 4,
        model_path: str | None = None,
        features_per_group: int = 2,
    ):
        super().__init__()
        self.max_classes = max_classes

        model, _criterion, _config = load_model_criterion_config(
            model_path=model_path,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which="classifier",
            version="v2",
            download=True,
        )
        model.features_per_group = features_per_group

        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        self.tabpfn = model

    def train(self, mode: bool = True) -> "FrozenTabPFNBackbone":
        """Keep TabPFN always in eval mode."""
        return super().train(False)

    def forward(self, x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor) -> torch.Tensor:
        """Run TabPFN in-context learning; returns test logits (n_test, max_classes)."""
        x_all = torch.cat([x_train, x_test], dim=0)
        x_seq = x_all.unsqueeze(1)
        y_seq = y_train.float().unsqueeze(-1).unsqueeze(-1)
        logits = self.tabpfn(x_seq, y_seq, only_return_standard_out=True)
        return logits.squeeze(1)[:, : self.max_classes]
