"""ICMIL view ensemble: one in-context model, averaged over random views of a dataset.

ICMIL learns in context, so an ensemble member costs one forward pass and needs
no training. Each member sees its own view of the dataset, and the members' test
probabilities are averaged. A view is

* a stratified random subsample of the context bags (about 90 % of them), and
* a feature transform fitted on the instances of those bags and applied to
  context and test bags alike. By default members alternate between a
  Yeo-Johnson power transform and a random rotation of the standardised features.

Features are zero-padded or truncated to the model's ``in_features`` first, and
the transforms act on all of those columns. A rotation therefore also spreads
narrow inputs over the model's otherwise empty feature columns.
"""

from __future__ import annotations

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import PowerTransformer
from torch import nn

# Instances used to fit the power transform, to bound its cost on large datasets.
_MAX_FIT_ROWS = 100_000


class EnsembleTransform:
    """A feature transform fitted on the context instances of one ensemble member.

    ``name`` selects the transform:

    * ``"power"``: Yeo-Johnson power transform with standardisation;
    * ``"rotation"``: standardisation followed by a random orthogonal rotation;
    * ``"none"``: the features are left as they are.

    ``context`` holds the instances it is fitted on, shape ``(n, F)``. Calling the
    transform applies it to a tensor of any shape ``(..., F)``.
    """

    def __init__(self, name: str, context: torch.Tensor, generator: torch.Generator) -> None:
        self.name = name
        if name == "power":
            if len(context) > _MAX_FIT_ROWS:
                rows = torch.randperm(len(context), generator=generator, device=context.device)[:_MAX_FIT_ROWS]
                context = context[rows]
            context = context.double().cpu().numpy()
            self.power = PowerTransformer().fit(context)
            fitted = self.power.transform(context)
            self.low, self.high = fitted.min(axis=0), fitted.max(axis=0)
        elif name == "rotation":
            self.mean = context.mean(dim=0)
            self.std = context.std(dim=0).clamp_min(1e-6)  # padded columns have zero spread
            n_features = context.shape[-1]
            gaussian = torch.randn(n_features, n_features, generator=generator, device=context.device)
            self.rotation, _ = torch.linalg.qr(gaussian)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        if self.name == "power":
            # Values far outside the context's range can overflow to infinity, which scikit-learn
            # rejects by default. Let them through and replace them by the context's extremes.
            with np.errstate(over="ignore"), sklearn.config_context(assume_finite=True):
                out = self.power.transform(X.reshape(-1, X.shape[-1]).double().cpu().numpy())
            out = np.where(np.isposinf(out), self.high, np.where(np.isneginf(out), self.low, out))
            return torch.as_tensor(out, dtype=X.dtype, device=X.device).reshape(X.shape)
        if self.name == "rotation":
            return ((X - self.mean) / self.std) @ self.rotation
        return X


class ICMILEnsemble(nn.Module):
    """Average an in-context MIL model over random views of each dataset.

    ``model`` maps ``X_train (B, n_train, n_instances, in_features)``,
    ``y_train (B, n_train)`` and ``X_test (B, n_test, n_instances, in_features)``
    to class logits ``(B, n_test, n_classes)`` and exposes ``in_features``. The
    ensemble takes the same inputs and returns log-probabilities.

    Args:
        model: The in-context MIL model to ensemble.
        n_members: Number of views.
        transforms: Feature transforms the members cycle through: ``"power"``,
            ``"rotation"`` or ``"none"``. Repeat a name to give it a larger
            share, e.g. ``("power", "rotation", "rotation")``.
        bag_fraction_mean: Mean share of the context bags a member sees.
        bag_fraction_std: Standard deviation of that share across members.
        seed: Seed of the views. Successive calls draw fresh views.
    """

    def __init__(
        self,
        model: nn.Module,
        n_members: int = 16,
        transforms: tuple[str, ...] = ("power", "rotation"),
        bag_fraction_mean: float = 0.9,
        bag_fraction_std: float = 0.01,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.n_members = n_members
        self.transforms = transforms
        # The Beta(a, b) distribution with the requested mean and standard deviation.
        concentration = bag_fraction_mean * (1 - bag_fraction_mean) / bag_fraction_std**2 - 1
        self.beta = (bag_fraction_mean * concentration, (1 - bag_fraction_mean) * concentration)
        self.seed = seed
        self.n_calls = 0

    def _predict(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
        """Member-averaged class probabilities ``(n_test, n_classes)`` for one dataset."""
        seed = self.seed * 1_000_003 + self.n_calls * 7919 + 99  # a distinct seed for every call
        self.n_calls += 1
        rng = np.random.default_rng(seed)
        generator = torch.Generator(device=X_train.device).manual_seed(seed)

        width = self.model.in_features
        X_train = F.pad(X_train, (0, max(0, width - X_train.shape[-1])))[..., :width]
        X_test = F.pad(X_test, (0, max(0, width - X_test.shape[-1])))[..., :width]
        labels = y_train.cpu().numpy()

        probabilities = 0
        for member in range(self.n_members):
            fraction = rng.beta(*self.beta)
            splitter = StratifiedShuffleSplit(1, train_size=fraction, random_state=int(rng.integers(2**31)))
            keep, _ = next(splitter.split(labels, labels))
            keep = torch.as_tensor(np.sort(keep), device=X_train.device)
            X_context, y_context = X_train[keep], y_train[keep]

            name = self.transforms[member % len(self.transforms)]
            transform = EnsembleTransform(name, X_context.reshape(-1, width), generator)
            logits = self.model(transform(X_context)[None], y_context[None], transform(X_test)[None])[0]
            probabilities = probabilities + torch.softmax(logits, dim=-1)
        return probabilities / self.n_members

    @torch.no_grad()
    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
        """Log of the member-averaged class probabilities, ``(B, n_test, n_classes)``."""
        return torch.stack([self._predict(*dataset) for dataset in zip(X_train, y_train, X_test, strict=True)]).log()
