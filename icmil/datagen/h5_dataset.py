"""Readers for the synthetic prior H5 files written by :mod:`icmil.datagen.generate`.

``BaggedPriorH5Dataset`` streams one prior file; ``MultiPriorH5Dataset`` mixes several
according to per-arm weights. Both hand out whole pre-batched groups rather than 
individual samples, and both preserve batch order when the file carries a curriculum.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def _passthrough_collate(x):
    """Pass through data as-is (dataset already returns batches)."""
    return x


class BaggedPriorH5Dataset(Dataset):
    """Simple Map-style dataset for loading pre-generated HDF5 data.

    Each index maps to one batch group ``batch_{idx}``.

    Args:
        filename: Path to the HDF5 file.
        shuffle_seed: Seed for deterministic batch shuffling.

    Example:
        dataset = BaggedPriorH5Dataset("data.h5")

        for epoch in range(num_epochs):
            loader = dataset.get_loader_for_epoch(epoch, steps_per_epoch, num_workers=4)
            for step, batch in enumerate(loader):
                X, y, num_features, n_bags, bag_sizes, n_train_bags = batch
                # ... training step ...
    """

    def __init__(self, filename: str, shuffle_seed: int = 42) -> None:
        self.filename = filename
        self._file_handle: h5py.File | None = None

        # Read metadata once at init
        with h5py.File(filename, "r") as f:
            self.total_samples = int(f["total_samples"][0])
            self.batch_size = int(f["batch_size"][0])
            self.max_n_bags = int(f["max_n_bags"][0])
            self.max_bag_size = int(f["max_bag_size"][0])
            self.max_features = int(f["max_features"][0])
            self.max_classes = int(f["max_classes"][0])
            self.num_batches = int(f["num_batches"][0])

            # Prior name metadata
            configs = json.loads(f["prior_configs_json"][()])
            self.prior_names = [c.get("name", "unknown") for c in configs]

            # Curriculum-enabled H5 files store batches in a deliberate order
            # (small bag_size first, increasing over training).  Shuffling
            # would destroy that ordering, so we skip it.
            self.has_curriculum = "curriculum_schedule" in f

        if self.has_curriculum:
            self.batch_order = list(range(self.num_batches))
        else:
            rng = np.random.default_rng(shuffle_seed)
            self.batch_order = list(range(self.num_batches))
            rng.shuffle(self.batch_order)

    def __len__(self) -> int:
        return self.num_batches

    def _ensure_file_open(self) -> h5py.File:
        """Lazily open file handle (once per worker process)."""
        if self._file_handle is None:
            self._file_handle = h5py.File(self.filename, "r", swmr=True)
        return self._file_handle

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Load one full batch by global index.

        ``idx`` cycles through ``batch_order`` so repeated epochs of training
        continue past ``num_batches``.
        """
        batch_idx = self.batch_order[idx % self.num_batches]

        f = self._ensure_file_open()
        batch = f[f"batch_{batch_idx}"]

        X = torch.from_numpy(batch["X"][:].astype(np.float32))
        y = torch.from_numpy(batch["y"][:].astype(np.int64))
        num_features = torch.from_numpy(batch["num_features"][:].astype(np.int64))
        n_bags = torch.from_numpy(batch["n_bags"][:].astype(np.int64))
        bag_sizes = torch.from_numpy(batch["bag_sizes"][:].astype(np.int64))
        n_train_bags = torch.from_numpy(batch["n_train_bags"][:].astype(np.int64))

        return X, y, num_features, n_bags, bag_sizes, n_train_bags

    def get_loader_for_epoch(
        self,
        epoch: int,
        steps_per_epoch: int,
        num_workers: int = 0,
        prefetch_factor: int | None = 2,
        pin_memory: bool = True,
    ) -> DataLoader:
        """Create a DataLoader for a specific epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            steps_per_epoch: Number of batches to load this epoch.
            num_workers: Number of worker processes.
            prefetch_factor: Batches to prefetch per worker.
            pin_memory: Pin memory for faster GPU transfer.

        Returns:
            DataLoader that yields (X, y, num_features, n_bags, bag_sizes, n_train_bags) tuples.
        """
        start_idx = epoch * steps_per_epoch
        indices = list(range(start_idx, start_idx + steps_per_epoch))

        if num_workers == 0:
            prefetch_factor = None

        return DataLoader(
            self,
            batch_size=None,  # Dataset already returns batches
            sampler=indices,  # Sequential sampler from list
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            collate_fn=_passthrough_collate,
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )

    def __repr__(self) -> str:
        lines = [
            "BaggedPriorH5Dataset(",
            f"  filename: {self.filename}",
            f"  num_batches: {self.num_batches}",
            f"  batch_size: {self.batch_size}",
            f"  max_n_bags: {self.max_n_bags}",
            f"  max_bag_size: {self.max_bag_size}",
            f"  max_features: {self.max_features}",
            f"  max_classes: {self.max_classes}",
        ]
        lines.append(f"  prior: {', '.join(self.prior_names)}")
        lines.append(")")
        return "\n".join(lines)


class MultiPriorH5Dataset(Dataset):
    """Training dataset that mixes multiple per-prior H5 files at training time.

    Each epoch, a prior is sampled based on weights. The selected prior's
    BaggedPriorH5Dataset serves that epoch's data.

    Args:
        prior_dir: Directory containing per-prior .h5 files.
        weights: Optional dict mapping file stem -> weight. If None, equal weights.
        shuffle_seed: Seed for deterministic batch shuffling within each prior.
        mix_seed: Seed for deterministic prior selection across epochs.
    """

    def __init__(
        self,
        prior_dir: str,
        weights: dict[str, float] | None = None,
        shuffle_seed: int = 42,
        mix_seed: int = 0,
    ) -> None:
        prior_path = Path(prior_dir)
        if not prior_path.is_dir():
            raise FileNotFoundError(f"Prior directory does not exist: {prior_dir}")

        h5_files = sorted(prior_path.glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found in {prior_dir}")

        self.prior_names: list[str] = [f.stem for f in h5_files]
        self.datasets: list[BaggedPriorH5Dataset] = [
            BaggedPriorH5Dataset(str(f), shuffle_seed=shuffle_seed) for f in h5_files
        ]
        self.mix_seed = mix_seed

        # Validate metadata consistency across all files
        ref = self.datasets[0]
        for ds in self.datasets[1:]:
            if ds.max_features != ref.max_features:
                raise ValueError(
                    f"max_features mismatch: {ref.filename} has {ref.max_features}, "
                    f"{ds.filename} has {ds.max_features}"
                )
            if ds.max_classes != ref.max_classes:
                raise ValueError(
                    f"max_classes mismatch: {ref.filename} has {ref.max_classes}, "
                    f"{ds.filename} has {ds.max_classes}"
                )
            if ds.batch_size != ref.batch_size:
                raise ValueError(
                    f"batch_size mismatch: {ref.filename} has {ref.batch_size}, {ds.filename} has {ds.batch_size}"
                )

        # Compute normalized weights
        if weights is not None:
            missing = set(weights.keys()) - set(self.prior_names)
            if missing:
                raise ValueError(f"Weight keys not matching any .h5 file: {missing}")
            raw = np.array([weights.get(name, 0.0) for name in self.prior_names])
        else:
            raw = np.ones(len(self.datasets))
        total = raw.sum()
        if total <= 0:
            raise ValueError("Weights must sum to a positive value")
        self.weights: np.ndarray = raw / total

        # Per-prior epoch counters (each prior tracks its own epoch independently)
        self._prior_epoch_counters: list[int] = [0] * len(self.datasets)

        # Expose unified metadata from first dataset
        self.max_features: int = ref.max_features
        self.max_classes: int = ref.max_classes
        self.batch_size: int = ref.batch_size

    def __len__(self) -> int:
        return sum(len(ds) for ds in self.datasets)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Map global index across concatenated sub-datasets."""
        for ds in self.datasets:
            if idx < len(ds):
                return ds[idx]
            idx -= len(ds)
        raise IndexError("Index out of range")

    def get_loader_for_epoch(
        self,
        epoch: int,
        steps_per_epoch: int,
        num_workers: int = 0,
        prefetch_factor: int | None = 2,
        pin_memory: bool = True,
    ) -> DataLoader:
        """Create a DataLoader for a specific epoch, sampling a prior by weight.

        The prior selection is deterministic given mix_seed and epoch.
        Each prior maintains its own epoch counter so it consumes its
        batches sequentially across the global epochs where it is selected.
        """
        # Deterministic prior selection for this epoch
        epoch_rng = np.random.default_rng(self.mix_seed + epoch)
        prior_idx = int(epoch_rng.choice(len(self.datasets), p=self.weights))

        # Use this prior's internal epoch counter
        prior_epoch = self._prior_epoch_counters[prior_idx]
        self._prior_epoch_counters[prior_idx] += 1

        return self.datasets[prior_idx].get_loader_for_epoch(
            epoch=prior_epoch,
            steps_per_epoch=steps_per_epoch,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )

    def __repr__(self) -> str:
        lines = [
            "MultiPriorH5Dataset(",
            f"  num_priors: {len(self.datasets)}",
            f"  max_features: {self.max_features}",
            f"  max_classes: {self.max_classes}",
            f"  batch_size: {self.batch_size}",
            f"  mix_seed: {self.mix_seed}",
            "  priors:",
        ]
        for name, weight, ds in zip(self.prior_names, self.weights, self.datasets, strict=False):
            lines.append(f"    - {name} (weight={weight:.3f}, batches={ds.num_batches})")
        lines.append(")")
        return "\n".join(lines)
