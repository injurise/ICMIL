"""HDF5 writer for the synthetic prior corpus.

Defines the on-disk format that :mod:`icmil.datagen.h5_dataset` reads.
"""

import json
from typing import Self

import h5py
import numpy as np
import torch
from tqdm import tqdm


def _to_native(obj):
    """Recursively convert config objects to JSON-serialisable Python types."""
    if isinstance(obj, dict) or hasattr(obj, "keys"):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple) or (hasattr(obj, "__iter__") and not isinstance(obj, str | bytes)):
        return [_to_native(v) for v in obj]
    return obj


class BaggedPriorH5Writer:
    """Context manager for writing bagged prior data to HDF5.

    Each batch is stored as a separate HDF5 group ``batch_{idx}`` with its
    exact ``(n_bags, bag_size)`` dimensions.
    """

    def __init__(
        self,
        save_path: str,
        batch_size: int,
        max_n_bags: int,
        max_bag_size: int,
        max_features: int,
        max_classes: int,
    ) -> None:
        self.save_path = save_path
        self.batch_size = batch_size
        self.max_n_bags = max_n_bags
        self.max_bag_size = max_bag_size
        self.max_features = max_features
        self.max_classes = max_classes
        self._file = None
        self._current_batch = None
        self._current_batch_datasets = {}
        self._batch_count = 0
        self._total_samples = 0

    def __enter__(self) -> Self:
        self._file = h5py.File(self.save_path, "w")
        self._write_dimension_metadata()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
        return False

    def _write_dimension_metadata(self) -> None:
        f = self._file
        f.create_dataset("max_classes", data=np.array([self.max_classes]), dtype="i4")
        f.create_dataset("max_n_bags", data=np.array([self.max_n_bags]), dtype="i4")
        f.create_dataset("max_bag_size", data=np.array([self.max_bag_size]), dtype="i4")
        f.create_dataset("max_features", data=np.array([self.max_features]), dtype="i4")
        f.create_dataset("batch_size", data=np.array([self.batch_size]), dtype="i4")

    def write_metadata(self, metadata: dict) -> None:
        """Write arbitrary metadata to the HDF5 file."""
        f = self._file
        for key, value in metadata.items():
            if key in f:
                continue
            if isinstance(value, int | np.integer):
                f.create_dataset(key, data=np.array([value]), dtype="i4")
            elif isinstance(value, float | np.floating):
                f.create_dataset(key, data=np.array([value]), dtype="f4")
            elif isinstance(value, str):
                f.create_dataset(key, data=value, dtype=h5py.string_dtype())
            elif isinstance(value, np.ndarray):
                f.create_dataset(key, data=value)
            elif isinstance(value, list | tuple | dict) or (hasattr(value, "__iter__") and not isinstance(value, str)):
                f.create_dataset(key, data=json.dumps(_to_native(value)), dtype=h5py.string_dtype())
            else:
                raise TypeError(f"Unsupported metadata type for {key}: {type(value)}")

    def start_batch(self, batch_idx: int, n_bags: int, bag_size: int) -> None:
        """Start a new batch group with specific dimensions."""
        self._current_batch = self._file.create_group(f"batch_{batch_idx}")
        self._current_batch.attrs["n_bags"] = n_bags
        self._current_batch.attrs["bag_size"] = bag_size

        bs = self.batch_size
        self._current_batch_datasets = {
            "X": self._current_batch.create_dataset(
                "X",
                shape=(0, n_bags, bag_size, self.max_features),
                maxshape=(None, n_bags, bag_size, self.max_features),
                chunks=(bs, n_bags, bag_size, self.max_features),
                compression="lzf",
                dtype="f4",
            ),
            "y": self._current_batch.create_dataset(
                "y",
                shape=(0, n_bags),
                maxshape=(None, n_bags),
                chunks=(bs, n_bags),
                dtype="i4",
            ),
        }
        for name in ["num_features", "n_bags", "bag_sizes", "n_train_bags"]:
            self._current_batch_datasets[name] = self._current_batch.create_dataset(
                name, shape=(0,), maxshape=(None,), chunks=(bs,), dtype="i4"
            )

        self._batch_count += 1

    def append_batch(
        self,
        X: "torch.Tensor",
        y: "torch.Tensor",
        num_features: "torch.Tensor",
        n_bags: "torch.Tensor",
        bag_sizes: "torch.Tensor",
        n_train_bags: "torch.Tensor",
    ) -> None:
        """Append data to the current batch group."""
        if self._current_batch is None:
            raise RuntimeError("Must call start_batch() before append_batch()")

        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        d_np = num_features.cpu().numpy()
        n_bags_np = n_bags.cpu().numpy()
        bag_sizes_np = bag_sizes.cpu().numpy()
        n_train_np = n_train_bags.cpu().numpy()

        batch_size = X_np.shape[0]
        ds = self._current_batch_datasets

        current_size = ds["X"].shape[0]
        new_size = current_size + batch_size

        ds["X"].resize(new_size, axis=0)
        ds["X"][current_size:new_size] = X_np

        ds["y"].resize(new_size, axis=0)
        ds["y"][current_size:new_size] = y_np

        ds["num_features"].resize(new_size, axis=0)
        ds["num_features"][current_size:new_size] = d_np

        ds["n_bags"].resize(new_size, axis=0)
        ds["n_bags"][current_size:new_size] = n_bags_np

        ds["bag_sizes"].resize(new_size, axis=0)
        ds["bag_sizes"][current_size:new_size] = bag_sizes_np

        ds["n_train_bags"].resize(new_size, axis=0)
        ds["n_train_bags"][current_size:new_size] = n_train_np

        self._total_samples += batch_size

    def append_batch_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_features: np.ndarray,
        n_bags: np.ndarray,
        bag_sizes: np.ndarray,
        n_train_bags: np.ndarray,
    ) -> None:
        """Append data (numpy arrays) to the current batch group.

        Same as :meth:`append_batch` but accepts pre-converted numpy arrays.

        Args:
            X: Features, shape ``(batch_size, n_bags, bag_size, max_features)``.
            y: Labels, shape ``(batch_size, n_bags)``.
            num_features: Active feature counts, shape ``(batch_size,)``.
            n_bags: Bag counts, shape ``(batch_size,)``.
            bag_sizes: Bag sizes, shape ``(batch_size,)``.
            n_train_bags: Train split positions, shape ``(batch_size,)``.
        """
        if self._current_batch is None:
            raise RuntimeError("Must call start_batch() before append_batch_numpy()")

        batch_size = X.shape[0]
        ds = self._current_batch_datasets

        current_size = ds["X"].shape[0]
        new_size = current_size + batch_size

        ds["X"].resize(new_size, axis=0)
        ds["X"][current_size:new_size] = X

        ds["y"].resize(new_size, axis=0)
        ds["y"][current_size:new_size] = y

        ds["num_features"].resize(new_size, axis=0)
        ds["num_features"][current_size:new_size] = num_features

        ds["n_bags"].resize(new_size, axis=0)
        ds["n_bags"][current_size:new_size] = n_bags

        ds["bag_sizes"].resize(new_size, axis=0)
        ds["bag_sizes"][current_size:new_size] = bag_sizes

        ds["n_train_bags"].resize(new_size, axis=0)
        ds["n_train_bags"][current_size:new_size] = n_train_bags

        self._total_samples += batch_size

    def finalize(self) -> int:
        """Write total_samples and batch count, return total sample count."""
        self._file.create_dataset("total_samples", data=np.array([self._total_samples]), dtype="i4")
        self._file.create_dataset("num_batches", data=np.array([self._batch_count]), dtype="i4")
        return self._total_samples


def dump_bagged_prior_to_h5(
    prior_dataset,  # a prior generator: needs .get_batch() and .sample_params()
    save_path: str,
    num_batches: int,
    max_n_bags: int,
    max_bag_size: int,
    max_features: int,
    max_classes: int,
) -> None:
    """Dump synthetic bagged prior data to HDF5 as a flat sequence of batches."""
    with BaggedPriorH5Writer(
        save_path, prior_dataset.batch_size, max_n_bags, max_bag_size, max_features, max_classes
    ) as writer:
        writer.write_metadata(
            {
                "prior_type": prior_dataset.prior_type,
                "bag_label_strategy": prior_dataset.bag_label_strategy,
                "prior_configs_json": [
                    {"name": f"{prior_dataset.prior_type}_{prior_dataset.bag_label_strategy}".strip("_")}
                ],
            }
        )

        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            batch_params = prior_dataset.sample_params()
            writer.start_batch(
                batch_idx,
                n_bags=batch_params.n_bags,
                bag_size=batch_params.bag_size,
            )
            X, y, d, n_bags, bag_sizes, n_train = prior_dataset.get_batch(params=batch_params)
            writer.append_batch(X, y, d, n_bags, bag_sizes, n_train)

        writer.finalize()
