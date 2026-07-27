"""Dataset utilities for MLflow dataset tracking.

This module provides utilities for registering datasets with MLflow's
dataset tracking feature, supporting PyTorch, NumPy, and Pandas datasets.
"""

import hashlib
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

if TYPE_CHECKING:
    import pandas as pd


def compute_dataset_digest(
    dataset: Union[Dataset, np.ndarray, "pd.DataFrame"],
    num_samples: int = 100,
    seed: int = 42,
) -> str:
    """Compute a hash digest for a dataset.

    For large datasets, samples a subset for efficiency.

    Args:
        dataset: PyTorch Dataset, NumPy array, or Pandas DataFrame
        num_samples: Number of samples to use for hashing
        seed: Random seed for reproducible sampling

    Returns:
        SHA256 hex digest string
    """
    hasher = hashlib.sha256()

    if isinstance(dataset, np.ndarray):
        # Sample from numpy array
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        sample = dataset[sorted(indices)]
        hasher.update(sample.tobytes())

    elif hasattr(dataset, "to_numpy"):  # Pandas DataFrame
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        sample = dataset.iloc[sorted(indices)].to_numpy()
        hasher.update(sample.tobytes())

    elif isinstance(dataset, (Dataset, Subset)):
        # PyTorch dataset - sample and hash
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        for idx in sorted(indices):
            item = dataset[idx]
            if isinstance(item, tuple):
                for elem in item:
                    if isinstance(elem, torch.Tensor):
                        hasher.update(elem.numpy().tobytes())
                    elif isinstance(elem, np.ndarray):
                        hasher.update(elem.tobytes())
                    else:
                        hasher.update(str(elem).encode())
            elif isinstance(item, torch.Tensor):
                hasher.update(item.numpy().tobytes())
            else:
                hasher.update(str(item).encode())
    else:
        # Fallback: hash string representation
        hasher.update(str(dataset)[:1000].encode())

    return hasher.hexdigest()[:16]


def get_dataset_info(
    dataset: Union[Dataset, np.ndarray, "pd.DataFrame"],
    name: Optional[str] = None,
) -> dict:
    """Extract metadata from a dataset.

    Args:
        dataset: PyTorch Dataset, NumPy array, or Pandas DataFrame
        name: Optional name for the dataset

    Returns:
        Dictionary with dataset metadata
    """
    info = {
        "name": name or "unknown",
        "type": type(dataset).__name__,
        "size": len(dataset) if hasattr(dataset, "__len__") else "unknown",
    }

    if isinstance(dataset, np.ndarray):
        info["shape"] = str(dataset.shape)
        info["dtype"] = str(dataset.dtype)

    elif hasattr(dataset, "columns"):  # Pandas DataFrame
        info["columns"] = list(dataset.columns)
        info["shape"] = str(dataset.shape)

    elif isinstance(dataset, Dataset):
        # Try to get sample shape
        try:
            sample = dataset[0]
            if isinstance(sample, tuple) and len(sample) >= 1:
                if isinstance(sample[0], torch.Tensor):
                    info["sample_shape"] = str(tuple(sample[0].shape))
                elif isinstance(sample[0], np.ndarray):
                    info["sample_shape"] = str(sample[0].shape)
            elif isinstance(sample, torch.Tensor):
                info["sample_shape"] = str(tuple(sample.shape))
        except Exception:
            pass

    return info


def create_mlflow_dataset(
    dataset: Union[Dataset, np.ndarray, "pd.DataFrame"],
    name: Optional[str] = None,
    source: Optional[str] = None,
    context: str = "training",
):
    """Create an MLflow Dataset object from various dataset types.

    Args:
        dataset: PyTorch Dataset, NumPy array, or Pandas DataFrame
        name: Name for the dataset
        source: Source path or description
        context: Context string (e.g., "training", "validation", "test")

    Returns:
        MLflow Dataset object suitable for logging with mlflow.log_input()
    """
    import mlflow.data

    digest = compute_dataset_digest(dataset)

    # NumPy array
    if isinstance(dataset, np.ndarray):
        return mlflow.data.from_numpy(
            features=dataset,
            source=source or "numpy_array",
            name=name,
            digest=digest,
        )

    # Pandas DataFrame
    if hasattr(dataset, "to_numpy") and hasattr(dataset, "columns"):
        import pandas as pd

        if isinstance(dataset, pd.DataFrame):
            return mlflow.data.from_pandas(
                df=dataset,
                source=source or "pandas_dataframe",
                name=name,
                digest=digest,
            )

    # PyTorch Dataset - extract samples and create numpy dataset
    if isinstance(dataset, (Dataset, Subset)):
        # Sample a subset for the MLflow dataset representation
        # Full dataset may be too large
        max_samples = min(1000, len(dataset))
        features_list = []
        targets_list = []

        for i in range(max_samples):
            item = dataset[i]
            if isinstance(item, tuple) and len(item) >= 2:
                feat, target = item[0], item[1]
                if isinstance(feat, torch.Tensor):
                    feat = feat.numpy()
                if isinstance(target, torch.Tensor):
                    target = target.numpy()
                elif isinstance(target, int):
                    target = np.array(target)
                features_list.append(feat)
                targets_list.append(target)
            elif isinstance(item, torch.Tensor):
                features_list.append(item.numpy())

        if features_list:
            features = np.stack(features_list)
            targets = np.stack(targets_list) if targets_list else None

            return mlflow.data.from_numpy(
                features=features,
                targets=targets,
                source=source or f"torch_dataset:{type(dataset).__name__}",
                name=name or type(dataset).__name__,
                digest=digest,
            )

    # Fallback: create a meta dataset with just info
    # This handles cases where we can't easily convert to numpy
    info = get_dataset_info(dataset, name)
    return mlflow.data.meta_dataset.MetaDataset(
        source=mlflow.data.dataset_source.DatasetSource._resolve(
            source or f"dataset:{info['type']}"
        ),
        name=name or info["type"],
        digest=digest,
    )


def log_dataset_to_mlflow(
    dataset: Union[Dataset, np.ndarray, "pd.DataFrame"],
    name: str,
    context: str = "training",
    source: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    """Log a dataset to MLflow tracking.

    Args:
        dataset: PyTorch Dataset, NumPy array, or Pandas DataFrame
        name: Name for the dataset
        context: Context string (e.g., "training", "validation", "test")
        source: Source path or description
        run_id: Optional MLflow run ID (uses active run if not specified)
    """
    import mlflow

    mlflow_dataset = create_mlflow_dataset(
        dataset=dataset,
        name=name,
        source=source,
        context=context,
    )

    mlflow.log_input(mlflow_dataset, context=context)
