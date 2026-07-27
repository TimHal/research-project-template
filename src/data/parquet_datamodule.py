"""Parquet Dataset DataModule.

This module provides a Lightning DataModule for loading datasets from
Parquet files, supporting both tabular data and image-path datasets.

Requires: pip install pyarrow

Supports two modes:
- tabular: numeric feature columns + label column → TensorDataset
- image_path: image file path column + label column → loads images on the fly
"""

import os
from typing import Callable, Optional

import lightning as L
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, TensorDataset


def _import_pandas():
    """Lazy import of pandas with clear error message."""
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise ImportError(
            "ParquetDataModule requires the 'pandas' and 'pyarrow' packages. "
            "Install them with: pip install pandas pyarrow"
        ) from None


class _ImagePathDataset(Dataset):
    """Dataset that loads images from file paths stored in a DataFrame."""

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        transform: Callable,
        base_dir: Optional[str] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.base_dir = base_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        if self.base_dir is not None:
            path = os.path.join(self.base_dir, path)

        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image)

        return image, self.labels[idx]


class ParquetDataModule(L.LightningDataModule):
    """DataModule for datasets stored in Parquet files.

    Supports two modes:

    **Tabular mode** (``mode="tabular"``):
    Loads numeric feature columns and a label column into tensors.
    Suitable for classification with MLP or tabular models.

    **Image path mode** (``mode="image_path"``):
    Reads image file paths from a column, loads and transforms images on the fly.
    Suitable for image classification with CNN/ViT models.

    Args:
        train_file: Path to training parquet file
        val_file: Path to validation parquet file (if None, split from train)
        test_file: Path to test parquet file (optional)
        label_column: Column containing class labels
        mode: Loading mode — 'tabular' or 'image_path'
        feature_columns: Columns to use as features (tabular mode; None = all except label)
        image_column: Column containing image file paths (image_path mode)
        image_base_dir: Base directory to prepend to image paths
        root_dir: Default base directory (used as dataset_name for logging)
        img_size: Target image size (height, width) — image_path mode only
        img_mode: Image mode, 'RGB' or 'L' — image_path mode only
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        train_transform: Optional custom transform for training (image_path mode)
        val_transform: Optional custom transform for validation/test (image_path mode)
        norm_mean: Mean values for normalization — image_path mode only
        norm_std: Standard deviation values for normalization — image_path mode only
        train_val_split: Tuple of (train_ratio, val_ratio)
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
        seed: Random seed for reproducibility
        augmentation: Augmentation mode ('none' or 'basic') — image_path mode only
    """

    def __init__(
        self,
        train_file: str = "./data/train.parquet",
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        label_column: str = "label",
        mode: str = "tabular",
        feature_columns: Optional[list[str]] = None,
        image_column: Optional[str] = None,
        image_base_dir: Optional[str] = None,
        root_dir: str = "./data",
        img_size: tuple[int, int] = (224, 224),
        img_mode: str = "RGB",
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        norm_mean: Optional[list[float]] = None,
        norm_std: Optional[list[float]] = None,
        train_val_split: tuple[float, float] = (0.9, 0.1),
        pin_memory: bool = True,
        persistent_workers: bool = True,
        seed: int = 42,
        augmentation: str = "none",
    ):
        super().__init__()

        assert mode in ["tabular", "image_path"], "mode must be 'tabular' or 'image_path'"
        if mode == "image_path":
            assert img_mode in ["RGB", "L"], "img_mode must be 'RGB' or 'L'"
            if image_column is None:
                raise ValueError("image_column is required when mode='image_path'")

        self.dataset_name = os.path.basename(os.path.dirname(os.path.abspath(train_file)))
        self.root_dir = root_dir
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.label_column = label_column
        self.mode = mode
        self.feature_columns = feature_columns
        self.image_column = image_column
        self.image_base_dir = image_base_dir
        self.img_size = img_size
        self.img_mode = img_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed
        self.augmentation = augmentation

        self.num_channels = 1 if img_mode == "L" else 3
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        if mode == "image_path":
            self.train_transform = train_transform or self._get_default_train_transform()
            self.val_transform = val_transform or self._get_default_val_transform()
        else:
            self.train_transform = None
            self.val_transform = None

        self._num_classes = None

    def _get_default_train_transform(self) -> transforms.Compose:
        """Get default training transforms."""
        transform_list = [transforms.Resize(self.img_size)]

        if self.img_mode == "L":
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        else:
            transform_list.append(transforms.Lambda(lambda x: x.convert("RGB")))

        if self.augmentation == "basic":
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
            ])

        transform_list.append(transforms.ToTensor())

        if self.norm_mean is not None and self.norm_std is not None:
            transform_list.append(
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
            )

        return transforms.Compose(transform_list)

    def _get_default_val_transform(self) -> transforms.Compose:
        """Get default validation/test transforms."""
        transform_list = [transforms.Resize(self.img_size)]

        if self.img_mode == "L":
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        else:
            transform_list.append(transforms.Lambda(lambda x: x.convert("RGB")))

        transform_list.append(transforms.ToTensor())

        if self.norm_mean is not None and self.norm_std is not None:
            transform_list.append(
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
            )

        return transforms.Compose(transform_list)

    def _build_tabular_dataset(self, df) -> TensorDataset:
        """Build a TensorDataset from a DataFrame for tabular mode."""
        labels = torch.tensor(df[self.label_column].values, dtype=torch.long)

        if self.feature_columns is not None:
            features = df[self.feature_columns].values
        else:
            features = df.drop(columns=[self.label_column]).select_dtypes(
                include=["number"]
            ).values

        features = torch.tensor(features, dtype=torch.float32)
        return TensorDataset(features, labels)

    def _build_image_dataset(self, df, transform: Callable) -> _ImagePathDataset:
        """Build an image dataset from a DataFrame for image_path mode."""
        # Encode string labels to integers if needed
        labels = df[self.label_column]
        if labels.dtype == object:
            if not hasattr(self, "_label_map"):
                unique_labels = sorted(labels.unique())
                self._label_map = {label: i for i, label in enumerate(unique_labels)}
            labels = labels.map(self._label_map).tolist()
        else:
            labels = labels.tolist()

        image_paths = df[self.image_column].tolist()
        return _ImagePathDataset(image_paths, labels, transform, self.image_base_dir)

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage."""
        pd = _import_pandas()

        if stage == "fit" or stage is None:
            train_df = pd.read_parquet(self.train_file)

            # Detect num_classes
            label_vals = train_df[self.label_column]
            if label_vals.dtype == object:
                self._num_classes = label_vals.nunique()
            else:
                self._num_classes = int(label_vals.max()) + 1

            if self.val_file is not None:
                val_df = pd.read_parquet(self.val_file)
                if self.mode == "tabular":
                    self.train_dataset = self._build_tabular_dataset(train_df)
                    self.val_dataset = self._build_tabular_dataset(val_df)
                else:
                    self.train_dataset = self._build_image_dataset(train_df, self.train_transform)
                    self.val_dataset = self._build_image_dataset(val_df, self.val_transform)
            else:
                # Split train into train/val
                if self.mode == "tabular":
                    full_dataset = self._build_tabular_dataset(train_df)
                else:
                    full_train = self._build_image_dataset(train_df, self.train_transform)
                    full_val = self._build_image_dataset(train_df, self.val_transform)

                train_size = int(len(train_df) * self.train_val_split[0])
                val_size = len(train_df) - train_size

                if self.mode == "tabular":
                    self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                        full_dataset,
                        [train_size, val_size],
                        generator=torch.Generator().manual_seed(self.seed),
                    )
                else:
                    self.train_dataset, _ = torch.utils.data.random_split(
                        full_train,
                        [train_size, val_size],
                        generator=torch.Generator().manual_seed(self.seed),
                    )
                    _, self.val_dataset = torch.utils.data.random_split(
                        full_val,
                        [train_size, val_size],
                        generator=torch.Generator().manual_seed(self.seed),
                    )

        if stage == "test" or stage is None:
            if self.test_file is not None:
                test_df = pd.read_parquet(self.test_file)
                if self.mode == "tabular":
                    self.test_dataset = self._build_tabular_dataset(test_df)
                else:
                    self.test_dataset = self._build_image_dataset(test_df, self.val_transform)
            else:
                self.test_dataset = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes for the dataset."""
        if self._num_classes is None:
            raise RuntimeError("Call setup() before accessing num_classes")
        return self._num_classes

    def train_dataloader(self):
        """Get training dataloader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        """Get test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError(
                "No test_file was provided. Set test_file to a parquet file path."
            )
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
