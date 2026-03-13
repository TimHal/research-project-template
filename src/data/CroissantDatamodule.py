"""Croissant Dataset DataModule.

This module provides a Lightning DataModule for loading datasets described
by Croissant JSON-LD metadata files. Croissant is a standardized format for
dataset metadata adopted by HuggingFace, Kaggle, OpenML, and others.

Requires: pip install mlcroissant

Croissant spec: https://github.com/mlcommons/croissant
"""

import io
from typing import Callable, Optional

import lightning as L
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def _import_mlcroissant():
    """Lazy import of mlcroissant with clear error message."""
    try:
        import mlcroissant
        return mlcroissant
    except ImportError:
        raise ImportError(
            "CroissantDatamodule requires the 'mlcroissant' package. "
            "Install it with: pip install mlcroissant"
        ) from None


class _CroissantImageDataset(Dataset):
    """Dataset wrapper for Croissant records with image data."""

    def __init__(
        self,
        records: list[dict],
        image_column: str,
        label_column: str,
        label_map: dict,
        transform: Callable,
    ):
        self.records = records
        self.image_column = image_column
        self.label_column = label_column
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # Decode image from bytes
        image_data = record[self.image_column]
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            raise TypeError(
                f"Unexpected image type: {type(image_data)}. "
                f"Expected bytes or PIL.Image."
            )

        if self.transform is not None:
            image = self.transform(image)

        # Map label to integer
        label = record[self.label_column]
        if isinstance(label, str):
            label = self.label_map[label]
        else:
            label = int(label)

        return image, label


class CroissantDatamodule(L.LightningDataModule):
    """DataModule for datasets described by Croissant JSON-LD metadata.

    Loads datasets using the mlcroissant library, which reads standardized
    metadata files to locate, download, and parse dataset records.

    Note:
        All records are materialized into memory during ``setup()``. For very
        large datasets, consider using WebDatasetDatamodule instead.

    Args:
        metadata_path: Path or URL to the Croissant JSON-LD metadata file
        record_set: Name of the record set to load from the metadata
        image_column: Column/field containing images (auto-detected if None)
        label_column: Column/field containing labels (auto-detected if None)
        num_classes: Number of classes (auto-detected from unique labels if None)
        root_dir: Cache directory for downloaded data
        img_size: Target image size (height, width)
        img_mode: Image mode, either 'RGB' or 'L' (grayscale)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        train_transform: Optional custom transform for training
        val_transform: Optional custom transform for validation/test
        norm_mean: Mean values for normalization
        norm_std: Standard deviation values for normalization
        train_val_split: Tuple of (train_ratio, val_ratio)
        test_ratio: Fraction of data to hold out for testing (0.0 to skip)
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
        seed: Random seed for reproducibility
        augmentation: Augmentation mode ('none' or 'basic')
    """

    def __init__(
        self,
        metadata_path: str = "",
        record_set: str = "",
        image_column: Optional[str] = None,
        label_column: Optional[str] = None,
        num_classes: Optional[int] = None,
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
        test_ratio: float = 0.0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        seed: int = 42,
        augmentation: str = "none",
    ):
        super().__init__()

        assert img_mode in ["RGB", "L"], "img_mode must be 'RGB' or 'L'"

        if not metadata_path:
            raise ValueError("metadata_path is required")
        if not record_set:
            raise ValueError("record_set is required")

        self.dataset_name = record_set
        self.root_dir = root_dir
        self.metadata_path = metadata_path
        self.record_set = record_set
        self._image_column = image_column
        self._label_column = label_column
        self._explicit_num_classes = num_classes
        self.img_size = img_size
        self.img_mode = img_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.test_ratio = test_ratio
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed
        self.augmentation = augmentation

        self.num_channels = 1 if img_mode == "L" else 3
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.train_transform = train_transform or self._get_default_train_transform()
        self.val_transform = val_transform or self._get_default_val_transform()

        self._num_classes = None
        self._label_map = {}
        self.class_names = []

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

    def _detect_columns(self, record: dict) -> tuple[str, str]:
        """Auto-detect image and label columns from a sample record."""
        image_col = self._image_column
        label_col = self._label_column

        if image_col is None:
            for key, value in record.items():
                if isinstance(value, (bytes, Image.Image)):
                    image_col = key
                    break
            if image_col is None:
                raise ValueError(
                    f"Could not auto-detect image column. "
                    f"Available keys: {list(record.keys())}. "
                    f"Set image_column explicitly."
                )

        if label_col is None:
            for key, value in record.items():
                if key != image_col and isinstance(value, (str, int)):
                    label_col = key
                    break
            if label_col is None:
                raise ValueError(
                    f"Could not auto-detect label column. "
                    f"Available keys: {list(record.keys())}. "
                    f"Set label_column explicitly."
                )

        return image_col, label_col

    def setup(self, stage: Optional[str] = None):
        """Set up datasets by loading all records from Croissant metadata."""
        mlcroissant = _import_mlcroissant()

        ds = mlcroissant.Dataset(self.metadata_path)
        records = list(ds.records(self.record_set))

        if not records:
            raise RuntimeError(
                f"No records found for record_set '{self.record_set}' "
                f"in {self.metadata_path}"
            )

        # Detect columns from first record
        image_col, label_col = self._detect_columns(records[0])

        # Build label mapping for string labels
        all_labels = [r[label_col] for r in records]
        if isinstance(all_labels[0], str):
            unique_labels = sorted(set(all_labels))
            self._label_map = {label: i for i, label in enumerate(unique_labels)}
            self.class_names = unique_labels
            self._num_classes = len(unique_labels)
        else:
            self._label_map = {}
            self.class_names = []
            self._num_classes = max(int(l) for l in all_labels) + 1

        if self._explicit_num_classes is not None:
            self._num_classes = self._explicit_num_classes

        # Split records into train(+val) and test
        generator = torch.Generator().manual_seed(self.seed)

        if self.test_ratio > 0:
            test_size = int(len(records) * self.test_ratio)
            trainval_size = len(records) - test_size
            indices = torch.randperm(len(records), generator=generator).tolist()
            trainval_indices = indices[:trainval_size]
            test_indices = indices[trainval_size:]
        else:
            trainval_indices = list(range(len(records)))
            test_indices = []

        trainval_records = [records[i] for i in trainval_indices]
        test_records = [records[i] for i in test_indices]

        if stage == "fit" or stage is None:
            # Split trainval into train and val
            train_size = int(len(trainval_records) * self.train_val_split[0])
            val_size = len(trainval_records) - train_size

            gen2 = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(len(trainval_records), generator=gen2).tolist()
            train_records = [trainval_records[i] for i in perm[:train_size]]
            val_records = [trainval_records[i] for i in perm[train_size:]]

            self.train_dataset = _CroissantImageDataset(
                train_records, image_col, label_col, self._label_map, self.train_transform
            )
            self.val_dataset = _CroissantImageDataset(
                val_records, image_col, label_col, self._label_map, self.val_transform
            )

        if stage == "test" or stage is None:
            if test_records:
                self.test_dataset = _CroissantImageDataset(
                    test_records, image_col, label_col, self._label_map, self.val_transform
                )
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
                "No test data available. Set test_ratio > 0 to hold out test data."
            )
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
