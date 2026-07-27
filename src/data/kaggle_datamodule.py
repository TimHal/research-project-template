"""Kaggle Dataset DataModule.

This module provides a Lightning DataModule for downloading and loading
datasets from Kaggle using the Kaggle API.

Requires: pip install kaggle
Auth: Place your API token at ~/.kaggle/kaggle.json
Docs: https://github.com/Kaggle/kaggle-api

Supports two loading formats after download:
- imagefolder: directory of class subdirectories (most image datasets)
- csv: CSV file with label column and optional image paths or pixel data
"""

import os
import zipfile
from typing import Callable, Optional

import lightning as L
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from data.transforms import build_image_transform


def _import_kaggle():
    """Lazy import of kaggle with clear error message."""
    try:
        from kaggle import api as kaggle_api
        return kaggle_api
    except ImportError:
        raise ImportError(
            "KaggleDataModule requires the 'kaggle' package. "
            "Install it with: pip install kaggle\n"
            "You also need a Kaggle API token at ~/.kaggle/kaggle.json. "
            "See: https://github.com/Kaggle/kaggle-api#api-credentials"
        ) from None
    except OSError as e:
        raise OSError(
            f"Kaggle API authentication failed: {e}\n"
            f"Place your API token at ~/.kaggle/kaggle.json. "
            f"See: https://github.com/Kaggle/kaggle-api#api-credentials"
        ) from None


class _CSVImageDataset(Dataset):
    """Dataset that loads images from file paths stored in a CSV."""

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        transform: Callable,
        base_dir: str,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.base_dir = base_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.base_dir, self.image_paths[idx])
        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


class _PixelDataset(Dataset):
    """Dataset for CSV files with flat pixel data (e.g., Kaggle digit-recognizer)."""

    def __init__(self, pixels: torch.Tensor, labels: torch.Tensor, transform: Optional[Callable] = None):
        self.pixels = pixels
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.pixels[idx]
        if self.transform is not None:
            # Convert tensor back to PIL for transform pipeline
            image = transforms.ToPILImage()(image)
            image = self.transform(image)
        return image, self.labels[idx]


class KaggleDataModule(L.LightningDataModule):
    """DataModule for Kaggle datasets and competitions.

    Downloads data via the Kaggle API, then loads it as either an ImageFolder
    dataset or from a CSV file.

    Args:
        dataset_slug: Kaggle dataset slug (e.g., "zalando-research/fashionmnist")
        competition: Kaggle competition name (e.g., "digit-recognizer")
        load_format: How to load data after download — 'imagefolder' or 'csv'
        csv_file: CSV filename within the downloaded archive (for load_format='csv')
        label_column: Label column in CSV
        image_column: Image file path column in CSV (for CSV with image files)
        pixel_columns: Pixel value columns in CSV (None = all columns except label)
        image_shape: Shape to reshape pixel columns into (C, H, W)
        num_classes: Number of classes (auto-detected if None)
        root_dir: Directory to download and extract data into
        img_size: Target image size (height, width)
        img_mode: Image mode, either 'RGB' or 'L' (grayscale)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        train_transform: Optional custom transform for training
        val_transform: Optional custom transform for validation/test
        norm_mean: Mean values for normalization
        norm_std: Standard deviation values for normalization
        train_val_split: Tuple of (train_ratio, val_ratio)
        download: Whether to download (skips if data already exists)
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
        seed: Random seed for reproducibility
        augmentation: Augmentation mode ('none' or 'basic')
    """

    def __init__(
        self,
        dataset_slug: Optional[str] = None,
        competition: Optional[str] = None,
        load_format: str = "imagefolder",
        csv_file: Optional[str] = None,
        label_column: str = "label",
        image_column: Optional[str] = None,
        pixel_columns: Optional[list[str]] = None,
        image_shape: Optional[tuple[int, int, int]] = None,
        num_classes: Optional[int] = None,
        root_dir: str = "./data/kaggle",
        img_size: tuple[int, int] = (224, 224),
        img_mode: str = "RGB",
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        norm_mean: Optional[list[float]] = None,
        norm_std: Optional[list[float]] = None,
        train_val_split: tuple[float, float] = (0.9, 0.1),
        download: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        seed: int = 42,
        augmentation: str = "none",
    ):
        super().__init__()

        if dataset_slug is None and competition is None:
            raise ValueError("One of dataset_slug or competition must be provided")

        assert load_format in ["imagefolder", "csv"], (
            "load_format must be 'imagefolder' or 'csv'"
        )
        assert img_mode in ["RGB", "L"], "img_mode must be 'RGB' or 'L'"

        if load_format == "csv" and csv_file is None:
            raise ValueError("csv_file is required when load_format='csv'")

        self.dataset_name = dataset_slug or competition
        self.root_dir = root_dir
        self.dataset_slug = dataset_slug
        self.competition = competition
        self.load_format = load_format
        self.csv_file = csv_file
        self.label_column = label_column
        self.image_column = image_column
        self.pixel_columns = pixel_columns
        self.image_shape = image_shape
        self._explicit_num_classes = num_classes
        self.img_size = img_size
        self.img_mode = img_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.download = download
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed
        self.augmentation = augmentation

        self.num_channels = 1 if img_mode == "L" else 3
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.train_transform = train_transform or build_image_transform(
            self.img_size, self.img_mode, self.augmentation,
            self.norm_mean, self.norm_std, train=True,
        )
        self.val_transform = val_transform or build_image_transform(
            self.img_size, self.img_mode, self.augmentation,
            self.norm_mean, self.norm_std, train=False,
        )

        self._num_classes = num_classes

    def _is_downloaded(self) -> bool:
        """Check if data has already been downloaded."""
        if not os.path.isdir(self.root_dir):
            return False
        # Consider downloaded if directory is non-empty
        return len(os.listdir(self.root_dir)) > 0

    def prepare_data(self):
        """Download dataset from Kaggle."""
        if not self.download or self._is_downloaded():
            return

        api = _import_kaggle()
        os.makedirs(self.root_dir, exist_ok=True)

        if self.dataset_slug is not None:
            api.dataset_download_files(
                self.dataset_slug, path=self.root_dir, unzip=True
            )
        else:
            api.competition_download_files(
                self.competition, path=self.root_dir
            )
            # Competition downloads are zip files — unzip them
            for fname in os.listdir(self.root_dir):
                if fname.endswith(".zip"):
                    zip_path = os.path.join(self.root_dir, fname)
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(self.root_dir)
                    os.remove(zip_path)

    def _find_imagefolder_root(self) -> str:
        """Find the ImageFolder root within downloaded data."""
        # Check if root_dir directly contains class subdirectories
        subdirs = [
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]

        # Check for common split directory names
        for split_name in ["train", "training"]:
            if split_name in subdirs:
                return os.path.join(self.root_dir, split_name)

        # If subdirs look like class names (more than 1), use root_dir directly
        if len(subdirs) >= 2:
            # Check if subdirs contain image files
            first_subdir = os.path.join(self.root_dir, subdirs[0])
            files = os.listdir(first_subdir)
            image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
            has_images = any(
                os.path.splitext(f)[1].lower() in image_exts for f in files
            )
            if has_images:
                return self.root_dir

        # Single subdirectory — go one level deeper
        if len(subdirs) == 1:
            return os.path.join(self.root_dir, subdirs[0])

        return self.root_dir

    def _load_csv(self, csv_path: str):
        """Load data from a CSV file. Returns (features_or_paths, labels, is_pixel_data)."""
        import csv as csv_module

        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            raise RuntimeError(f"CSV file is empty: {csv_path}")

        labels = [row[self.label_column] for row in rows]

        # Encode string labels
        if not labels[0].lstrip("-").isdigit():
            unique = sorted(set(labels))
            label_map = {lbl: i for i, lbl in enumerate(unique)}
            labels = [label_map[lbl] for lbl in labels]
        else:
            labels = [int(lbl) for lbl in labels]

        if self.image_column is not None:
            # CSV with image file paths
            paths = [row[self.image_column] for row in rows]
            return paths, labels, False
        else:
            # CSV with pixel data
            if self.pixel_columns is not None:
                cols = self.pixel_columns
            else:
                cols = [k for k in rows[0].keys() if k != self.label_column]

            pixels = [[float(row[c]) for c in cols] for row in rows]
            return pixels, labels, True

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage."""
        if self.load_format == "imagefolder":
            self._setup_imagefolder(stage)
        else:
            self._setup_csv(stage)

    def _setup_imagefolder(self, stage: Optional[str] = None):
        """Set up datasets from ImageFolder layout."""
        folder_root = self._find_imagefolder_root()

        # Check for train/test split directories
        train_dir = None
        test_dir = None
        for name in ["train", "training"]:
            candidate = os.path.join(folder_root, name)
            if os.path.isdir(candidate):
                train_dir = candidate
                break
        for name in ["test", "testing", "val", "validation"]:
            candidate = os.path.join(folder_root, name)
            if os.path.isdir(candidate):
                test_dir = candidate
                break

        if train_dir is None:
            train_dir = folder_root

        if stage == "fit" or stage is None:
            full_train = torchvision.datasets.ImageFolder(
                train_dir, transform=self.train_transform
            )

            train_size = int(len(full_train) * self.train_val_split[0])
            val_size = len(full_train) - train_size

            self.train_dataset, _ = torch.utils.data.random_split(
                full_train,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

            full_val = torchvision.datasets.ImageFolder(
                train_dir, transform=self.val_transform
            )
            _, self.val_dataset = torch.utils.data.random_split(
                full_val,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

            source = full_train
            if self._num_classes is None:
                self._num_classes = len(source.classes)

        if stage == "test" or stage is None:
            if test_dir is not None:
                self.test_dataset = torchvision.datasets.ImageFolder(
                    test_dir, transform=self.val_transform
                )
            else:
                self.test_dataset = getattr(self, "val_dataset", None)

    def _setup_csv(self, stage: Optional[str] = None):
        """Set up datasets from CSV file."""
        csv_path = os.path.join(self.root_dir, self.csv_file)
        data, labels, is_pixel = self._load_csv(csv_path)

        if self._num_classes is None:
            self._num_classes = max(labels) + 1

        if is_pixel:
            pixels_tensor = torch.tensor(data, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)

            if self.image_shape is not None:
                pixels_tensor = pixels_tensor.view(-1, *self.image_shape)
                # Normalize pixel values to [0, 1] if they look like 0-255
                if pixels_tensor.max() > 1.0:
                    pixels_tensor = pixels_tensor / 255.0

            full_dataset = _PixelDataset(pixels_tensor, labels_tensor, self.train_transform)
        else:
            full_dataset = _CSVImageDataset(
                data, labels, self.train_transform, self.root_dir
            )

        if stage == "fit" or stage is None:
            train_size = int(len(full_dataset) * self.train_val_split[0])
            val_size = len(full_dataset) - train_size

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage == "test" or stage is None:
            # Check for a test CSV
            test_csv = self.csv_file.replace("train", "test")
            test_path = os.path.join(self.root_dir, test_csv)
            if os.path.isfile(test_path) and test_csv != self.csv_file:
                test_data, test_labels, test_is_pixel = self._load_csv(test_path)
                if test_is_pixel:
                    t = torch.tensor(test_data, dtype=torch.float32)
                    label_tensor = torch.tensor(test_labels, dtype=torch.long)
                    if self.image_shape is not None:
                        t = t.view(-1, *self.image_shape)
                        if t.max() > 1.0:
                            t = t / 255.0
                    self.test_dataset = _PixelDataset(t, label_tensor, self.val_transform)
                else:
                    self.test_dataset = _CSVImageDataset(
                        test_data, test_labels, self.val_transform, self.root_dir
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
            raise RuntimeError("No test data available.")
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
