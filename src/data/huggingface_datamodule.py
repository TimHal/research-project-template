"""HuggingFace Datasets DataModule.

This module provides a Lightning DataModule for loading datasets from the
HuggingFace Hub using the `datasets` library.

Requires: pip install datasets

Available datasets: https://huggingface.co/datasets
"""

from typing import Callable, Optional

import lightning as L
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def _import_datasets():
    """Lazy import of datasets with clear error message."""
    try:
        import datasets
        return datasets
    except ImportError:
        raise ImportError(
            "HuggingFaceDataModule requires the 'datasets' package. "
            "Install it with: pip install datasets"
        ) from None


class _HFImageDataset(Dataset):
    """Wrapper that applies transforms to a HuggingFace image dataset."""

    def __init__(self, hf_dataset, image_column: str, label_column: str, transform: Callable):
        self.hf_dataset = hf_dataset
        self.image_column = image_column
        self.label_column = label_column
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item[self.image_column]
        label = item[self.label_column]

        # HF datasets return PIL images when using datasets.Image feature
        if self.transform is not None:
            image = self.transform(image)

        return image, label


class HuggingFaceDataModule(L.LightningDataModule):
    """DataModule for HuggingFace Hub datasets.

    Loads any image classification dataset from the HuggingFace Hub.
    Automatically detects image and label columns from the dataset features.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "beans", "cifar10")
        dataset_config: Dataset configuration name (for multi-config datasets)
        image_column: Column containing PIL images (auto-detected if None)
        label_column: Column containing labels (auto-detected if None)
        train_split: Name of the training split
        test_split: Name of the test split (None if dataset has no test split)
        trust_remote_code: Whether to trust remote code in dataset scripts
        root_dir: Cache directory for downloaded datasets
        img_size: Target image size (height, width)
        img_mode: Image mode, either 'RGB' or 'L' (grayscale)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        train_transform: Optional custom transform for training
        val_transform: Optional custom transform for validation/test
        norm_mean: Mean values for normalization (ImageNet default for RGB)
        norm_std: Standard deviation values for normalization
        train_val_split: Tuple of (train_ratio, val_ratio)
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
        seed: Random seed for reproducibility
        augmentation: Augmentation mode ('none' or 'basic')
    """

    def __init__(
        self,
        dataset_name: str = "beans",
        dataset_config: Optional[str] = None,
        image_column: Optional[str] = None,
        label_column: Optional[str] = None,
        train_split: str = "train",
        test_split: Optional[str] = "test",
        trust_remote_code: bool = False,
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

        assert img_mode in ["RGB", "L"], "img_mode must be 'RGB' or 'L'"

        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self._image_column = image_column
        self._label_column = label_column
        self.train_split = train_split
        self.test_split = test_split
        self.trust_remote_code = trust_remote_code
        self.root_dir = root_dir
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

        if norm_mean is not None and norm_std is not None:
            if len(norm_mean) != self.num_channels or len(norm_std) != self.num_channels:
                raise ValueError(
                    f"norm_mean and norm_std must have length {self.num_channels} "
                    f"for img_mode {img_mode}"
                )

        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.train_transform = train_transform or self._get_default_train_transform()
        self.val_transform = val_transform or self._get_default_val_transform()

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

    def _detect_columns(self, features) -> tuple[str, str]:
        """Auto-detect image and label columns from dataset features."""
        datasets = _import_datasets()

        image_col = self._image_column
        label_col = self._label_column

        if image_col is None:
            for name, feat in features.items():
                if isinstance(feat, datasets.Image):
                    image_col = name
                    break
            if image_col is None:
                raise ValueError(
                    f"Could not auto-detect image column. "
                    f"Available columns: {list(features.keys())}. "
                    f"Set image_column explicitly."
                )

        if label_col is None:
            for name, feat in features.items():
                if isinstance(feat, datasets.ClassLabel):
                    label_col = name
                    break
            if label_col is None:
                raise ValueError(
                    f"Could not auto-detect label column. "
                    f"Available columns: {list(features.keys())}. "
                    f"Set label_column explicitly."
                )

        return image_col, label_col

    def prepare_data(self):
        """Download the dataset."""
        datasets = _import_datasets()
        datasets.load_dataset(
            self.dataset_name,
            self.dataset_config,
            cache_dir=self.root_dir,
            trust_remote_code=self.trust_remote_code,
        )

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage."""
        datasets = _import_datasets()
        ds = datasets.load_dataset(
            self.dataset_name,
            self.dataset_config,
            cache_dir=self.root_dir,
            trust_remote_code=self.trust_remote_code,
        )

        # Detect columns
        image_col, label_col = self._detect_columns(ds[self.train_split].features)

        # Detect num_classes
        feat = ds[self.train_split].features[label_col]
        if isinstance(feat, datasets.ClassLabel):
            self._num_classes = feat.num_classes
        else:
            # Fallback: count unique values
            self._num_classes = len(set(ds[self.train_split][label_col]))

        if stage == "fit" or stage is None:
            train_hf = ds[self.train_split]

            # Wrap in PyTorch datasets
            full_train = _HFImageDataset(train_hf, image_col, label_col, self.train_transform)

            # Split into train and validation
            train_size = int(len(full_train) * self.train_val_split[0])
            val_size = len(full_train) - train_size

            self.train_dataset, _ = torch.utils.data.random_split(
                full_train,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

            full_val = _HFImageDataset(train_hf, image_col, label_col, self.val_transform)
            _, self.val_dataset = torch.utils.data.random_split(
                full_val,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage == "test" or stage is None:
            if self.test_split is not None and self.test_split in ds:
                test_hf = ds[self.test_split]
                self.test_dataset = _HFImageDataset(
                    test_hf, image_col, label_col, self.val_transform
                )
            elif "validation" in ds:
                val_hf = ds["validation"]
                self.test_dataset = _HFImageDataset(
                    val_hf, image_col, label_col, self.val_transform
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
                f"Dataset '{self.dataset_name}' has no test split. "
                f"Set test_split to an available split name."
            )
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
