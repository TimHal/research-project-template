"""Torchvision Dataset DataModule wrapper.

This module provides a Lightning DataModule wrapper for standard
torchvision datasets (CIFAR10, CIFAR100, MNIST, EMNIST, etc.).
"""

from typing import Callable, Literal, Optional

import lightning as L
import torch
import torchvision
import torchvision.transforms as transforms

# Valid EMNIST splits
EMNISTSplit = Literal["byclass", "bymerge", "balanced", "letters", "digits", "mnist"]

# Number of classes for each EMNIST split
EMNIST_NUM_CLASSES = {
    "byclass": 62,
    "bymerge": 47,
    "balanced": 47,
    "letters": 26,
    "digits": 10,
    "mnist": 10,
}


class TorchvisionDatamodule(L.LightningDataModule):
    """DataModule wrapper for torchvision datasets.

    Supports common torchvision datasets:
    - CIFAR10, CIFAR100
    - MNIST, FashionMNIST, KMNIST
    - EMNIST (with splits: byclass, bymerge, balanced, letters, digits, mnist)
    - STL10, SVHN

    Args:
        dataset_name: Name of the torchvision dataset
        root_dir: Root directory for dataset storage
        img_size: Target image size (height, width)
        img_mode: Image mode, either 'RGB' or 'L' (grayscale)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        train_transform: Optional custom transform for training
        val_transform: Optional custom transform for validation/test
        norm_mean: Mean values for normalization (ImageNet default for RGB)
        norm_std: Standard deviation values for normalization
        download: Whether to download the dataset
        train_val_split: Tuple of (train_ratio, val_ratio)
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
        seed: Random seed for reproducibility
        emnist_split: Split to use for EMNIST dataset
        augmentation: Augmentation mode ('none' or 'basic')
    """

    SUPPORTED_DATASETS = {
        "CIFAR10": torchvision.datasets.CIFAR10,
        "CIFAR100": torchvision.datasets.CIFAR100,
        "MNIST": torchvision.datasets.MNIST,
        "FashionMNIST": torchvision.datasets.FashionMNIST,
        "KMNIST": torchvision.datasets.KMNIST,
        "EMNIST": torchvision.datasets.EMNIST,
        "STL10": torchvision.datasets.STL10,
        "SVHN": torchvision.datasets.SVHN,
    }

    # Datasets that are natively grayscale
    GRAYSCALE_DATASETS = {"MNIST", "FashionMNIST", "KMNIST", "EMNIST"}

    def __init__(
        self,
        dataset_name: str = "CIFAR10",
        root_dir: str = "./data",
        img_size: tuple[int, int] = (224, 224),
        img_mode: str = "RGB",
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        norm_mean: Optional[list[float]] = None,
        norm_std: Optional[list[float]] = None,
        download: bool = True,
        train_val_split: tuple[float, float] = (0.9, 0.1),
        pin_memory: bool = True,
        persistent_workers: bool = True,
        seed: int = 42,
        emnist_split: EMNISTSplit = "balanced",
        augmentation: str = "none",
    ):
        super().__init__()

        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset {dataset_name} not supported. "
                f"Supported datasets: {list(self.SUPPORTED_DATASETS.keys())}"
            )

        if dataset_name == "EMNIST" and emnist_split not in EMNIST_NUM_CLASSES:
            raise ValueError(
                f"EMNIST split '{emnist_split}' not supported. "
                f"Supported splits: {list(EMNIST_NUM_CLASSES.keys())}"
            )

        assert img_mode in ["RGB", "L"], "img_mode must be 'RGB' or 'L'"

        self.dataset_name = dataset_name
        self.dataset_class = self.SUPPORTED_DATASETS[dataset_name]
        self.root_dir = root_dir
        self.img_size = img_size
        self.img_mode = img_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.train_val_split = train_val_split
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.emnist_split = emnist_split
        self.augmentation = augmentation

        # Determine number of channels based on img_mode
        self.num_channels = 1 if img_mode == "L" else 3
        self._native_channels = 1 if dataset_name in self.GRAYSCALE_DATASETS else 3

        if norm_mean is not None and norm_std is not None:
            if len(norm_mean) != self.num_channels or len(norm_std) != self.num_channels:
                raise ValueError(
                    f"norm_mean and norm_std must have length {self.num_channels} "
                    f"for img_mode {img_mode}"
                )

        # Set transforms
        self.train_transform = train_transform or self._get_default_train_transform()
        self.val_transform = val_transform or self._get_default_val_transform()

    def _get_default_train_transform(self) -> transforms.Compose:
        """Get default training transforms."""
        transform_list = [transforms.Resize(self.img_size)]

        # Convert to target img_mode if needed
        if self.img_mode == "RGB" and self._native_channels == 1:
            transform_list.append(transforms.Lambda(lambda x: x.convert("RGB")))
        elif self.img_mode == "L" and self._native_channels == 3:
            transform_list.append(transforms.Lambda(lambda x: x.convert("L")))

        if self.dataset_name == "EMNIST":
            # EMNIST images are rotated by 90 degrees and flipped by default
            transform_list.append(
                transforms.Lambda(lambda x: x.rotate(-90).transpose(method=0))
            )

        # Basic augmentation
        if self.augmentation == "basic":
            transform_list.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                ]
            )

        transform_list.append(transforms.ToTensor())

        # Normalize
        if self.norm_mean is not None and self.norm_std is not None:
            transform_list.append(
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
            )

        return transforms.Compose(transform_list)

    def _get_default_val_transform(self) -> transforms.Compose:
        """Get default validation/test transforms."""
        transform_list = [transforms.Resize(self.img_size)]

        # Convert to target img_mode if needed
        if self.img_mode == "RGB" and self._native_channels == 1:
            transform_list.append(transforms.Lambda(lambda x: x.convert("RGB")))
        elif self.img_mode == "L" and self._native_channels == 3:
            transform_list.append(transforms.Lambda(lambda x: x.convert("L")))

        if self.dataset_name == "EMNIST":
            # EMNIST images are rotated by 90 degrees and flipped by default
            transform_list.append(
                transforms.Lambda(lambda x: x.rotate(-90).transpose(method=0))
            )

        transform_list.append(transforms.ToTensor())

        # Normalize
        if self.norm_mean is not None and self.norm_std is not None:
            transform_list.append(
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
            )

        return transforms.Compose(transform_list)

    def prepare_data(self):
        """Download the dataset if needed."""
        if self.dataset_name == "SVHN":
            self.dataset_class(self.root_dir, split="train", download=self.download)
            self.dataset_class(self.root_dir, split="test", download=self.download)
        elif self.dataset_name == "EMNIST":
            self.dataset_class(
                self.root_dir,
                split=self.emnist_split,
                train=True,
                download=self.download,
            )
            self.dataset_class(
                self.root_dir,
                split=self.emnist_split,
                train=False,
                download=self.download,
            )
        else:
            self.dataset_class(self.root_dir, train=True, download=self.download)
            self.dataset_class(self.root_dir, train=False, download=self.download)

    def _create_dataset(self, train: bool, transform: Callable):
        """Create a dataset instance with the appropriate parameters."""
        if self.dataset_name == "SVHN":
            split = "train" if train else "test"
            return self.dataset_class(
                self.root_dir, split=split, transform=transform, download=False
            )
        elif self.dataset_name == "EMNIST":
            return self.dataset_class(
                self.root_dir,
                split=self.emnist_split,
                train=train,
                transform=transform,
                download=False,
            )
        else:
            return self.dataset_class(
                self.root_dir, train=train, transform=transform, download=False
            )

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage."""
        if stage == "fit" or stage is None:
            # Load training data
            full_train = self._create_dataset(train=True, transform=self.train_transform)

            # Split into train and validation
            train_size = int(len(full_train) * self.train_val_split[0])
            val_size = len(full_train) - train_size

            self.train_dataset, _ = torch.utils.data.random_split(
                full_train,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

            # Create validation dataset with different transform
            full_val = self._create_dataset(train=True, transform=self.val_transform)
            _, self.val_dataset = torch.utils.data.random_split(
                full_val,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage == "test" or stage is None:
            self.test_dataset = self._create_dataset(
                train=False, transform=self.val_transform
            )

    @property
    def num_classes(self) -> int:
        """Get the number of classes for the dataset."""
        if self.dataset_name == "EMNIST":
            return EMNIST_NUM_CLASSES[self.emnist_split]
        elif self.dataset_name == "CIFAR10":
            return 10
        elif self.dataset_name == "CIFAR100":
            return 100
        elif self.dataset_name in ("MNIST", "FashionMNIST", "KMNIST"):
            return 10
        elif self.dataset_name == "STL10":
            return 10
        elif self.dataset_name == "SVHN":
            return 10
        else:
            raise ValueError(f"Unknown number of classes for {self.dataset_name}")

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
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
