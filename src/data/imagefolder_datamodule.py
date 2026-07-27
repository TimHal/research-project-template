"""ImageFolder Dataset DataModule.

This module provides a Lightning DataModule for loading image datasets
organized in a directory structure where each class has its own subdirectory.

Supports two layouts:
- Split directories: root_dir/train/class_name/, root_dir/val/class_name/, etc.
- Single directory: root_dir/class_name/ (auto-split into train/val)
"""

import os
from typing import Callable, Optional

import lightning as L
import torch
import torchvision
import torchvision.transforms as transforms


class ImageFolderDataModule(L.LightningDataModule):
    """DataModule for image datasets organized as class subdirectories.

    Expected directory layouts:

    **Split directories** (default)::

        root_dir/
        ├── train/
        │   ├── class_a/
        │   │   ├── img1.jpg
        │   │   └── img2.jpg
        │   └── class_b/
        │       └── img3.jpg
        ├── val/          # optional
        │   └── ...
        └── test/         # optional
            └── ...

    **Single directory** (all classes in one folder, auto-split)::

        root_dir/
        ├── class_a/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── class_b/
            └── img3.jpg

    Args:
        root_dir: Root directory containing image folders
        train_dir: Explicit path to training folder (overrides root_dir/train)
        val_dir: Explicit path to validation folder (overrides root_dir/val)
        test_dir: Explicit path to test folder (overrides root_dir/test)
        img_size: Target image size (height, width)
        img_mode: Image mode, either 'RGB' or 'L' (grayscale)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        train_transform: Optional custom transform for training
        val_transform: Optional custom transform for validation/test
        norm_mean: Mean values for normalization (ImageNet default for RGB)
        norm_std: Standard deviation values for normalization
        train_val_split: Tuple of (train_ratio, val_ratio) for auto-splitting
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
        seed: Random seed for reproducibility
        augmentation: Augmentation mode ('none' or 'basic')
    """

    def __init__(
        self,
        root_dir: str = "./data",
        train_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
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

        self.dataset_name = os.path.basename(os.path.normpath(root_dir))
        self.root_dir = root_dir
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
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

    def _resolve_dirs(self) -> dict[str, Optional[str]]:
        """Resolve train/val/test directory paths."""
        dirs = {
            "train": self.train_dir or os.path.join(self.root_dir, "train"),
            "val": self.val_dir or os.path.join(self.root_dir, "val"),
            "test": self.test_dir or os.path.join(self.root_dir, "test"),
        }

        # Check if split directories exist
        has_splits = os.path.isdir(dirs["train"])

        if not has_splits:
            # Single directory mode: root_dir contains class folders directly
            return {"train": self.root_dir, "val": None, "test": None}

        return {
            k: v if os.path.isdir(v) else None
            for k, v in dirs.items()
        }

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage."""
        dirs = self._resolve_dirs()

        if stage == "fit" or stage is None:
            if dirs["val"] is not None:
                # Separate train and val directories
                self.train_dataset = torchvision.datasets.ImageFolder(
                    dirs["train"], transform=self.train_transform
                )
                self.val_dataset = torchvision.datasets.ImageFolder(
                    dirs["val"], transform=self.val_transform
                )
            else:
                # Single directory — split into train/val
                full_train = torchvision.datasets.ImageFolder(
                    dirs["train"], transform=self.train_transform
                )
                train_size = int(len(full_train) * self.train_val_split[0])
                val_size = len(full_train) - train_size

                self.train_dataset, _ = torch.utils.data.random_split(
                    full_train,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(self.seed),
                )

                full_val = torchvision.datasets.ImageFolder(
                    dirs["train"], transform=self.val_transform
                )
                _, self.val_dataset = torch.utils.data.random_split(
                    full_val,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(self.seed),
                )

            # Detect num_classes from training data
            source = self.train_dataset
            if isinstance(source, torch.utils.data.Subset):
                source = source.dataset
            self._num_classes = len(source.classes)

        if stage == "test" or stage is None:
            test_dir = dirs["test"]
            if test_dir is not None:
                self.test_dataset = torchvision.datasets.ImageFolder(
                    test_dir, transform=self.val_transform
                )
            else:
                # No test directory — use validation set
                self.test_dataset = getattr(self, "val_dataset", None)

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
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
