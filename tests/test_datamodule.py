"""Tests for TorchvisionDataModule construction and validation.

Only construction-time behavior is tested here (no setup()/prepare_data()),
so nothing is downloaded. Tests that need real data should download a small
dataset in a fixture and be marked @pytest.mark.slow.
"""

import pytest
from PIL import Image

from data import ImageFolderDataModule, TorchvisionDataModule


@pytest.mark.parametrize(
    "dataset_name,expected",
    [("CIFAR10", 10), ("CIFAR100", 100), ("MNIST", 10)],
)
def test_num_classes(dataset_name, expected):
    dm = TorchvisionDataModule(dataset_name=dataset_name, download=False)
    assert dm.num_classes == expected


def test_unsupported_dataset_raises():
    with pytest.raises(ValueError):
        TorchvisionDataModule(dataset_name="NotADataset", download=False)


def test_invalid_img_mode_raises():
    with pytest.raises(AssertionError):
        TorchvisionDataModule(dataset_name="CIFAR10", img_mode="RGBA", download=False)


def test_imagefolder_end_to_end(tmp_path):
    """Synthetic ImageFolder exercises the shared transform + a full loader pass."""
    for cls in ("cat", "dog"):
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(3):
            Image.new("RGB", (16, 16), color=(i * 20, 0, 0)).save(cls_dir / f"{i}.png")

    dm = ImageFolderDataModule(root_dir=str(tmp_path), img_size=(8, 8), batch_size=2, num_workers=0)
    dm.setup("fit")
    assert dm.num_classes == 2

    images, labels = next(iter(dm.train_dataloader()))
    assert images.shape[1:] == (3, 8, 8)
    assert labels.max() < 2
