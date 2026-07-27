"""WebDataset DataModule.

This module provides a Lightning DataModule for streaming large-scale
datasets from tar archives using the WebDataset library.

Requires: pip install webdataset

WebDataset format stores samples as tar archives where each sample is a set
of files sharing the same basename (e.g., sample001.jpg, sample001.cls).
Supports local files and URLs, with shard patterns for distributed training.

Format docs: https://webdataset.github.io/webdataset/
"""

from typing import Callable, Optional

import lightning as L

from data.transforms import build_image_transform


def _import_webdataset():
    """Lazy import of webdataset with clear error message."""
    try:
        import webdataset as wds
        return wds
    except ImportError:
        raise ImportError(
            "WebDatasetDataModule requires the 'webdataset' package. "
            "Install it with: pip install webdataset"
        ) from None


class WebDatasetDataModule(L.LightningDataModule):
    """DataModule for streaming datasets from tar archives via WebDataset.

    Designed for large-scale datasets stored as sharded tar files. Samples
    are streamed on-the-fly without loading the entire dataset into memory.

    Tar archives should contain files like::

        sample000.jpg    # image
        sample000.cls    # class label (integer as text)
        sample001.jpg
        sample001.cls
        ...

    Shard patterns use brace expansion::

        train-{0000..0099}.tar   →  train-0000.tar, train-0001.tar, ...

    Note:
        MLflow dataset tracking (LogDatasetCallback) is not supported for
        streaming datasets. WebDataset uses IterableDataset internally.

    Args:
        train_urls: Shard URL pattern for training data
        val_urls: Shard URL pattern for validation data
        test_urls: Shard URL pattern for test data (optional)
        num_classes: Number of classes (must be specified — cannot auto-detect)
        num_train_samples: Total number of training samples (needed for epoch length)
        num_val_samples: Total number of validation samples
        num_test_samples: Total number of test samples
        image_key: File extension key for images in tar (e.g., 'jpg', 'png')
        label_key: File extension key for labels in tar (e.g., 'cls', 'json')
        shardshuffle: Whether to shuffle shards each epoch
        shuffle_buffer: Size of the sample shuffle buffer
        root_dir: Not used for data loading (kept for LogDatasetCallback compatibility)
        img_size: Target image size (height, width)
        img_mode: Image mode, either 'RGB' or 'L' (grayscale)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        train_transform: Optional custom transform for training
        val_transform: Optional custom transform for validation/test
        norm_mean: Mean values for normalization
        norm_std: Standard deviation values for normalization
        pin_memory: Whether to pin memory for faster GPU transfer
        augmentation: Augmentation mode ('none' or 'basic')
    """

    def __init__(
        self,
        train_urls: str = "",
        val_urls: Optional[str] = None,
        test_urls: Optional[str] = None,
        num_classes: int = 10,
        num_train_samples: int = 50000,
        num_val_samples: Optional[int] = None,
        num_test_samples: Optional[int] = None,
        image_key: str = "jpg",
        label_key: str = "cls",
        shardshuffle: bool = True,
        shuffle_buffer: int = 1000,
        root_dir: str = "./data",
        img_size: tuple[int, int] = (224, 224),
        img_mode: str = "RGB",
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        norm_mean: Optional[list[float]] = None,
        norm_std: Optional[list[float]] = None,
        pin_memory: bool = True,
        augmentation: str = "none",
    ):
        super().__init__()

        assert img_mode in ["RGB", "L"], "img_mode must be 'RGB' or 'L'"

        if not train_urls:
            raise ValueError("train_urls is required")
        if val_urls is None:
            raise ValueError(
                "val_urls is required for WebDataset (cannot auto-split streaming data). "
                "Provide separate validation shards."
            )

        self.dataset_name = "webdataset"
        self.root_dir = root_dir
        self.train_urls = train_urls
        self.val_urls = val_urls
        self.test_urls = test_urls
        self._num_classes = num_classes
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.image_key = image_key
        self.label_key = label_key
        self.shardshuffle = shardshuffle
        self.shuffle_buffer = shuffle_buffer
        self.img_size = img_size
        self.img_mode = img_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
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


    def _make_pipeline(self, urls: str, transform: Callable, shuffle: bool = False):
        """Build a WebDataset pipeline."""
        wds = _import_webdataset()

        pipeline = wds.WebDataset(urls, shardshuffle=shuffle)

        if shuffle:
            pipeline = pipeline.shuffle(self.shuffle_buffer)

        pipeline = (
            pipeline
            .decode("pil")
            .to_tuple(self.image_key, self.label_key)
            .map_tuple(transform, lambda y: int(y))
        )

        return pipeline

    def setup(self, stage: Optional[str] = None):
        """Set up datasets (no-op for WebDataset — pipelines built in dataloaders)."""
        pass

    @property
    def num_classes(self) -> int:
        """Get the number of classes for the dataset."""
        return self._num_classes

    def train_dataloader(self):
        """Get training dataloader."""
        wds = _import_webdataset()

        pipeline = self._make_pipeline(self.train_urls, self.train_transform, shuffle=True)
        pipeline = pipeline.batched(self.batch_size)

        loader = wds.WebLoader(pipeline, num_workers=self.num_workers, pin_memory=self.pin_memory)
        loader = loader.with_epoch(self.num_train_samples // self.batch_size)

        return loader

    def val_dataloader(self):
        """Get validation dataloader."""
        wds = _import_webdataset()

        pipeline = self._make_pipeline(self.val_urls, self.val_transform, shuffle=False)
        pipeline = pipeline.batched(self.batch_size)

        loader = wds.WebLoader(pipeline, num_workers=self.num_workers, pin_memory=self.pin_memory)
        if self.num_val_samples is not None:
            loader = loader.with_epoch(self.num_val_samples // self.batch_size)

        return loader

    def test_dataloader(self):
        """Get test dataloader."""
        if self.test_urls is None:
            raise RuntimeError("No test_urls provided.")

        wds = _import_webdataset()

        pipeline = self._make_pipeline(self.test_urls, self.val_transform, shuffle=False)
        pipeline = pipeline.batched(self.batch_size)

        loader = wds.WebLoader(pipeline, num_workers=self.num_workers, pin_memory=self.pin_memory)
        if self.num_test_samples is not None:
            loader = loader.with_epoch(self.num_test_samples // self.batch_size)

        return loader
