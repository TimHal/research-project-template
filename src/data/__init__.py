"""DataModules for PyTorch Lightning training.

Every DataModule is re-exported here so experiment configs can reference it by
the short path ``data.<ClassName>`` (e.g. ``data.TorchvisionDataModule``).
Optional third-party dependencies (datasets, webdataset, mlcroissant, kaggle,
pyarrow) are imported lazily *inside* each DataModule, so importing this
package never requires them.
"""

from data.croissant_datamodule import CroissantDataModule
from data.huggingface_datamodule import HuggingFaceDataModule
from data.imagefolder_datamodule import ImageFolderDataModule
from data.kaggle_datamodule import KaggleDataModule
from data.parquet_datamodule import ParquetDataModule
from data.torchvision_datamodule import TorchvisionDataModule
from data.webdataset_datamodule import WebDatasetDataModule

__all__ = [
    "CroissantDataModule",
    "HuggingFaceDataModule",
    "ImageFolderDataModule",
    "KaggleDataModule",
    "ParquetDataModule",
    "TorchvisionDataModule",
    "WebDatasetDataModule",
]
