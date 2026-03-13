"""Data modules for PyTorch Lightning training."""

from data.ImageFolderDatamodule import ImageFolderDatamodule
from data.TorchvisionDatamodule import TorchvisionDatamodule

# Modules with optional dependencies are loaded on demand via class_path in YAML.
# They do not need to be imported here.

__all__ = [
    "CroissantDatamodule",
    "HuggingFaceDatamodule",
    "ImageFolderDatamodule",
    "KaggleDatamodule",
    "ParquetDatamodule",
    "TorchvisionDatamodule",
    "WebDatasetDatamodule",
]
