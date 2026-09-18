"""Shared pytest fixtures for the test suite.

Fixtures defined here are available to every test file without importing.
Keep them small, fast, and dependency-light so the suite runs in seconds on CPU.
"""

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from util import paths

# Small, fixed problem size shared across tests. Tiny on purpose: fast on CPU,
# no dataset downloads, no GPU.
NUM_CLASSES = 3
IMG_SHAPE = (3, 8, 8)  # (channels, height, width)


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Keep every test hermetic — nothing escapes a per-test temp directory.

    Applied automatically to all tests. It:

    - Runs each test in a throwaway working directory, so any *relative* output
      (``lightning_logs/``, ``mlruns/``, checkpoints, ``./sweeps``, ``./data``)
      lands in ``tmp_path``, which pytest deletes afterwards.
    - Repoints the default output root at ``tmp_path``, so tests that fall
      back to it never write into the developer's real home directory.
    - Redirects experiment trackers and caches to temp/offline, so tests never
      touch a real MLflow server, W&B account, or shared cache — even if a new
      test configures a logger.

    Config discovery still works: configs are found via absolute paths.
    """
    monkeypatch.chdir(tmp_path)

    # Generated run outputs -> temp, never the developer's real output root.
    monkeypatch.setattr(paths, "DEFAULT_OUTPUT_ROOT", tmp_path / "outputs")

    # MLflow -> local temp file store, never a real tracking server.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())

    # Weights & Biases -> disabled (no files, no network, no login).
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))

    # HuggingFace caches -> temp (guards any accidental download).
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    yield


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def num_classes() -> int:
    """Number of classes used by the tiny model / fake data fixtures."""
    return NUM_CLASSES


@pytest.fixture
def tiny_classifier() -> nn.Module:
    """A minimal nn.Module classifier for fast, download-free tests.

    Stand-in for a real architecture (TIMMModel, TorchvisionModel, ...) that
    would otherwise pull pretrained weights over the network.
    """
    in_features = IMG_SHAPE[0] * IMG_SHAPE[1] * IMG_SHAPE[2]
    return nn.Sequential(nn.Flatten(), nn.Linear(in_features, NUM_CLASSES))


@pytest.fixture
def fake_image_loader() -> DataLoader:
    """A DataLoader yielding (image, label) batches of random data.

    Matches the (x, y) batch format the tasks expect, so it can drive a
    fast_dev_run without any real dataset.
    """
    n = 4
    images = torch.randn(n, *IMG_SHAPE)
    labels = torch.randint(0, NUM_CLASSES, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=2, num_workers=0)
