"""Model loading utilities for config-based instantiation.

Thin wrappers over :mod:`util.instantiate` for notebook-based analysis: load a
model and datamodule from a YAML experiment config, built exactly as the
Lightning CLI (and the Optuna study) would build them.
"""

import torch
from omegaconf import OmegaConf

from util.instantiate import import_class, instantiate_from_config

__all__ = ["import_class", "instantiate_from_config", "load_from_config"]


def load_from_config(config_path: str, checkpoint_path: str | None = None):
    """Load model and datamodule from a config file.

    Args:
        config_path: Path to the experiment YAML config
        checkpoint_path: Optional path to checkpoint file (.ckpt)

    Returns:
        tuple: (model/task, datamodule)

    Example:
        >>> model, dm = load_from_config("conf/experiment/cifar10.yaml")
        >>> model.eval()
        >>> dm.setup("test")
        >>> batch = next(iter(dm.test_dataloader()))
    """
    config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(config, resolve=True)

    if not isinstance(config_dict, dict):
        raise ValueError("Config file must contain a dictionary at the top level.")

    # Instantiate model/task and datamodule (same resolution as the CLI/study).
    model = instantiate_from_config(config_dict["model"])
    datamodule = instantiate_from_config(config_dict["data"])

    # Load checkpoint weights
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded weights from: {checkpoint_path}")

    return model, datamodule
