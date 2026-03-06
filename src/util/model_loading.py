"""Model loading utilities for config-based instantiation.

This module provides utilities for loading models and datamodules
from YAML configuration files, useful for notebook-based analysis.
"""

import importlib

import torch
from omegaconf import OmegaConf


def import_class(class_path: str):
    """Import a class from a dotted path string.

    Args:
        class_path: Dotted path to class (e.g., 'model.TIMMModel.TIMMModel')

    Returns:
        The imported class
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def instantiate_from_config(config_dict: dict):
    """Recursively instantiate a class from a config dict with class_path/init_args.

    Args:
        config_dict: Dict with 'class_path' and 'init_args' keys

    Returns:
        Instantiated object

    Example:
        >>> config = {
        ...     "class_path": "model.TIMMModel.TIMMModel",
        ...     "init_args": {"model_name": "resnet18", "num_classes": 10}
        ... }
        >>> model = instantiate_from_config(config)
    """
    if not isinstance(config_dict, dict):
        return config_dict

    if "class_path" not in config_dict:
        # Not a class config, return as-is
        return config_dict

    class_path = config_dict["class_path"]
    init_args = config_dict.get("init_args", {})

    # Recursively instantiate nested class configs
    processed_args = {}
    for key, value in init_args.items():
        if isinstance(value, dict) and "class_path" in value:
            processed_args[key] = instantiate_from_config(value)
        else:
            processed_args[key] = value

    # Import and instantiate the class
    cls = import_class(class_path)
    return cls(**processed_args)


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

    # Instantiate model/task
    model = instantiate_from_config(config_dict["model"])

    # Instantiate datamodule
    datamodule = instantiate_from_config(config_dict["data"])

    # Load checkpoint weights
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded weights from: {checkpoint_path}")

    return model, datamodule
