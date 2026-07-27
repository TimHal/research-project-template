"""Tests for the config-based instantiation helpers used by notebooks."""

import pytest
import torch.nn as nn

from util.model_loading import import_class, instantiate_from_config


def test_import_class_returns_class():
    assert import_class("torch.nn.Linear") is nn.Linear


def test_import_class_invalid_raises():
    with pytest.raises((ImportError, AttributeError, ModuleNotFoundError, ValueError)):
        import_class("torch.nn.DefinitelyNotAClass")


def test_instantiate_from_config_simple():
    obj = instantiate_from_config(
        {
            "class_path": "torch.nn.Linear",
            "init_args": {"in_features": 4, "out_features": 2},
        }
    )
    assert isinstance(obj, nn.Linear)
    assert obj.in_features == 4
    assert obj.out_features == 2


def test_instantiate_from_config_nested():
    """Nested class_path/init_args (task wrapping a model) resolve recursively."""
    task = instantiate_from_config(
        {
            "class_path": "task.ClassificationTask",
            "init_args": {
                "num_classes": 3,
                "model": {"class_path": "torch.nn.Flatten", "init_args": {}},
            },
        }
    )
    assert isinstance(task.model, nn.Flatten)
    assert task.num_classes == 3
