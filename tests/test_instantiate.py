"""Tests for the shared config instantiation helpers.

`fit`-style loading (notebooks) and the Optuna study go through the same code
path here, so they must resolve configs identically.
"""

import torch

from util.instantiate import (
    import_class,
    instantiate,
    instantiate_from_config,
    resolve_init_args,
    resolve_value,
)


def test_nested_class_path_instantiated():
    task = instantiate_from_config(
        {
            "class_path": "task.ClassificationTask",
            "init_args": {
                "num_classes": 2,
                "model": {"class_path": "torch.nn.Flatten", "init_args": {}},
            },
        }
    )
    assert isinstance(task.model, torch.nn.Flatten)
    assert task.num_classes == 2


def test_bare_class_reference_resolved():
    """`optimizer_class`/`scheduler_class` given as dotted strings become classes.

    This is the divergence the shared instantiator fixes: previously the
    notebook loader passed these through as raw strings, so the loaded task
    would fail in `configure_optimizers`.
    """
    task = instantiate_from_config(
        {
            "class_path": "task.ClassificationTask",
            "init_args": {
                "num_classes": 2,
                "model": {"class_path": "torch.nn.Flatten", "init_args": {}},
                "optimizer_class": "torch.optim.SGD",
                "scheduler_class": "torch.optim.lr_scheduler.StepLR",
            },
        }
    )
    assert task.optimizer_class is torch.optim.SGD
    assert task.scheduler_class is torch.optim.lr_scheduler.StepLR


def test_import_class():
    assert import_class("torch.nn.Linear") is torch.nn.Linear


def test_instantiate_resolves_init_args():
    layer = instantiate("torch.nn.Linear", {"in_features": 4, "out_features": 2})
    assert isinstance(layer, torch.nn.Linear)
    assert (layer.in_features, layer.out_features) == (4, 2)


def test_resolve_value_leaves_ordinary_strings_untouched():
    # File paths, URIs, and dotted data values must NOT be treated as imports.
    for value in [
        "./data/my_dataset/croissant.json",
        "http://example.com/data",
        "0.001",
        "resnet50.a1_in1k",  # dotted identifiers, but not importable
        "CIFAR10",
    ]:
        assert resolve_value(value) == value


def test_resolve_value_ignores_non_callable_imports():
    # Importable but not a class/callable -> stays a string.
    assert resolve_value("torch.pi") == "torch.pi"


def test_resolve_init_args_passes_through_trainer_settings():
    # Plain trainer settings survive unchanged (parity with `fit`); nested
    # class configs get instantiated.
    resolved = resolve_init_args(
        {
            "max_epochs": 10,
            "accumulate_grad_batches": 4,
            "precision": "16-mixed",
            "num_sanity_val_steps": 0,
        }
    )
    assert resolved == {
        "max_epochs": 10,
        "accumulate_grad_batches": 4,
        "precision": "16-mixed",
        "num_sanity_val_steps": 0,
    }
