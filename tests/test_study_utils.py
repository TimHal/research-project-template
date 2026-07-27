"""Tests for the Optuna study runner's config helpers and parameter suggestion."""

import pytest

optuna = pytest.importorskip("optuna")  # skip the module if optuna isn't installed

from study import _deep_set, _suggest_param  # noqa: E402


def test_deep_set_nested():
    cfg = {"model": {"init_args": {"learning_rate": 0.1}}}
    _deep_set(cfg, "model.init_args.learning_rate", 0.5)
    assert cfg["model"]["init_args"]["learning_rate"] == 0.5


def test_deep_set_invalid_path_raises():
    with pytest.raises(KeyError):
        _deep_set({"model": {}}, "model.init_args.learning_rate", 0.5)


def test_suggest_categorical_returns_actual_value():
    """Categorical params are suggested by value, not by index.

    A FixedTrial fixing the param to 64 only works if the choices passed to
    Optuna are the real values [32, 64, 128]. Under the old index encoding the
    choices were [0, 1, 2] and this would raise.
    """
    trial = optuna.trial.FixedTrial({"batch_size": 64})
    value = _suggest_param(
        trial, "batch_size", {"type": "categorical", "choices": [32, 64, 128]}
    )
    assert value == 64


def test_suggest_float_and_int():
    trial = optuna.trial.FixedTrial({"lr": 1e-3, "layers": 4})
    assert _suggest_param(trial, "lr", {"type": "float", "low": 1e-5, "high": 1e-2}) == 1e-3
    assert _suggest_param(trial, "layers", {"type": "int", "low": 1, "high": 8}) == 4
