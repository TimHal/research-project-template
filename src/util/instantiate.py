"""Canonical config instantiation, shared by the notebook loader and the study.

Lightning's CLI (``fit``, ``test``, ...) builds objects from
``{class_path, init_args}`` config blocks and resolves bare dotted class
references for ``type[...]``-annotated arguments
(e.g. ``optimizer_class: torch.optim.AdamW``). This module reproduces that
behavior so that instantiating a config *outside* the CLI — in a notebook via
``load_from_config``, or per-trial in an Optuna study — yields the same objects
the CLI would build. Keeping a single implementation here prevents ``fit`` and
``study`` from silently diverging.
"""

import importlib
from typing import Any, Optional


def import_class(class_path: str):
    """Import a class or callable from a dotted path, e.g. 'torch.optim.AdamW'."""
    module_path, name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, name)


def _looks_like_class_path(value: str) -> bool:
    """True if ``value`` is a dotted path of identifiers (a plausible class ref).

    Guards against treating ordinary string values as import targets: file
    paths (``./data/x.json``), URIs (``http://host``), dotted data values
    (``0.001``, ``resnet50.a1_in1k``), etc. do not qualify.
    """
    if "." not in value:
        return False
    return all(part.isidentifier() for part in value.split("."))


def resolve_value(value: Any) -> Any:
    """Resolve a single config value to a runtime object.

    - ``{class_path, init_args}``   -> instantiated object (recursively)
    - ``"pkg.mod.Class"`` (type ref) -> the imported class/callable
    - anything else                 -> returned unchanged
    """
    if isinstance(value, dict) and "class_path" in value:
        return instantiate(value["class_path"], value.get("init_args", {}))

    if isinstance(value, str) and _looks_like_class_path(value):
        try:
            obj = import_class(value)
        except (ImportError, AttributeError, ValueError):
            return value
        # Only substitute genuine class/callable references — not modules or
        # plain constants that happen to be importable (e.g. "torch.pi").
        return obj if callable(obj) else value

    return value


def resolve_init_args(init_args: dict) -> dict:
    """Resolve every value in an init_args dict (see :func:`resolve_value`)."""
    return {key: resolve_value(value) for key, value in init_args.items()}


def instantiate(class_path: str, init_args: Optional[dict] = None):
    """Instantiate ``class_path`` with resolved ``init_args``."""
    cls = import_class(class_path)
    return cls(**resolve_init_args(init_args or {}))


def instantiate_from_config(config: Any):
    """Instantiate a top-level ``{class_path, init_args}`` config block.

    Values that are not class configs (no ``class_path``) are returned
    unchanged.
    """
    if not isinstance(config, dict) or "class_path" not in config:
        return config
    return instantiate(config["class_path"], config.get("init_args", {}))
