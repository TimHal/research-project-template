"""Default locations for the files a run produces, kept out of the source tree.

Lightning resolves checkpoint paths against ``Trainer.default_root_dir``, which
defaults to the *current working directory*. With a remote MLflow tracking URI
``MLFlowLogger.save_dir`` is ``None``, so a run started from the repo root writes
``{experiment_id}/{run_id}/checkpoints/`` straight into the project. Everything
that needs a default location for generated files goes through here instead.

Set ``$OUTPUT_DIR`` to move all of it somewhere else, e.g. a scratch disk::

    export OUTPUT_DIR=/data/<user>/runs

When you fork this template, rename :data:`ENV_VAR` and :data:`DEFAULT_ROOT` to
something project-specific so several projects can coexist on one machine.
"""

import os
from pathlib import Path

ENV_VAR = "OUTPUT_DIR"
DEFAULT_ROOT = Path.home() / ".cache" / "research-project"


def output_root() -> Path:
    """Root for generated files, from ``$OUTPUT_DIR`` or ``~/.cache/research-project``.

    Returns:
        The root directory. It is not created here; callers create the
        subdirectory they actually use.
    """
    return Path(os.environ.get(ENV_VAR) or DEFAULT_ROOT).expanduser()


def output_path(*parts: str) -> str:
    """Build a path under :func:`output_root`.

    Args:
        *parts: Path segments appended to the root, e.g. ``"runs"``.

    Returns:
        The joined path as a string, which is what configs, argparse defaults
        and Lightning all expect.
    """
    return str(output_root().joinpath(*parts))
