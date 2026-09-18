"""Default location for the files a run produces, kept out of the source tree.

Lightning resolves checkpoint paths against ``Trainer.default_root_dir``, which
defaults to the *current working directory*. With a remote MLflow tracking URI
``MLFlowLogger.save_dir`` is ``None``, so a run started from the repo root would
write ``{experiment_id}/{run_id}/checkpoints/`` straight into the project. The
CLI defaults ``default_root_dir`` to :func:`output_path` instead.

This is only the *default*. Override it per experiment in the config, which is
where a path that matters to a run belongs::

    trainer:
      default_root_dir: /data/<user>/runs

Change :data:`DEFAULT_OUTPUT_ROOT` when you fork this template, so several
projects do not share one directory.
"""

from pathlib import Path

DEFAULT_OUTPUT_ROOT = Path.home() / ".cache" / "research-project"


def output_path(*parts: str) -> str:
    """Build a default path under :data:`DEFAULT_OUTPUT_ROOT`.

    Args:
        *parts: Path segments appended to the root, e.g. ``"runs"``.

    Returns:
        The joined path as a string, which is what configs, argparse defaults
        and Lightning all expect. The directory is not created here; callers
        create the one they actually use.
    """
    return str(DEFAULT_OUTPUT_ROOT.joinpath(*parts))
