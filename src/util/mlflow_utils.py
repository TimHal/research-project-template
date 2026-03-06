"""MLflow utilities for resolving experiment paths and artifacts.

This module provides utilities to dynamically resolve checkpoint and config paths
from MLflow experiments, eliminating the need for hardcoded paths in notebooks.

Example:
    >>> from util.mlflow_utils import resolve_experiment_path
    >>> conf_path, ckpt_path = resolve_experiment_path(
    ...     experiment="My-Experiment",
    ...     run="my_run_2024-01-15_10_30"
    ... )
"""

import os
from glob import glob
from pathlib import Path
from typing import Literal, Optional

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

# Default tracking URI (can be overridden via MLFLOW_TRACKING_URI env var)
DEFAULT_TRACKING_URI = "http://localhost:8090"


def resolve_experiment_path(
    experiment: str,
    run: str,
    checkpoint: Literal["best", "latest", "only"] | str = "only",
    tracking_uri: Optional[str] = None,
    download_dir: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve config.yaml and checkpoint paths from MLflow experiment.

    Args:
        experiment: MLflow experiment name
        run: MLflow run name
        checkpoint: Checkpoint selection strategy:
            - "best": Select checkpoint with best validation metric
            - "latest": Select most recent checkpoint
            - "only": Auto-select if there's only one checkpoint
            - Or provide exact checkpoint name
        tracking_uri: MLflow tracking URI. Defaults to env var or localhost.
        download_dir: Directory to download artifacts to.

    Returns:
        tuple[str, str]: (config_path, checkpoint_path)

    Raises:
        ValueError: If experiment/run not found
        FileNotFoundError: If config or checkpoint not found
    """
    # Set up MLflow client
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Find the experiment
    experiment_obj = client.get_experiment_by_name(experiment)
    if experiment_obj is None:
        available = [e.name for e in client.search_experiments()]
        raise ValueError(
            f"Experiment '{experiment}' not found. Available: {available}"
        )

    # Find the run by name
    runs = client.search_runs(
        experiment_ids=[experiment_obj.experiment_id],
        filter_string=f"attributes.run_name = '{run}'",
        max_results=1,
    )

    if not runs:
        raise ValueError(f"Run '{run}' not found in experiment '{experiment}'")

    run_obj = runs[0]

    # Download config.yaml
    config_path = _download_artifact(client, run_obj, "config.yaml", download_dir)

    # Find and download checkpoint
    checkpoint_path = _resolve_checkpoint(client, run_obj, checkpoint, download_dir)

    return config_path, checkpoint_path


def _download_artifact(
    client: MlflowClient,
    run: Run,
    artifact_path: str,
    download_dir: Optional[str] = None,
) -> str:
    """Download an artifact and return its local path."""
    try:
        if download_dir:
            local_path = client.download_artifacts(
                run.info.run_id, artifact_path, dst_path=download_dir
            )
        else:
            local_path = client.download_artifacts(run.info.run_id, artifact_path)

        return str(Path(local_path).resolve())
    except Exception as e:
        raise FileNotFoundError(
            f"Artifact '{artifact_path}' not found in run. Error: {e}"
        ) from e


def _resolve_checkpoint(
    client: MlflowClient,
    run: Run,
    checkpoint: str,
    download_dir: Optional[str] = None,
) -> str:
    """Resolve and download checkpoint based on selection strategy."""
    # Find all checkpoint files
    artifact_uri = run.info.artifact_uri
    if artifact_uri.startswith("file://"):
        artifact_uri = artifact_uri.replace("file://", "")

    ckpt_files = glob("**/*.ckpt", root_dir=artifact_uri, recursive=True)

    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints found in run {run.info.run_name}")

    if checkpoint == "only":
        if len(ckpt_files) == 1:
            return _download_artifact(client, run, ckpt_files[0], download_dir)
        raise ValueError(
            f"Multiple checkpoints found ({len(ckpt_files)}). "
            f"Specify 'best', 'latest', or exact name. Available: {ckpt_files}"
        )

    elif checkpoint == "latest":
        return _download_artifact(
            client, run, _select_latest_checkpoint(ckpt_files), download_dir
        )

    elif checkpoint == "best":
        return _download_artifact(
            client, run, _select_best_checkpoint(ckpt_files), download_dir
        )

    else:
        # Exact match or partial match
        if checkpoint in ckpt_files:
            return _download_artifact(client, run, checkpoint, download_dir)

        matches = [c for c in ckpt_files if checkpoint in c]
        if len(matches) == 1:
            return _download_artifact(client, run, matches[0], download_dir)
        elif len(matches) > 1:
            raise ValueError(f"Ambiguous checkpoint '{checkpoint}'. Matches: {matches}")
        else:
            raise ValueError(
                f"Checkpoint '{checkpoint}' not found. Available: {ckpt_files}"
            )


def _select_latest_checkpoint(ckpt_files: list[str]) -> str:
    """Select checkpoint with highest epoch/step number."""
    import re

    def extract_numbers(path: str) -> tuple[int, int]:
        epoch_match = re.search(r"epoch=(\d+)", path)
        step_match = re.search(r"step=(\d+)", path)
        epoch = int(epoch_match.group(1)) if epoch_match else -1
        step = int(step_match.group(1)) if step_match else -1
        return (epoch, step)

    return max(ckpt_files, key=extract_numbers)


def _select_best_checkpoint(ckpt_files: list[str]) -> str:
    """Select checkpoint with 'best' in name, or fall back to latest."""
    best_ckpts = [c for c in ckpt_files if "best" in c.lower()]
    if best_ckpts:
        return best_ckpts[0]

    last_ckpts = [c for c in ckpt_files if "last" in c.lower()]
    if last_ckpts:
        return last_ckpts[0]

    return _select_latest_checkpoint(ckpt_files)


def list_experiments(tracking_uri: Optional[str] = None) -> list[str]:
    """List all available MLflow experiments.

    Args:
        tracking_uri: MLflow tracking URI

    Returns:
        List of experiment names
    """
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiments = client.search_experiments()
    return [e.name for e in experiments if e.name != "Default"]


def list_runs(experiment: str, tracking_uri: Optional[str] = None) -> list[str]:
    """List all run names in an experiment.

    Args:
        experiment: Experiment name
        tracking_uri: MLflow tracking URI

    Returns:
        List of run names
    """
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment_obj = client.get_experiment_by_name(experiment)
    if experiment_obj is None:
        raise ValueError(f"Experiment '{experiment}' not found")

    runs = client.search_runs(experiment_ids=[experiment_obj.experiment_id])
    return [run.info.run_name for run in runs]
