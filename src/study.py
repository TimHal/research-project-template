"""Optuna hyperparameter optimization for Lightning experiments.

Runs an Optuna study using a base experiment config and a study config
that defines the search space, sampler, pruner, and optimization settings.

Usage:
    PYTHONPATH=src python src/cli.py study \
        --config conf/experiment/cifar10_classification.yaml \
        --study_config conf/study/example_cifar10_classification.yaml

    # Continue an existing study with more trials:
    PYTHONPATH=src python src/cli.py study \
        --config conf/experiment/cifar10_classification.yaml \
        --study_config conf/study/example_cifar10_classification.yaml \
        --n_trials 50
"""

import argparse
import copy
import gc
import os
import sys
import tempfile
import traceback
from pathlib import Path

import optuna
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from omegaconf import OmegaConf
from optuna.integration import PyTorchLightningPruningCallback

from util.instantiate import instantiate, resolve_init_args


def _deep_set(d: dict, dotpath: str, value) -> None:
    """Set a value in a nested dict using dot-separated keys.

    Example: _deep_set(cfg, "model.init_args.learning_rate", 0.001)
    """
    keys = dotpath.split(".")
    for i, key in enumerate(keys[:-1]):
        if key not in d:
            traversed = ".".join(keys[: i + 1])
            raise KeyError(f"Key '{key}' not found at '{traversed}' in config. Check your search_space/overrides paths.")
        d = d[key]
    d[keys[-1]] = value


def _suggest_param(trial: optuna.Trial, name: str, spec: dict):
    """Suggest a parameter value from an Optuna trial based on the spec."""
    ptype = spec["type"]
    if ptype == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    elif ptype == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
    elif ptype == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    else:
        raise ValueError(f"Unknown search space type: {ptype}")


def _resolve_objectives(study_settings: dict) -> tuple[list[str], list[str]]:
    """Resolve metric(s) and direction(s) from study config.

    Single-objective: metric + direction
    Multi-objective:  metrics + directions
    """
    if "metrics" in study_settings and "directions" in study_settings:
        return study_settings["metrics"], study_settings["directions"]
    return [study_settings["metric"]], [study_settings["direction"]]


def _build_callbacks(study_cfg: dict, trial: optuna.Trial, metrics: list[str], directions: list[str]) -> list:
    """Build pruning + optional early stopping callbacks."""
    monitor = metrics[0]
    callbacks = [PyTorchLightningPruningCallback(trial, monitor=monitor)]

    es_cfg = study_cfg["study"].get("early_stopping")
    if es_cfg:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                patience=es_cfg.get("patience", 10),
                mode="min" if directions[0] == "minimize" else "max",
                verbose=False,
            )
        )
    return callbacks


def _build_trial_logger(trainer_cfg: dict, study_name: str, trial_number: int):
    """Build a logger for a trial, re-using the experiment config's settings.

    Supports MLFlowLogger and WandbLogger. Returns False if no supported
    logger is configured.
    """
    logger_cfg = trainer_cfg.get("logger")
    if not isinstance(logger_cfg, dict):
        return False

    class_path = logger_cfg.get("class_path", "")
    init_args = dict(logger_cfg.get("init_args", {}))
    run_name = f"{study_name}_trial_{trial_number:04d}"

    if "MLFlowLogger" in class_path:
        from lightning.pytorch.loggers import MLFlowLogger

        init_args["run_name"] = run_name
        init_args["log_model"] = False
        return MLFlowLogger(**init_args)

    if "WandbLogger" in class_path:
        from lightning.pytorch.loggers import WandbLogger

        init_args["name"] = run_name
        init_args["group"] = study_name
        init_args["reinit"] = True
        init_args.pop("log_model", None)
        return WandbLogger(**init_args)

    return False


def _build_storage(study_cfg: dict):
    """Create study storage for persistent/distributed studies.

    Supports two modes:
    - storage_dir (default): JournalStorage with append-only journal files,
      safe for concurrent multi-process writes.
    - storage_url: Direct Optuna storage URL (e.g. sqlite:///path/to/db.sqlite3)

    Returns None if neither is configured (in-memory study).
    """
    storage_url = study_cfg["study"].get("storage_url")
    if storage_url:
        return optuna.storages.RDBStorage(storage_url)

    storage_dir = study_cfg["study"].get("storage_dir")
    if not storage_dir:
        return None

    study_name = study_cfg["study"].get("name", "hpo_study")
    os.makedirs(storage_dir, exist_ok=True)
    journal_path = os.path.join(storage_dir, f"{study_name}.journal")
    lock_path = journal_path + ".lock"
    return optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            journal_path,
            lock_obj=optuna.storages.journal.JournalFileOpenLock(lock_path),
        )
    )


def _build_sampler(study_cfg: dict) -> optuna.samplers.BaseSampler:
    """Build an Optuna sampler from the study config."""
    sampler_name = study_cfg["study"].get("sampler", "TPESampler")
    sampler_map = {
        "TPESampler": optuna.samplers.TPESampler,
        "RandomSampler": optuna.samplers.RandomSampler,
        "CmaEsSampler": optuna.samplers.CmaEsSampler,
        "NSGAIISampler": optuna.samplers.NSGAIISampler,
    }
    sampler_cls = sampler_map.get(sampler_name)
    if sampler_cls is None:
        raise ValueError(f"Unknown sampler: {sampler_name}. Choose from: {list(sampler_map.keys())}")
    return sampler_cls()


def _build_pruner(study_cfg: dict) -> optuna.pruners.BasePruner:
    """Build an Optuna pruner from the study config."""
    pruner_cfg = study_cfg["study"].get("pruner")
    if pruner_cfg is None:
        return optuna.pruners.MedianPruner()

    pruner_map = {
        "MedianPruner": optuna.pruners.MedianPruner,
        "PercentilePruner": optuna.pruners.PercentilePruner,
        "NopPruner": optuna.pruners.NopPruner,
        "SuccessiveHalvingPruner": optuna.pruners.SuccessiveHalvingPruner,
        "HyperbandPruner": optuna.pruners.HyperbandPruner,
    }
    pruner_cls = pruner_map.get(pruner_cfg["type"])
    if pruner_cls is None:
        raise ValueError(f"Unknown pruner: {pruner_cfg['type']}. Choose from: {list(pruner_map.keys())}")

    kwargs = {k: v for k, v in pruner_cfg.items() if k != "type"}
    return pruner_cls(**kwargs)


def _log_trial_metadata(trainer, trial: optuna.Trial, study_name: str, cfg: dict, error: str | None = None) -> None:
    """Log Optuna trial metadata to the active logger (MLflow or W&B)."""
    if trainer.logger is False or trainer.logger is None:
        return

    logger_name = type(trainer.logger).__name__
    if "MLFlow" in logger_name:
        _log_trial_to_mlflow(trainer, trial, study_name, cfg, error)
    elif "Wandb" in logger_name:
        _log_trial_to_wandb(trainer, trial, study_name, cfg, error)


def _log_trial_to_mlflow(trainer, trial: optuna.Trial, study_name: str, cfg: dict, error: str | None = None) -> None:
    """Log Optuna params, config artifact, and optional error to MLflow."""
    mlf = trainer.logger.experiment
    run_id = trainer.logger.run_id

    # Log Optuna-suggested params with hpo/ prefix
    for param_name, param_value in trial.params.items():
        mlf.log_param(run_id, f"hpo/{param_name}", param_value)
    mlf.set_tag(run_id, "hpo_study", study_name)
    mlf.set_tag(run_id, "hpo_trial", str(trial.number))

    if error:
        mlf.set_tag(run_id, "hpo_status", "FAILED")
        with tempfile.TemporaryDirectory() as tmp_dir:
            error_path = Path(tmp_dir) / "trial_error.txt"
            error_path.write_text(error)
            mlf.log_artifact(local_path=str(error_path), run_id=run_id)
    else:
        mlf.set_tag(run_id, "hpo_status", "OK")

    # Log full resolved config as artifact
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "config.yaml"
        config_path.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg)))
        mlf.log_artifact(local_path=str(config_path), run_id=run_id)


def _log_trial_to_wandb(trainer, trial: optuna.Trial, study_name: str, cfg: dict, error: str | None = None) -> None:
    """Log Optuna params, config artifact, and optional error to W&B."""
    import wandb

    run = trainer.logger.experiment

    # Log Optuna-suggested params with hpo/ prefix
    for param_name, param_value in trial.params.items():
        run.config[f"hpo/{param_name}"] = param_value
    run.config["hpo_study"] = study_name
    run.config["hpo_trial"] = trial.number

    if error:
        run.summary["hpo_status"] = "FAILED"
        with tempfile.TemporaryDirectory() as tmp_dir:
            error_path = Path(tmp_dir) / "trial_error.txt"
            error_path.write_text(error)
            artifact = wandb.Artifact(f"trial_{trial.number:04d}_error", type="error")
            artifact.add_file(str(error_path))
            run.log_artifact(artifact)
    else:
        run.summary["hpo_status"] = "OK"

    # Log full resolved config as artifact
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "config.yaml"
        config_path.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg)))
        artifact = wandb.Artifact(f"trial_{trial.number:04d}_config", type="config")
        artifact.add_file(str(config_path))
        run.log_artifact(artifact)


def run_study(base_config_path: str, study_config_path: str, n_trials_override: int | None = None) -> None:
    """Run an Optuna hyperparameter optimization study.

    Supports single-objective and multi-objective optimization.
    Uses JournalStorage for safe concurrent multi-process execution.
    Automatically continues an existing study if one exists with the same name.
    """
    base_cfg: dict = OmegaConf.to_container(OmegaConf.load(base_config_path), resolve=True)  # type: ignore[assignment]
    study_cfg: dict = OmegaConf.to_container(OmegaConf.load(study_config_path), resolve=True)  # type: ignore[assignment]

    study_settings = study_cfg["study"]
    search_space = study_cfg["search_space"]
    overrides = study_cfg.get("overrides", {})
    metrics, directions = _resolve_objectives(study_settings)
    multi_objective = len(metrics) > 1
    n_trials = n_trials_override if n_trials_override is not None else int(study_settings["n_trials"])

    study_name = study_settings.get("name", "hpo_study")
    storage = _build_storage(study_cfg)

    create_kwargs: dict = dict(
        study_name=study_name,
        sampler=_build_sampler(study_cfg),
        pruner=_build_pruner(study_cfg),
        storage=storage,
        load_if_exists=True,
    )
    if multi_objective:
        create_kwargs["directions"] = directions
    else:
        create_kwargs["direction"] = directions[0]

    study = optuna.create_study(**create_kwargs)

    # Report status
    existing_trials = len(study.trials)
    is_continuation = existing_trials > 0
    storage_dir = study_settings.get("storage_dir")
    storage_info = os.path.join(storage_dir, f"{study_name}.journal") if storage_dir else "in-memory"

    if is_continuation:
        completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        print(f"Continuing study: {study_name} ({existing_trials} existing trials, {completed} completed)")
    else:
        print(f"Creating study: {study_name}")
    print(f"  Direction(s): {directions}")
    print(f"  Metric(s): {metrics}")
    print(f"  Trials: {n_trials} (this worker)")
    print(f"  Storage: {storage_info}")
    print(f"  Search space: {list(search_space.keys())}")
    if overrides:
        print(f"  Overrides: {overrides}")
    print()

    if n_trials == 0:
        print("n_trials=0, study created/loaded. Exiting.")
        return

    def objective(trial: optuna.Trial) -> float | list[float]:
        cfg = copy.deepcopy(base_cfg)

        # Apply fixed overrides
        for dotpath, value in overrides.items():
            _deep_set(cfg, dotpath, value)

        # Suggest and apply hyperparameters
        for param_path, spec in search_space.items():
            value = _suggest_param(trial, param_path, spec)
            _deep_set(cfg, param_path, value)

        seed = cfg.get("seed_everything", 42)
        seed_everything(seed, workers=True)

        model = instantiate(cfg["model"]["class_path"], cfg["model"].get("init_args", {}))
        datamodule = instantiate(cfg["data"]["class_path"], cfg["data"].get("init_args", {}))

        # Build the Trainer from the *full* trainer config so HPO trials train
        # under the same conditions as `fit` (accumulate_grad_batches,
        # num_sanity_val_steps, gradient_clip_val, strategy, precision, ...).
        # Only logger, callbacks, and checkpointing are managed by the study:
        # each trial gets a pruning callback and its own run, and checkpointing
        # is disabled to avoid writing artifacts for every trial.
        trainer_cfg = cfg.get("trainer", {})
        trainer_kwargs = resolve_init_args(
            {k: v for k, v in trainer_cfg.items() if k not in ("logger", "callbacks")}
        )
        trainer_kwargs["callbacks"] = _build_callbacks(study_cfg, trial, metrics, directions)
        trainer_kwargs["enable_checkpointing"] = False
        trainer_kwargs["logger"] = _build_trial_logger(trainer_cfg, study_name, trial.number)

        trainer = Trainer(**trainer_kwargs)
        result = None

        try:
            try:
                trainer.fit(model, datamodule=datamodule)
            except optuna.TrialPruned:
                _log_trial_metadata(trainer, trial, study_name, cfg)
                raise
            except Exception as e:
                error_msg = traceback.format_exc()
                print(f"Trial {trial.number} FAILED: {e}")
                trial.set_user_attr("error", error_msg)
                _log_trial_metadata(trainer, trial, study_name, cfg, error=error_msg)
                raise optuna.TrialPruned() from None

            _log_trial_metadata(trainer, trial, study_name, cfg)

            # Extract metrics before cleanup
            callback_metrics = trainer.callback_metrics
            values = []
            for m in metrics:
                if m not in callback_metrics:
                    raise ValueError(f"Metric '{m}' not found. Available: {list(callback_metrics.keys())}")
                values.append(callback_metrics[m].item())

            result = values if multi_objective else values[0]
        finally:
            # Clean up to prevent file descriptor exhaustion across trials
            if hasattr(trainer, "logger") and hasattr(trainer.logger, "finalize"):
                try:
                    trainer.logger.finalize("success")
                except Exception:
                    pass
            # W&B requires explicit run finalization between trials
            if hasattr(trainer, "logger") and "Wandb" in type(trainer.logger).__name__:
                try:
                    import wandb

                    wandb.finish(quiet=True)
                except Exception:
                    pass
            del trainer, model, datamodule
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return result

    study.optimize(objective, n_trials=n_trials)

    # Print results
    print("\n" + "=" * 60)
    print("STUDY COMPLETE")
    print("=" * 60)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"Total trials: {len(study.trials)} ({len(completed)} completed, {len(pruned)} pruned, {len(failed)} failed)")

    if not completed:
        print("No completed trials.")
        return

    if multi_objective:
        pareto = study.best_trials
        print(f"\nPareto-optimal trials: {len(pareto)}")
        for t in pareto:
            vals = ", ".join(f"{m}={v:.6f}" for m, v in zip(metrics, t.values, strict=True))
            print(f"  Trial #{t.number}: {vals}")
            for key, value in t.params.items():
                print(f"    {key}: {value}")
    else:
        best = study.best_trial

        print(f"\nBest trial: #{best.number}")
        print(f"Best value ({metrics[0]}): {study.best_value:.6f}")
        print("Best params:")
        for key, value in best.params.items():
            print(f"  {key}: {value}")

        override_args = [f"  --{k}={v}" for k, v in best.params.items()]
        print("\nTo train with the best params, run:")
        print(f"  PYTHONPATH=src python src/cli.py fit --config {base_config_path} \\")
        print(" \\\n".join(override_args))


def parse_study_args() -> tuple[str, str, int | None]:
    """Parse CLI arguments for the study command."""
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter study")
    parser.add_argument("--config", required=True, help="Base experiment config path")
    parser.add_argument("--study_config", required=True, help="Study config path")
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Override number of trials (use to extend an existing study)",
    )
    argv = [a for a in sys.argv[1:] if a != "study"]
    args = parser.parse_args(argv)
    return args.config, args.study_config, args.n_trials


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    base_path, study_path, n_trials = parse_study_args()
    run_study(base_path, study_path, n_trials_override=n_trials)
