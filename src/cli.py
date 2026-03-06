"""Research CLI - Command Line Interface for PyTorch Lightning experiments.

This CLI provides a flexible interface for running experiments with:
- Automatic run ID generation with timestamps
- MLFlow experiment tracking
- Configuration-driven experiments (YAML + OmegaConf)
- Learning rate monitoring and early stopping
- Automatic stdout/stderr capture and logging
- MLflow dataset tracking and registration

Usage:
    # Training
    PYTHONPATH=src python src/cli.py fit --config conf/experiment/my_experiment.yaml

    # Testing with checkpoint
    PYTHONPATH=src python src/cli.py test --config conf/experiment/my_experiment.yaml \\
        --ckpt_path path/to/checkpoint.ckpt

    # Override config values
    PYTHONPATH=src python src/cli.py fit --config conf/experiment/my_experiment.yaml \\
        --trainer.max_epochs=100 --data.init_args.batch_size=128

    # Custom run ID
    PYTHONPATH=src python src/cli.py fit --config conf/experiment/my_experiment.yaml \\
        --run_id my_custom_run
"""

from datetime import datetime
from typing import Optional

import torch
from lightning.pytorch.cli import LightningCLI

from core.callbacks.SaveConfigCallback import SaveMLFlowConfigCallback

# Constants
RUN_ID_TIMESTAMP_FORMAT = "%Y-%m-%d_%H_%M"


class ResearchCLI(LightningCLI):
    """Enhanced Lightning CLI with MLFlow support and sensible defaults.

    Features:
    - OmegaConf parser for flexible YAML configuration
    - Automatic run_id generation with timestamp
    - MLFlow logger auto-configuration
    - Learning rate monitoring
    - Optional early stopping
    """

    def __init__(self, *args, **kwargs):
        """Initialize the CLI with enhanced defaults."""
        self._set_default_kwargs(kwargs)
        super().__init__(*args, **kwargs)

    def _set_default_kwargs(self, kwargs: dict) -> None:
        """Set default keyword arguments if not provided."""
        kwargs.setdefault("save_config_callback", SaveMLFlowConfigCallback)
        kwargs.setdefault(
            "parser_kwargs",
            {"parser_mode": "omegaconf", "error_handler": None},
        )

    def add_arguments_to_parser(self, parser) -> None:
        """Add custom arguments for experiment management."""
        self._add_early_stopping_arguments(parser)
        self._add_experiment_arguments(parser)

    def _add_early_stopping_arguments(self, parser) -> None:
        """Add early stopping-related CLI arguments."""
        parser.add_argument(
            "--early_stopping",
            type=bool,
            default=False,
            help="Enable early stopping",
        )
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=10,
            help="Patience for early stopping",
        )
        parser.add_argument(
            "--early_stopping_monitor",
            type=str,
            default="val/loss",
            help="Metric to monitor for early stopping",
        )

    def _add_experiment_arguments(self, parser) -> None:
        """Add experiment management CLI arguments."""
        parser.add_argument(
            "--experiment_name",
            type=Optional[str],
            default=None,
            help="MLFlow experiment name (overrides config)",
        )
        parser.add_argument(
            "--run_id",
            type=Optional[str],
            default=None,
            help=(
                "Custom run ID. Defaults to <experiment_name>_<timestamp> "
                f"format ({RUN_ID_TIMESTAMP_FORMAT})"
            ),
        )
        parser.add_argument(
            "--log_output",
            type=bool,
            default=True,
            help="Capture stdout/stderr and log as MLFlow artifact (default: True)",
        )
        parser.add_argument(
            "--log_datasets",
            type=bool,
            default=True,
            help="Register datasets with MLflow dataset tracking (default: True)",
        )

    def before_instantiate_classes(self) -> None:
        """Set up callbacks and experiment metadata before instantiation."""
        config = self._get_subcommand_config()
        callbacks = self._ensure_callbacks_list(config)

        # Generate experiment identifiers
        experiment_name = self._resolve_experiment_name(config)
        run_id = self._resolve_run_id(experiment_name)

        # Configure MLFlow logger
        self._configure_mlflow_logger(config, run_id)

        # Add standard callbacks
        self._add_lr_monitor_callback(callbacks, config)
        self._add_log_output_callback(callbacks, config)
        self._add_log_dataset_callback(callbacks, config)
        self._add_early_stopping_callback(callbacks)

        config["trainer"]["callbacks"] = callbacks

    def _get_subcommand_config(self) -> dict:
        """Get the configuration for the current subcommand."""
        return self.config[self.config["subcommand"]]

    def _ensure_callbacks_list(self, config: dict) -> list:
        """Ensure trainer callbacks is an initialized list."""
        if config["trainer"].get("callbacks") is None:
            config["trainer"]["callbacks"] = []
        return config["trainer"]["callbacks"]

    def _resolve_experiment_name(self, config: dict) -> str:
        """Resolve experiment name from CLI args or config.

        Priority: CLI argument > logger config > default
        """
        # CLI argument takes precedence
        if self.config.get("experiment_name"):
            return self.config["experiment_name"]

        # Extract from MLFlow logger config
        logger_config = config["trainer"].get("logger")
        if isinstance(logger_config, dict):
            experiment_name = logger_config.get("init_args", {}).get("experiment_name")
            if experiment_name:
                return experiment_name

        return "experiment"

    def _resolve_run_id(self, experiment_name: str) -> str:
        """Resolve run_id from CLI args or generate timestamp-based one."""
        if self.config.get("run_id"):
            return self.config["run_id"]

        timestamp = datetime.now().strftime(RUN_ID_TIMESTAMP_FORMAT)
        return f"{experiment_name}_{timestamp}"

    def _configure_mlflow_logger(self, config: dict, run_id: str) -> None:
        """Configure MLFlow logger with run_name if present."""
        logger_config = config["trainer"].get("logger")
        if not isinstance(logger_config, dict):
            return

        if "MLFlowLogger" not in str(logger_config.get("class_path", "")):
            return

        logger_config.setdefault("init_args", {})
        logger_config["init_args"]["run_name"] = run_id

    def _add_lr_monitor_callback(self, callbacks: list, config: dict) -> None:
        """Add LearningRateMonitor callback if logger is enabled."""
        # Only add LR monitor if there's a logger
        logger_config = config["trainer"].get("logger")
        if logger_config is False or logger_config is None:
            return

        callbacks.append(
            {
                "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                "init_args": {"logging_interval": "step"},
            }
        )

    def _add_log_output_callback(self, callbacks: list, config: dict) -> None:
        """Add LogOutputCallback to capture stdout/stderr if enabled."""
        if not self.config.get("log_output", True):
            return

        # Only add if there's a logger that supports artifacts
        logger_config = config["trainer"].get("logger")
        if logger_config is False or logger_config is None:
            return

        callbacks.append(
            {
                "class_path": "core.callbacks.LogOutputCallback.LogOutputCallback",
                "init_args": {"artifact_name": "output.log"},
            }
        )

    def _add_log_dataset_callback(self, callbacks: list, config: dict) -> None:
        """Add LogDatasetCallback to register datasets with MLflow."""
        if not self.config.get("log_datasets", True):
            return

        # Only add if there's an MLflow logger
        logger_config = config["trainer"].get("logger")
        if logger_config is False or logger_config is None:
            return

        if not isinstance(logger_config, dict):
            return

        if "MLFlowLogger" not in str(logger_config.get("class_path", "")):
            return

        callbacks.append(
            {
                "class_path": "core.callbacks.LogDatasetCallback.LogDatasetCallback",
                "init_args": {
                    "log_train": True,
                    "log_val": True,
                    "log_test": True,
                },
            }
        )

    def _add_early_stopping_callback(self, callbacks: list) -> None:
        """Add EarlyStopping callback if enabled."""
        if not self.config.get("early_stopping", False):
            return

        monitor = self.config.get("early_stopping_monitor", "val/loss")
        patience = self.config.get("early_stopping_patience", 10)

        callbacks.append(
            {
                "class_path": "lightning.pytorch.callbacks.EarlyStopping",
                "init_args": {
                    "monitor": monitor,
                    "patience": patience,
                    "mode": "min",
                    "verbose": True,
                },
            }
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli = ResearchCLI()
