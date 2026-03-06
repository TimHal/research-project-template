"""Callback to save experiment configuration to MLFlow artifacts."""

import tempfile
from pathlib import Path

import lightning as L
from lightning.pytorch.cli import SaveConfigCallback


class SaveMLFlowConfigCallback(SaveConfigCallback):
    """Save experiment config.yaml as an MLFlow artifact.

    This callback automatically saves the full experiment configuration
    to MLFlow artifacts, making it easy to reproduce experiments.
    """

    def __init__(
        self,
        parser,
        config,
        config_filename="config.yaml",
        overwrite=False,
        multifile=False,
    ):
        super().__init__(
            parser, config, config_filename, overwrite, multifile, save_to_log_dir=False
        )

    def save_config(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        """Save config to MLFlow artifacts."""
        if trainer.is_global_zero:
            # Check if logger has MLFlow-compatible interface
            if not hasattr(trainer.logger, "experiment") or not hasattr(
                trainer.logger.experiment, "log_artifact"
            ):
                # Fall back to default behavior for non-MLFlow loggers
                return

            with tempfile.TemporaryDirectory() as tmp_dir:
                config_path = Path(tmp_dir) / "config.yaml"
                self.parser.save(
                    self.config,
                    config_path,
                    skip_none=False,
                    overwrite=self.overwrite,
                    multifile=self.multifile,
                )
                trainer.logger.experiment.log_artifact(
                    local_path=config_path, run_id=trainer.logger.run_id
                )
