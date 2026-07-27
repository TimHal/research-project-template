"""Callback to save experiment configuration as a logger artifact."""

import tempfile
from pathlib import Path

import lightning as L
from lightning.pytorch.cli import SaveConfigCallback


class SaveConfigArtifactCallback(SaveConfigCallback):
    """Save experiment config.yaml as a logger artifact.

    This callback automatically saves the full experiment configuration
    as an artifact (MLFlow or W&B), making it easy to reproduce experiments.
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
        """Save config as a logger artifact (MLFlow or W&B)."""
        if not trainer.is_global_zero:
            return

        if not hasattr(trainer.logger, "experiment"):
            return

        logger_name = type(trainer.logger).__name__

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            self.parser.save(
                self.config,
                config_path,
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )

            if "MLFlow" in logger_name and hasattr(trainer.logger, "run_id"):
                trainer.logger.experiment.log_artifact(
                    local_path=config_path, run_id=trainer.logger.run_id
                )
            elif "Wandb" in logger_name:
                import wandb

                artifact = wandb.Artifact("config", type="config")
                artifact.add_file(str(config_path))
                trainer.logger.experiment.log_artifact(artifact)
