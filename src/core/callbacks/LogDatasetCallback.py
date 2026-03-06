"""Callback to register datasets with MLflow dataset tracking."""

from typing import Optional

import lightning as L

from util.dataset_utils import get_dataset_info, log_dataset_to_mlflow


class LogDatasetCallback(L.Callback):
    """Register datasets with MLflow dataset tracking for reproducibility.

    This callback automatically logs dataset metadata and samples to MLflow
    at the start of training, enabling dataset versioning and lineage tracking.

    Supports:
    - PyTorch Dataset/Subset
    - NumPy arrays
    - Pandas DataFrames

    Args:
        log_train: Whether to log training dataset (default: True)
        log_val: Whether to log validation dataset (default: True)
        log_test: Whether to log test dataset (default: True)
        source: Optional source description (e.g., path, URL)

    Example:
        Add to trainer callbacks in config:

        trainer:
          callbacks:
            - class_path: core.callbacks.LogDatasetCallback.LogDatasetCallback
              init_args:
                log_train: true
                log_val: true
                log_test: false
    """

    def __init__(
        self,
        log_train: bool = True,
        log_val: bool = True,
        log_test: bool = True,
        source: Optional[str] = None,
    ):
        super().__init__()
        self.log_train = log_train
        self.log_val = log_val
        self.log_test = log_test
        self.source = source
        self._logged = False

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log datasets at the start of training."""
        if not trainer.is_global_zero:
            return

        if self._logged:
            return

        # Check if MLflow logger is available
        if not self._has_mlflow_logger(trainer):
            return

        datamodule = trainer.datamodule
        if datamodule is None:
            return

        self._log_datasets(trainer, datamodule, stage="fit")
        self._logged = True

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log test dataset at the start of testing."""
        if not trainer.is_global_zero:
            return

        if not self.log_test:
            return

        if not self._has_mlflow_logger(trainer):
            return

        datamodule = trainer.datamodule
        if datamodule is None:
            return

        # Only log test dataset if not already logged during fit
        if not self._logged:
            self._log_datasets(trainer, datamodule, stage="test")

    def _has_mlflow_logger(self, trainer: L.Trainer) -> bool:
        """Check if trainer has an MLflow logger."""
        if trainer.logger is None:
            return False

        # Check for MLflow logger
        logger_class = type(trainer.logger).__name__
        return "MLFlow" in logger_class or hasattr(trainer.logger, "run_id")

    def _log_datasets(self, trainer: L.Trainer, datamodule: L.LightningDataModule, stage: str) -> None:
        """Log datasets from the datamodule."""
        import mlflow

        # Ensure MLflow is using the same run as Lightning
        if hasattr(trainer.logger, "run_id"):
            mlflow.start_run(run_id=trainer.logger.run_id)

        try:
            # Get dataset name from datamodule if available
            dataset_name = getattr(datamodule, "dataset_name", type(datamodule).__name__)
            source = self.source or getattr(datamodule, "root_dir", None)

            # Log training dataset
            if self.log_train and stage == "fit" and hasattr(datamodule, "train_dataset"):
                train_dataset = datamodule.train_dataset
                if train_dataset is not None:
                    self._log_single_dataset(
                        dataset=train_dataset,
                        name=f"{dataset_name}_train",
                        context="training",
                        source=source,
                    )

            # Log validation dataset
            if self.log_val and stage == "fit" and hasattr(datamodule, "val_dataset"):
                val_dataset = datamodule.val_dataset
                if val_dataset is not None:
                    self._log_single_dataset(
                        dataset=val_dataset,
                        name=f"{dataset_name}_val",
                        context="validation",
                        source=source,
                    )

            # Log test dataset
            if self.log_test and hasattr(datamodule, "test_dataset"):
                test_dataset = datamodule.test_dataset
                if test_dataset is not None:
                    self._log_single_dataset(
                        dataset=test_dataset,
                        name=f"{dataset_name}_test",
                        context="test",
                        source=source,
                    )

            # Also log dataset metadata as parameters
            self._log_dataset_params(datamodule, dataset_name)

        except Exception as e:
            print(f"Warning: Failed to log datasets to MLflow: {e}")

    def _log_single_dataset(
        self,
        dataset,
        name: str,
        context: str,
        source: Optional[str],
    ) -> None:
        """Log a single dataset to MLflow."""
        try:
            log_dataset_to_mlflow(
                dataset=dataset,
                name=name,
                context=context,
                source=source,
            )
            info = get_dataset_info(dataset, name)
            print(f"Logged dataset '{name}' to MLflow ({context}, size={info.get('size', '?')})")
        except Exception as e:
            print(f"Warning: Failed to log dataset '{name}': {e}")

    def _log_dataset_params(self, datamodule: L.LightningDataModule, dataset_name: str) -> None:
        """Log dataset-related parameters."""
        import mlflow

        params = {"dataset_name": dataset_name}

        # Extract common datamodule attributes
        for attr in ["batch_size", "num_workers", "img_size", "num_classes", "train_val_split"]:
            if hasattr(datamodule, attr):
                value = getattr(datamodule, attr)
                if value is not None:
                    params[f"data_{attr}"] = str(value)

        mlflow.log_params(params)
