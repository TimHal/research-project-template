"""Classification Task - Standard image classification training.

This module provides a LightningModule for image classification with:
- Cross-entropy loss
- Accuracy tracking with torchmetrics
- Configurable optimizer and scheduler
- Proper metric logging
"""

from typing import Any, Dict, Optional

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class ClassificationTask(L.LightningModule):
    """Standard classification LightningModule.

    Args:
        model: Neural network module (e.g., TIMMModel)
        num_classes: Number of output classes
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay (L2 regularization)
        optimizer_class: Optimizer class (default: AdamW)
        scheduler_class: Optional LR scheduler class
        scheduler_params: Parameters for scheduler

    Example:
        >>> from model.TIMMModel import TIMMModel
        >>> model = TIMMModel("resnet18", num_classes=10)
        >>> task = ClassificationTask(model, num_classes=10, learning_rate=1e-3)
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        optimizer_class: type[Optimizer] = torch.optim.AdamW,
        scheduler_class: Optional[type[LRScheduler]] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_class = optimizer_class
        self.scheduler_class = scheduler_class
        self.scheduler_params = scheduler_params or {}

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Update metrics
        self.train_acc(logits, y)

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """Validation step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Update metrics
        self.val_acc(logits, y)

        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """Test step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Update metrics
        self.test_acc(logits, y)

        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and optional scheduler."""
        optimizer = self.optimizer_class(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_class is None:
            return optimizer

        scheduler = self.scheduler_class(optimizer, **self.scheduler_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
            },
        }
