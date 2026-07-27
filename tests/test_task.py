"""Tests for ClassificationTask: a forward-shape check and an end-to-end
fast_dev_run smoke test that exercises the full training loop on random data.
"""

import lightning as L
import torch

from task import ClassificationTask


def test_forward_shape(tiny_classifier, num_classes):
    task = ClassificationTask(tiny_classifier, num_classes=num_classes)
    logits = task(torch.randn(2, 3, 8, 8))
    assert logits.shape == (2, num_classes)


def test_fit_smoke(tiny_classifier, num_classes, fake_image_loader, tmp_path):
    """One train batch through a real Trainer — catches wiring/logging errors.

    ``logger=False`` and ``enable_checkpointing=False`` keep it quiet, and
    ``default_root_dir`` pins any Trainer output to the test's temp dir.
    """
    task = ClassificationTask(tiny_classifier, num_classes=num_classes)
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(task, train_dataloaders=fake_image_loader)
