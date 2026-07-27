"""Tasks (LightningModules) that define training, loss, and metrics.

Re-exported so configs can reference the short path ``task.<ClassName>``
(e.g. ``task.ClassificationTask``).
"""

from task.classification_task import ClassificationTask

__all__ = ["ClassificationTask"]
