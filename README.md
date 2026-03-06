# Research Project Template

Minimal PyTorch Lightning template for ML research experiments with MLflow tracking.

## Quick Start

```bash
# Activate environment
conda activate mlresearch

# Run training (quick test)
PYTHONPATH=src python src/cli.py fit \
    --config conf/experiment/cifar10_classification.yaml \
    --trainer.fast_dev_run=true

# Full training
PYTHONPATH=src python src/cli.py fit \
    --config conf/experiment/cifar10_classification.yaml

# Test with checkpoint
PYTHONPATH=src python src/cli.py test \
    --config conf/experiment/cifar10_classification.yaml \
    --ckpt_path path/to/checkpoint.ckpt
```

## Project Structure

```
src/
├── cli.py              # CLI entry point (LightningCLI)
├── core/callbacks/     # Custom callbacks (SaveConfig, LogOutput)
├── data/               # DataModules (TorchvisionDatamodule)
├── model/              # Model architectures (TIMMModel wrapper)
├── task/               # LightningModules (ClassificationTask)
└── util/               # Utilities (viz, model_loading, mlflow)

conf/experiment/        # Experiment YAML configs
nbs/                    # Analysis notebooks
logs/                   # Local logs (gitignored)
```

## Creating New Experiments

1. Create a YAML config in `conf/experiment/`
2. Define your model in `src/model/` (or use TIMMModel)
3. Define your task in `src/task/` (or use ClassificationTask)
4. Run with CLI

### Example: Custom Config

```yaml
seed_everything: 42

trainer:
  max_epochs: 100
  accelerator: auto
  logger:
    class_path: lightning.pytorch.loggers.MLFlowLogger
    init_args:
      experiment_name: "My-Experiment"

model:
  class_path: task.ClassificationTask.ClassificationTask
  init_args:
    num_classes: 10
    learning_rate: 1e-3
    model:
      class_path: model.TIMMModel.TIMMModel
      init_args:
        model_name: "efficientnet_b0"
        pretrained: true
        num_classes: 10

data:
  class_path: data.TorchvisionDatamodule.TorchvisionDatamodule
  init_args:
    dataset_name: "CIFAR10"
    batch_size: 64
```

## CLI Features

### Override Config Values

```bash
PYTHONPATH=src python src/cli.py fit \
    --config conf/experiment/cifar10_classification.yaml \
    --trainer.max_epochs=100 \
    --data.init_args.batch_size=128 \
    --model.init_args.learning_rate=2e-4
```

### Custom Run ID

```bash
PYTHONPATH=src python src/cli.py fit \
    --config conf/experiment/cifar10_classification.yaml \
    --run_id my_experiment_v1
```

### Early Stopping

```bash
PYTHONPATH=src python src/cli.py fit \
    --config conf/experiment/cifar10_classification.yaml \
    --early_stopping=true \
    --early_stopping_patience=5
```

### Output Logging

By default, all stdout/stderr output is captured and logged as an MLFlow artifact (`output.log`) for reproducibility. Disable with:

```bash
PYTHONPATH=src python src/cli.py fit \
    --config conf/experiment/cifar10_classification.yaml \
    --log_output=false
```

### Dataset Tracking

Datasets are automatically registered with MLflow's dataset tracking for reproducibility. This logs:
- Dataset metadata (name, size, shape)
- Sample digest for versioning
- Train/val/test split information

Supports PyTorch datasets, NumPy arrays, and Pandas DataFrames. Disable with:

```bash
PYTHONPATH=src python src/cli.py fit \
    --config conf/experiment/cifar10_classification.yaml \
    --log_datasets=false
```

## Loading Models in Notebooks

```python
from util.model_loading import load_from_config
from util.mlflow_utils import resolve_experiment_path

# Option 1: Load directly from config
model, dm = load_from_config("conf/experiment/cifar10_classification.yaml")

# Option 2: Load from MLflow run
conf_path, ckpt_path = resolve_experiment_path(
    experiment="CIFAR10-Classification",
    run="CIFAR10-Classification_2024-01-15_10_30"
)
model, dm = load_from_config(conf_path, ckpt_path)

# Use the model
model.eval()
dm.setup("test")
batch = next(iter(dm.test_dataloader()))
```

## Visualization

```python
from util.viz_utils import batch_to_grid
import matplotlib.pyplot as plt

# Get a batch
batch, labels = next(iter(dm.train_dataloader()))

# Display as grid
grid = batch_to_grid(batch[:16], nrow=4)
plt.figure(figsize=(10, 10))
plt.imshow(grid)
plt.axis("off")
plt.show()
```

## Extending the Template

### Adding a New Model

Create `src/model/MyModel.py`:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x.flatten(1))
```

Reference in config:

```yaml
model:
  class_path: task.ClassificationTask.ClassificationTask
  init_args:
    model:
      class_path: model.MyModel.MyModel
      init_args:
        in_features: 3072
        num_classes: 10
```

### Adding a New Task

Create `src/task/MyTask.py`:

```python
import lightning as L
import torch.nn as nn

class MyTask(L.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = ...
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
```

### Adding a New DataModule

Create `src/data/MyDatamodule.py`:

```python
import lightning as L
from torch.utils.data import DataLoader

class MyDatamodule(L.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 32):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Load your data
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
```

## Available Models (via timm)

The template uses [timm](https://huggingface.co/timm) for model architectures:

- **ResNet**: `resnet18`, `resnet50`, `resnet101`
- **EfficientNet**: `efficientnet_b0`, `efficientnet_b4`
- **ViT**: `vit_base_patch16_224`, `vit_small_patch16_224`
- **ConvNeXt**: `convnext_tiny`, `convnext_small`
- **Swin**: `swin_tiny_patch4_window7_224`

See [timm model list](https://huggingface.co/timm) for all available models.

## MLflow Setup

Start MLflow server:

```bash
mlflow server --host 0.0.0.0 --port 8090
```

Or use file-based tracking (no server needed):

```yaml
trainer:
  logger:
    class_path: lightning.pytorch.loggers.MLFlowLogger
    init_args:
      tracking_uri: "file:./mlruns"
      experiment_name: "My-Experiment"
```

## Requirements

Assumes `mlresearch` conda environment with:
- torch, torchvision
- lightning
- timm
- torchmetrics
- mlflow
- omegaconf
- numpy, matplotlib
