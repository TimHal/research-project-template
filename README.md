# Research Project Template

Minimal [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) template for ML research experiments with [MLflow](https://mlflow.org/) or [Weights & Biases](https://wandb.ai/) tracking and [Optuna](https://optuna.org/) hyperparameter optimization.

## Quick Start

```bash
# Activate environment
conda activate mlresearch

# Run training (quick test — single batch, no GPU wait)
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
├── cli.py              # CLI entry point (extends LightningCLI)
├── study.py            # Optuna hyperparameter study runner
├── core/callbacks/     # Custom callbacks (SaveConfigArtifact, LogOutput, LogDataset)
├── data/               # DataModules (Torchvision, HuggingFace, ImageFolder, …) + transforms.py
├── model/              # Model wrappers (TIMM, torchvision, HuggingFace, generic)
├── task/               # LightningModules (ClassificationTask)
└── util/               # Utilities (viz, model_loading, instantiate, mlflow)

conf/experiment/        # Experiment YAML configs
conf/study/             # Optuna study configs
nbs/                    # Analysis notebooks
tests/                  # pytest suite (see tests/README.md)
docs/                   # Conventions & cheat sheet (docs/CONVENTIONS.md)
```

> **Conventions:** naming, the file → re-export → short `class_path` workflow,
> and test rules are documented in [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md).

## How Configs Work

Experiments are configured entirely in YAML. The config has four sections:

```yaml
seed_everything: 42          # Reproducibility seed

trainer:                     # PyTorch Lightning Trainer settings
  max_epochs: 100
  accelerator: auto          # "auto", "gpu", "cpu"
  logger:                    # MLFlowLogger or WandbLogger (see "Experiment Tracking")
    class_path: lightning.pytorch.loggers.MLFlowLogger
    init_args:
      experiment_name: "My-Experiment"

model:                       # Task (LightningModule) wrapping a model architecture
  class_path: task.ClassificationTask
  init_args:
    num_classes: 10
    learning_rate: 1e-3
    model:                   # <-- The actual neural network (nn.Module)
      class_path: model.TIMMModel
      init_args:
        model_name: "resnet18"
        pretrained: true
        num_classes: 10

data:                        # DataModule
  class_path: data.TorchvisionDataModule
  init_args:
    dataset_name: "CIFAR10"
    batch_size: 64
```

> **Note:** The `model` section contains the *task* (training logic), which wraps the *architecture* (neural network). This is a standard [Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) pattern — the task handles loss, metrics, and optimizer; the architecture handles the forward pass.

See `conf/experiment/cifar10_classification.yaml` for a complete working example.

## CLI Features

### Override Config Values

Any config value can be overridden from the command line:

```bash
PYTHONPATH=src python src/cli.py fit \
    --config conf/experiment/cifar10_classification.yaml \
    --trainer.max_epochs=100 \
    --data.init_args.batch_size=128 \
    --model.init_args.learning_rate=2e-4
```

### Early Stopping

```bash
PYTHONPATH=src python src/cli.py fit \
    --config conf/experiment/cifar10_classification.yaml \
    --early_stopping=true \
    --early_stopping_patience=5 \
    --early_stopping_monitor=val/acc \
    --early_stopping_mode=max              # "min" for loss, "max" for accuracy
```

### Custom Run ID

```bash
PYTHONPATH=src python src/cli.py fit \
    --config conf/experiment/cifar10_classification.yaml \
    --run_id my_experiment_v1
```

### Output Logging

By default, stdout/stderr is captured and logged as an artifact (`output.log`). Works with both MLflow and W&B. Disable with `--log_output=false`.

### Where Generated Files Go

Nothing a run produces is written into the project directory. Checkpoints and
Optuna journals go under a single output root:

```bash
export OUTPUT_DIR=/data/<user>/runs
```

When unset it defaults to `~/.cache/research-project`. The layout underneath is:

| Path | Written by |
| ---- | ---------- |
| `<root>/runs/<experiment_id>/<run_id>/checkpoints/` | Lightning's `ModelCheckpoint` |
| `<root>/sweeps/` | Optuna journals (a relative `storage_dir`) |

This matters because Lightning resolves checkpoints against
`Trainer.default_root_dir`, which defaults to the *current working directory*.
With a remote MLflow `tracking_uri` the logger reports no `save_dir`, so a run
started from the repo root would otherwise drop
`<repo>/<experiment_id>/<run_id>/checkpoints/` into your source tree. The CLI
sets `default_root_dir` for you; an explicit value in a config or on the command
line still wins.

Artifacts destined for the tracker (the captured log, the resolved config, any
figures) are written to a temp directory and uploaded from there, so they are
unaffected.

When you fork this template, rename `ENV_VAR` and `DEFAULT_ROOT` in
`src/util/paths.py` so several projects can coexist on one machine.

### Dataset Tracking

Datasets are automatically registered with [MLflow dataset tracking](https://mlflow.org/docs/latest/tracking/data-api.html) (metadata, digest, split info). MLflow only. Disable with `--log_datasets=false`.

### Hyperparameter Study (Optuna)

Run an [Optuna](https://optuna.readthedocs.io/) hyperparameter search combining a base experiment config with a study config:

```bash
PYTHONPATH=src python src/cli.py study \
    --config conf/experiment/cifar10_classification.yaml \
    --study_config conf/study/example_cifar10_classification.yaml
```

Continue an existing study with more trials:

```bash
PYTHONPATH=src python src/cli.py study \
    --config conf/experiment/cifar10_classification.yaml \
    --study_config conf/study/example_cifar10_classification.yaml \
    --n_trials 50
```

The study config (`conf/study/example_cifar10_classification.yaml`) defines:
- **search_space** — parameters to optimize, using dot-paths into the base config
- **sampler** — TPESampler (default), RandomSampler, CmaEsSampler, NSGAIISampler
- **pruner** — MedianPruner (default), stops unpromising trials early
- **overrides** — fixed config changes for all trials (e.g. fewer epochs during search)

> **Training parity:** Each trial builds its model, datamodule, and `Trainer`
> from the *same* config resolution as `fit`, so all `trainer:` settings
> (`accumulate_grad_batches`, `precision`, `gradient_clip_val`, `strategy`, ...)
> apply during the search exactly as they would in a full run. The study only
> takes over `logger`, `callbacks`, and checkpointing: every trial gets a
> pruning callback and its own logged run, and checkpointing is disabled so
> trials don't litter artifacts. Adjust anything else via the base config or
> `overrides`.

Study state is persisted to journal files in `./sweeps/` for safe concurrent multi-process execution. After completion, the best parameters and a ready-to-run CLI command are printed.

See the [Optuna docs](https://optuna.readthedocs.io/en/stable/tutorial/index.html) for more on samplers, pruners, and multi-objective optimization.

## Available Model Wrappers

The template provides four model wrappers under `src/model/`. All support `forward()` for inference and `get_features()` for extracting intermediate representations (useful for visualization, transfer learning, and analysis).

> **Tip:** To list available layer names for `get_features(layer=...)`, run:
> `[n for n, _ in model.model.named_modules() if n]`

### TIMMModel — [timm](https://huggingface.co/timm)

1000+ vision models. Best for image classification research.

```yaml
model:
  class_path: model.TIMMModel
  init_args:
    model_name: "resnet18"         # or vit_base_patch16_224, efficientnet_b0, swin_base_patch4_window7_224, ...
    pretrained: true
    num_classes: 10
```

### TorchvisionModel — [torchvision](https://pytorch.org/vision/stable/models.html)

Standard PyTorch vision models. No extra dependencies beyond torchvision.

```yaml
model:
  class_path: model.TorchvisionModel
  init_args:
    model_name: "resnet50"         # or vit_b_16, efficientnet_b0, swin_t, mobilenet_v3_small, ...
    weights: "IMAGENET1K_V2"       # or "DEFAULT", true (= DEFAULT), null (random init)
    num_classes: 10
```

### HuggingFaceModel — [transformers](https://huggingface.co/models)

Vision and text models from HuggingFace. Requires `pip install transformers`.

```yaml
model:
  class_path: model.HuggingFaceModel
  init_args:
    model_name: "google/vit-base-patch16-224"
    task: "image-classification"   # or "sequence-classification", "feature-extraction"
    num_labels: 10
```

### TorchModel — generic

Instantiate any `nn.Module` by Python import path. Escape hatch for models not covered above.

```yaml
model:
  class_path: model.TorchModel
  init_args:
    class_path: "torchvision.models.resnet18"
    init_args:
      num_classes: 10
    features_layer: "layer4"       # optional: named layer for get_features()
```

> **Remember:** Model wrappers go inside the task's `model` argument. See the full config example above.

## Available DataModules

The template provides seven DataModules under `src/data/` for loading data from common sources. All are configured via the `data:` section in experiment YAML configs.

### TorchvisionDataModule — built-in datasets

Standard torchvision datasets (CIFAR10/100, MNIST, FashionMNIST, KMNIST, EMNIST, STL10, SVHN). No extra dependencies.

```yaml
data:
  class_path: data.TorchvisionDataModule
  init_args:
    dataset_name: "CIFAR10"
    batch_size: 64
    img_size: [224, 224]
    augmentation: "basic"
```

### ImageFolderDataModule — local image directories

Load images organized as `root/class_name/image.jpg`. Auto-detects `train/`/`val/`/`test/` subdirectories, or auto-splits a single directory. No extra dependencies.

```yaml
data:
  class_path: data.ImageFolderDataModule
  init_args:
    root_dir: "./data/my_dataset"     # contains train/, val/, test/ or class folders directly
    img_size: [224, 224]
    batch_size: 32
    augmentation: "basic"
```

### HuggingFaceDataModule — [HuggingFace Hub](https://huggingface.co/datasets)

Any image classification dataset from the HuggingFace Hub. Auto-detects image and label columns. Requires `pip install datasets`.

```yaml
data:
  class_path: data.HuggingFaceDataModule
  init_args:
    dataset_name: "beans"             # or "cifar10", "imagenet-1k", any HF dataset
    img_size: [224, 224]
    batch_size: 32
    augmentation: "basic"
```

### ParquetDataModule — Parquet files

Load data from Parquet files in two modes. Requires `pip install pyarrow`.

**Tabular mode** — numeric feature columns + label:

```yaml
data:
  class_path: data.ParquetDataModule
  init_args:
    train_file: "./data/tabular/train.parquet"
    test_file: "./data/tabular/test.parquet"
    label_column: "target"
    mode: "tabular"
    batch_size: 128
```

**Image path mode** — image file paths stored in a column:

```yaml
data:
  class_path: data.ParquetDataModule
  init_args:
    train_file: "./data/images/metadata.parquet"
    label_column: "label"
    mode: "image_path"
    image_column: "file_path"
    image_base_dir: "./data/images/"
    img_size: [224, 224]
    batch_size: 32
```

### WebDatasetDataModule — [WebDataset](https://webdataset.github.io/webdataset/) tar archives

Stream large-scale datasets from sharded tar files without loading everything into memory. Requires `pip install webdataset`.

```yaml
data:
  class_path: data.WebDatasetDataModule
  init_args:
    train_urls: "./data/wds/train-{0000..0099}.tar"
    val_urls: "./data/wds/val-{0000..0009}.tar"
    num_classes: 1000
    num_train_samples: 1281167        # required for epoch length
    num_val_samples: 50000
    img_size: [224, 224]
    batch_size: 64
    num_workers: 8
```

### CroissantDataModule — [Croissant](https://github.com/mlcommons/croissant) metadata

Load datasets described by Croissant JSON-LD metadata files — the standard format adopted by HuggingFace, Kaggle, and OpenML. Requires `pip install mlcroissant`.

```yaml
data:
  class_path: data.CroissantDataModule
  init_args:
    metadata_path: "./data/my_dataset/croissant.json"
    record_set: "images"              # which record set to load
    img_size: [224, 224]
    batch_size: 32
    test_ratio: 0.1                   # hold out 10% for testing
```

### KaggleDataModule — [Kaggle](https://www.kaggle.com/) datasets & competitions

Download and load datasets from Kaggle. Supports ImageFolder and CSV loading formats. Requires `pip install kaggle` and a [Kaggle API token](https://github.com/Kaggle/kaggle-api#api-credentials) at `~/.kaggle/kaggle.json`.

```yaml
# ImageFolder mode (most image datasets)
data:
  class_path: data.KaggleDataModule
  init_args:
    dataset_slug: "moltean/fruits"
    root_dir: "./data/kaggle_fruits"
    load_format: "imagefolder"
    img_size: [224, 224]
    batch_size: 32

# CSV mode (competitions like digit-recognizer)
data:
  class_path: data.KaggleDataModule
  init_args:
    competition: "digit-recognizer"
    root_dir: "./data/kaggle_digits"
    load_format: "csv"
    csv_file: "train.csv"
    label_column: "label"
    image_shape: [1, 28, 28]
    num_classes: 10
```

> See `conf/experiment/example_*.yaml` for complete working configs for each DataModule.

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

batch, labels = next(iter(dm.train_dataloader()))
grid = batch_to_grid(batch[:16], nrow=4)
plt.figure(figsize=(10, 10))
plt.imshow(grid)
plt.axis("off")
plt.show()
```

## Extending the Template

Adding a model, task, or DataModule is always the same three steps (see
[`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) for the full conventions):

1. Create a `snake_case.py` module holding one `PascalCase` class.
2. Re-export the class from the package's `__init__.py`.
3. Reference it in a config by the short path `package.ClassName`.

### Adding a New Model

Create `src/model/my_model.py` with any `nn.Module`:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x.flatten(1))
```

Re-export it in `src/model/__init__.py`:

```python
from model.my_model import MyModel  # add to __all__ too
```

Use it in a config — note the short `model.MyModel` path:

```yaml
model:
  class_path: task.ClassificationTask
  init_args:
    model:
      class_path: model.MyModel
      init_args:
        in_features: 3072
        num_classes: 10
```

### Adding a New Task

Create `src/task/my_task.py` extending [LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html), then re-export it from `src/task/__init__.py`:

```python
import lightning as L
import torch

class MyTask(L.LightningModule):
    def __init__(self, model, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = ...  # your loss function
        self.log("train/loss", loss)  # use "train/…", "val/…", "test/…" metric names
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
```

### Adding a New DataModule

Create `src/data/my_datamodule.py` extending [LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html), then re-export it from `src/data/__init__.py`. Reuse [`build_image_transform`](src/data/transforms.py) for the standard image pipeline:

```python
import lightning as L
from torch.utils.data import DataLoader

from data.transforms import build_image_transform

class MyDataModule(L.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 32):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_transform = build_image_transform((224, 224), train=True)

    def setup(self, stage=None):
        pass  # load your data here

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
```

## Running Tests

A minimal [pytest](https://docs.pytest.org/) suite covers the core building blocks (config parsing, model/datamodule construction, a `fast_dev_run` training smoke test). It runs on CPU in seconds with no dataset downloads.

```bash
conda activate mlresearch
pytest                # run everything
pytest -v             # one line per test
pytest -m "not slow"  # skip tests marked slow
```

No `PYTHONPATH=src` prefix is needed for tests — it's configured in `pyproject.toml`. See [`tests/README.md`](tests/README.md) for what's covered and how to add your own.

## Experiment Tracking

Switch between MLflow and Weights & Biases by changing the `trainer.logger` section in your experiment config.

### MLflow

Start the MLflow tracking server:

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

See the [MLflow quickstart](https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html) for more.

### Weights & Biases

```bash
pip install wandb
wandb login
```

```yaml
trainer:
  logger:
    class_path: lightning.pytorch.loggers.WandbLogger
    init_args:
      project: "My-Experiment"
```

See `conf/experiment/cifar10_classification_wandb.yaml` for a complete example. See the [W&B quickstart](https://docs.wandb.ai/quickstart/) for more.

## Requirements

Assumes `mlresearch` conda environment with:
- [torch](https://pytorch.org/), torchvision
- [lightning](https://lightning.ai/docs/pytorch/stable/)
- [timm](https://huggingface.co/timm)
- [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/)
- [mlflow](https://mlflow.org/)
- [omegaconf](https://omegaconf.readthedocs.io/)
- [optuna](https://optuna.readthedocs.io/)
- numpy, matplotlib
- [pytest](https://docs.pytest.org/) (for running the test suite)
- [transformers](https://huggingface.co/docs/transformers/) (optional, for HuggingFaceModel)
- [datasets](https://huggingface.co/docs/datasets/) (optional, for HuggingFaceDataModule)
- [webdataset](https://webdataset.github.io/webdataset/) (optional, for WebDatasetDataModule)
- [mlcroissant](https://github.com/mlcommons/croissant) (optional, for CroissantDataModule)
- [kaggle](https://github.com/Kaggle/kaggle-api) (optional, for KaggleDataModule)
- [pyarrow](https://arrow.apache.org/docs/python/) (optional, for ParquetDataModule)
- [wandb](https://docs.wandb.ai/) (optional, for W&B tracking)

## Further Reading

- [PyTorch Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) — how config-driven training works
- [timm model zoo](https://huggingface.co/timm) — available vision models
- [HuggingFace model hub](https://huggingface.co/models) — available transformer models
- [Optuna tutorial](https://optuna.readthedocs.io/en/stable/tutorial/index.html) — hyperparameter optimization guide
- [MLflow tracking](https://mlflow.org/docs/latest/tracking.html) — experiment tracking and artifact logging
- [Weights & Biases](https://docs.wandb.ai/) — alternative experiment tracking platform
