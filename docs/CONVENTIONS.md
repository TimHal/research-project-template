# Conventions & Cheat Sheet

A quick reference for how this template is organized and the conventions to
follow when extending it. Keep this file in sync when conventions change.

---

## Layout at a glance

```
src/
├── cli.py                 # LightningCLI entry point (+ `study` subcommand)
├── study.py               # Optuna hyperparameter search
├── core/callbacks/        # Custom Lightning callbacks
├── data/                  # DataModules + transforms.py (shared image pipeline)
├── model/                 # nn.Module wrappers (architectures)
├── task/                  # LightningModules (training logic, loss, metrics)
└── util/                  # instantiate, model_loading, viz, mlflow, dataset helpers
conf/experiment/           # One YAML per experiment
conf/study/                # One YAML per Optuna study
tests/                     # pytest suite (offline, hermetic)
docs/                      # This file and other docs
```

**Task vs. model.** A *task* (in `task/`) is a `LightningModule` that owns the
loss, metrics, and optimizer. A *model* (in `model/`) is a plain `nn.Module`
that only does the forward pass. In a config the `model:` block is the *task*,
and the task's `init_args.model` is the *architecture*. See the README's
"How Configs Work".

---

## Naming rules

| Thing              | Convention            | Example                                    |
| ------------------ | --------------------- | ------------------------------------------ |
| Module file        | `snake_case.py`       | `timm_model.py`, `torchvision_datamodule.py` |
| Class              | `PascalCase`          | `TIMMModel`, `TorchvisionDataModule`       |
| DataModule class   | ends in `DataModule`  | `ImageFolderDataModule`                    |
| Task class         | ends in `Task`        | `ClassificationTask`                       |
| Callback class     | ends in `Callback`    | `LogOutputCallback`                        |
| Private helper     | `_leading_underscore` | `_emnist_orient`, `_build_pruner`          |
| Metric log name    | `split/metric`        | `train/loss`, `val/acc`, `test/loss`       |
| Config file        | `snake_case.yaml`     | `cifar10_classification.yaml`              |

One public class per module file; the file is named after the class in
`snake_case`. Match `torch`/`lightning` spelling for compound words
(`DataModule`, `DataLoader`).

---

## The golden rule: file → re-export → short `class_path`

Every model / task / DataModule / callback follows the **same three steps** so
configs can reference it by a short, unambiguous path.

1. **Create** `src/<package>/<snake_case>.py` with one `PascalCase` class.
2. **Re-export** it from `src/<package>/__init__.py` (add to the imports *and*
   `__all__`).
3. **Reference** it in YAML as `class_path: <package>.<ClassName>`.

```python
# src/model/my_model.py
import torch.nn as nn

class MyModel(nn.Module):
    ...
```

```python
# src/model/__init__.py
from model.my_model import MyModel
__all__ = [..., "MyModel"]
```

```yaml
# in a config
model:
  class_path: model.MyModel        # NOT model.my_model.MyModel
  init_args: { ... }
```

Because the class is re-exported, `class_path: model.MyModel` resolves to the
class while `model.my_model` remains the module — no ambiguity.

> **Optional dependencies:** import heavy third-party packages *lazily inside*
> the class (see `HuggingFaceModel._import_transformers`), never at module top
> level. That keeps `import model` / `import data` cheap and dependency-free, so
> re-exporting every class in `__init__.py` is safe.

---

## Config anatomy (`conf/experiment/*.yaml`)

```yaml
seed_everything: 42
trainer:            # → lightning.pytorch.Trainer kwargs
  logger:           # MLFlowLogger or WandbLogger (class_path/init_args)
model:              # the TASK (LightningModule); its init_args.model is the architecture
data:               # the DataModule
early_stopping: true   # CLI-level flags (see cli.py add_arguments_to_parser)
```

Anything can be overridden on the command line:
`--trainer.max_epochs=100 --data.init_args.batch_size=128`.

---

## Instantiating configs outside the CLI

`util/instantiate.py` is the **single** place that turns config dicts into
objects, used by both the notebook loader (`util/model_loading.load_from_config`)
and the Optuna study. It handles nested `{class_path, init_args}` blocks and
bare dotted class references (e.g. `optimizer_class: torch.optim.AdamW`). Never
reimplement instantiation elsewhere — `fit` and `study` must stay in sync.

---

## Images: use the shared transform

`data/transforms.py :: build_image_transform(...)` builds the standard pipeline
(resize → channel conversion → augmentation → ToTensor → normalize). Reuse it in
new image DataModules instead of copying transform code. Callables are
module-level functions (not lambdas) so transforms stay picklable and work with
`num_workers > 0`. Dataset-specific steps go through `extra_transforms=[...]`
(see the EMNIST orientation fix in `torchvision_datamodule.py`).

---

## Tests

- Run: `pytest` (no `PYTHONPATH=src` needed — set in `pyproject.toml`).
- **Offline & hermetic:** the default suite downloads nothing and writes nothing
  outside a temp dir (autouse `_isolated_env` fixture). Keep it that way.
- Test construction/validation and shape logic; mark anything that downloads or
  trains for real with `@pytest.mark.slow`.
- Skip optional deps gracefully: `pytest.importorskip("webdataset")`.
- See `tests/README.md` for fixtures and a copy-paste template.

---

## Style

- `ruff check src/ tests/` must pass (config in `pyproject.toml`; line length 88,
  E/F/I/W rules). Run `ruff check --fix` to auto-sort imports.
- Metric names are always `split/metric` so loggers and early-stopping monitors
  line up (`--early_stopping_monitor=val/acc`).
- Docstrings: module docstring stating purpose; Google-style `Args:` on public
  classes/functions. Match the surrounding density. Types belong in the
  annotations, not in the `Args:` entries.
- VS Code generates this style for you: install the recommended
  [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autoDocstring)
  extension, then type `"""` under a `def`/`class` and press Enter. The template
  lives in `.vscode/docstring.mustache`; `moddoc`, `clsdoc`, and `cfgex`
  snippets cover module headers and YAML examples.
