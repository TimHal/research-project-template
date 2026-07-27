# Tests

Minimal pytest suite covering the template's core building blocks. It runs on
CPU in a few seconds with **no dataset downloads and no network access** — the
tests use a tiny in-memory model and random tensors instead of real data.

## Running

```bash
conda activate mlresearch
pytest                     # run everything
pytest -v                  # verbose: one line per test
pytest tests/test_task.py  # a single file
pytest -k datamodule       # tests matching a keyword
pytest -m "not slow"       # skip tests marked slow
```

No `PYTHONPATH=src` prefix is needed — `pythonpath = ["src"]` in
`pyproject.toml` puts the source packages on the import path for tests.

## What's covered

| File                     | What it checks                                             |
| ------------------------ | ---------------------------------------------------------- |
| `test_configs.py`        | Every `conf/experiment/*.yaml` parses and has model/data   |
| `test_instantiate.py`    | Shared config instantiation (nested + bare class refs)     |
| `test_model_loading.py`  | `import_class` / `instantiate_from_config` (incl. nested)  |
| `test_transforms.py`     | `build_image_transform` output shapes + picklability       |
| `test_datamodule.py`     | `TorchvisionDataModule` validation + synthetic ImageFolder |
| `test_task.py`           | `ClassificationTask` forward shape + a `fast_dev_run` fit  |
| `test_viz_utils.py`      | `denormalize` / `batch_to_grid` shapes and ranges          |
| `test_study_utils.py`    | Optuna helpers (`_deep_set`) + parameter suggestion        |

Shared fixtures live in `conftest.py`: `tiny_classifier` (a small `nn.Module`),
`fake_image_loader` (random `(image, label)` batches), `num_classes`, and
`repo_root`.

## Isolation (no pollution)

An autouse fixture (`_isolated_env` in `conftest.py`) makes every test hermetic:
it runs in a throwaway temp working directory and redirects experiment trackers
and caches to temp/offline. So nothing leaks into your repo or environment —
no `lightning_logs/`, no `mlruns/`, no checkpoints, no MLflow-server writes, and
W&B is disabled (no network, no login). Everything is cleaned up automatically.

You get this for free. When writing a test that produces files, write them under
the `tmp_path` fixture (or a Trainer's `default_root_dir=str(tmp_path)`) to keep
the guarantee explicit.

## Adding tests

Create a `test_*.py` file; pytest discovers `test_` functions automatically.
Reuse the fixtures so new tests stay fast and download-free.

```python
# tests/test_my_feature.py
from data import MyDataModule


def test_my_datamodule_num_classes():
    dm = MyDataModule(root_dir="…", download=False)
    assert dm.num_classes == 5


def test_my_task_trains(tiny_classifier, num_classes, fake_image_loader):
    import lightning as L
    from task import MyTask

    task = MyTask(tiny_classifier, num_classes=num_classes)
    trainer = L.Trainer(
        fast_dev_run=True, accelerator="cpu", logger=False, enable_checkpointing=False
    )
    trainer.fit(task, train_dataloaders=fake_image_loader)
```

Guidelines:

- **Keep the default suite fast and offline.** Construct objects and test
  validation/shape logic; avoid `setup()`/`prepare_data()` that download data.
- **Mark anything slow.** Tests that download a dataset or train for real should
  be decorated with `@pytest.mark.slow` so `pytest -m "not slow"` skips them.
- **Skip optional dependencies gracefully** so the suite still passes without
  them installed:
  ```python
  import pytest
  pytest.importorskip("webdataset")
  ```
- **Add a fixture to `conftest.py`** when the same setup is needed in more than
  one file.
