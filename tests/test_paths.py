"""Tests for the default output root.

The point of these is a regression guard: a run must never write generated
files into the project directory, which is what happens when Lightning falls
back to ``default_root_dir = os.getcwd()``.
"""

from pathlib import Path

from cli import ResearchCLI
from study import _build_storage
from util import paths
from util.paths import output_path


def test_output_path_joins_under_the_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DEFAULT_OUTPUT_ROOT", tmp_path / "elsewhere")
    assert output_path("runs") == str(tmp_path / "elsewhere" / "runs")


def test_cli_default_root_dir_is_absolute_and_outside_the_repo(repo_root):
    """The regression this guards: a relative default lands in the source tree."""
    config = {"trainer": {}}
    ResearchCLI._set_default_root_dir(None, config)

    # Absolute is the point: a relative default is resolved against the cwd,
    # which is the repo root whenever a run is started from there.
    root = Path(config["trainer"]["default_root_dir"])
    assert root.is_absolute()
    assert root != repo_root and repo_root not in root.parents


def test_cli_fills_in_an_absent_default_root_dir():
    config = {"trainer": {}}
    ResearchCLI._set_default_root_dir(None, config)
    assert config["trainer"]["default_root_dir"] == output_path("runs")


def test_config_default_root_dir_wins():
    """The config is the override: the CLI must not overwrite an explicit value."""
    config = {"trainer": {"default_root_dir": "/somewhere/explicit"}}
    ResearchCLI._set_default_root_dir(None, config)
    assert config["trainer"]["default_root_dir"] == "/somewhere/explicit"


def test_relative_study_storage_resolves_under_the_output_root():
    """`storage_dir: ./sweeps` must not put journals next to the source."""
    _build_storage({"study": {"name": "s", "storage_dir": "./sweeps"}})
    assert Path(output_path("sweeps")).is_dir()
    assert not (Path.cwd() / "sweeps").exists()


def test_absolute_study_storage_is_left_alone(tmp_path):
    target = tmp_path / "explicit_sweeps"
    _build_storage({"study": {"name": "s", "storage_dir": str(target)}})
    assert (target / "s.journal").exists()
