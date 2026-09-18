"""Tests for the shared output root.

The point of these is a regression guard: a run must never write generated
files into the project directory, which is what happens when Lightning falls
back to ``default_root_dir = os.getcwd()``.
"""

from pathlib import Path

from cli import ResearchCLI
from study import _build_storage
from util.paths import DEFAULT_ROOT, output_path, output_root


def test_env_var_sets_the_root(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "elsewhere"))
    assert output_root() == tmp_path / "elsewhere"
    assert output_path("runs") == str(tmp_path / "elsewhere" / "runs")


def test_default_root_is_outside_the_project(monkeypatch, repo_root):
    """Without the env var the root is under ~/.cache, never the working dir."""
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    root = output_root()

    assert root == DEFAULT_ROOT
    assert Path.cwd() not in root.parents and root != Path.cwd()
    assert repo_root not in root.parents


def test_cli_fills_in_an_absent_default_root_dir():
    config = {"trainer": {}}
    ResearchCLI._set_default_root_dir(None, config)
    assert config["trainer"]["default_root_dir"] == output_path("runs")


def test_cli_keeps_an_explicit_default_root_dir():
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
