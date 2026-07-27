"""Smoke tests for experiment configs.

Every YAML under conf/experiment/ must parse and expose the sections the CLI
relies on. This catches broken YAML and structural regressions without
instantiating anything (no model downloads, no dataset access).
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[1] / "conf" / "experiment"
CONFIG_FILES = sorted(CONFIG_DIR.glob("*.yaml"))


def test_experiment_configs_exist():
    assert CONFIG_FILES, f"No experiment configs found under {CONFIG_DIR}"


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.name)
def test_experiment_config_parses(config_path):
    cfg = OmegaConf.load(config_path)

    assert "model" in cfg, f"{config_path.name}: missing 'model' section"
    assert "data" in cfg, f"{config_path.name}: missing 'data' section"
    assert cfg.model.get("class_path"), f"{config_path.name}: model missing class_path"
    assert cfg.data.get("class_path"), f"{config_path.name}: data missing class_path"
