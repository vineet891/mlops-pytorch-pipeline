# Copyright (c) 2026 Vineet Venkatesh
"""Checks that the checked-in training config has the required shape.

These tests do not import PyTorch. They exist so Part A has a real
pytest target before the model code is added.
"""

from pathlib import Path

import yaml

from config import get_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "training_config.yaml"

REQUIRED_SECTIONS = ("model", "training", "data", "output")
REQUIRED_KEYS = {
    "model": ("architecture", "num_classes"),
    "training": (
        "epochs",
        "batch_size",
        "learning_rate",
        "early_stopping_patience",
    ),
    "data": ("dataset", "data_dir"),
    "output": ("checkpoint_dir", "model_name"),
}


def _load_config():
  """Read the repository training configuration.

  Returns:
    dict: Parsed YAML mapping.

  Raises:
    AssertionError: If the file is missing or is not a mapping.
  """
  assert CONFIG_PATH.is_file(), f"Missing config at {CONFIG_PATH}"
  with CONFIG_PATH.open(encoding="utf-8") as handle:
    parsed = yaml.safe_load(handle)
  assert isinstance(parsed, dict), "Config must be a YAML mapping"
  return parsed


def test_config_has_required_sections():
  """Every top-level section used by train.py must be present."""
  config = _load_config()
  for section in REQUIRED_SECTIONS:
    assert section in config, f"Missing section: {section}"
    assert isinstance(config[section], dict), (
        f"Section {section} must be a mapping")


def test_config_has_required_keys():
  """Every key the assignment names must be present and non-empty."""
  config = _load_config()
  for section, keys in REQUIRED_KEYS.items():
    for key in keys:
      assert key in config[section], f"Missing key: {section}.{key}"
      assert config[section][key] not in (None, ""), (
          f"Empty value: {section}.{key}")


def test_config_numeric_fields_are_positive():
  """Epochs, batch size, learning rate and class count must be usable."""
  config = _load_config()
  assert config["model"]["num_classes"] > 0
  assert config["training"]["epochs"] > 0
  assert config["training"]["batch_size"] > 0
  assert config["training"]["learning_rate"] > 0
  assert config["training"]["early_stopping_patience"] > 0


def test_get_config_loads_repo_file():
  """The loader accepts the checked-in YAML file."""
  config = get_config(str(CONFIG_PATH))
  assert config["model"]["architecture"] == "resnet18"
  assert config["data"]["dataset"] == "cifar10"


def test_env_override_epochs(monkeypatch):
  """TRAIN_EPOCHS replaces the YAML value without editing the file."""
  monkeypatch.setenv("TRAIN_EPOCHS", "2")
  config = get_config(str(CONFIG_PATH))
  assert config["training"]["epochs"] == 2
