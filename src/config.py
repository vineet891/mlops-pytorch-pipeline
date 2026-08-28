# Copyright (c) 2026 Vineet Venkatesh
"""Load, validate and optionally override the training configuration.

Training and serving read the same YAML file. Inside a container the
file is expected at /app/configs/training_config.yaml (ConfigMap).
Locally it is the copy in configs/. CONFIG_PATH or --config wins.
"""

import os

import yaml

from logging_utils import log_event

CONFIG_PATH_ENV = "CONFIG_PATH"
CHECKPOINT_PATH_ENV = "CHECKPOINT_PATH"
CANDIDATE_CONFIG_PATHS = (
    "/app/configs/training_config.yaml",
    "configs/training_config.yaml",
)

DEFAULTS = {
    "data": {
        "num_workers": 2,
        "subset_fraction": 1.0,
        "download": True,
    },
    "training": {
        "seed": 42,
    },
}

ENV_OVERRIDES = (
    ("TRAIN_EPOCHS", "training", "epochs", int),
    ("TRAIN_BATCH_SIZE", "training", "batch_size", int),
    ("TRAIN_LEARNING_RATE", "training", "learning_rate", float),
    ("TRAIN_PATIENCE", "training", "early_stopping_patience", int),
    ("TRAIN_SEED", "training", "seed", int),
    ("TRAIN_DATASET", "data", "dataset", str),
    ("TRAIN_DATA_DIR", "data", "data_dir", str),
    ("TRAIN_SUBSET_FRACTION", "data", "subset_fraction", float),
    ("TRAIN_NUM_WORKERS", "data", "num_workers", int),
    ("CHECKPOINT_DIR", "output", "checkpoint_dir", str),
    ("MODEL_NAME", "output", "model_name", str),
)

REQUIRED_FIELDS = (
    ("model", "architecture", str),
    ("model", "num_classes", int),
    ("training", "epochs", int),
    ("training", "batch_size", int),
    ("training", "learning_rate", float),
    ("training", "early_stopping_patience", int),
    ("data", "dataset", str),
    ("data", "data_dir", str),
    ("output", "checkpoint_dir", str),
    ("output", "model_name", str),
)

POSITIVE_FIELDS = (
    ("model", "num_classes"),
    ("training", "epochs"),
    ("training", "batch_size"),
    ("training", "learning_rate"),
    ("training", "early_stopping_patience"),
)


def resolve_config_path(explicit_path=None):
  """Choose which configuration file to read.

  Order: explicit path, CONFIG_PATH, container path, repo path.

  Args:
    explicit_path (str): Path from --config, or None.

  Returns:
    str: Path to an existing configuration file.

  Raises:
    FileNotFoundError: If none of the candidate paths exist.
  """
  candidates = []
  if explicit_path:
    candidates.append(explicit_path)
  env_path = os.environ.get(CONFIG_PATH_ENV)
  if env_path:
    candidates.append(env_path)
  candidates.extend(CANDIDATE_CONFIG_PATHS)
  for candidate in candidates:
    if os.path.isfile(candidate):
      return candidate
  joined = ", ".join(candidates)
  raise FileNotFoundError(f"No configuration file in: {joined}")


def load_config(config_path):
  """Parse a YAML configuration file.

  Args:
    config_path (str): Path to the YAML file.

  Returns:
    dict: The parsed configuration.

  Raises:
    ValueError: If the file is empty or is not a mapping.
    yaml.YAMLError: If the file is not valid YAML.
  """
  with open(config_path, encoding="utf-8") as handle:
    parsed = yaml.safe_load(handle)
  if not isinstance(parsed, dict):
    raise ValueError(f"Configuration in {config_path} is not a mapping")
  return parsed


def apply_defaults(config):
  """Fill optional keys that the file does not set.

  Args:
    config (dict): Parsed configuration.

  Returns:
    dict: The same dictionary with defaults applied.
  """
  for section, defaults in DEFAULTS.items():
    section_values = config.setdefault(section, {})
    for key, value in defaults.items():
      section_values.setdefault(key, value)
  return config


def apply_env_overrides(config):
  """Override configuration values from the environment.

  A container can change epochs or paths without a rebuild.

  Args:
    config (dict): Parsed configuration.

  Returns:
    dict: The same dictionary with environment values applied.

  Raises:
    ValueError: If an environment value cannot be cast.
  """
  for variable, section, key, caster in ENV_OVERRIDES:
    raw_value = os.environ.get(variable)
    if raw_value is None or raw_value == "":
      continue
    try:
      config.setdefault(section, {})[key] = caster(raw_value)
    except ValueError as error:
      type_name = caster.__name__
      raise ValueError(
          f"{variable}={raw_value} is not a valid {type_name}"
      ) from error
    log_event(
        "config_override",
        key=f"{section}.{key}",
        source=variable,
        value=config[section][key],
    )
  return config


def validate_config(config):
  """Check that every required field is present and usable.

  Args:
    config (dict): Parsed configuration.

  Returns:
    None

  Raises:
    ValueError: If a section or key is missing, has the wrong type,
      or holds a value outside its allowed range.
  """
  for section, key, expected in REQUIRED_FIELDS:
    if not isinstance(config.get(section), dict):
      raise ValueError(f"Missing configuration section: {section}")
    if key not in config[section]:
      raise ValueError(f"Missing configuration key: {section}.{key}")
    value = config[section][key]
    if isinstance(value, bool):
      raise ValueError(f"Key {section}.{key} must not be a boolean")
    if expected is float and isinstance(value, int):
      config[section][key] = float(value)
      continue
    if not isinstance(value, expected):
      expected_name = expected.__name__
      raise ValueError(
          f"Key {section}.{key} must be of type {expected_name}")
  for section, key in POSITIVE_FIELDS:
    if config[section][key] <= 0:
      raise ValueError(f"Key {section}.{key} must be greater than zero")
  fraction = config["data"]["subset_fraction"]
  if not 0 < fraction <= 1:
    raise ValueError("Key data.subset_fraction must be in (0, 1]")


def get_config(explicit_path=None):
  """Load, complete, override and validate the configuration.

  Args:
    explicit_path (str): Path passed on the command line, or None.

  Returns:
    dict: A validated configuration dictionary.

  Raises:
    FileNotFoundError: If no configuration file can be located.
    ValueError: If the configuration is invalid.
  """
  config_path = resolve_config_path(explicit_path)
  config = load_config(config_path)
  config = apply_defaults(config)
  config = apply_env_overrides(config)
  validate_config(config)
  log_event("config_loaded", path=config_path)
  return config


def get_checkpoint_path(config):
  """Build the full path of the model checkpoint.

  CHECKPOINT_PATH overrides the path derived from the YAML file.

  Args:
    config (dict): Validated configuration.

  Returns:
    str: Path to the checkpoint file.
  """
  override = os.environ.get(CHECKPOINT_PATH_ENV)
  if override:
    return override
  directory = config["output"]["checkpoint_dir"]
  name = config["output"]["model_name"]
  return os.path.join(directory, name)
