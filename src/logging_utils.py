# Copyright (c) 2026 Vineet Venkatesh
"""JSON-line logging used by training and serving.

The assignment asks for metrics as one JSON object per line on stdout.
This module writes those lines through the logging library so the
format stays structured without using print.
"""

import json
import logging
import sys

LOGGER_NAME = "mlops"
LOG_FORMAT = "%(message)s"


def configure_logging(level=logging.INFO):
  """Attach a stdout handler that writes the message only.

  Args:
    level (int): Logging level for the pipeline logger.

  Returns:
    logging.Logger: The configured logger.
  """
  handler = logging.StreamHandler(sys.stdout)
  handler.setFormatter(logging.Formatter(LOG_FORMAT))
  logger = logging.getLogger(LOGGER_NAME)
  logger.handlers.clear()
  logger.addHandler(handler)
  logger.setLevel(level)
  logger.propagate = False
  return logger


def get_logger():
  """Return the shared pipeline logger.

  Returns:
    logging.Logger: The logger used by all pipeline modules.
  """
  return logging.getLogger(LOGGER_NAME)


def log_event(event, **fields):
  """Emit one JSON line describing an event.

  Args:
    event (str): Short name of the event, for example epoch_metrics.
    **fields: Extra key/value pairs included in the JSON object.

  Returns:
    None
  """
  payload = {"event": event}
  payload.update(fields)
  get_logger().info(json.dumps(payload))
