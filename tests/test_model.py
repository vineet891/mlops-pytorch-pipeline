# Copyright (c) 2026 Vineet Venkatesh
"""Unit tests for model factories and forward-pass shapes."""

import pytest
import torch

from model import (
    SUPPORTED_ARCHITECTURES,
    count_parameters,
    get_model,
)

BATCH_SIZE = 2
NUM_CLASSES = 10
IMAGE_SIZE = 32
CHANNELS = 3


def _dummy_batch():
  """Build a random image batch matching CIFAR-10 shape.

  Returns:
    torch.Tensor: Tensor of shape (N, C, H, W).
  """
  return torch.randn(BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)


def test_resnet18_output_shape():
  """ResNet-18 maps a CIFAR-10 batch to class logits."""
  model = get_model("resnet18", NUM_CLASSES, CHANNELS)
  model.eval()
  with torch.no_grad():
    outputs = model(_dummy_batch())
  assert outputs.shape == (BATCH_SIZE, NUM_CLASSES)


def test_simple_cnn_output_shape():
  """SimpleCNN maps a CIFAR-10 batch to class logits."""
  model = get_model("simple_cnn", NUM_CLASSES, CHANNELS)
  model.eval()
  with torch.no_grad():
    outputs = model(_dummy_batch())
  assert outputs.shape == (BATCH_SIZE, NUM_CLASSES)


def test_unknown_architecture_raises():
  """An unsupported name must fail with a clear error."""
  with pytest.raises(ValueError, match="Unsupported architecture"):
    get_model("not_a_model", NUM_CLASSES, CHANNELS)


def test_supported_architectures_are_constructable():
  """Every advertised architecture builds and has parameters."""
  for architecture in SUPPORTED_ARCHITECTURES:
    model = get_model(architecture, NUM_CLASSES, CHANNELS)
    assert count_parameters(model) > 0
