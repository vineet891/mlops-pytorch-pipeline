# Copyright (c) 2026 Vineet Venkatesh
"""Tests for dataset specs and transforms. No CIFAR-10 download."""

import pytest
import torch
from PIL import Image

from dataset import (
    CIFAR10_CHANNELS,
    CIFAR10_IMAGE_SIZE,
    get_dataset_spec,
    get_inference_transform,
    get_transforms,
    take_subset,
)

PIXEL_COLOR = (10, 20, 30)


def _rgb_image():
  """Create a dummy CIFAR-sized RGB image.

  Returns:
    PIL.Image.Image: A 32x32 RGB image.
  """
  return Image.new(
      "RGB",
      (CIFAR10_IMAGE_SIZE, CIFAR10_IMAGE_SIZE),
      color=PIXEL_COLOR,
  )


def test_cifar10_spec():
  """CIFAR-10 spec matches the assignment dataset."""
  spec = get_dataset_spec("cifar10")
  assert spec.image_size == CIFAR10_IMAGE_SIZE
  assert spec.channels == CIFAR10_CHANNELS
  assert len(spec.classes) == 10


def test_unknown_dataset_raises():
  """An unsupported dataset name must fail."""
  with pytest.raises(ValueError, match="Unsupported dataset"):
    get_dataset_spec("imagenet")


def test_eval_transform_shape():
  """Evaluation transform yields a (C, H, W) tensor."""
  spec = get_dataset_spec("cifar10")
  tensor = get_transforms(spec, train=False)(_rgb_image())
  assert tensor.shape == (
      CIFAR10_CHANNELS, CIFAR10_IMAGE_SIZE, CIFAR10_IMAGE_SIZE)
  assert tensor.dtype == torch.float32


def test_train_transform_shape():
  """Training augmentation still returns a 32x32 tensor."""
  spec = get_dataset_spec("cifar10")
  tensor = get_transforms(spec, train=True)(_rgb_image())
  assert tensor.shape == (
      CIFAR10_CHANNELS, CIFAR10_IMAGE_SIZE, CIFAR10_IMAGE_SIZE)


def test_inference_transform_resizes():
  """Uploaded images of any size are resized to 32x32."""
  spec = get_dataset_spec("cifar10")
  large = Image.new("RGB", (64, 48), color=PIXEL_COLOR)
  tensor = get_inference_transform(spec)(large)
  assert tensor.shape == (
      CIFAR10_CHANNELS, CIFAR10_IMAGE_SIZE, CIFAR10_IMAGE_SIZE)


def test_take_subset_keeps_requested_fraction():
  """Subsampling keeps a seeded slice of a fake dataset."""
  fake = list(range(100))
  subset = take_subset(fake, fraction=0.1, seed=0)
  assert len(subset) == 10
