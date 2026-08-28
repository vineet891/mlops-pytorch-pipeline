# Copyright (c) 2026 Vineet Venkatesh
"""Model factories for training and serving.

resnet18 is the torchvision model with its stem changed for 32x32
CIFAR-10 images. The default 7x7 stride-2 convolution plus max-pool
would shrink a 32x32 input to 8x8 before the residual blocks; a 3x3
stride-1 stem keeps the spatial size. simple_cnn is a small network
used in unit tests so they do not need ImageNet-scale weights.
"""

import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHITECTURES = ("resnet18", "simple_cnn")
STEM_KERNEL_SIZE = 3
STEM_STRIDE = 1
STEM_PADDING = 1
SIMPLE_CNN_WIDTHS = (32, 64, 128)
CONV_KERNEL_SIZE = 3
CONV_PADDING = 1
POOL_KERNEL_SIZE = 2


class SimpleCNN(nn.Module):
  """Three-block convolutional classifier for 32x32 images."""

  def __init__(self, num_classes, in_channels=3):
    """Build the network.

    Args:
      num_classes (int): Number of output classes.
      in_channels (int): Number of input image channels.
    """
    super().__init__()
    blocks = []
    channels = in_channels
    for width in SIMPLE_CNN_WIDTHS:
      blocks.append(self._build_block(channels, width))
      channels = width
    self.features = nn.Sequential(*blocks)
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(channels, num_classes)

  @staticmethod
  def _build_block(in_channels, out_channels):
    """Create convolution, normalisation, activation and pooling.

    Args:
      in_channels (int): Channels entering the block.
      out_channels (int): Channels leaving the block.

    Returns:
      nn.Sequential: The assembled block.
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=CONV_KERNEL_SIZE,
            padding=CONV_PADDING,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(POOL_KERNEL_SIZE),
    )

  def forward(self, inputs):
    """Run a forward pass.

    Args:
      inputs (torch.Tensor): Batch of images, shape (N, C, H, W).

    Returns:
      torch.Tensor: Class logits, shape (N, num_classes).
    """
    features = self.pool(self.features(inputs))
    return self.classifier(features.flatten(1))


def build_resnet18(num_classes, in_channels=3):
  """Create a ResNet-18 adapted to small input images.

  Args:
    num_classes (int): Number of output classes.
    in_channels (int): Number of input image channels.

  Returns:
    nn.Module: The adapted ResNet-18.
  """
  network = models.resnet18(weights=None, num_classes=num_classes)
  network.conv1 = nn.Conv2d(
      in_channels,
      network.conv1.out_channels,
      kernel_size=STEM_KERNEL_SIZE,
      stride=STEM_STRIDE,
      padding=STEM_PADDING,
      bias=False,
  )
  network.maxpool = nn.Identity()
  return network


def get_model(architecture, num_classes, in_channels=3):
  """Create a model by name.

  Args:
    architecture (str): One of SUPPORTED_ARCHITECTURES.
    num_classes (int): Number of output classes.
    in_channels (int): Number of input image channels.

  Returns:
    nn.Module: The requested model.

  Raises:
    ValueError: If the architecture name is not supported.
  """
  if architecture == "resnet18":
    return build_resnet18(num_classes, in_channels)
  if architecture == "simple_cnn":
    return SimpleCNN(num_classes, in_channels)
  names = ", ".join(SUPPORTED_ARCHITECTURES)
  raise ValueError(
      f"Unsupported architecture '{architecture}'. Choose one of {names}")


def count_parameters(model):
  """Count the trainable parameters of a model.

  Args:
    model (nn.Module): Model to inspect.

  Returns:
    int: Number of trainable parameters.
  """
  return sum(
      parameter.numel()
      for parameter in model.parameters()
      if parameter.requires_grad
  )
