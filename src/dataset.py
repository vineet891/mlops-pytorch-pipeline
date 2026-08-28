# Copyright (c) 2026 Vineet Venkatesh
"""CIFAR-10 loading, transforms and DataLoader construction.

Mean and standard deviation are the standard CIFAR-10 channel
statistics. Training uses a random crop and horizontal flip; evaluation
and serving only normalise.
"""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CROP_PADDING = 4
CIFAR10_IMAGE_SIZE = 32
CIFAR10_CHANNELS = 3
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 2
DEFAULT_SUBSET_FRACTION = 1.0
DEFAULT_SEED = 42


@dataclass(frozen=True)
class DatasetSpec:
  """Static properties of CIFAR-10."""

  factory: object
  image_size: int
  channels: int
  mean: tuple
  std: tuple
  classes: tuple


CIFAR10_SPEC = DatasetSpec(
    factory=datasets.CIFAR10,
    image_size=CIFAR10_IMAGE_SIZE,
    channels=CIFAR10_CHANNELS,
    mean=CIFAR10_MEAN,
    std=CIFAR10_STD,
    classes=CIFAR10_CLASSES,
)


def get_dataset_spec(dataset_name):
  """Return the specification for a supported dataset.

  Args:
    dataset_name (str): Dataset key. Only cifar10 is supported.

  Returns:
    DatasetSpec: The matching specification.

  Raises:
    ValueError: If the dataset is not supported.
  """
  if dataset_name.lower() != "cifar10":
    raise ValueError(
        f"Unsupported dataset '{dataset_name}'. Only cifar10 is supported")
  return CIFAR10_SPEC


def get_transforms(spec, train=True):
  """Build the transform pipeline for one split.

  Args:
    spec (DatasetSpec): Specification of the dataset in use.
    train (bool): Whether to include training augmentation.

  Returns:
    transforms.Compose: The transform pipeline.
  """
  steps = []
  if train:
    steps.extend([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(spec.image_size, padding=CROP_PADDING),
    ])
  steps.extend([
      transforms.ToTensor(),
      transforms.Normalize(mean=spec.mean, std=spec.std),
  ])
  return transforms.Compose(steps)


def get_inference_transform(spec):
  """Build the transform used on a single uploaded image.

  Args:
    spec (DatasetSpec): Specification of the dataset in use.

  Returns:
    transforms.Compose: Resize, tensor conversion and normalisation.
  """
  return transforms.Compose([
      transforms.Resize((spec.image_size, spec.image_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean=spec.mean, std=spec.std),
  ])


def take_subset(dataset, fraction, seed):
  """Return a random subset so laptop runs can stay short.

  Args:
    dataset (torch.utils.data.Dataset): Dataset to sample from.
    fraction (float): Share of samples to keep, in (0, 1].
    seed (int): Seed controlling which samples are kept.

  Returns:
    torch.utils.data.Dataset: The original dataset or a subset of it.
  """
  if fraction >= 1:
    return dataset
  generator = torch.Generator().manual_seed(seed)
  keep = max(1, int(len(dataset) * fraction))
  indices = torch.randperm(len(dataset), generator=generator)[:keep]
  return Subset(dataset, indices.tolist())


def get_dataloaders(
    spec,
    data_dir,
    batch_size=DEFAULT_BATCH_SIZE,
    num_workers=DEFAULT_NUM_WORKERS,
    subset_fraction=DEFAULT_SUBSET_FRACTION,
    seed=DEFAULT_SEED,
    download=True,
):
  """Create the training and validation dataloaders.

  Args:
    spec (DatasetSpec): Specification of the dataset in use.
    data_dir (str): Directory holding or receiving the raw data.
    batch_size (int): Samples per batch.
    num_workers (int): Worker processes per loader.
    subset_fraction (float): Share of each split to use, in (0, 1].
    seed (int): Seed used when subsampling.
    download (bool): Whether to download the data if it is missing.

  Returns:
    tuple: A (train_loader, val_loader) pair.
  """
  train_dataset = spec.factory(
      root=data_dir,
      train=True,
      download=download,
      transform=get_transforms(spec, train=True),
  )
  val_dataset = spec.factory(
      root=data_dir,
      train=False,
      download=download,
      transform=get_transforms(spec, train=False),
  )
  train_dataset = take_subset(train_dataset, subset_fraction, seed)
  val_dataset = take_subset(val_dataset, subset_fraction, seed)
  train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=False,
  )
  val_loader = DataLoader(
      val_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=False,
  )
  return train_loader, val_loader
