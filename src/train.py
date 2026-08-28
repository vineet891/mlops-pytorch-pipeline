# Copyright (c) 2026 Vineet Venkatesh
"""Training entry point for the image classification pipeline.

Hyperparameters come from configs/training_config.yaml. Each epoch is
logged as one JSON object on stdout. The best checkpoint is written to
the configured output path. Training stops early when validation loss
does not improve for early_stopping_patience epochs.

Usage:
  python src/train.py
  python src/train.py --config configs/training_config.yaml
"""

import argparse
import os
import time

import torch
import torch.nn as nn

from config import get_checkpoint_path, get_config
from dataset import get_dataloaders, get_dataset_spec
from logging_utils import configure_logging, log_event
from model import count_parameters, get_model

METRIC_PRECISION = 4


def parse_args():
  """Parse the command line arguments.

  Returns:
    argparse.Namespace: Parsed arguments.
  """
  parser = argparse.ArgumentParser(
      description="Train an image classifier from a YAML config")
  parser.add_argument(
      "--config",
      default=None,
      help="Path to the training configuration file")
  return parser.parse_args()


def select_device():
  """Pick the best device available on the current machine.

  Returns:
    torch.device: CUDA if present, otherwise Apple MPS, otherwise CPU.
  """
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def set_seed(seed):
  """Seed the random number generators used during training.

  Args:
    seed (int): Seed value.

  Returns:
    None
  """
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
  """Run one pass over the training data.

  Args:
    model (nn.Module): Model being trained.
    loader (torch.utils.data.DataLoader): Training data.
    optimizer (torch.optim.Optimizer): Optimiser to step.
    criterion (nn.Module): Loss function.
    device (torch.device): Device to compute on.

  Returns:
    tuple: Average loss and accuracy over the epoch.
  """
  model.train()
  total_loss = 0.0
  correct = 0
  total = 0
  for inputs, targets in loader:
    inputs = inputs.to(device)
    targets = targets.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    total_loss += loss.item() * inputs.size(0)
    correct += outputs.argmax(1).eq(targets).sum().item()
    total += targets.size(0)
  return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
  """Score the model on the validation data.

  Args:
    model (nn.Module): Model being evaluated.
    loader (torch.utils.data.DataLoader): Validation data.
    criterion (nn.Module): Loss function.
    device (torch.device): Device to compute on.

  Returns:
    tuple: Average loss and accuracy over the split.
  """
  model.eval()
  total_loss = 0.0
  correct = 0
  total = 0
  for inputs, targets in loader:
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    total_loss += loss.item() * inputs.size(0)
    correct += outputs.argmax(1).eq(targets).sum().item()
    total += targets.size(0)
  return total_loss / total, correct / total


def save_checkpoint(path, model, optimizer, config, spec, metrics):
  """Write the model and metadata needed to rebuild it at serve time.

  Args:
    path (str): Destination file.
    model (nn.Module): Trained model.
    optimizer (torch.optim.Optimizer): Optimiser state to keep.
    config (dict): Validated configuration.
    spec: DatasetSpec of the dataset in use.
    metrics (dict): Epoch number and validation metrics.

  Returns:
    None
  """
  torch.save({
      "epoch": metrics["epoch"],
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "val_loss": metrics["val_loss"],
      "val_accuracy": metrics["val_accuracy"],
      "architecture": config["model"]["architecture"],
      "num_classes": config["model"]["num_classes"],
      "in_channels": spec.channels,
      "dataset": config["data"]["dataset"],
      "classes": list(spec.classes),
  }, path)


def run_training(config):
  """Train the model described by the configuration.

  Args:
    config (dict): Validated configuration.

  Returns:
    dict: Metrics of the best epoch.
  """
  set_seed(config["training"]["seed"])
  device = select_device()
  spec = get_dataset_spec(config["data"]["dataset"])
  model = get_model(
      architecture=config["model"]["architecture"],
      num_classes=config["model"]["num_classes"],
      in_channels=spec.channels,
  ).to(device)
  log_event(
      "training_started",
      device=str(device),
      architecture=config["model"]["architecture"],
      dataset=config["data"]["dataset"],
      parameters=count_parameters(model),
      epochs=config["training"]["epochs"],
      batch_size=config["training"]["batch_size"],
      subset_fraction=config["data"]["subset_fraction"],
  )
  train_loader, val_loader = get_dataloaders(
      spec=spec,
      data_dir=config["data"]["data_dir"],
      batch_size=config["training"]["batch_size"],
      num_workers=config["data"]["num_workers"],
      subset_fraction=config["data"]["subset_fraction"],
      seed=config["training"]["seed"],
      download=config["data"]["download"],
  )
  optimizer = torch.optim.Adam(
      model.parameters(),
      lr=config["training"]["learning_rate"],
  )
  criterion = nn.CrossEntropyLoss()
  checkpoint_dir = config["output"]["checkpoint_dir"]
  os.makedirs(checkpoint_dir, exist_ok=True)
  checkpoint_path = get_checkpoint_path(config)
  patience = config["training"]["early_stopping_patience"]
  best = {"val_loss": float("inf")}
  patience_counter = 0

  for epoch in range(1, config["training"]["epochs"] + 1):
    started = time.time()
    train_loss, train_accuracy = train_one_epoch(
        model, train_loader, optimizer, criterion, device)
    val_loss, val_accuracy = evaluate(
        model, val_loader, criterion, device)
    metrics = {
        "epoch": epoch,
        "train_loss": round(train_loss, METRIC_PRECISION),
        "train_accuracy": round(train_accuracy, METRIC_PRECISION),
        "val_loss": round(val_loss, METRIC_PRECISION),
        "val_accuracy": round(val_accuracy, METRIC_PRECISION),
        "seconds": round(time.time() - started, 1),
    }
    log_event("epoch_metrics", **metrics)
    if val_loss < best["val_loss"]:
      best = metrics
      patience_counter = 0
      save_checkpoint(
          checkpoint_path, model, optimizer, config, spec, metrics)
      log_event(
          "checkpoint_saved", path=checkpoint_path, epoch=epoch)
    else:
      patience_counter += 1
      if patience_counter >= patience:
        log_event("early_stopping", epoch=epoch, patience=patience)
        break

  log_event(
      "training_complete",
      best_epoch=best.get("epoch"),
      best_val_loss=best["val_loss"],
      best_val_accuracy=best.get("val_accuracy"),
      checkpoint=checkpoint_path,
  )
  return best


def main():
  """Load the configuration and run training.

  Returns:
    None
  """
  configure_logging()
  args = parse_args()
  config = get_config(args.config)
  run_training(config)


if __name__ == "__main__":
  main()
