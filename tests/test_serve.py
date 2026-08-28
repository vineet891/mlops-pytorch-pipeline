# Copyright (c) 2026 Vineet Venkatesh
"""HTTP tests for the serving app using an in-memory checkpoint."""

import io

import torch
from fastapi.testclient import TestClient
from PIL import Image

from dataset import CIFAR10_CLASSES
from model import get_model
from serve import create_app

NUM_CLASSES = 10
CHANNELS = 3
IMAGE_SIZE = 32


def _write_checkpoint(path):
  """Write a randomly initialised simple_cnn checkpoint.

  Args:
    path (pathlib.Path): Destination file.

  Returns:
    None
  """
  model = get_model("simple_cnn", NUM_CLASSES, CHANNELS)
  torch.save({
      "epoch": 1,
      "model_state_dict": model.state_dict(),
      "architecture": "simple_cnn",
      "num_classes": NUM_CLASSES,
      "in_channels": CHANNELS,
      "dataset": "cifar10",
      "classes": list(CIFAR10_CLASSES),
      "val_loss": 2.3,
      "val_accuracy": 0.1,
  }, path)


def _png_bytes():
  """Encode a 32x32 PNG for the predict endpoint.

  Returns:
    bytes: PNG file contents.
  """
  buffer = io.BytesIO()
  Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(255, 0, 0)).save(
      buffer, format="PNG")
  return buffer.getvalue()


def test_health_without_checkpoint(monkeypatch):
  """Missing weights make /health return 503."""
  monkeypatch.setenv("CHECKPOINT_PATH", "/tmp/missing_classifier.pt")
  client = TestClient(create_app())
  response = client.get("/health")
  assert response.status_code == 503
  assert response.json()["model_loaded"] is False


def test_health_and_predict_with_checkpoint(monkeypatch, tmp_path):
  """A valid checkpoint makes /health 200 and /predict return classes."""
  checkpoint = tmp_path / "classifier_v1.pt"
  _write_checkpoint(checkpoint)
  monkeypatch.setenv("CHECKPOINT_PATH", str(checkpoint))
  client = TestClient(create_app())
  health = client.get("/health")
  assert health.status_code == 200
  body = health.json()
  assert body["model_loaded"] is True
  assert body["architecture"] == "simple_cnn"

  prediction = client.post(
      "/predict",
      files={"image": ("test.png", _png_bytes(), "image/png")},
  )
  assert prediction.status_code == 200
  payload = prediction.json()
  assert payload["predicted_class"] in CIFAR10_CLASSES
  assert len(payload["probabilities"]) == NUM_CLASSES
  assert abs(sum(payload["probabilities"].values()) - 1.0) < 1e-5


def test_predict_rejects_non_image(monkeypatch, tmp_path):
  """A non-image upload is rejected with 400."""
  checkpoint = tmp_path / "classifier_v1.pt"
  _write_checkpoint(checkpoint)
  monkeypatch.setenv("CHECKPOINT_PATH", str(checkpoint))
  client = TestClient(create_app())
  response = client.post(
      "/predict",
      files={"image": ("notes.txt", b"not an image", "text/plain")},
  )
  assert response.status_code == 400
