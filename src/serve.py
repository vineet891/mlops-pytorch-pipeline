# Copyright (c) 2026 Vineet Venkatesh
"""FastAPI app that serves predictions from a training checkpoint.

GET /health returns 200 when the checkpoint is loaded and 503
otherwise. POST /predict accepts a multipart field named image and
returns class probabilities.

Usage:
  python src/serve.py
"""

import io
import os

import torch
import torch.nn.functional as functional
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from config import get_checkpoint_path, get_config
from dataset import get_dataset_spec, get_inference_transform
from logging_utils import configure_logging, log_event
from model import get_model

HOST_ENV = "SERVE_HOST"
PORT_ENV = "SERVE_PORT"
TOP_K_ENV = "TOP_K"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_TOP_K = 3
PROBABILITY_PRECISION = 6
RGB_MODE = "RGB"


class ModelService:
  """Holds the loaded model and classifies uploaded image bytes."""

  def __init__(self, config):
    """Prepare the service without loading weights yet.

    Args:
      config (dict): Validated configuration.
    """
    self.config = config
    self.checkpoint_path = get_checkpoint_path(config)
    self.device = torch.device("cpu")
    self.model = None
    self.transform = None
    self.classes = []
    self.metadata = {}

  @property
  def is_ready(self):
    """Whether a model has been loaded.

    Returns:
      bool: True once the checkpoint is in memory.
    """
    return self.model is not None

  def load(self):
    """Read the checkpoint and rebuild the model it describes.

    Returns:
      None

    Raises:
      FileNotFoundError: If the checkpoint file does not exist.
      KeyError: If the checkpoint has no model_state_dict.
    """
    if not os.path.isfile(self.checkpoint_path):
      raise FileNotFoundError(
          f"Checkpoint not found at {self.checkpoint_path}")
    checkpoint = torch.load(
        self.checkpoint_path,
        map_location=self.device,
        weights_only=False,
    )
    dataset_name = checkpoint.get(
        "dataset", self.config["data"]["dataset"])
    spec = get_dataset_spec(dataset_name)
    architecture = checkpoint.get(
        "architecture", self.config["model"]["architecture"])
    num_classes = checkpoint.get(
        "num_classes", self.config["model"]["num_classes"])
    in_channels = checkpoint.get("in_channels", spec.channels)
    model = get_model(architecture, num_classes, in_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    self.model = model
    self.transform = get_inference_transform(spec)
    self.classes = checkpoint.get("classes", list(spec.classes))
    self.metadata = {
        "architecture": architecture,
        "dataset": dataset_name,
        "num_classes": num_classes,
        "trained_epochs": checkpoint.get("epoch"),
        "val_accuracy": checkpoint.get("val_accuracy"),
    }

  def predict(self, image_bytes, top_k=DEFAULT_TOP_K):
    """Classify one image.

    Args:
      image_bytes (bytes): Raw contents of the uploaded file.
      top_k (int): Number of ranked predictions to return.

    Returns:
      dict: Predicted class, probability map and top-k ranking.

    Raises:
      RuntimeError: If the model has not been loaded.
      ValueError: If the bytes cannot be decoded as an image.
    """
    if not self.is_ready:
      raise RuntimeError("Model is not loaded")
    try:
      image = Image.open(io.BytesIO(image_bytes)).convert(RGB_MODE)
    except (UnidentifiedImageError, OSError) as error:
      raise ValueError(
          "Uploaded file is not a readable image") from error
    tensor = self.transform(image).unsqueeze(0).to(self.device)
    with torch.no_grad():
      logits = self.model(tensor)
      probabilities = functional.softmax(logits, dim=1)[0]
    ranked = torch.topk(probabilities, min(top_k, len(self.classes)))
    predicted_index = int(probabilities.argmax())
    return {
        "predicted_class": self.classes[predicted_index],
        "predicted_index": predicted_index,
        "probabilities": {
            name: round(float(value), PROBABILITY_PRECISION)
            for name, value in zip(self.classes, probabilities)
        },
        "top_k": [
            {
                "class": self.classes[int(index)],
                "probability": round(
                    float(score), PROBABILITY_PRECISION),
            }
            for score, index in zip(ranked.values, ranked.indices)
        ],
    }


def create_app():
  """Build the FastAPI application and try to load the checkpoint.

  Returns:
    FastAPI: The configured application.
  """
  configure_logging()
  service = ModelService(get_config())
  application = FastAPI(
      title="MLOps PyTorch image classifier",
      version="1.0.0",
  )
  try:
    service.load()
    log_event(
        "model_loaded",
        checkpoint=service.checkpoint_path,
        **service.metadata,
    )
  except (FileNotFoundError, KeyError, ValueError) as error:
    log_event(
        "model_load_failed",
        checkpoint=service.checkpoint_path,
        reason=str(error),
    )
  application.state.service = service

  @application.get("/health")
  def health():
    """Report whether the model is loaded.

    Returns:
      JSONResponse: 200 when ready, 503 otherwise.
    """
    if service.is_ready:
      return JSONResponse(
          status_code=200,
          content={
              "status": "ok",
              "model_loaded": True,
              "checkpoint": service.checkpoint_path,
              **service.metadata,
          },
      )
    return JSONResponse(
        status_code=503,
        content={
            "status": "unavailable",
            "model_loaded": False,
            "checkpoint": service.checkpoint_path,
        },
    )

  @application.post("/predict")
  async def predict(image: UploadFile = File(...)):
    """Return class probabilities for one uploaded image.

    Args:
      image (UploadFile): Multipart form field named image.

    Returns:
      dict: Predicted class, probability map and top-k ranking.

    Raises:
      HTTPException: 400 for an unreadable image, 503 if unloaded.
    """
    if not service.is_ready:
      raise HTTPException(status_code=503, detail="Model is not loaded")
    payload = await image.read()
    top_k = int(os.environ.get(TOP_K_ENV, DEFAULT_TOP_K))
    try:
      result = service.predict(payload, top_k=top_k)
    except ValueError as error:
      raise HTTPException(status_code=400, detail=str(error)) from error
    log_event(
        "prediction",
        filename=image.filename,
        predicted_class=result["predicted_class"],
    )
    return result

  return application


def main():
  """Start the HTTP server.

  Returns:
    None
  """
  uvicorn.run(
      create_app(),
      host=os.environ.get(HOST_ENV, DEFAULT_HOST),
      port=int(os.environ.get(PORT_ENV, DEFAULT_PORT)),
      log_level="info",
  )


if __name__ == "__main__":
  main()
