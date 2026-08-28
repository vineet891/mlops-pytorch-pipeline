# Copyright (c) 2026 Vineet Venkatesh
"""Write a small PNG used to exercise POST /predict.

This does not download CIFAR-10. The serving transform resizes any
RGB image to 32x32, so a solid-colour PNG is enough to prove the
endpoint works.

Usage:
  python scripts/make_test_image.py
"""

import argparse
from pathlib import Path

from PIL import Image

DEFAULT_PATH = Path("test_image.png")
DEFAULT_SIZE = 32
DEFAULT_COLOR = (255, 0, 0)


def parse_args():
  """Parse the command line arguments.

  Returns:
    argparse.Namespace: Parsed arguments.
  """
  parser = argparse.ArgumentParser(
      description="Write a small PNG for the predict endpoint")
  parser.add_argument(
      "--output",
      default=str(DEFAULT_PATH),
      help="Destination PNG path")
  parser.add_argument(
      "--size",
      type=int,
      default=DEFAULT_SIZE,
      help="Width and height in pixels")
  return parser.parse_args()


def write_test_image(path, size):
  """Create a solid RGB PNG.

  Args:
    path (str): Destination file.
    size (int): Width and height in pixels.

  Returns:
    None
  """
  image = Image.new("RGB", (size, size), color=DEFAULT_COLOR)
  image.save(path, format="PNG")


def main():
  """Write the test image to disk.

  Returns:
    None
  """
  args = parse_args()
  write_test_image(args.output, args.size)


if __name__ == "__main__":
  main()
