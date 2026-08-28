# Copyright (c) 2026 Vineet Venkatesh
"""Docker HEALTHCHECK helper for the serving container.

Exits 0 when GET /health returns HTTP 200, otherwise exits 1. A 503
means the checkpoint was not loaded, so the container is not ready.
"""

import sys
import urllib.error
import urllib.request

HEALTH_URL = "http://127.0.0.1:8080/health"
REQUEST_TIMEOUT_SECONDS = 5


def main():
  """Probe the local health endpoint.

  Returns:
    None

  Raises:
    SystemExit: 0 on HTTP 200, 1 on any failure.
  """
  try:
    response = urllib.request.urlopen(
        HEALTH_URL, timeout=REQUEST_TIMEOUT_SECONDS)
  except (urllib.error.URLError, urllib.error.HTTPError):
    sys.exit(1)
  sys.exit(0 if response.status == 200 else 1)


if __name__ == "__main__":
  main()
