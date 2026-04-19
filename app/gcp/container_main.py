"""
Single container image entrypoint: FastAPI or Pub/Sub worker.

Environment:
  SPOTBALLER_CONTAINER_ROLE — ``worker`` (default) or ``api``

API-only:
  SPOTBALLER_API_HOST (default 0.0.0.0)
  SPOTBALLER_API_PORT (default 8000)
"""
from __future__ import annotations

import logging
import os
import sys

LOG = logging.getLogger("spotballer.gcp.container")


def main() -> None:
    role = (os.environ.get("SPOTBALLER_CONTAINER_ROLE") or "worker").strip().lower()
    if role in ("worker", "gpu-worker"):
        from app.gcp.worker_main import main as worker_main

        worker_main()
        return
    if role in ("api", "server", "http"):
        import uvicorn

        host = os.environ.get("SPOTBALLER_API_HOST", "0.0.0.0").strip() or "0.0.0.0"
        port = int(os.environ.get("SPOTBALLER_API_PORT", "8000"))
        log_level = os.environ.get("UVICORN_LOG_LEVEL", "info").strip().lower() or "info"
        LOG.info("starting API on %s:%s", host, port)
        uvicorn.run("app.api.main:app", host=host, port=port, log_level=log_level)
        return
    LOG.error("Unknown SPOTBALLER_CONTAINER_ROLE=%r (use worker or api)", role)
    sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
