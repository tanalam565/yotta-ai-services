"""
Cross-platform backend startup entrypoint.

Linux/macOS: starts Gunicorn with Uvicorn workers.
Windows/IIS: starts Uvicorn directly with multi-worker mode.
"""

import sys
import os
import multiprocessing
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_worker_count():
    """Return recommended worker count using the standard `(2 * CPU) + 1` formula."""
    return multiprocessing.cpu_count() * 2 + 1


def start_linux():
    """
    Start API server using Gunicorn with Uvicorn workers on Linux/macOS.

    Falls back to direct Uvicorn startup if Gunicorn is not installed.
    """
    try:
        import gunicorn.app.base
    except ImportError:
        logger.warning("Gunicorn not found, falling back to Uvicorn")
        start_uvicorn()
        return

    logger.info(f"Starting with Gunicorn ({get_worker_count()} workers) on Linux")
    os.execlp(
        "gunicorn",
        "gunicorn",
        "-c", "gunicorn.conf.py",
        "main:app"
    )


def start_uvicorn():
    """
    Start API server with Uvicorn directly.

    Used for Windows/IIS environments and as a fallback when Gunicorn
    is unavailable.
    """
    workers = get_worker_count()
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = os.getenv("APP_PORT", "8000")

    logger.info(f"Starting with Uvicorn ({workers} workers) on Windows/IIS")
    logger.info(f"Binding to {host}:{port}")

    try:
        import uvicorn
    except ImportError:
        logger.error("Uvicorn not found. Run: pip install uvicorn[standard]")
        sys.exit(1)

    # On Windows, uvicorn --workers requires Python 3.8+ and no --reload
    os.execlp(
        sys.executable,
        sys.executable,
        "-m", "uvicorn",
        "main:app",
        "--host", host,
        "--port", port,
        "--workers", str(workers)
    )


if __name__ == "__main__":
    logger.info(
        f"Starting YottaReal backend on {sys.platform} with {get_worker_count()} workers"
    )

    if sys.platform.startswith("win"):
        start_uvicorn()
    else:
        start_linux()