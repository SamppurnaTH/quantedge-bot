"""
Environment variable loader.
Reads .env file from the project root and injects into os.environ.
Must be called before any module that reads env vars (e.g. Telegram).
"""

import os
import logging

logger = logging.getLogger(__name__)

_ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")


def load_env() -> None:
    """
    Parse .env file and set variables into os.environ.
    Skips lines that are comments (#) or empty.
    Does NOT override variables already set in the environment.
    """
    if not os.path.exists(_ENV_FILE):
        logger.debug(".env file not found at %s — skipping", _ENV_FILE)
        return

    loaded = 0
    with open(_ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip()
            if key and value and key not in os.environ:
                os.environ[key] = value
                loaded += 1

    if loaded:
        logger.debug("Loaded %d variable(s) from .env", loaded)
