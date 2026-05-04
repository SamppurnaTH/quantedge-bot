"""
Phase 7 — Logging Setup
Configures both file and console logging for the entire system.
"""

import os
import logging
import logging.handlers
from config.settings import LOG_CONFIG


def setup_logging() -> logging.Logger:
    """
    Configure root logger with:
      - Console handler (INFO level)
      - Rotating file handler (DEBUG level, max 5 MB × 3 backups)

    Returns:
        Root logger instance
    """
    os.makedirs(LOG_CONFIG["log_dir"], exist_ok=True)
    log_path = os.path.join(LOG_CONFIG["log_dir"], LOG_CONFIG["log_file"])

    level = getattr(logging, LOG_CONFIG["level"].upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # capture everything; handlers filter

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)

    # ── File handler (rotating) ───────────────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)

    # Avoid duplicate handlers on re-import
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

    return root_logger
