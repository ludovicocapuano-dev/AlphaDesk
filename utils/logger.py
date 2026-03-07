"""
AlphaDesk — Logging Configuration
Structured logging with file rotation and Telegram alerts.
"""

import logging
import logging.handlers
import os
from datetime import datetime


def setup_logging(log_path: str = "logs/alphadesk.log", level: str = "INFO"):
    """Configure structured logging with console + file + rotation."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Root logger
    root = logging.getLogger("alphadesk")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Format
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler with rotation (10MB, keep 30 files)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10_000_000, backupCount=30
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Trade-specific log
    trade_logger = logging.getLogger("alphadesk.trades")
    trade_path = log_path.replace(".log", "_trades.log")
    trade_handler = logging.handlers.RotatingFileHandler(
        trade_path, maxBytes=5_000_000, backupCount=50
    )
    trade_handler.setFormatter(fmt)
    trade_logger.addHandler(trade_handler)

    root.info("Logging initialized")
    return root
