"""
AetherForge Centralized Logging
===============================
Manages async-friendly, rotating file logs.
Splits streams into stdout (console), ops (all events), and safety (warnings/criticals).
"""

import os
import logging
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"

def setup_logging():
    """Initializes the AetherForge root logger and handlers."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # File Formatter (Detailed dates)
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-18s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Formatter (Shorter dates for readability)
    console_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-18s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # 1. Ops Log (Tracks all API requests, swaps, and decisions)
    ops_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "aetherforge.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5
    )
    ops_handler.setLevel(logging.INFO)
    ops_handler.setFormatter(file_fmt)

    # 2. Safety Audit Log (CRITICAL: Only WARNING, ERROR, CRITICAL)
    safety_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "hardware_safety.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    safety_handler.setLevel(logging.WARNING)
    safety_handler.setFormatter(file_fmt)

    # 3. Console Stream
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_fmt)

    # Configure the base 'aether' logger
    aether_logger = logging.getLogger("aether")
    aether_logger.setLevel(logging.INFO)
    aether_logger.propagate = False  # Prevent duplicate logs from hitting the root uvicorn logger

    # Attach all streams
    if not aether_logger.handlers:
        aether_logger.addHandler(ops_handler)
        aether_logger.addHandler(safety_handler)
        aether_logger.addHandler(console_handler)

def get_logger(namespace: str) -> logging.Logger:
    """
    Returns a configured logger for a specific subsystem.
    Example: get_logger('api') -> returns logger named 'aether.api'
    """
    return logging.getLogger(f"aether.{namespace}")