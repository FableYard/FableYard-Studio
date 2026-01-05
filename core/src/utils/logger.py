"""
Simple logger utility.

Provides a clean logging format: [HH:MM:SS - filename]: message
with color-coded log levels for better readability.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


class CustomFormatter(logging.Formatter):
    """
    Custom formatter that outputs: [HH:MM:SS - filename]: symbol message

    Features:
    - Time without date (HH:MM:SS format)
    - Just filename (not full path)
    - Color-coded by level
    """

    # ANSI color codes for terminal output
    COLORS = {
        'DEBUG': '\033[90m',    # Gray
        'INFO': '\033[94m',     # Blue
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'RESET': '\033[0m'
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with a custom format.

        Args:
            record: LogRecord to format

        Returns:
            Formatted string: [HH:MM:SS - filename]: symbol message
        """
        # Get just the filename from the caller's file path
        filename = Path(record.pathname).name

        # Format time as HH:MM:SS (no date)
        timestamp = datetime.now().strftime('%H:%M:%S')

        # Get color and symbol for this log level
        color = self.COLORS.get(record.levelname, '')
        reset = self.COLORS['RESET']

        # Build final formatted message
        formatted = f"{color}[{timestamp} - {filename}]{reset}: {record.getMessage()}"

        return formatted


def get_logger(name: str = "FableYard", level: int = logging.DEBUG) -> logging.Logger:
    """
    Get or create a logger with custom formatting.

    Uses singleton pattern - multiple calls with the same name return the same logger instance.

    Args:
        name: Logger name (default: "FableYard")
        level: Logging level (default: logging.INFO)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger("MyModule", logging.DEBUG)
        >>> logger.info("Starting process")
        [14:32:15 - myfile.py]: ℹ️ Starting process
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured (singleton pattern)
    if not logger.handlers:
        logger.setLevel(level)

        # Create console handler with custom formatter
        handler = logging.StreamHandler()
        handler.setFormatter(CustomFormatter())
        logger.addHandler(handler)

        # Prevent propagation to root logger (avoid duplicate logs)
        logger.propagate = False

    return logger


# Create default logger instance for convenience functions
_default_logger = get_logger()


def debug(msg: str) -> None:
    """
    Log a debug message.

    Args:
        msg: Message to log

    Example:
        >>> debug("Tensor shape: (B, S, D)")
        [14:32:15 - myfile.py]: Tensor shape: (B, S, D)
    """
    _default_logger.debug(msg)


def info(msg: str) -> None:
    """
    Log an info message.

    Args:
        msg: Message to log

    Example:
        >>> info("Model initialized successfully")
        [14:32:15 - myfile.py]: Model initialized successfully
    """
    _default_logger.info(msg)


def warning(msg: str) -> None:
    """
    Log a warning message.

    Args:
        msg: Message to log

    Example:
        >>> warning("Using default configuration")
        [14:32:15 - myfile.py]: Using default configuration
    """
    _default_logger.warning(msg)


def error(msg: str) -> None:
    """
    Log an error message.

    Args:
        msg: Message to log

    Example:
        >>> error("Failed to load checkpoint")
        [14:32:15 - myfile.py]: Failed to load checkpoint
    """
    _default_logger.error(msg)


def set_level(level: int) -> None:
    """
    Set the logging level for the default logger.

    Args:
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR)

    Example:
        >>> set_level(logging.DEBUG)
    """
    _default_logger.setLevel(level)
