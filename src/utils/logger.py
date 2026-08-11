"""Project-wide logger factory."""

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Create and configure a named logger.

    Args:
        name: Logger name (usually __name__)
        level: Log level string ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional path to write logs to disk

    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # Optional file handler
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger
