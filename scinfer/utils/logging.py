"""Logging configuration using loguru.

Provides a centralised logging setup for scInfer that can be configured
via the framework's YAML configuration or programmatically.
"""

from __future__ import annotations

import sys
from typing import Optional, Union

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    colorize: bool = True,
) -> None:
    """Configure the scInfer logging system.

    Removes the default loguru handler and adds a new one with the
    specified settings.  Optionally also logs to a file.

    Parameters
    ----------
    level : str
        Minimum log level: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``, ``"CRITICAL"``.
    log_file : str, optional
        Path to a log file.  If ``None``, only stderr output is used.
    format_string : str, optional
        Custom log format string.  Defaults to a concise coloured format.
    colorize : bool
        Whether to colourise stderr output.
    """
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[module]}</cyan> - "
            "<level>{message}</level>"
        )

    # Remove the default loguru handler
    logger.remove()

    # Add stderr handler with optional colorization
    logger.add(
        sys.stderr,
        level=level,
        format=format_string,
        colorize=colorize,
    )

    # Optionally add a file handler
    if log_file is not None:
        logger.add(
            log_file,
            level=level,
            format=format_string,
            colorize=False,
            rotation="10 MB",
            retention="7 days",
        )

    logger.debug("Logging configured: level={}, log_file={}", level, log_file)


def get_logger(name: str) -> "logger":  # type: ignore[valid-type]
    """Return a logger bound with the given module name.

    Parameters
    ----------
    name : str
        Module or component name (e.g. ``"scinfer.adapters.scgpt"``).

    Returns
    -------
    loguru.logger
        A logger instance with the ``module`` context bound.
    """
    return logger.bind(module=name)
