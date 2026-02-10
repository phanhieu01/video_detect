"""Utility functions for Video Detect Pro

This module provides helper functions for logging, configuration,
and file operations with proper error handling.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from rich.logging import RichHandler
from rich.console import Console
from datetime import datetime
import os

# Module level logger
logger = logging.getLogger(__name__)

# Default logging configuration
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log levels mapping
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Global flag to track if logging has been setup
_logging_configured = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None,
    use_rich: bool = True,
    log_to_console: bool = True,
    log_to_file: bool = False,
) -> None:
    """
    Setup comprehensive logging for the application.

    Args:
        level: Logging level (default: INFO)
        log_file: Path to log file (optional)
        log_format: Custom log format string
        use_rich: Use RichHandler for console output
        log_to_console: Enable console logging
        log_to_file: Enable file logging
    """
    global _logging_configured

    # Only configure logging once
    if _logging_configured:
        return

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatters
    format_str = log_format or DEFAULT_LOG_FORMAT
    date_format = DEFAULT_DATE_FORMAT
    formatter = logging.Formatter(format_str, datefmt=date_format)

    # Console handler
    if log_to_console:
        if use_rich:
            try:
                console = Console(stderr=True)
                console_handler = RichHandler(
                    rich_tracebacks=True,
                    tracebacks_show_locals=True,
                    console=console,
                    show_time=True,
                    show_path=True,
                )
                # Rich has its own formatting
                root_logger.addHandler(console_handler)
            except Exception as e:
                logger.warning(f"Failed to setup RichHandler: {e}, using standard handler")
                console_handler = logging.StreamHandler(sys.stderr)
                console_handler.setFormatter(formatter)
                console_handler.setLevel(level)
                root_logger.addHandler(console_handler)
        else:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(level)
            root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_to_file or log_file:
        if log_file is None:
            # Create logs directory
            log_dir = Path.cwd() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"video_detect_{datetime.now().strftime('%Y%m%d')}.log"

        try:
            # Rotating file handler - 10MB max, keep 5 backups
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            root_logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.warning(f"Failed to setup file logging: {e}")

    # Set specific log levels for noisy libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    _logging_configured = True
    logger.info(f"Logging configured at level: {logging.getLevelName(level)}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name"""
    return logging.getLogger(name)


def set_log_level(level: int | str) -> None:
    """
    Change the logging level at runtime.

    Args:
        level: Either a logging constant (logging.INFO) or string ("INFO")
    """
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.upper(), logging.INFO)

    logging.getLogger().setLevel(level)
    logger.info(f"Log level changed to: {logging.getLevelName(level)}")


def log_function_call(logger: logging.Logger, func_name: str, **kwargs) -> None:
    """Log function call with parameters"""
    params_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.debug(f"Calling {func_name}({params_str})")


def log_execution_time(logger: logging.Logger, func_name: str, duration: float) -> None:
    """Log function execution time"""
    logger.debug(f"{func_name} completed in {duration:.2f} seconds")


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with proper error handling.

    Args:
        config_path: Path to config file. If None, uses default location.

    Returns:
        Dictionary containing configuration, or empty dict if file not found.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    try:
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded config from {config_path}")
            return config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def save_config(config: Dict[str, Any], config_path: Optional[Path] = None) -> bool:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary to save
        config_path: Path to save config. If None, uses default location.

    Returns:
        True if successful, False otherwise.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Config saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        The path that was ensured to exist
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise


def get_file_size(file_path: Path) -> int:
    """Get file size in bytes with error handling"""
    try:
        return file_path.stat().st_size
    except Exception as e:
        logger.error(f"Failed to get file size for {file_path}: {e}")
        return 0


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def validate_file_path(file_path: Path, must_exist: bool = True) -> bool:
    """
    Validate file path with proper error logging.

    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist

    Returns:
        True if valid, False otherwise
    """
    try:
        resolved = file_path.resolve()

        if must_exist and not resolved.exists():
            logger.error(f"File does not exist: {resolved}")
            return False

        if must_exist and not resolved.is_file():
            logger.error(f"Path is not a file: {resolved}")
            return False

        # Check for parent directory write access
        if not must_exist:
            parent = resolved.parent
            if not parent.exists():
                try:
                    parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.error(f"Cannot create parent directory: {e}")
                    return False

        return True

    except Exception as e:
        logger.error(f"Path validation failed for {file_path}: {e}")
        return False


def load_config(config_path: Optional[Path] = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    
    if not config_path.exists():
        return {}
    
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
