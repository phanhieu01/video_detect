"""Resource management utilities for Video Detect Pro

This module provides context managers and utilities for proper resource
cleanup and management throughout the application.
"""

import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Callable, Any
from contextlib import contextmanager
import threading
import atexit
import weakref

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Centralized resource manager for tracking and cleaning up resources.

    This class ensures proper cleanup of temporary files, file handles,
    and other resources even when errors occur.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern to ensure only one resource manager exists"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._temp_files: List[Path] = []
        self._temp_dirs: List[Path] = []
        self._cleanup_handlers: List[Callable] = []
        self._lock = threading.Lock()
        self._initialized = True

        # Register cleanup on exit
        atexit.register(self.cleanup_all)

    def track_temp_file(self, path: Path) -> None:
        """Track a temporary file for cleanup"""
        with self._lock:
            if path not in self._temp_files:
                self._temp_files.append(path)
                logger.debug(f"Tracking temp file: {path}")

    def track_temp_dir(self, path: Path) -> None:
        """Track a temporary directory for cleanup"""
        with self._lock:
            if path not in self._temp_dirs:
                self._temp_dirs.append(path)
                logger.debug(f"Tracking temp directory: {path}")

    def register_cleanup_handler(self, handler: Callable) -> None:
        """Register a cleanup function to be called on exit"""
        with self._lock:
            self._cleanup_handlers.append(handler)
            logger.debug(f"Registered cleanup handler: {handler.__name__}")

    def untrack_temp_file(self, path: Path) -> bool:
        """Remove a file from tracking (already cleaned up)"""
        with self._lock:
            if path in self._temp_files:
                self._temp_files.remove(path)
                logger.debug(f"Untracked temp file: {path}")
                return True
            return False

    def cleanup_temp_file(self, path: Path) -> bool:
        """Clean up a specific temporary file"""
        try:
            if path.exists():
                path.unlink()
                logger.debug(f"Cleaned up temp file: {path}")
            self.untrack_temp_file(path)
            return True
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {path}: {e}")
            return False

    def cleanup_temp_dir(self, path: Path) -> bool:
        """Clean up a specific temporary directory"""
        try:
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
                logger.debug(f"Cleaned up temp directory: {path}")
            with self._lock:
                if path in self._temp_dirs:
                    self._temp_dirs.remove(path)
            return True
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory {path}: {e}")
            return False

    def cleanup_all(self) -> None:
        """Clean up all tracked resources"""
        logger.info("Cleaning up all tracked resources...")

        # Run cleanup handlers
        for handler in self._cleanup_handlers:
            try:
                handler()
            except Exception as e:
                logger.warning(f"Cleanup handler {handler.__name__} failed: {e}")

        # Clean up temp files
        with self._lock:
            files_to_cleanup = self._temp_files.copy()
            for temp_file in files_to_cleanup:
                self.cleanup_temp_file(temp_file)
            self._temp_files.clear()

            # Clean up temp directories
            dirs_to_cleanup = self._temp_dirs.copy()
            for temp_dir in dirs_to_cleanup:
                self.cleanup_temp_dir(temp_dir)
            self._temp_dirs.clear()

        logger.info("Resource cleanup complete")


def get_resource_manager() -> ResourceManager:
    """Get the singleton resource manager instance"""
    return ResourceManager()


@contextmanager
def temp_file(suffix: Optional[str] = None, prefix: str = "vdl_"):
    """
    Context manager for temporary file creation with automatic cleanup.

    Args:
        suffix: File suffix/extension
        prefix: File name prefix

    Yields:
        Path to the temporary file
    """
    manager = get_resource_manager()
    temp_path: Optional[Path] = None

    try:
        # Create temporary file
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        import os
        os.close(fd)  # Close the file descriptor
        temp_path = Path(temp_path)

        # Track for cleanup
        manager.track_temp_file(temp_path)

        logger.debug(f"Created temp file: {temp_path}")
        yield temp_path

    except Exception as e:
        logger.error(f"Error in temp_file context: {e}")
        raise
    finally:
        # Cleanup
        if temp_path:
            manager.cleanup_temp_file(temp_path)


@contextmanager
def temp_dir(suffix: Optional[str] = None, prefix: str = "vdl_"):
    """
    Context manager for temporary directory creation with automatic cleanup.

    Args:
        suffix: Directory suffix
        prefix: Directory name prefix

    Yields:
        Path to the temporary directory
    """
    manager = get_resource_manager()
    temp_path: Optional[Path] = None

    try:
        # Create temporary directory
        temp_path = Path(tempfile.mkdtemp(suffix=suffix, prefix=prefix))

        # Track for cleanup
        manager.track_temp_dir(temp_path)

        logger.debug(f"Created temp directory: {temp_path}")
        yield temp_path

    except Exception as e:
        logger.error(f"Error in temp_dir context: {e}")
        raise
    finally:
        # Cleanup
        if temp_path:
            manager.cleanup_temp_dir(temp_path)


@contextmanager
def open_file_safe(path: Path, mode: str = "r", **kwargs):
    """
    Context manager for safe file opening with guaranteed closure.

    Args:
        path: File path to open
        mode: File open mode
        **kwargs: Additional arguments for open()

    Yields:
        File object
    """
    file_obj = None
    try:
        file_obj = open(path, mode, **kwargs)
        logger.debug(f"Opened file: {path} (mode: {mode})")
        yield file_obj
    except Exception as e:
        logger.error(f"Error opening file {path}: {e}")
        raise
    finally:
        if file_obj:
            try:
                file_obj.close()
                logger.debug(f"Closed file: {path}")
            except Exception as e:
                logger.warning(f"Error closing file {path}: {e}")


class FileLock:
    """
    Simple file-based lock mechanism for coordinating file access.

    Uses lock files to prevent concurrent access to the same file.
    """

    def __init__(self, path: Path, timeout: float = 30.0):
        self.path = path
        self.lock_path = Path(f"{path}.lock")
        self.timeout = timeout
        self._locked = False

    def acquire(self) -> bool:
        """Attempt to acquire the lock"""
        import time

        start = time.time()
        while time.time() - start < self.timeout:
            if not self.lock_path.exists():
                try:
                    self.lock_path.touch()
                    self._locked = True
                    logger.debug(f"Acquired lock for: {self.path}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to create lock file: {e}")
            time.sleep(0.1)

        logger.warning(f"Lock acquisition timeout for: {self.path}")
        return False

    def release(self) -> None:
        """Release the lock"""
        if self._locked and self.lock_path.exists():
            try:
                self.lock_path.unlink()
                self._locked = False
                logger.debug(f"Released lock for: {self.path}")
            except Exception as e:
                logger.warning(f"Failed to remove lock file: {e}")

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError(f"Could not acquire lock for {self.path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def cleanup_old_temp_dirs(base_dir: Path, max_age_hours: int = 24) -> int:
    """
    Clean up old temporary directories older than specified age.

    Args:
        base_dir: Base directory containing temp directories
        max_age_hours: Maximum age in hours before cleanup

    Returns:
        Number of directories cleaned up
    """
    import time

    if not base_dir.exists():
        return 0

    cleaned = 0
    max_age_seconds = max_age_hours * 3600
    now = time.time()

    try:
        for item in base_dir.iterdir():
            if item.is_dir() and item.name.startswith("vdl_"):
                try:
                    stat = item.stat()
                    age = now - stat.st_mtime
                    if age > max_age_seconds:
                        shutil.rmtree(item)
                        cleaned += 1
                        logger.info(f"Cleaned up old temp directory: {item}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup old temp dir {item}: {e}")
    except Exception as e:
        logger.error(f"Error scanning for old temp directories: {e}")

    if cleaned > 0:
        logger.info(f"Cleaned up {cleaned} old temporary directories")

    return cleaned
