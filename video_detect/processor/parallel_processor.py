"""Parallel processing module for handling multiple files concurrently

This module provides parallel processing capabilities with proper resource
cleanup and error handling for batch processing operations.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, Executor
import logging
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import multiprocessing
import time
import traceback
import gc

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing mode for parallel execution"""
    THREAD = "thread"  # Use threading (good for I/O bound)
    PROCESS = "process"  # Use multiprocessing (good for CPU bound)
    ASYNC = "async"  # Use asyncIO (good for I/O bound with many tasks)


@dataclass
class ProcessingResult:
    """Result of a single file processing"""
    file_path: Path
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    duration: float = 0.0


class ParallelProcessor:

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.max_workers = self.config.get(
            "max_workers",
            min(multiprocessing.cpu_count(), 4),
        )
        self.mode = ProcessingMode(self.config.get("mode", "thread"))

    def process_files(
        self,
        file_paths: List[Path],
        processor_func: Callable[[Path, Optional[Path]], bool],
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ProcessingResult]:
        """Process multiple files in parallel"""
        results = []

        if self.mode == ProcessingMode.PROCESS:
            results = self._process_with_multiprocessing(
                file_paths, processor_func, output_dir, progress_callback
            )
        else:
            results = self._process_with_threading(
                file_paths, processor_func, output_dir, progress_callback
            )

        return results

    def _process_with_threading(
        self,
        file_paths: List[Path],
        processor_func: Callable,
        output_dir: Optional[Path],
        progress_callback: Optional[Callable],
    ) -> List[ProcessingResult]:
        """Process files using ThreadPoolExecutor with proper resource cleanup"""
        results = []
        executor: Optional[ThreadPoolExecutor] = None

        try:
            executor = ThreadPoolExecutor(max_workers=self.max_workers)
            future_to_file = {}

            # Submit all tasks
            for file_path in file_paths:
                output_path = None
                if output_dir:
                    output_path = output_dir / f"processed_{file_path.name}"

                future = executor.submit(self._process_single, processor_func, file_path, output_path)
                future_to_file[future] = file_path

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_file, timeout=3600):  # 1 hour timeout
                file_path = future_to_file[future]
                try:
                    result = future.result(timeout=600)  # 10 min per file
                    results.append(result)

                    # Update progress
                    completed += 1
                    if progress_callback:
                        try:
                            progress_callback(completed, len(file_paths))
                        except Exception as e:
                            logger.warning(f"Progress callback failed: {e}")

                except TimeoutError:
                    logger.error(f"Timeout processing {file_path.name}")
                    results.append(ProcessingResult(
                        file_path=file_path,
                        success=False,
                        error="Processing timeout",
                    ))
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
                    logger.debug(traceback.format_exc())
                    results.append(ProcessingResult(
                        file_path=file_path,
                        success=False,
                        error=str(e),
                    ))

        except Exception as e:
            logger.error(f"Thread pool execution failed: {e}")
            logger.debug(traceback.format_exc())
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
                logger.debug("Thread pool shut down")

        return results

    def _process_with_multiprocessing(
        self,
        file_paths: List[Path],
        processor_func: Callable,
        output_dir: Optional[Path],
        progress_callback: Optional[Callable],
    ) -> List[ProcessingResult]:
        """Process files using ProcessPoolExecutor with proper resource cleanup"""
        results = []
        executor: Optional[ProcessPoolExecutor] = None

        try:
            executor = ProcessPoolExecutor(max_workers=self.max_workers)
            future_to_file = {}

            # Submit all tasks
            for file_path in file_paths:
                output_path = None
                if output_dir:
                    output_path = output_dir / f"processed_{file_path.name}"

                future = executor.submit(self._process_single, processor_func, file_path, output_path)
                future_to_file[future] = file_path

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_file, timeout=3600):  # 1 hour timeout
                file_path = future_to_file[future]
                try:
                    result = future.result(timeout=600)  # 10 min per file
                    results.append(result)

                    # Update progress
                    completed += 1
                    if progress_callback:
                        try:
                            progress_callback(completed, len(file_paths))
                        except Exception as e:
                            logger.warning(f"Progress callback failed: {e}")

                except TimeoutError:
                    logger.error(f"Timeout processing {file_path.name}")
                    results.append(ProcessingResult(
                        file_path=file_path,
                        success=False,
                        error="Processing timeout",
                    ))
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
                    logger.debug(traceback.format_exc())
                    results.append(ProcessingResult(
                        file_path=file_path,
                        success=False,
                        error=str(e),
                    ))

        except Exception as e:
            logger.error(f"Process pool execution failed: {e}")
            logger.debug(traceback.format_exc())
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
                logger.debug("Process pool shut down")

        return results

    def _process_single(
        self,
        processor_func: Callable,
        input_path: Path,
        output_path: Optional[Path],
    ) -> ProcessingResult:
        """
        Process a single file with proper resource cleanup and error handling.

        This method ensures resources are cleaned up even if processing fails.
        """
        start_time = time.time()
        temp_files: List[Path] = []

        try:
            # Validate input file exists
            if not input_path.exists():
                return ProcessingResult(
                    file_path=input_path,
                    success=False,
                    error="Input file not found",
                    duration=time.time() - start_time,
                )

            # Process the file
            success = processor_func(input_path, output_path)
            duration = time.time() - start_time

            # Validate output if processing succeeded
            if success and output_path:
                if not output_path.exists():
                    logger.warning(f"Processing reported success but output file missing: {output_path}")
                    return ProcessingResult(
                        file_path=input_path,
                        success=False,
                        error="Output file not created",
                        duration=duration,
                    )

            return ProcessingResult(
                file_path=input_path,
                success=success,
                output_path=output_path if success else None,
                duration=duration,
            )

        except MemoryError as e:
            logger.error(f"Memory error processing {input_path.name}: {e}")
            gc.collect()  # Force garbage collection
            return ProcessingResult(
                file_path=input_path,
                success=False,
                error="Out of memory",
                duration=time.time() - start_time,
            )

        except KeyboardInterrupt:
            logger.info(f"Processing cancelled for {input_path.name}")
            raise

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error processing {input_path.name}: {e}")
            logger.debug(traceback.format_exc())

            # Cleanup temp files on error
            for temp_file in temp_files:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file {temp_file}: {cleanup_error}")

            return ProcessingResult(
                file_path=input_path,
                success=False,
                error=str(e),
                duration=duration,
            )

        finally:
            # Ensure memory is freed
            gc.collect()


class BatchProcessingPipeline:
    """
    Batch processing pipeline with proper error handling and resource cleanup.

    This class provides high-level batch processing capabilities with
    comprehensive error recovery and resource management.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

        # Import pipeline
        try:
            from .pipeline import ProcessingPipeline
            self.pipeline = ProcessingPipeline(config)
        except ImportError as e:
            logger.error(f"Failed to import ProcessingPipeline: {e}")
            raise

        # Setup parallel processor
        parallel_config = self.config.get("parallel", {})
        self.parallel_processor = ParallelProcessor(parallel_config)

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        pattern: str = "*",
        show_progress: bool = True,
    ) -> List[ProcessingResult]:
        """
        Process all files in a directory with proper error handling.

        Args:
            input_dir: Directory containing input files
            output_dir: Directory for output files (created if needed)
            pattern: Glob pattern for file matching
            show_progress: Whether to log progress updates

        Returns:
            List of ProcessingResult objects
        """
        # Validate input directory
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return []

        if not input_dir.is_dir():
            logger.error(f"Input path is not a directory: {input_dir}")
            return []

        # Setup output directory
        if output_dir is None:
            output_dir = Path(self.config.get("output", {}).get("output_dir", "./downloads/processed"))

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            return []

        # Find files matching pattern
        try:
            file_paths = list(input_dir.glob(pattern))
            file_paths = [f for f in file_paths if f.is_file()]
        except Exception as e:
            logger.error(f"Failed to list files in {input_dir}: {e}")
            return []

        if not file_paths:
            logger.warning(f"No files found matching pattern: {pattern} in {input_dir}")
            return []

        logger.info(f"Found {len(file_paths)} files to process in {input_dir}")

        # Progress callback with error handling
        def progress_callback(completed: int, total: int):
            if show_progress:
                try:
                    percentage = (completed * 100) // total if total > 0 else 0
                    logger.info(f"Progress: {completed}/{total} ({percentage}%)")
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

        # Process files with error wrapper
        def processor_func(input_path: Path, output_path: Optional[Path]) -> bool:
            try:
                return self.pipeline.process(input_path, output_path)
            except Exception as e:
                logger.error(f"Pipeline error for {input_path.name}: {e}")
                return False

        # Execute parallel processing
        results = []
        try:
            results = self.parallel_processor.process_files(
                file_paths=file_paths,
                processor_func=processor_func,
                output_dir=output_dir,
                progress_callback=progress_callback,
            )
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            logger.debug(traceback.format_exc())

        # Log summary
        self._log_summary(results)

        return results

    def _log_summary(self, results: List[ProcessingResult]) -> None:
        """Log processing summary with error details"""
        total = len(results)
        if total == 0:
            return

        success_count = sum(1 for r in results if r.success)
        fail_count = total - success_count
        success_rate = (success_count / total * 100) if total > 0 else 0

        logger.info(f"Processing complete: {success_count}/{total} succeeded ({success_rate:.1f}%)")

        if fail_count > 0:
            logger.warning(f"Failed to process {fail_count} files:")
            for result in results:
                if not result.success:
                    logger.warning(f"  - {result.file_path.name}: {result.error or 'Unknown error'}")

    def get_processing_summary(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Get summary of processing results"""
        total = len(results)
        success = sum(1 for r in results if r.success)
        failed = total - success

        total_duration = sum(r.duration for r in results)
        avg_duration = total_duration / total if total > 0 else 0

        return {
            "total_files": total,
            "successful": success,
            "failed": failed,
            "success_rate": success / total if total > 0 else 0,
            "total_duration": total_duration,
            "average_duration": avg_duration,
        }
