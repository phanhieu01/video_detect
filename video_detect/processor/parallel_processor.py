"""Parallel processing module for handling multiple files concurrently"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
from dataclasses import dataclass
from enum import Enum
import multiprocessing

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
        """Process files using ThreadPoolExecutor"""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {}
            for file_path in file_paths:
                output_path = None
                if output_dir:
                    output_path = output_dir / f"processed_{file_path.name}"

                future = executor.submit(self._process_single, processor_func, file_path, output_path)
                future_to_file[future] = file_path

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Update progress
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(file_paths))

                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
                    results.append(ProcessingResult(
                        file_path=file_path,
                        success=False,
                        error=str(e),
                    ))

        return results

    def _process_with_multiprocessing(
        self,
        file_paths: List[Path],
        processor_func: Callable,
        output_dir: Optional[Path],
        progress_callback: Optional[Callable],
    ) -> List[ProcessingResult]:
        """Process files using ProcessPoolExecutor"""
        results = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {}
            for file_path in file_paths:
                output_path = None
                if output_dir:
                    output_path = output_dir / f"processed_{file_path.name}"

                future = executor.submit(self._process_single, processor_func, file_path, output_path)
                future_to_file[future] = file_path

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Update progress
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(file_paths))

                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
                    results.append(ProcessingResult(
                        file_path=file_path,
                        success=False,
                        error=str(e),
                    ))

        return results

    def _process_single(
        self,
        processor_func: Callable,
        input_path: Path,
        output_path: Optional[Path],
    ) -> ProcessingResult:
        """Process a single file"""
        import time
        start_time = time.time()

        try:
            success = processor_func(input_path, output_path)
            duration = time.time() - start_time

            return ProcessingResult(
                file_path=input_path,
                success=success,
                output_path=output_path if success else None,
                duration=duration,
            )
        except Exception as e:
            duration = time.time() - start_time
            return ProcessingResult(
                file_path=input_path,
                success=False,
                error=str(e),
                duration=duration,
            )


class BatchProcessingPipeline:

    def __init__(self, config: dict = None):
        self.config = config or {}

        # Import pipeline
        from .pipeline import ProcessingPipeline
        self.pipeline = ProcessingPipeline(config)

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
        """Process all files in a directory"""
        if output_dir is None:
            output_dir = Path(self.config.get("output", {}).get("output_dir", "./downloads/processed"))

        output_dir.mkdir(parents=True, exist_ok=True)

        # Find files
        file_paths = list(input_dir.glob(pattern))
        file_paths = [f for f in file_paths if f.is_file()]

        if not file_paths:
            logger.warning(f"No files found matching pattern: {pattern}")
            return []

        logger.info(f"Found {len(file_paths)} files to process")

        # Progress callback
        def progress_callback(completed: int, total: int):
            if show_progress:
                logger.info(f"Progress: {completed}/{total} ({completed*100//total}%)")

        # Process files in parallel
        def processor_func(input_path: Path, output_path: Path) -> bool:
            return self.pipeline.process(input_path, output_path)

        results = self.parallel_processor.process_files(
            file_paths=file_paths,
            processor_func=processor_func,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )

        # Log summary
        success_count = sum(1 for r in results if r.success)
        fail_count = len(results) - success_count

        logger.info(f"Processing complete: {success_count} succeeded, {fail_count} failed")

        return results

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
