"""Processing pipeline - coordinate all processing steps"""

from pathlib import Path
from typing import Optional, List
import tempfile
import shutil
import logging
from .watermark_remover import WatermarkRemover
from .fingerprint_remover import FingerprintRemover
from .video_transform import VideoTransformer
from .image_transform import ImageTransformer
from .metadata_cleaner import MetadataCleaner
from ..utils import ensure_dir

logger = logging.getLogger(__name__)


class ProcessingPipeline:

    def __init__(self, config: dict = None):
        self.config = config or {}

        watermark_config = self.config.get("watermark", {})
        fingerprint_config = self.config.get("fingerprint", {})
        transforms_config = self.config.get("transforms", {})
        metadata_config = self.config.get("metadata", {})

        self.watermark_remover = WatermarkRemover(watermark_config)
        self.fingerprint_remover = FingerprintRemover(fingerprint_config)
        self.video_transformer = VideoTransformer(transforms_config)
        self.image_transformer = ImageTransformer(transforms_config)
        self.metadata_cleaner = MetadataCleaner(metadata_config)

    def process(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        steps: Optional[List[str]] = None,
    ) -> bool:
        if output_path is None:
            output_path = self._get_output_path(input_path)

        ensure_dir(output_path.parent)

        current_path = input_path
        suffix = input_path.suffix.lower()
        temp_files = []

        try:
            if steps is None:
                steps = self._default_steps(suffix)

            for step in steps:
                new_temp = Path(tempfile.mktemp(suffix=suffix))
                temp_files.append(new_temp)
                current_path = self._apply_step(current_path, new_temp, step)

            # Copy final result to output
            if current_path != input_path:
                shutil.copy(current_path, output_path)

            return True
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}", exc_info=True)
            return False
        finally:
            # Cleanup all temp files
            for f in temp_files:
                if f.exists():
                    try:
                        f.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {f}: {e}")
    
    def _default_steps(self, suffix: str) -> List[str]:
        steps = []

        # Distinguish between video and images for fingerprint removal
        video_extensions = [".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv"]
        image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]

        if self.config.get("fingerprint", {}).get("enabled", True):
            if suffix in video_extensions:
                steps.append("fingerprint_remove")  # Video only
            elif suffix in image_extensions:
                steps.append("image_fingerprint_remove")  # Images have separate processing

        # Watermark removal only for video
        if suffix in video_extensions:
            if self.config.get("watermark", {}).get("enabled", True):
                steps.append("watermark_remove")
            if self.config.get("transforms", {}).get("enabled", True):
                steps.append("video_transform")
        else:
            if self.config.get("transforms", {}).get("enabled", True):
                steps.append("image_transform")

        steps.append("metadata_clean")
        return steps
    
    def _apply_step(self, input_path: Path, output_path: Path, step: str) -> Path:
        if step == "watermark_remove":
            self.watermark_remover.remove_watermark(input_path, output_path)
        elif step == "fingerprint_remove":
            self.fingerprint_remover.remove_fingerprint(input_path, output_path)
        elif step == "image_fingerprint_remove":
            self.fingerprint_remover.remove_fingerprint(input_path, output_path)
        elif step == "video_transform":
            self.video_transformer.transform(input_path, output_path)
        elif step == "image_transform":
            self.image_transformer.transform(input_path, output_path)
        elif step == "metadata_clean":
            self.metadata_cleaner.clean(input_path, output_path)

        return output_path
    
    def _get_output_path(self, input_path: Path) -> Path:
        output_dir = self.config.get("output", {}).get("output_dir", "./downloads/processed")
        return Path(output_dir) / f"processed_{input_path.name}"
