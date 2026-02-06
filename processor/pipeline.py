"""Processing pipeline - coordinate all processing steps"""

from pathlib import Path
from typing import Optional, List
from .watermark_remover import WatermarkRemover
from .fingerprint_remover import FingerprintRemover
from .video_transform import VideoTransformer
from .image_transform import ImageTransformer
from .metadata_cleaner import MetadataCleaner
from utils import ensure_dir


class ProcessingPipeline:
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        download_config = self.config.get("download", {})
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
        
        if steps is None:
            steps = self._default_steps(suffix)
        
        try:
            for step in steps:
                current_path = self._apply_step(current_path, output_path, step)
            
            return True
        except Exception:
            return False
    
    def _default_steps(self, suffix: str) -> List[str]:
        steps = ["metadata_clean"]
        
        if self.config.get("fingerprint", {}).get("enabled", True):
            steps.extend(["fingerprint_remove"])
        
        if suffix in [".mp4", ".mkv", ".mov", ".avi"]:
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
        temp_path = output_path.parent / f"temp_{step}_{output_path.name}"
        
        if step == "watermark_remove":
            self.watermark_remover.remove_watermark(input_path, temp_path)
        elif step == "fingerprint_remove":
            self.fingerprint_remover.remove_fingerprint(input_path, temp_path)
        elif step == "video_transform":
            self.video_transformer.transform(input_path, temp_path)
        elif step == "image_transform":
            self.image_transformer.transform(input_path, temp_path)
        elif step == "metadata_clean":
            self.metadata_cleaner.clean(input_path, temp_path)
        
        return temp_path
    
    def _get_output_path(self, input_path: Path) -> Path:
        output_dir = self.config.get("download", {}).get("output_dir", "./downloads/processed")
        return Path(output_dir) / f"processed_{input_path.name}"
