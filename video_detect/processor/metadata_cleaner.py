"""Metadata cleaner - strip EXIF and video metadata"""

from pathlib import Path
from datetime import datetime, timedelta
import random
from PIL import Image
from ..utils.ffmpeg_wrapper import FFmpegWrapper


class MetadataCleaner:
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.strip_metadata = self.config.get("strip_metadata", True)
        self.randomize_date = self.config.get("randomize_date", True)
    
    def clean(self, input_path: Path, output_path: Path) -> bool:
        suffix = input_path.suffix.lower()
        
        if suffix in [".jpg", ".jpeg", ".png"]:
            return self._clean_image_metadata(input_path, output_path)
        elif suffix in [".mp4", ".mkv", ".mov", ".avi"]:
            return self._clean_video_metadata(input_path, output_path)
        
        return False
    
    def _clean_image_metadata(self, input_path: Path, output_path: Path) -> bool:
        try:
            img = Image.open(input_path)
            img_data = list(img.getdata())
            
            new_img = Image.new(img.mode, img.size)
            new_img.putdata(img_data)
            
            if self.randomize_date:
                random_date = datetime.now() - timedelta(days=random.randint(0, 365))
                exif = new_img.info.get("exif", b"")
            
            new_img.save(output_path, format=img.format, quality=95)
            return True
        except Exception:
            return False
    
    def _clean_video_metadata(self, input_path: Path, output_path: Path) -> bool:
        try:
            FFmpegWrapper.strip_metadata(input_path, output_path)
            return True
        except Exception:
            return False
