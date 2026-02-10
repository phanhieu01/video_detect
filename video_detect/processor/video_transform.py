"""Video transformation - change video signature to avoid content ID"""

from pathlib import Path
import random
import tempfile
import shutil
import logging
from ..utils.ffmpeg_wrapper import FFmpegWrapper

logger = logging.getLogger(__name__)


class VideoTransformer:

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.speed_change = self.config.get("speed_change", True)
        self.speed_factor = self.config.get("speed_factor", 1.03)
        self.pitch_shift = self.config.get("pitch_shift", True)
        self.horizontal_flip = self.config.get("horizontal_flip", False)
        self.crop_percent = self.config.get("crop_percent", 2)
        self.brightness_shift = self.config.get("brightness_shift", 2)
        self.contrast_shift = self.config.get("contrast_shift", 2)
        self.saturation_shift = self.config.get("saturation_shift", 5)
        self.add_silence = self.config.get("add_silence", True)
        self.silence_duration = self.config.get("silence_duration", 0.2)

    def transform(
        self,
        input_path: Path,
        output_path: Path,
    ) -> bool:
        temp_files = []

        try:
            temp_path = input_path

            if self.speed_change:
                new_temp = Path(tempfile.mktemp(suffix=".mp4"))
                temp_files.append(new_temp)
                temp_path = self._apply_speed_change(temp_path, new_temp)

            if self.horizontal_flip:
                new_temp = Path(tempfile.mktemp(suffix=".mp4"))
                temp_files.append(new_temp)
                temp_path = self._apply_flip(temp_path, new_temp)

            if self.crop_percent > 0:
                new_temp = Path(tempfile.mktemp(suffix=".mp4"))
                temp_files.append(new_temp)
                temp_path = self._apply_crop(temp_path, new_temp)

            if any([self.brightness_shift, self.contrast_shift, self.saturation_shift]):
                new_temp = Path(tempfile.mktemp(suffix=".mp4"))
                temp_files.append(new_temp)
                temp_path = self._apply_color_adjust(temp_path, new_temp)

            if self.add_silence:
                new_temp = Path(tempfile.mktemp(suffix=".mp4"))
                temp_files.append(new_temp)
                temp_path = self._apply_silence(temp_path, new_temp)

            # Copy final result to output
            if temp_path != input_path:
                shutil.copy(temp_path, output_path)

            return True
        except Exception as e:
            logger.error(f"Video transform failed: {e}", exc_info=True)
            return False
        finally:
            # Cleanup all temp files
            for f in temp_files:
                if f.exists():
                    try:
                        f.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {f}: {e}")
    
    def _apply_speed_change(self, input_path: Path, output_path: Path) -> Path:
        factor = self.speed_factor + random.uniform(-0.01, 0.01)
        FFmpegWrapper.change_speed(input_path, output_path, factor)
        return output_path
    
    def _apply_flip(self, input_path: Path, output_path: Path) -> Path:
        FFmpegWrapper.horizontal_flip(input_path, output_path)
        return output_path
    
    def _apply_crop(self, input_path: Path, output_path: Path) -> Path:
        crop = self.crop_percent + random.randint(-1, 1)
        FFmpegWrapper.crop(input_path, output_path, crop)
        return output_path
    
    def _apply_color_adjust(self, input_path: Path, output_path: Path) -> Path:
        brightness = self.brightness_shift / 100 + random.uniform(-0.01, 0.01)
        contrast = 1 + (self.contrast_shift / 100) + random.uniform(-0.01, 0.01)
        saturation = 1 + (self.saturation_shift / 100) + random.uniform(-0.02, 0.02)
        
        FFmpegWrapper.color_adjust(input_path, output_path, brightness, contrast, saturation)
        return output_path
    
    def _apply_silence(self, input_path: Path, output_path: Path) -> Path:
        duration = self.silence_duration + random.uniform(-0.05, 0.05)
        FFmpegWrapper.add_silence(input_path, output_path, max(0.1, duration))
        return output_path
