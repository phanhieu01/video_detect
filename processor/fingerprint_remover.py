"""Fingerprint remover - re-encode and modify video signature"""

from pathlib import Path
import random
from datetime import datetime, timedelta
from utils.ffmpeg_wrapper import FFmpegWrapper


class FingerprintRemover:
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.re_encode = self.config.get("re_encode", True)
        self.change_bitrate = self.config.get("change_bitrate", True)
        self.change_resolution = self.config.get("change_resolution", True)
        self.change_colorspace = self.config.get("change_colorspace", True)
        self.add_noise = self.config.get("add_noise", True)
    
    def remove_fingerprint(
        self,
        input_path: Path,
        output_path: Path,
    ) -> bool:
        try:
            temp_path = output_path
            
            if self.re_encode:
                temp_path = self._apply_re_encode(input_path, temp_path)
            
            if self.change_bitrate:
                temp_path = self._apply_bitrate_change(temp_path, temp_path)
            
            if self.change_resolution:
                temp_path = self._apply_resolution_change(temp_path, temp_path)
            
            if self.change_colorspace:
                temp_path = self._apply_colorspace_change(temp_path, temp_path)
            
            return True
        except Exception:
            return False
    
    def _apply_re_encode(self, input_path: Path, output_path: Path) -> Path:
        codec = random.choice(["libx264", "libx265", "libvpx-vp9"])
        crf = random.randint(20, 26)
        FFmpegWrapper.re_encode(input_path, output_path, codec=codec, crf=crf)
        return output_path
    
    def _apply_bitrate_change(self, input_path: Path, output_path: Path) -> Path:
        bitrate = random.choice(["3000k", "3500k", "4000k", "4500k", "5000k"])
        FFmpegWrapper.re_encode(input_path, output_path, bitrate=bitrate)
        return output_path
    
    def _apply_resolution_change(self, input_path: Path, output_path: Path) -> Path:
        scale_factors = ["iw-2:ih-2", "iw-4:ih-4", "iw+2:ih+2", "iw-6:ih-6"]
        scale = random.choice(scale_factors)
        
        try:
            FFmpegWrapper.run_command([
                "-i", str(input_path),
                "-vf", f"scale={scale}",
                "-c:a", "copy",
                str(output_path),
            ])
        except Exception:
            FFmpegWrapper.re_encode(input_path, output_path)
        
        return output_path
    
    def _apply_colorspace_change(self, input_path: Path, output_path: Path) -> Path:
        try:
            FFmpegWrapper.run_command([
                "-i", str(input_path),
                "-vf", "eq=contrast=1.02:brightness=0.02:saturation=1.05",
                "-c:a", "copy",
                str(output_path),
            ])
        except Exception:
            FFmpegWrapper.re_encode(input_path, output_path)
        
        return output_path
