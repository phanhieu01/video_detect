"""Video Processor - watermark and fingerprint removal"""

from pathlib import Path
from typing import Optional
try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from ..utils.ffmpeg_wrapper import FFmpegWrapper
from ..utils import ensure_dir


class WatermarkRemover:
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.method = self.config.get("method", "inpaint")
        self.auto_detect = self.config.get("auto_detect", False)
        self.watermark_region = self.config.get("watermark_region", {})
        
        if self.auto_detect:
            from ..detector import WatermarkDetector
            self.detector = WatermarkDetector(self.config)
        else:
            self.detector = None
    
    def remove_watermark(
        self,
        input_path: Path,
        output_path: Path,
    ) -> bool:
        if self.auto_detect and self.detector:
            detection_result = self.detector.detect_watermark(input_path)
            if detection_result and detection_result.get("found"):
                self.watermark_region = detection_result["region"]
        
        if self.method == "inpaint":
            return self._remove_with_inpaint(input_path, output_path)
        elif self.method == "delogo":
            return self._remove_with_delogo(input_path, output_path)
        return False
    
    def _remove_with_inpaint(
        self,
        input_path: Path,
        output_path: Path,
    ) -> bool:
        if cv2 is None or np is None:
            print("Warning: OpenCV not installed, skipping inpainting")
            return False

        # Check if there's a watermark region to process
        if not self.watermark_region or any(v is None for v in self.watermark_region.values()):
            # No watermark region, just copy the file
            import shutil
            shutil.copy(input_path, output_path)
            return True

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            return False

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        ensure_dir(output_path.parent)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        mask = self._create_watermark_mask(width, height)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if mask is not None:
                inpainted = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
                out.write(inpainted)
            else:
                out.write(frame)

        cap.release()
        out.release()

        # After OpenCV processing (which only handles video), add audio back using FFmpeg
        temp_output = output_path.with_suffix('.temp.mp4')
        import shutil
        shutil.move(output_path, temp_output)

        try:
            # Use FFmpeg to copy audio from original and merge with processed video
            FFmpegWrapper.run_command([
                "-i", str(temp_output),  # Video from OpenCV (no audio)
                "-i", str(input_path),   # Original file (with audio)
                "-map", "0:v",           # Use video from processed file
                "-map", "1:a?",          # Use audio from original if exists
                "-c:v", "copy",
                "-c:a", "copy",
                str(output_path),
            ])
            return True
        finally:
            # Clean up temp file
            if temp_output.exists():
                temp_output.unlink()

        return True
    
    def _remove_with_delogo(
        self,
        input_path: Path,
        output_path: Path,
    ) -> bool:
        region = self.watermark_region
        # Fixed: Check for None values instead of using all() which fails for x=0 or y=0
        if not region or any(v is None for v in region.values()):
            return False

        try:
            FFmpegWrapper.delogo(
                input_path,
                output_path,
                region["x"],
                region["y"],
                region["width"],
                region["height"],
            )
            return True
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Delogo removal failed: {e}", exc_info=True)
            return False
    
    def _create_watermark_mask(
        self,
        width: int,
        height: int,
    ) -> Optional:
        if np is None:
            return None

        # Fixed: Check for None values instead of using all() which fails for x=0 or y=0
        if self.watermark_region and all(v is not None for v in self.watermark_region.values()):
            x = self.watermark_region["x"]
            y = self.watermark_region["y"]
            w = self.watermark_region["width"]
            h = self.watermark_region["height"]

            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y:y+h, x:x+w] = 255
            return mask

        return None
