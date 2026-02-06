"""Video Processor - watermark and fingerprint removal"""

from pathlib import Path
from typing import Optional
try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from utils.ffmpeg_wrapper import FFmpegWrapper
from utils import ensure_dir


class WatermarkRemover:
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.method = self.config.get("method", "inpaint")
        self.auto_detect = self.config.get("auto_detect", False)
        self.watermark_region = self.config.get("watermark_region", {})
        
        if self.auto_detect:
            from detector import WatermarkDetector
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
        return True
    
    def _remove_with_delogo(
        self,
        input_path: Path,
        output_path: Path,
    ) -> bool:
        region = self.watermark_region
        if not all(region.values()):
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
        except Exception:
            return False
    
    def _create_watermark_mask(
        self,
        width: int,
        height: int,
    ) -> Optional:
        if np is None:
            return None
        
        if self.watermark_region and all(self.watermark_region.values()):
            x = self.watermark_region["x"]
            y = self.watermark_region["y"]
            w = self.watermark_region["width"]
            h = self.watermark_region["height"]
            
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y:y+h, x:x+w] = 255
            return mask
        
        return None
