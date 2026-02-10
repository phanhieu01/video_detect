"""Text watermark detector using OCR"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class TextDetector:

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.languages = self.config.get("ocr_languages", ["en", "vi"])
        self.confidence_threshold = self.config.get("ocr_confidence", 0.7)

        if EASYOCR_AVAILABLE:
            self.reader = easyocr.Reader(self.languages, gpu=False)
        else:
            self.reader = None

    def detect_text(
        self,
        image_path: Path,
    ) -> Dict[str, Any]:
        """Detect text in image using OCR"""
        if not EASYOCR_AVAILABLE or not CV2_AVAILABLE:
            return {"error": "EasyOCR or OpenCV not available"}

        img = cv2.imread(str(image_path))
        if img is None:
            return {"error": "Cannot read image"}

        result = {
            "image_path": str(image_path),
            "texts": [],
            "potential_watermarks": [],
        }

        # Run OCR
        ocr_results = self.reader.readtext(img)

        for detection in ocr_results:
            bbox, text, confidence = detection

            if confidence >= self.confidence_threshold:
                result["texts"].append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": bbox,
                })

        # Identify potential username watermarks
        result["potential_watermarks"] = self._identify_watermarks(result["texts"])

        return result

    def detect_text_video(
        self,
        video_path: Path,
        frame_count: int = 5,
    ) -> Dict[str, Any]:
        """Detect text in video by sampling frames"""
        if not EASYOCR_AVAILABLE or not CV2_AVAILABLE:
            return {"error": "EasyOCR or OpenCV not available"}

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": "Cannot open video"}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = np.linspace(0, total_frames - 1, min(frame_count, total_frames), dtype=int)

        all_texts = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                ocr_results = self.reader.readtext(frame)
                for detection in ocr_results:
                    bbox, text, confidence = detection
                    if confidence >= self.confidence_threshold:
                        all_texts.append({
                            "text": text,
                            "confidence": float(confidence),
                            "frame": int(idx),
                        })

        cap.release()

        # Remove duplicates
        seen = set()
        unique_texts = []
        for item in all_texts:
            text_lower = item["text"].lower()
            if text_lower not in seen:
                seen.add(text_lower)
                unique_texts.append(item)

        return {
            "video_path": str(video_path),
            "texts": unique_texts,
            "potential_watermarks": self._identify_watermarks(unique_texts),
        }

    def _identify_watermarks(self, texts: List[Dict]) -> List[Dict]:
        """Identify potential username watermarks from detected text"""
        watermarks = []
        username_keywords = [
            "@", "follow", "tiktok", "instagram", "fb", "facebook",
            "twitter", "x.com", "yt", "youtube", "twitch", "douyin",
        ]

        for text_item in texts:
            text = text_item["text"]
            text_lower = text.lower()

            # Check for username patterns
            if any(keyword in text_lower for keyword in username_keywords):
                watermarks.append({
                    "type": "username_watermark",
                    "text": text,
                    "confidence": text_item["confidence"],
                    "severity": "high",
                })
            # Check for short alphanumeric strings (usernames)
            elif 3 <= len(text) <= 20 and any(c.isalnum() for c in text):
                if sum(c.isalnum() for c in text) / len(text) > 0.7:
                    watermarks.append({
                        "type": "potential_username",
                        "text": text,
                        "confidence": text_item["confidence"],
                        "severity": "medium",
                    })

        return watermarks
