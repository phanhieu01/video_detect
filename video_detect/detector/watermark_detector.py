"""Automatic watermark detector using template matching and edge detection"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class WatermarkDetector:
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.detection_method = self.config.get("detection_method", "auto")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        
        self._load_common_patterns()
    
    def detect_watermark(
        self,
        video_path: Path,
        frame_count: int = 30,
    ) -> Optional[Dict[str, Any]]:
        if cv2 is None:
            return None
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = min(frame_count, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            return None
        
        result = self._detect_from_frames(frames)
        return result
    
    def _detect_from_frames(
        self,
        frames: list,
    ) -> Dict[str, Any]:
        height, width = frames[0].shape[:2]
        
        result = {
            "found": False,
            "method": None,
            "region": None,
            "confidence": 0.0,
        }
        
        detected_regions = []
        
        for pattern_name, pattern_info in self.common_patterns.items():
            if pattern_info.get("type") == "logo":
                region = self._detect_logo(frames, pattern_info)
                if region:
                    detected_regions.append({
                        "pattern": pattern_name,
                        "region": region,
                        "confidence": pattern_info.get("confidence", 0.8),
                    })
        
        if detected_regions:
            best_match = max(detected_regions, key=lambda x: x["confidence"])
            result.update({
                "found": True,
                "method": "template_matching",
                "region": best_match["region"],
                "confidence": best_match["confidence"],
                "pattern": best_match["pattern"],
            })
        else:
            region = self._detect_by_edge_detection(frames)
            if region:
                result.update({
                    "found": True,
                    "method": "edge_detection",
                    "region": region,
                    "confidence": 0.6,
                })
            else:
                region = self._detect_by_static_overlay(frames)
                if region:
                    result.update({
                        "found": True,
                        "method": "static_overlay",
                        "region": region,
                        "confidence": 0.5,
                    })
        
        return result
    
    def _detect_logo(
        self,
        frames: list,
        pattern_info: dict,
    ) -> Optional[Dict[str, int]]:
        if cv2 is None:
            return None
        
        template = pattern_info.get("template")
        if template is None:
            return None
        
        height, width = frames[0].shape[:2]
        template_h, template_w = template.shape[:2]
        
        positions = []
        
        for frame in frames[:3]:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            res = cv2.matchTemplate(gray_frame, template_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if max_val >= self.confidence_threshold:
                positions.append(max_loc)
        
        if len(positions) >= 2:
            avg_x = int(np.mean([p[0] for p in positions]))
            avg_y = int(np.mean([p[1] for p in positions]))
            
            return {
                "x": max(0, avg_x - 10),
                "y": max(0, avg_y - 10),
                "width": min(width - avg_x, template_w + 20),
                "height": min(height - avg_y, template_h + 20),
            }
        
        return None
    
    def _detect_by_edge_detection(
        self,
        frames: list,
    ) -> Optional[Dict[str, int]]:
        if cv2 is None:
            return None
        
        height, width = frames[0].shape[:2]
        combined_edges = None
        
        for frame in frames[:3]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            if combined_edges is None:
                combined_edges = edges
            else:
                combined_edges = cv2.bitwise_and(combined_edges, edges)
        
        if combined_edges is None:
            return None
        
        kernel = np.ones((50, 50), np.uint8)
        dilated = cv2.dilate(combined_edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            if 0.5 < aspect_ratio < 2.0 and 100 < area < 50000:
                candidates.append((x, y, w, h, area))
        
        if candidates:
            candidates.sort(key=lambda x: x[4], reverse=True)
            x, y, w, h = candidates[0][:4]
            return {"x": x, "y": y, "width": w, "height": h}
        
        return None
    
    def _detect_by_static_overlay(
        self,
        frames: list,
    ) -> Optional[Dict[str, int]]:
        if cv2 is None:
            return None
        
        height, width = frames[0].shape[:2]
        
        if len(frames) < 2:
            return None
        
        reference_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        diff_accum = np.zeros_like(reference_gray, dtype=np.float32)
        
        for frame in frames[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(reference_gray, gray)
            diff_accum += diff
        
        diff_accum = diff_accum.astype(np.uint8)
        static_mask = 255 - cv2.threshold(diff_accum, 30, 255, cv2.THRESH_BINARY)[1]
        
        kernel = np.ones((30, 30), np.uint8)
        dilated = cv2.dilate(static_mask, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if 0 < x < width and 0 < y < height and w > 50 and h > 20:
                is_corner = (
                    x < 100 or x > width - 100 or
                    y < 100 or y > height - 100
                )
                
                if is_corner:
                    candidates.append((x, y, w, h))
        
        if candidates:
            x, y, w, h = candidates[0]
            return {"x": x, "y": y, "width": w, "height": h}
        
        return None
    
    def _load_common_patterns(self):
        self.common_patterns = {
            "tiktok_logo": {
                "type": "logo",
                "confidence": 0.85,
                "template": self._create_tiktok_template(),
                "expected_position": "bottom_right",
            },
            "tiktok_note": {
                "type": "logo",
                "confidence": 0.80,
                "template": self._create_tiktok_note_template(),
                "expected_position": "bottom_right",
            },
            "instagram_logo": {
                "type": "logo",
                "confidence": 0.75,
                "template": self._create_instagram_gradient_template(),
                "expected_position": "bottom_right",
            },
            "instagram_handle": {
                "type": "logo",
                "confidence": 0.70,
                "template": self._create_instagram_template(),
                "expected_position": "bottom_right",
            },
            "tiktok_handle": {
                "type": "text",
                "confidence": 0.70,
                "expected_position": "bottom_center",
            },
        }

        # Try to load custom templates from file
        self._load_custom_templates()

    def _load_custom_templates(self):
        """Load custom templates from templates directory"""
        from pathlib import Path

        templates_dir = Path(__file__).parent / "templates"
        if templates_dir.exists():
            for template_file in templates_dir.glob("*.png"):
                name = template_file.stem
                if CV2_AVAILABLE:
                    template_img = cv2.imread(str(template_file))
                    if template_img is not None:
                        self.common_patterns[f"custom_{name}"] = {
                            "type": "logo",
                            "confidence": 0.90,
                            "template": template_img,
                            "expected_position": "auto",
                        }

    def _create_tiktok_template(self) -> Optional[np.ndarray]:
        if cv2 is None or np is None:
            return None

        # Create more realistic TikTok logo template
        template = np.ones((50, 50, 3), dtype=np.uint8) * 240

        # TikTok logo colors (cyan and pink/red)
        cyan = (0, 255, 255)
        pink = (254, 0, 100)

        # Draw 8 note shape
        center_x, center_y = 25, 25

        # Left cyan curve
        cv2.ellipse(template, (center_x - 5, center_y), (8, 12), 0, 0, 180, cyan, 3)

        # Right pink/red curve
        cv2.ellipse(template, (center_x + 5, center_y), (8, 12), 0, 0, 180, pink, 3)

        return template

    def _create_tiktok_note_template(self) -> Optional[np.ndarray]:
        """Create TikTok musical note template"""
        if cv2 is None or np is None:
            return None

        template = np.ones((60, 60, 3), dtype=np.uint8) * 255

        # Draw musical note symbol
        cyan = (0, 255, 255)
        pink = (254, 0, 100)

        # Note head
        cv2.circle(template, (30, 45), 8, cyan, -1)

        # Note stem
        cv2.rectangle(template, (36, 15), (38, 45), cyan, -1)

        # Note flag (curved)
        cv2.ellipse(template, (38, 20), (8, 12), 0, 180, 270, pink, 3)

        return template

    def _create_instagram_gradient_template(self) -> Optional[np.ndarray]:
        """Create Instagram gradient logo template"""
        if cv2 is None or np is None:
            return None

        template = np.ones((50, 50, 3), dtype=np.uint8) * 255

        # Create gradient effect (yellow to purple)
        for i in range(15):
            color = (
                int(255 - i * 5),   # R: yellow to purple
                int(100 + i * 7),  # G
                int(150 + i * 7),  # B
            )
            cv2.circle(template, (25, 25), 15 - i, color, -1)

        # Camera outline
        cv2.circle(template, (25, 25), 15, (50, 50, 50), 2)

        return template
    
    def _create_tiktok_template(self) -> Optional[np.ndarray]:
        if cv2 is None or np is None:
            return None
        
        template = np.ones((40, 40, 3), dtype=np.uint8) * 255
        
        cv2.circle(template, (20, 20), 15, (0, 0, 0), -1)
        cv2.circle(template, (20, 20), 10, (255, 255, 255), -1)
        cv2.putText(
            template,
            "TT",
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
        
        return template
    
    def _create_instagram_template(self) -> Optional[np.ndarray]:
        if cv2 is None or np is None:
            return None
        
        template = np.ones((30, 30, 3), dtype=np.uint8) * 255
        
        cv2.circle(template, (15, 15), 12, (0, 0, 0), 2)
        cv2.circle(template, (15, 15), 2, (0, 0, 0), -1)
        
        return template
    
    def update_config(self, video_path: Path) -> bool:
        result = self.detect_watermark(video_path)
        
        if result and result["found"]:
            region = result["region"]
            print(f"Watermark detected: {result['method']}")
            print(f"Region: x={region['x']}, y={region['y']}, w={region['width']}, h={region['height']}")
            
            watermark_config = {
                "enabled": True,
                "method": "inpaint",
                "watermark_region": region,
            }
            
            return True
        
        return False
