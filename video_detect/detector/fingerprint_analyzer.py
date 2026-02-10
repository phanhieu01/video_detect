"""Invisible fingerprint analyzer - detect steganographic fingerprints"""

from pathlib import Path
from typing import Optional, Dict, Any
import hashlib
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class FingerprintAnalyzer:
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def analyze(self, video_path: Path) -> Dict[str, Any]:
        if cv2 is None:
            return {"error": "OpenCV not available"}
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": "Cannot open video"}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        result = {
            "video_path": str(video_path),
            "total_frames": total_frames,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "potential_fingerprints": [],
        }
        
        sample_indices = np.linspace(0, total_frames - 1, min(30, total_frames), dtype=int)
        frames_hash = []
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_hash = self._compute_frame_hash(gray)
                frames_hash.append(frame_hash)
        
        cap.release()
        
        if len(frames_hash) > 1:
            consistency = self._analyze_hash_consistency(frames_hash)
            result["hash_consistency"] = consistency
            
            if consistency > 0.9:
                result["potential_fingerprints"].append({
                    "type": "static_pattern",
                    "severity": "high",
                    "description": "Very consistent frame patterns detected",
                })
        
        lsb_analysis = self._analyze_lsb(video_path)
        result["lsb_analysis"] = lsb_analysis
        
        if lsb_analysis.get("suspicious"):
            result["potential_fingerprints"].append({
                "type": "lsb_steganography",
                "severity": lsb_analysis.get("severity", "medium"),
                "description": "Suspicious LSB patterns detected",
            })
        
        return result
    
    def _compute_frame_hash(self, frame: np.ndarray) -> str:
        resized = cv2.resize(frame, (32, 32))
        return hashlib.sha256(resized.tobytes()).hexdigest()
    
    def _analyze_hash_consistency(self, hashes: list) -> float:
        if len(hashes) < 2:
            return 0.0
        
        similarity_scores = []
        for i in range(len(hashes) - 1):
            similarity = self._hash_similarity(hashes[i], hashes[i + 1])
            similarity_scores.append(similarity)
        
        return np.mean(similarity_scores)
    
    def _hash_similarity(self, hash1: str, hash2: str) -> float:
        if len(hash1) != len(hash2):
            return 0.0
        
        matches = sum(1 for a, b in zip(hash1, hash2) if a == b)
        return matches / len(hash1)
    
    def _analyze_lsb(self, video_path: Path) -> Dict[str, Any]:
        if cv2 is None:
            return {"suspicious": False, "reason": "OpenCV not available"}
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"suspicious": False, "reason": "Cannot open video"}
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return {"suspicious": False, "reason": "No frames"}
        
        lsb_uniformity = self._check_lsb_uniformity(frame)
        
        if lsb_uniformity > 0.8:
            return {
                "suspicious": True,
                "severity": "high",
                "uniformity": lsb_uniformity,
            }
        elif lsb_uniformity > 0.6:
            return {
                "suspicious": True,
                "severity": "medium",
                "uniformity": lsb_uniformity,
            }
        
        return {"suspicious": False, "uniformity": lsb_uniformity}
    
    def _check_lsb_uniformity(self, frame: np.ndarray) -> float:
        lsb = frame & 1
        
        zero_count = np.sum(lsb == 0)
        one_count = np.sum(lsb == 1)
        total = zero_count + one_count
        
        if total == 0:
            return 0.0
        
        ratio = max(zero_count, one_count) / total
        return ratio
