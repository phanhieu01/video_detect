"""Example custom detector plugin"""

from pathlib import Path
from typing import Dict, Any
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plugins import DetectorPlugin


class CustomUsernameDetector(DetectorPlugin):
    """Example: Custom detector for username patterns"""

    __plugin_metadata__ = {
        "type": "detector",
        "name": "Custom Username Detector",
        "version": "1.0.0",
        "description": "Detect username watermarks using custom patterns",
        "author": "Your Name",
    }

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.patterns = self.config.get("patterns", [
            "@", "@.", "_", "tiktok.com", "instagram.com",
        ])
        self.confidence = self.config.get("confidence", 0.7)

    def detect(self, file_path: Path) -> Dict[str, Any]:
        """Detect username watermarks"""
        # This is a simple example - real implementation would use OCR
        return {
            "found": False,
            "method": "custom_username_detector",
            "confidence": self.confidence,
            "patterns": self.patterns,
        }

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "patterns": {
                "type": "list",
                "description": "Username patterns to search for",
                "default": ["@", "@."],
            },
            "confidence": {
                "type": "float",
                "description": "Confidence threshold",
                "default": 0.7,
                "minimum": 0.0,
                "maximum": 1.0,
            },
        }


class CustomFingerprintDetector(DetectorPlugin):
    """Example: Custom fingerprint detector"""

    __plugin_metadata__ = {
        "type": "analyzer",
        "name": "Custom Fingerprint Detector",
        "version": "1.0.0",
        "description": "Detect invisible fingerprints using custom algorithm",
        "author": "Your Name",
    }

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.algorithm = self.config.get("algorithm", "advanced")

    def analyze(self, file_path: Path) -> Dict[str, Any]:
        """Analyze file for fingerprints"""
        return {
            "file_path": str(file_path),
            "algorithm": self.algorithm,
            "potential_fingerprints": [],
        }

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "algorithm": {
                "type": "string",
                "description": "Algorithm to use",
                "default": "advanced",
                "enum": ["basic", "advanced", "experimental"],
            },
        }
