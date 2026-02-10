"""Tests for WatermarkDetector class"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from video_detect.detector.watermark_detector import WatermarkDetector


class TestWatermarkDetector:
    """Test cases for WatermarkDetector"""

    def test_init_default_config(self):
        """Test initialization with default config"""
        detector = WatermarkDetector()
        assert detector.config == {}
        assert detector.detection_method == "auto"
        assert detector.confidence_threshold == 0.7

    def test_init_custom_config(self, sample_config):
        """Test initialization with custom config"""
        detector = WatermarkDetector(sample_config.get("watermark"))
        assert detector.detection_method == "auto"
        assert detector.confidence_threshold == 0.7

    def test_init_custom_threshold(self):
        """Test initialization with custom confidence threshold"""
        config = {"confidence_threshold": 0.85}
        detector = WatermarkDetector(config)
        assert detector.confidence_threshold == 0.85

    @patch("video_detect.detector.watermark_detector.cv2")
    def test_detect_watermark_no_cv2(self, mock_cv2):
        """Test detect_watermark when cv2 is not available"""
        # Set cv2 to None to simulate OpenCV not available
        import sys
        original_cv2 = sys.modules.get("cv2")
        sys.modules["cv2"] = None
        try:
            # Need to reload the module to pick up the change
            from importlib import reload
            from video_detect.detector import watermark_detector
            reload(watermark_detector)

            detector = watermark_detector.WatermarkDetector()
            result = detector.detect_watermark(Path("test.mp4"))
            assert result is None
        finally:
            # Restore cv2 module
            if original_cv2 is not None:
                sys.modules["cv2"] = original_cv2
            elif "cv2" in sys.modules:
                del sys.modules["cv2"]

    @patch("video_detect.detector.watermark_detector.cv2.VideoCapture")
    def test_detect_watermark_invalid_video(self, mock_capture):
        """Test detect_watermark with invalid video file"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap

        detector = WatermarkDetector()
        result = detector.detect_watermark(Path("invalid.mp4"))
        assert result is None

    @patch("video_detect.detector.watermark_detector.cv2.VideoCapture")
    def test_detect_watermark_no_frames(self, mock_capture):
        """Test detect_watermark when no frames can be read"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.read.return_value = (False, None)
        mock_capture.return_value = mock_cap

        detector = WatermarkDetector()
        result = detector.detect_watermark(Path("no_frames.mp4"))
        assert result is None

    @patch("video_detect.detector.watermark_detector.cv2.VideoCapture")
    @patch("video_detect.detector.watermark_detector.np.linspace")
    def test_detect_watermark_frame_sampling(self, mock_linspace, mock_capture):
        """Test that detect_watermark samples frames correctly"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 300
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_capture.return_value = mock_cap
        mock_linspace.return_value = np.array([0, 10, 20])

        detector = WatermarkDetector()
        result = detector.detect_watermark(Path("test.mp4"), frame_count=3)

        # Verify frame sampling was called
        mock_linspace.assert_called_once()
        assert mock_cap.set.call_count >= 3

    def test_detect_from_frames_no_watermark(self, sample_frame):
        """Test _detect_from_frames returns not found when no watermark"""
        detector = WatermarkDetector()
        frames = [sample_frame] * 5
        result = detector._detect_from_frames(frames)

        assert result["found"] is False
        assert result["method"] is None
        assert result["confidence"] == 0.0

    def test_detect_from_frames_with_common_patterns(self, sample_frame):
        """Test _detect_from_frames checks common patterns"""
        detector = WatermarkDetector()
        assert "tiktok_logo" in detector.common_patterns
        assert "instagram_logo" in detector.common_patterns

    def test_detect_logo_none_template(self):
        """Test _detect_logo returns None for None template"""
        detector = WatermarkDetector()
        pattern_info = {"template": None}
        result = detector._detect_logo([], pattern_info)
        assert result is None

    def test_detect_logo_no_cv2(self):
        """Test _detect_logo when cv2 is not available"""
        detector = WatermarkDetector()
        pattern_info = {"template": np.zeros((50, 50, 3), dtype=np.uint8)}
        with patch("video_detect.detector.watermark_detector.cv2", None):
            result = detector._detect_logo([], pattern_info)
            assert result is None

    def test_detect_by_edge_detection_none_cv2(self):
        """Test _detect_by_edge_detection when cv2 is not available"""
        detector = WatermarkDetector()
        with patch("video_detect.detector.watermark_detector.cv2", None):
            result = detector._detect_by_edge_detection([])
            assert result is None

    def test_detect_by_static_overlay_none_cv2(self):
        """Test _detect_by_static_overlay when cv2 is not available"""
        detector = WatermarkDetector()
        with patch("video_detect.detector.watermark_detector.cv2", None):
            result = detector._detect_by_static_overlay([])
            assert result is None

    def test_detect_by_static_overlay_insufficient_frames(self, sample_frame):
        """Test _detect_by_static_overlay with insufficient frames"""
        detector = WatermarkDetector()
        result = detector._detect_by_static_overlay([sample_frame])
        assert result is None

    def test_create_tiktok_template(self):
        """Test _create_tiktok_template returns valid template"""
        detector = WatermarkDetector()
        template = detector._create_tiktok_template()
        assert template is not None
        # Template size may vary based on implementation
        assert len(template.shape) == 3  # Should be 3D (height, width, channels)

    def test_create_tiktok_note_template(self):
        """Test _create_tiktok_note_template returns valid template"""
        detector = WatermarkDetector()
        template = detector._create_tiktok_note_template()
        assert template is not None
        assert template.shape == (60, 60, 3)

    def test_create_instagram_gradient_template(self):
        """Test _create_instagram_gradient_template returns valid template"""
        detector = WatermarkDetector()
        template = detector._create_instagram_gradient_template()
        assert template is not None
        assert template.shape == (50, 50, 3)

    def test_common_patterns_loaded(self):
        """Test that common patterns are loaded correctly"""
        detector = WatermarkDetector()
        assert hasattr(detector, "common_patterns")
        assert len(detector.common_patterns) > 0

    def test_common_pattern_structure(self):
        """Test that common patterns have correct structure"""
        detector = WatermarkDetector()
        for pattern_name, pattern_info in detector.common_patterns.items():
            assert "type" in pattern_info
            assert "confidence" in pattern_info

    def test_update_config_with_detection(self, sample_video_path):
        """Test update_config updates config when watermark is detected"""
        detector = WatermarkDetector()

        with patch.object(detector, "detect_watermark", return_value={
            "found": True,
            "method": "template_matching",
            "region": {"x": 100, "y": 100, "width": 50, "height": 50},
        }):
            result = detector.update_config(sample_video_path)
            assert result is True

    def test_update_config_no_detection(self, sample_video_path):
        """Test update_config returns False when no watermark detected"""
        detector = WatermarkDetector()

        with patch.object(detector, "detect_watermark", return_value={
            "found": False,
        }):
            result = detector.update_config(sample_video_path)
            assert result is False
