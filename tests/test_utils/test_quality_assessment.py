"""Tests for QualityAssessment class"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from video_detect.utils.quality_assessment import QualityAssessment


class TestQualityAssessment:
    """Test cases for QualityAssessment"""

    @patch("video_detect.utils.quality_assessment.CV2_AVAILABLE", False)
    def test_assess_removal_no_cv2(self):
        """Test assess_removal when cv2 is not available"""
        result = QualityAssessment.assess_removal(Path("orig.mp4"), Path("proc.mp4"))
        assert "error" in result
        assert result["error"] == "OpenCV not available"

    @patch("video_detect.utils.quality_assessment.cv2")
    def test_assess_removal_invalid_files(self, mock_cv2):
        """Test assess_removal with invalid files"""
        mock_cv2.imread.return_value = None
        result = QualityAssessment.assess_removal(Path("orig.jpg"), Path("proc.jpg"))
        assert "error" in result

    @patch("video_detect.utils.quality_assessment.cv2")
    def test_assess_removal_success(self, mock_cv2):
        """Test assess_removal with valid images"""
        mock_cv2.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        result = QualityAssessment.assess_removal(Path("orig.jpg"), Path("proc.jpg"))

        assert "metrics" in result
        assert "mse" in result["metrics"]
        assert "psnr" in result["metrics"]
        assert "ssim" in result["metrics"]
        assert "overall_quality" in result

    def test_calculate_mse(self):
        """Test _calculate_mse"""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)

        mse = QualityAssessment._calculate_mse(img1, img2)
        assert mse == 0.0

    def test_calculate_psnr(self):
        """Test _calculate_psnr with identical images"""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)

        psnr = QualityAssessment._calculate_psnr(img1, img2)
        assert psnr == float('inf')

    def test_calculate_psnr_with_difference(self):
        """Test _calculate_psnr with different images"""
        img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 100

        psnr = QualityAssessment._calculate_psnr(img1, img2)
        assert psnr > 0

    @patch("video_detect.utils.quality_assessment.cv2")
    def test_calculate_ssim(self, mock_cv2):
        """Test _calculate_ssim with mock"""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock GaussianBlur to return predictable values
        mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.GaussianBlur.return_value = np.zeros((100, 100), dtype=np.float64)

        ssim = QualityAssessment._calculate_ssim(img1, img2)
        assert ssim >= 0 and ssim <= 1

    def test_determine_quality_excellent(self):
        """Test _determine_quality for excellent quality"""
        quality = QualityAssessment._determine_quality(40, 0.95)
        assert quality == "excellent"

    def test_determine_quality_good(self):
        """Test _determine_quality for good quality"""
        quality = QualityAssessment._determine_quality(30, 0.85)
        assert quality == "good"

    def test_determine_quality_fair(self):
        """Test _determine_quality for fair quality"""
        quality = QualityAssessment._determine_quality(25, 0.75)
        assert quality == "fair"

    def test_determine_quality_poor(self):
        """Test _determine_quality for poor quality"""
        quality = QualityAssessment._determine_quality(15, 0.5)
        assert quality == "poor"

    @patch("video_detect.utils.quality_assessment.cv2")
    def test_assess_video_quality(self, mock_cv2):
        """Test assess_video_quality"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [30, 300, 640, 480]  # fps, frame_count, width, height
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = mock_cap

        result = QualityAssessment.assess_video_quality(Path("test.mp4"))
        assert isinstance(result, dict)
        assert "fps" in result
        assert "resolution" in result
        assert "quality" in result
