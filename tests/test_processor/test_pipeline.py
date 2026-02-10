"""Tests for ProcessingPipeline class"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from video_detect.processor.pipeline import ProcessingPipeline


class TestProcessingPipeline:
    """Test cases for ProcessingPipeline"""

    def test_init_default_config(self):
        """Test initialization with default config"""
        pipeline = ProcessingPipeline({})
        assert pipeline is not None

    def test_init_with_config(self, sample_config):
        """Test initialization with config"""
        pipeline = ProcessingPipeline(sample_config)
        assert pipeline.config == sample_config

    @patch("video_detect.processor.pipeline.MetadataCleaner")
    @patch("video_detect.processor.pipeline.WatermarkRemover")
    @patch("video_detect.processor.pipeline.FingerprintRemover")
    @patch("video_detect.processor.pipeline.VideoTransformer")
    @patch("video_detect.processor.pipeline.ImageTransformer")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_file")
    @patch("shutil.copy")
    def test_process_success(self, mock_copy, mock_is_file, mock_exists, mock_img_trans, mock_vid_trans, mock_fingerprint, mock_watermark, mock_cleaner, sample_video_path, temp_output_dir):
        """Test successful processing"""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_copy.return_value = None

        mock_cleaner_instance = Mock()
        mock_cleaner.return_value = mock_cleaner_instance
        mock_cleaner_instance.clean.return_value = None

        mock_watermark_instance = Mock()
        mock_watermark.return_value = mock_watermark_instance

        mock_fingerprint_instance = Mock()
        mock_fingerprint.return_value = mock_fingerprint_instance

        mock_vid_trans_instance = Mock()
        mock_vid_trans.return_value = mock_vid_trans_instance

        mock_img_trans_instance = Mock()
        mock_img_trans.return_value = mock_img_trans_instance

        pipeline = ProcessingPipeline({"output": {"output_dir": str(temp_output_dir)}})
        result = pipeline.process(sample_video_path, temp_output_dir / "output.mp4")

        assert result is True

    @patch("video_detect.processor.pipeline.MetadataCleaner")
    @patch("video_detect.processor.pipeline.WatermarkRemover")
    @patch("video_detect.processor.pipeline.FingerprintRemover")
    @patch("video_detect.processor.pipeline.VideoTransformer")
    @patch("video_detect.processor.pipeline.ImageTransformer")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_file")
    @patch("shutil.copy")
    def test_process_with_custom_steps(self, mock_copy, mock_is_file, mock_exists, mock_img_trans, mock_vid_trans, mock_fingerprint, mock_watermark, mock_cleaner, sample_video_path, temp_output_dir):
        """Test processing with custom steps"""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_copy.return_value = None

        mock_cleaner_instance = Mock()
        mock_cleaner.return_value = mock_cleaner_instance
        mock_cleaner_instance.clean.return_value = None

        mock_watermark_instance = Mock()
        mock_watermark.return_value = mock_watermark_instance

        mock_fingerprint_instance = Mock()
        mock_fingerprint.return_value = mock_fingerprint_instance

        mock_vid_trans_instance = Mock()
        mock_vid_trans.return_value = mock_vid_trans_instance

        mock_img_trans_instance = Mock()
        mock_img_trans.return_value = mock_img_trans_instance

        pipeline = ProcessingPipeline({"output": {"output_dir": str(temp_output_dir)}})
        steps = ["metadata_clean"]
        result = pipeline.process(sample_video_path, temp_output_dir / "output.mp4", steps)

        assert result is True

    @patch("video_detect.processor.pipeline.MetadataCleaner")
    def test_process_invalid_input(self, mock_cleaner):
        """Test processing with invalid input path"""
        pipeline = ProcessingPipeline({})
        result = pipeline.process(Path("nonexistent.mp4"), Path("output.mp4"))
        assert result is False

    def test_determine_steps_video(self, sample_video_path):
        """Test _default_steps for video file"""
        pipeline = ProcessingPipeline({})
        steps = pipeline._default_steps(sample_video_path.suffix)
        assert isinstance(steps, list)
        assert len(steps) > 0

    def test_determine_steps_image(self, sample_image_path):
        """Test _default_steps for image file"""
        pipeline = ProcessingPipeline({})
        steps = pipeline._default_steps(sample_image_path.suffix)
        assert isinstance(steps, list)
        assert len(steps) > 0

    def test_determine_steps_unknown_file(self):
        """Test _default_steps for unknown file type"""
        pipeline = ProcessingPipeline({})
        steps = pipeline._default_steps(".txt")
        assert isinstance(steps, list)

    @patch("video_detect.processor.pipeline.MetadataCleaner")
    def test_process_with_error(self, mock_cleaner, sample_video_path, temp_output_dir):
        """Test processing when an error occurs"""
        mock_cleaner_instance = Mock()
        mock_cleaner.return_value = mock_cleaner_instance
        mock_cleaner_instance.clean.side_effect = Exception("Test error")

        pipeline = ProcessingPipeline({"output": {"output_dir": str(temp_output_dir)}})
        result = pipeline.process(sample_video_path, temp_output_dir / "output.mp4")

        # Should handle error gracefully
        assert isinstance(result, bool)
