"""Tests for FFmpegWrapper class"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, call
import subprocess

from video_detect.utils.ffmpeg_wrapper import FFmpegWrapper


class TestFFmpegWrapper:
    """Test cases for FFmpegWrapper"""

    @patch("subprocess.run")
    def test_get_version_success(self, mock_run):
        """Test get_version when FFmpeg is available"""
        mock_result = Mock()
        mock_result.stdout = "ffmpeg version 5.0\nmore info"
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        version = FFmpegWrapper.get_version()
        assert "ffmpeg version 5.0" in version

    @patch("subprocess.run")
    def test_get_version_not_found(self, mock_run):
        """Test get_version when FFmpeg is not found"""
        mock_run.side_effect = FileNotFoundError()
        version = FFmpegWrapper.get_version()
        assert version == "FFmpeg not found"

    @patch("subprocess.run")
    def test_get_version_error(self, mock_run):
        """Test get_version when FFmpeg returns error"""
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
        version = FFmpegWrapper.get_version()
        assert version == "FFmpeg not found"

    @patch.object(FFmpegWrapper, "_safe_run")
    def test_run_command_success(self, mock_safe_run):
        """Test run_command executes successfully"""
        mock_safe_run.return_value.__enter__.return_value = "output"

        result = FFmpegWrapper.run_command(["-i", "input.mp4", "output.mp4"])
        assert result == "output"

    @patch("subprocess.run")
    def test_run_command_failure(self, mock_run):
        """Test run_command handles failure"""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Error occurred"
        mock_run.return_value = mock_result

        with pytest.raises(RuntimeError, match="FFmpeg error"):
            FFmpegWrapper.run_command(["-i", "input.mp4"])

    @patch("subprocess.run")
    def test_has_audio_stream_true(self, mock_run):
        """Test has_audio_stream returns True when audio exists"""
        mock_result = Mock()
        mock_result.stdout = "audio\n"
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        result = FFmpegWrapper.has_audio_stream(Path("test.mp4"))
        assert result is True

    @patch("subprocess.run")
    def test_has_audio_stream_false(self, mock_run):
        """Test has_audio_stream returns False when no audio"""
        mock_result = Mock()
        mock_result.stdout = ""
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        result = FFmpegWrapper.has_audio_stream(Path("test.mp4"))
        assert result is False

    @patch("subprocess.run")
    def test_has_audio_stream_error(self, mock_run):
        """Test has_audio_stream handles errors"""
        mock_run.side_effect = FileNotFoundError()
        result = FFmpegWrapper.has_audio_stream(Path("test.mp4"))
        assert result is False

    @patch.object(FFmpegWrapper, "run_command")
    def test_strip_metadata(self, mock_run):
        """Test strip_metadata command"""
        FFmpegWrapper.strip_metadata(Path("input.mp4"), Path("output.mp4"))
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "-map_metadata" in call_args
        assert "-1" in call_args

    @patch.object(FFmpegWrapper, "run_command")
    def test_re_encode_default_params(self, mock_run):
        """Test re_encode with default parameters"""
        FFmpegWrapper.re_encode(Path("input.mp4"), Path("output.mp4"))
        mock_run.assert_called_once()

    @patch.object(FFmpegWrapper, "run_command")
    def test_re_encode_with_bitrate(self, mock_run):
        """Test re_encode with custom bitrate"""
        FFmpegWrapper.re_encode(
            Path("input.mp4"),
            Path("output.mp4"),
            bitrate="5M"
        )
        call_args = mock_run.call_args[0][0]
        assert "-b:v" in call_args
        assert "5M" in call_args

    @patch.object(FFmpegWrapper, "run_command")
    def test_delogo(self, mock_run):
        """Test delogo command"""
        FFmpegWrapper.delogo(Path("input.mp4"), Path("output.mp4"), 100, 100, 50, 50)
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "-vf" in call_args

    @patch.object(FFmpegWrapper, "has_audio_stream")
    @patch.object(FFmpegWrapper, "run_command")
    def test_change_speed_with_audio(self, mock_run, mock_has_audio):
        """Test change_speed with audio"""
        mock_has_audio.return_value = True
        FFmpegWrapper.change_speed(Path("input.mp4"), Path("output.mp4"), 1.5)
        mock_run.assert_called_once()

    @patch.object(FFmpegWrapper, "has_audio_stream")
    @patch.object(FFmpegWrapper, "run_command")
    def test_change_speed_without_audio(self, mock_run, mock_has_audio):
        """Test change_speed without audio"""
        mock_has_audio.return_value = False
        FFmpegWrapper.change_speed(Path("input.mp4"), Path("output.mp4"), 1.5)
        mock_run.assert_called_once()

    def test_build_atempo_filter_factor_2(self):
        """Test _build_atempo_filter with factor <= 2.0"""
        result = FFmpegWrapper._build_atempo_filter(1.5)
        assert result == "atempo=1.50"  # Format uses 2 decimal places

    def test_build_atempo_filter_factor_greater_2(self):
        """Test _build_atempo_filter with factor > 2.0"""
        result = FFmpegWrapper._build_atempo_filter(3.0)
        assert "atempo" in result
        # Should chain two atempo filters
        assert "," in result

    @patch.object(FFmpegWrapper, "run_command")
    def test_horizontal_flip(self, mock_run):
        """Test horizontal_flip command"""
        FFmpegWrapper.horizontal_flip(Path("input.mp4"), Path("output.mp4"))
        mock_run.assert_called_once()

    @patch.object(FFmpegWrapper, "run_command")
    def test_crop(self, mock_run):
        """Test crop command"""
        FFmpegWrapper.crop(Path("input.mp4"), Path("output.mp4"), 10)
        mock_run.assert_called_once()

    @patch.object(FFmpegWrapper, "run_command")
    def test_color_adjust(self, mock_run):
        """Test color_adjust command"""
        FFmpegWrapper.color_adjust(
            Path("input.mp4"),
            Path("output.mp4"),
            brightness=0.1,
            contrast=1.1,
            saturation=1.2
        )
        mock_run.assert_called_once()

    @patch.object(FFmpegWrapper, "has_audio_stream")
    @patch.object(FFmpegWrapper, "run_command")
    def test_add_silence_with_audio(self, mock_run, mock_has_audio):
        """Test add_silence with audio stream"""
        mock_has_audio.return_value = True
        FFmpegWrapper.add_silence(Path("input.mp4"), Path("output.mp4"), 2.0)
        mock_run.assert_called_once()

    @patch.object(FFmpegWrapper, "has_audio_stream")
    @patch.object(FFmpegWrapper, "run_command")
    def test_add_silence_without_audio(self, mock_run, mock_has_audio):
        """Test add_silence without audio stream"""
        mock_has_audio.return_value = False
        FFmpegWrapper.add_silence(Path("input.mp4"), Path("output.mp4"), 2.0)
        mock_run.assert_called_once()

    @patch("video_detect.utils.ffmpeg_wrapper.get_gpu_helper")
    def test_get_gpu_hwaccel_params(self, mock_get_gpu):
        """Test get_gpu_hwaccel_params"""
        mock_helper = Mock()
        mock_get_gpu.return_value = mock_helper
        mock_helper.get_ffmpeg_gpu_params.return_value = ["-hwaccel", "cuda"]

        result = FFmpegWrapper.get_gpu_hwaccel_params()
        assert result == ["-hwaccel", "cuda"]

    @patch("video_detect.utils.ffmpeg_wrapper.get_gpu_helper")
    @patch.object(FFmpegWrapper, "_safe_run")
    def test_run_command_gpu_no_gpu(self, mock_safe_run, mock_get_gpu):
        """Test run_command_gpu falls back to CPU when GPU unavailable"""
        mock_helper = Mock()
        mock_get_gpu.return_value = mock_helper
        mock_helper.is_available.return_value = False

        mock_safe_run.return_value.__enter__.return_value = "output"

        FFmpegWrapper.run_command_gpu(["-i", "input.mp4"])
        mock_safe_run.assert_called_once()

    @patch("video_detect.utils.ffmpeg_wrapper.get_gpu_helper")
    @patch.object(FFmpegWrapper, "_safe_run")
    def test_run_command_gpu_with_gpu(self, mock_safe_run, mock_get_gpu):
        """Test run_command_gpu with GPU available"""
        mock_helper = Mock()
        mock_get_gpu.return_value = mock_helper
        mock_helper.is_available.return_value = True
        mock_helper.get_ffmpeg_gpu_params.return_value = ["-hwaccel", "cuda"]

        mock_safe_run.return_value.__enter__.return_value = "output"

        FFmpegWrapper.run_command_gpu(["-i", "input.mp4"])
        # Verify GPU params were included
        call_args = mock_safe_run.call_args[0][0]
        assert "-hwaccel" in call_args
