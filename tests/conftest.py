"""Pytest configuration and shared fixtures for video_detect tests"""

import pytest
import logging
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np
from typing import Dict, Any

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing"""
    return {
        "watermark": {
            "enabled": True,
            "detection_method": "auto",
            "confidence_threshold": 0.7,
        },
        "fingerprint": {
            "enabled": True,
        },
        "output": {
            "output_dir": "./test_output",
        },
        "parallel": {
            "max_workers": 2,
            "mode": "thread",
        },
    }


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Temporary output directory for test outputs"""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sample_video_path(tmp_path: Path) -> Path:
    """Create a sample video file for testing"""
    video_path = tmp_path / "sample_video.mp4"
    # Create a minimal valid video file (1x1 pixel, 1 second)
    video_path.write_bytes(
        b'\x00\x00\x00\x20\x66\x74\x79\x70\x69\x73\x6f\x6d'
    )  # Minimal MP4 header
    return video_path


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    """Create a sample image file for testing"""
    image_path = tmp_path / "sample_image.jpg"
    # Create a minimal valid JPEG file
    image_path.write_bytes(
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        b'\xff\xdb\x00C\x00\x03\x02\x02\x03\x02\x02\x03\x03\x03\x03\x04\x03\x03'
        b'\x04\x05\x08\x05\x05\x04\x04\x05\n\x07\x07\x06\x08\x0c\n\x0c\x0c\x0b'
        b'\n\x0b\x0b\r\x0e\x12\x10\r\x0e\x11\x0e\x0b\x0b\x10\x16\x10\x11\x13\x14'
        b'\x15\x15\x15\x0c\x0f\x17\x18\x16\x14\x18\x12\x14\x15\x14\xff\xc0\x00'
        b'\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00'
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\xff\xc4\x00'
        b'\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\x9f\xff\xd9'
    )
    return image_path


@pytest.fixture
def mock_video_capture():
    """Mock cv2.VideoCapture for testing"""
    mock = Mock()
    mock.isOpened.return_value = True
    mock.get.return_value = 300  # 300 frames
    mock.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    return mock


@pytest.fixture
def sample_frame():
    """Create a sample video frame for testing"""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_watermark_region():
    """Sample watermark region for testing"""
    return {
        "x": 500,
        "y": 400,
        "width": 100,
        "height": 50,
    }


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run for testing FFmpeg commands"""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "FFmpeg version 5.0"
    mock_result.stderr = ""

    def mock_run(cmd, *args, **kwargs):
        return mock_result

    return mock_run


# Disable actual subprocess calls in tests
@pytest.fixture(autouse=True)
def disable_external_calls(monkeypatch):
    """Disable external subprocess calls during testing"""
    # Mock subprocess.run
    def mock_run(*args, **kwargs):
        mock = Mock()
        mock.returncode = 0
        mock.stdout = ""
        mock.stderr = ""
        return mock

    monkeypatch.setattr("subprocess.run", mock_run)


@pytest.fixture
def api_test_client():
    """Create FastAPI test client"""
    from fastapi.testclient import TestClient
    from video_detect.api import app

    return TestClient(app)
