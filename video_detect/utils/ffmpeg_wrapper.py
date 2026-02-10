"""FFmpeg wrapper for video processing

This module provides a secure wrapper around FFmpeg/FFprobe commands with:
- Input validation and sanitization
- No shell execution (uses subprocess with list arguments)
- Path validation to prevent directory traversal
- Comprehensive error handling and logging
"""

import subprocess
import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from .gpu_helper import get_gpu_helper

logger = logging.getLogger(__name__)

# Security: Define allowed characters for FFmpeg filter arguments
# This prevents injection of malicious shell commands
ALLOWED_FILTER_CHARS = re.compile(r'^[a-zA-Z0-9_=:\-\[\]\.,\*\/\+\s\{\}\(\)\%]+$')

# Allowed codecs for validation
ALLOWED_VIDEO_CODECS = {
    "libx264", "libx265", "libvpx", "libvpx-vp9", "h264_nvenc",
    "hevc_nvenc", "mpeg4", "copy"
}

ALLOWED_AUDIO_CODECS = {
    "aac", "libmp3lame", "libopus", "libvorbis", "copy"
}

# Numeric limits for validation
MAX_DIMENSION = 7680  # 8K resolution
MIN_DIMENSION = 1
MAX_DURATION = 86400  # 24 hours in seconds
MAX_BITRATE = 1000000  # 1 Gbps in kbps
MIN_SPEED_FACTOR = 0.25
MAX_SPEED_FACTOR = 4.0


def validate_path(path: Path) -> Path:
    """Validate path to prevent directory traversal attacks"""
    try:
        # Resolve to absolute path
        resolved = path.resolve()

        # Check for suspicious components
        if ".." in str(path):
            raise ValueError(f"Path traversal detected: {path}")

        return resolved
    except Exception as e:
        logger.error(f"Path validation failed for {path}: {e}")
        raise ValueError(f"Invalid path: {path}")


def validate_numeric(value: int, min_val: int, max_val: int, name: str) -> int:
    """Validate numeric parameter is within safe bounds"""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(value)}")

    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")

    return int(value)


def validate_codec(codec: str, allowed: set, name: str) -> str:
    """Validate codec is in allowed list"""
    if codec not in allowed:
        raise ValueError(f"Invalid {name}: {codec}. Allowed: {allowed}")
    return codec


def validate_filter_string(filter_str: str) -> str:
    """Validate filter string doesn't contain dangerous characters"""
    if not filter_str:
        raise ValueError("Filter string cannot be empty")

    # Check for suspicious patterns
    dangerous_patterns = [";", "&", "|", "$", "`", "\n", "\r", "\x00"]
    for pattern in dangerous_patterns:
        if pattern in filter_str:
            raise ValueError(f"Dangerous character in filter: {repr(pattern)}")

    # Validate characters using regex
    if not ALLOWED_FILTER_CHARS.match(filter_str):
        raise ValueError(f"Invalid characters in filter: {filter_str}")

    return filter_str


class FFmpegWrapper:

    @staticmethod
    def get_version() -> str:
        """Get FFmpeg version safely"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,  # Add timeout
            )
            if result.returncode == 0:
                return result.stdout.split("\n")[0]
            return "FFmpeg not found"
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Failed to get FFmpeg version: {e}")
            return "FFmpeg not found"

    @staticmethod
    @contextmanager
    def _safe_run(cmd: List[str], timeout: int = 300):
        """Context manager for safe subprocess execution"""
        logger.debug(f"Running command: {' '.join(cmd[:5])}...")  # Log first 5 args only

        proc = None
        try:
            # Never use shell=True - always use list argument form
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = proc.communicate(timeout=timeout)

            if proc.returncode != 0:
                error_msg = stderr.strip() if stderr else "Unknown error"
                logger.error(f"FFmpeg failed (code {proc.returncode}): {error_msg}")
                raise RuntimeError(f"FFmpeg error: {error_msg}")

            yield stdout

        except subprocess.TimeoutExpired:
            if proc:
                proc.kill()
                proc.communicate()
            logger.error(f"FFmpeg command timed out after {timeout} seconds")
            raise RuntimeError(f"FFmpeg command timed out")
        except FileNotFoundError:
            logger.error("FFmpeg executable not found")
            raise RuntimeError("FFmpeg is not installed or not in PATH")
        except Exception as e:
            logger.error(f"Unexpected error running FFmpeg: {e}")
            raise
        finally:
            if proc:
                proc.kill()
    
    @staticmethod
    def run_command(args: List[str], input_file: Optional[Path] = None) -> str:
        """
        Run FFmpeg command with validated arguments.

        Security: All arguments are passed as a list to prevent shell injection.
        """
        # Build command safely
        cmd = ["ffmpeg", "-y"]
        cmd.extend(args)

        # Validate paths in command
        for i, arg in enumerate(cmd):
            if i > 0 and cmd[i-1] in ["-i", "-c:v", "-c:a"]:
                # Validate input files and codecs
                if arg.endswith((".mp4", ".mkv", ".mov", ".avi", ".jpg", ".png", ".mp3")):
                    try:
                        validate_path(Path(arg))
                    except ValueError as e:
                        raise ValueError(f"Invalid file path in command: {e}")

        with FFmpegWrapper._safe_run(cmd, timeout=300) as output:
            return output

    @staticmethod
    def has_audio_stream(input_path: Path) -> bool:
        """Check if video has audio stream using ffprobe"""
        try:
            # Validate input path
            validated_path = validate_path(input_path)

            result = subprocess.run([
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(validated_path)
            ], capture_output=True, text=True, check=False, timeout=30)

            return result.returncode == 0 and bool(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Failed to check audio stream: {e}")
            return False
        except ValueError as e:
            logger.error(f"Invalid path in has_audio_stream: {e}")
            return False

    @staticmethod
    def strip_metadata(input_path: Path, output_path: Path) -> None:
        """Remove metadata from media file"""
        # Validate paths
        validated_input = validate_path(input_path)
        validated_output = validate_path(output_path)

        FFmpegWrapper.run_command([
            "-i", str(validated_input),
            "-map_metadata", "-1",
            "-map", "0:v",    # Map video stream
            "-map", "0:a?",   # Map audio stream if exists (? = optional)
            "-c:v", "copy",
            "-c:a", "copy",
            str(validated_output),
        ])
    
    @staticmethod
    def re_encode(
        input_path: Path,
        output_path: Path,
        codec: str = "libx264",
        bitrate: Optional[str] = None,
        crf: int = 23,
    ) -> None:
        """Re-encode video with specified codec and quality settings"""
        # Validate inputs
        validated_input = validate_path(input_path)
        validated_output = validate_path(output_path)
        validated_codec = validate_codec(codec, ALLOWED_VIDEO_CODECS, "video codec")
        validated_crf = validate_numeric(crf, 0, 51, "CRF")

        args = [
            "-i", str(validated_input),
            "-map", "0:v",    # Map video stream
            "-map", "0:a?",   # Map audio stream if exists (? = optional)
            "-c:v", validated_codec,
            "-crf", str(validated_crf)
        ]

        if bitrate:
            # Validate bitrate format (e.g., "5M", "2500k")
            if not re.match(r'^\d+[kKmMgG]?$', bitrate.strip()):
                raise ValueError(f"Invalid bitrate format: {bitrate}")
            args.extend(["-b:v", bitrate])

        args.extend(["-c:a", "aac", "-b:a", "192k", str(validated_output)])
        FFmpegWrapper.run_command(args)

    @staticmethod
    def delogo(
        input_path: Path,
        output_path: Path,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        """Remove logo from video using delogo filter"""
        # Validate inputs
        validated_input = validate_path(input_path)
        validated_output = validate_path(output_path)
        validated_x = validate_numeric(x, 0, MAX_DIMENSION, "x")
        validated_y = validate_numeric(y, 0, MAX_DIMENSION, "y")
        validated_width = validate_numeric(width, MIN_DIMENSION, MAX_DIMENSION, "width")
        validated_height = validate_numeric(height, MIN_DIMENSION, MAX_DIMENSION, "height")

        # Build and validate filter string
        filter_str = f"delogo=x={validated_x}:y={validated_y}:w={validated_width}:h={validated_height}:show=0"
        validate_filter_string(filter_str)

        FFmpegWrapper.run_command([
            "-i", str(validated_input),
            "-vf", filter_str,
            "-map", "0:v",    # Map video stream
            "-map", "0:a?",   # Map audio stream if exists (? = optional)
            "-c:a", "copy",
            str(validated_output),
        ])
    
    @staticmethod
    def change_speed(
        input_path: Path,
        output_path: Path,
        factor: float,
    ) -> None:
        """Change video playback speed"""
        # Validate inputs
        validated_input = validate_path(input_path)
        validated_output = validate_path(output_path)
        validated_factor = validate_numeric(factor, MIN_SPEED_FACTOR, MAX_SPEED_FACTOR, "speed factor")

        has_audio = FFmpegWrapper.has_audio_stream(validated_input)

        if has_audio:
            # Video with audio - apply filter to both video and audio
            atempo_filter = FFmpegWrapper._build_atempo_filter(validated_factor)
            validate_filter_string(atempo_filter)

            # Validate setpts filter
            setpts_filter = f"setpts={1/validated_factor}*PTS"
            validate_filter_string(setpts_filter)

            FFmpegWrapper.run_command([
                "-i", str(validated_input),
                "-filter:v", setpts_filter,
                "-filter:a", atempo_filter,
                str(validated_output),
            ])
        else:
            # Video without audio - only apply filter to video
            setpts_filter = f"setpts={1/validated_factor}*PTS"
            validate_filter_string(setpts_filter)

            FFmpegWrapper.run_command([
                "-i", str(validated_input),
                "-filter:v", setpts_filter,
                str(validated_output),
            ])

    @staticmethod
    def _build_atempo_filter(factor: float) -> str:
        """
        Build atempo filter chain (atempo max is 2.0, need to chain for higher values)

        Security: Returns validated filter string
        """
        if factor <= 2.0:
            return f"atempo={factor:.2f}"

        # Chain multiple atempo filters for factors > 2.0
        filters = []
        remaining = factor
        while remaining > 1.0:
            tempo = min(remaining, 2.0)
            filters.append(f"atempo={tempo:.2f}")
            remaining /= 2.0

        return ",".join(filters)
    
    @staticmethod
    def horizontal_flip(input_path: Path, output_path: Path) -> None:
        """Flip video horizontally"""
        # Validate paths
        validated_input = validate_path(input_path)
        validated_output = validate_path(output_path)

        FFmpegWrapper.run_command([
            "-i", str(validated_input),
            "-vf", "hflip",
            "-c:v", "libx264",
            "-map", "0:v",
            "-map", "0:a?",
            "-c:a", "copy",
            str(validated_output),
        ])

    @staticmethod
    def crop(
        input_path: Path,
        output_path: Path,
        crop_percent: int,
    ) -> None:
        """Crop video by specified percentage"""
        # Validate inputs
        validated_input = validate_path(input_path)
        validated_output = validate_path(output_path)
        validated_percent = validate_numeric(crop_percent, 0, 90, "crop percentage")

        # Calculate crop values
        crop_value = (100 - validated_percent) / 100
        filter_str = f"crop=iw*{crop_value}:ih*{crop_value}"
        validate_filter_string(filter_str)

        FFmpegWrapper.run_command([
            "-i", str(validated_input),
            "-vf", filter_str,
            "-c:v", "libx264",
            "-map", "0:v",
            "-map", "0:a?",
            "-c:a", "copy",
            str(validated_output),
        ])

    @staticmethod
    def color_adjust(
        input_path: Path,
        output_path: Path,
        brightness: float = 0,
        contrast: float = 1,
        saturation: float = 1,
    ) -> None:
        """Adjust video brightness, contrast, and saturation"""
        # Validate inputs
        validated_input = validate_path(input_path)
        validated_output = validate_path(output_path)

        # Clamp values to valid ranges
        validated_brightness = max(-1.0, min(1.0, brightness))
        validated_contrast = max(0.0, min(2.0, contrast))
        validated_saturation = max(0.0, min(3.0, saturation))

        filter_str = f"eq=brightness={validated_brightness}:contrast={validated_contrast}:saturation={validated_saturation}"
        validate_filter_string(filter_str)

        FFmpegWrapper.run_command([
            "-i", str(validated_input),
            "-vf", filter_str,
            "-c:v", "libx264",
            "-map", "0:v",
            "-map", "0:a?",
            "-c:a", "copy",
            str(validated_output),
        ])
    
    @staticmethod
    def add_silence(
        input_path: Path,
        output_path: Path,
        duration: float,
    ) -> None:
        """Add silence to audio stream"""
        # Validate inputs
        validated_input = validate_path(input_path)
        validated_output = validate_path(output_path)
        validated_duration = validate_numeric(duration, 0, MAX_DURATION, "duration")

        has_audio = FFmpegWrapper.has_audio_stream(validated_input)

        if has_audio:
            # Video with audio - add silence to audio stream
            filter_str = f"[0:a]apad=pad_dur={validated_duration}[aout]"
            validate_filter_string(filter_str)

            FFmpegWrapper.run_command([
                "-i", str(validated_input),
                "-filter_complex", filter_str,
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                str(validated_output),
            ])
        else:
            # Video without audio - only copy video (do nothing)
            FFmpegWrapper.run_command([
                "-i", str(validated_input),
                "-c:v", "copy",
                str(validated_output),
            ])

    @staticmethod
    def get_gpu_hwaccel_params() -> List[str]:
        """Get GPU hardware acceleration parameters for FFmpeg"""
        gpu_helper = get_gpu_helper()
        return gpu_helper.get_ffmpeg_gpu_params()

    @staticmethod
    def run_command_gpu(args: List[str], input_file: Optional[Path] = None) -> str:
        """Run FFmpeg command with GPU acceleration if available"""
        gpu_helper = get_gpu_helper()

        if not gpu_helper.is_available():
            logger.debug("GPU not available, using CPU")
            return FFmpegWrapper.run_command(args, input_file)

        cmd = ["ffmpeg", "-y"]
        gpu_params = gpu_helper.get_ffmpeg_gpu_params()
        cmd.extend(gpu_params)
        cmd.extend(args)

        with FFmpegWrapper._safe_run(cmd, timeout=300) as output:
            return output

    @staticmethod
    def re_encode_gpu(
        input_path: Path,
        output_path: Path,
        codec: str = "h264_nvenc",  # NVIDIA GPU encoding
        bitrate: Optional[str] = None,
        crf: int = 23,
    ) -> None:
        """Re-encode video using GPU acceleration"""
        # Validate inputs
        validated_input = validate_path(input_path)
        validated_output = validate_path(output_path)

        gpu_helper = get_gpu_helper()

        if not gpu_helper.cuda_available:
            # Fallback to CPU encoding
            logger.info("CUDA not available, falling back to CPU encoding")
            return FFmpegWrapper.re_encode(validated_input, validated_output, "libx264", bitrate, crf)

        validated_crf = validate_numeric(crf, 0, 51, "CRF")

        args = ["-i", str(validated_input)]
        args.extend(gpu_helper.get_ffmpeg_gpu_params())

        args.extend([
            "-c:v", codec,
            "-crf", str(validated_crf),
            "-preset", "p4",  # Medium preset for NVENC
            "-map", "0:v",
            "-map", "0:a?",  # Map audio stream if exists
        ])

        if bitrate:
            if not re.match(r'^\d+[kKmMgG]?$', bitrate.strip()):
                raise ValueError(f"Invalid bitrate format: {bitrate}")
            args.extend(["-b:v", bitrate])

        args.extend(["-c:a", "aac", "-b:a", "192k", str(validated_output)])
        FFmpegWrapper.run_command_gpu(args)

    @staticmethod
    def check_gpu_info() -> Dict[str, Any]:
        """Check GPU availability and info"""
        gpu_helper = get_gpu_helper()
        return gpu_helper.get_info()
