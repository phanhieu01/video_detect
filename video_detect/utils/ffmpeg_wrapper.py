"""FFmpeg wrapper for video processing"""

import subprocess
from pathlib import Path
from typing import List, Optional


class FFmpegWrapper:
    
    @staticmethod
    def get_version() -> str:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.split("\n")[0]
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "FFmpeg not found"
    
    @staticmethod
    def run_command(args: List[str], input_file: Optional[Path] = None) -> str:
        cmd = ["ffmpeg", "-y"]
        cmd.extend(args)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
        
        return result.stdout

    @staticmethod
    def has_audio_stream(input_path: Path) -> bool:
        """Check if video has audio stream using ffprobe"""
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(input_path)
            ], capture_output=True, text=True, check=True)
            return bool(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @staticmethod
    def strip_metadata(input_path: Path, output_path: Path) -> None:
        FFmpegWrapper.run_command([
            "-i", str(input_path),
            "-map_metadata", "-1",
            "-map", "0:v",    # Map video stream
            "-map", "0:a?",   # Map audio stream if exists (? = optional)
            "-c:v", "copy",
            "-c:a", "copy",
            str(output_path),
        ])
    
    @staticmethod
    def re_encode(
        input_path: Path,
        output_path: Path,
        codec: str = "libx264",
        bitrate: Optional[str] = None,
        crf: int = 23,
    ) -> None:
        args = [
            "-i", str(input_path),
            "-map", "0:v",    # Map video stream
            "-map", "0:a?",   # Map audio stream if exists (? = optional)
            "-c:v", codec,
            "-crf", str(crf)
        ]

        if bitrate:
            args.extend(["-b:v", bitrate])

        args.extend(["-c:a", "aac", "-b:a", "192k", str(output_path)])
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
        filter_str = f"delogo=x={x}:y={y}:w={width}:h={height}:show=0"
        FFmpegWrapper.run_command([
            "-i", str(input_path),
            "-vf", filter_str,
            "-map", "0:v",    # Map video stream
            "-map", "0:a?",   # Map audio stream if exists (? = optional)
            "-c:a", "copy",
            str(output_path),
        ])
    
    @staticmethod
    def change_speed(
        input_path: Path,
        output_path: Path,
        factor: float,
    ) -> None:
        has_audio = FFmpegWrapper.has_audio_stream(input_path)

        if has_audio:
            # Video with audio - apply filter to both video and audio
            atempo_filter = FFmpegWrapper._build_atempo_filter(factor)
            FFmpegWrapper.run_command([
                "-i", str(input_path),
                "-filter:v", f"setpts={1/factor}*PTS",
                "-filter:a", atempo_filter,
                str(output_path),
            ])
        else:
            # Video without audio - only apply filter to video
            FFmpegWrapper.run_command([
                "-i", str(input_path),
                "-filter:v", f"setpts={1/factor}*PTS",
                str(output_path),
            ])

    @staticmethod
    def _build_atempo_filter(factor: float) -> str:
        """Build atempo filter chain (atempo max is 2.0, need to chain for higher values)"""
        if factor <= 2.0:
            return f"atempo={factor}"

        # Chain multiple atempo filters for factors > 2.0
        filters = []
        remaining = factor
        while remaining > 1.0:
            filters.append(f"atempo={min(remaining, 2.0)}")
            remaining /= 2.0

        return ",".join(filters)
    
    @staticmethod
    def horizontal_flip(input_path: Path, output_path: Path) -> None:
        FFmpegWrapper.run_command([
            "-i", str(input_path),
            "-vf", "hflip",
            "-c:v", "libx264",  # Specify codec for video encoding
            "-map", "0:v",    # Map video stream
            "-map", "0:a?",   # Map audio stream if exists (? = optional)
            "-c:a", "copy",
            str(output_path),
        ])
    
    @staticmethod
    def crop(
        input_path: Path,
        output_path: Path,
        crop_percent: int,
    ) -> None:
        filter_str = f"crop=iw*{(100-crop_percent)/100}:ih*{(100-crop_percent)/100}"
        FFmpegWrapper.run_command([
            "-i", str(input_path),
            "-vf", filter_str,
            "-c:v", "libx264",  # Specify codec for video encoding
            "-map", "0:v",    # Map video stream
            "-map", "0:a?",   # Map audio stream if exists (? = optional)
            "-c:a", "copy",
            str(output_path),
        ])
    
    @staticmethod
    def color_adjust(
        input_path: Path,
        output_path: Path,
        brightness: float = 0,
        contrast: float = 1,
        saturation: float = 1,
    ) -> None:
        filter_str = f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}"
        FFmpegWrapper.run_command([
            "-i", str(input_path),
            "-vf", filter_str,
            "-c:v", "libx264",  # Specify codec for video encoding
            "-map", "0:v",    # Map video stream
            "-map", "0:a?",   # Map audio stream if exists (? = optional)
            "-c:a", "copy",
            str(output_path),
        ])
    
    @staticmethod
    def add_silence(
        input_path: Path,
        output_path: Path,
        duration: float,
    ) -> None:
        has_audio = FFmpegWrapper.has_audio_stream(input_path)

        if has_audio:
            # Video with audio - add silence to audio stream
            filter_str = f"[0:a]apad=pad_dur={duration}[aout]"
            FFmpegWrapper.run_command([
                "-i", str(input_path),
                "-filter_complex", filter_str,
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                str(output_path),
            ])
        else:
            # Video without audio - only copy video (do nothing)
            FFmpegWrapper.run_command([
                "-i", str(input_path),
                "-c:v", "copy",
                str(output_path),
            ])
