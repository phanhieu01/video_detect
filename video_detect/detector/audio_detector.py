"""Audio watermark detector using spectrogram analysis"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)


class AudioDetector:

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.sample_rate = self.config.get("audio_sample_rate", 22050)
        self.n_fft = self.config.get("audio_n_fft", 2048)
        self.hop_length = self.config.get("audio_hop_length", 512)

    def detect_watermark(
        self,
        video_path: Path,
    ) -> Dict[str, Any]:
        """Detect audio watermark in video file"""
        if not LIBROSA_AVAILABLE:
            return {"error": "Librosa not available - install with: pip install librosa"}

        # Extract audio from video using FFmpeg
        audio_path = self._extract_audio(video_path)
        if audio_path is None:
            return {"error": "Cannot extract audio from video"}

        try:
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate)

            result = {
                "video_path": str(video_path),
                "audio_duration": len(y) / sr,
                "sample_rate": sr,
                "potential_watermarks": [],
            }

            # Analyze spectrogram for patterns
            spectrogram_analysis = self._analyze_spectrogram(y, sr)
            result["spectrogram_analysis"] = spectrogram_analysis

            if spectrogram_analysis.get("suspicious"):
                result["potential_watermarks"].append({
                    "type": "spectrogram_pattern",
                    "severity": spectrogram_analysis.get("severity", "medium"),
                    "description": "Suspicious pattern detected in audio spectrogram",
                })

            # Analyze for repetitive patterns
            pattern_analysis = self._analyze_repetitive_patterns(y, sr)
            result["pattern_analysis"] = pattern_analysis

            if pattern_analysis.get("suspicious"):
                result["potential_watermarks"].append({
                    "type": "repetitive_pattern",
                    "severity": pattern_analysis.get("severity", "medium"),
                    "description": "Repetitive pattern detected in audio",
                })

            # Analyze frequency domain
            frequency_analysis = self._analyze_frequency_domain(y, sr)
            result["frequency_analysis"] = frequency_analysis

            if frequency_analysis.get("suspicious"):
                result["potential_watermarks"].append({
                    "type": "frequency_anomaly",
                    "severity": frequency_analysis.get("severity", "medium"),
                    "description": "Anomalous frequency pattern detected",
                })

            return result

        finally:
            # Cleanup temporary audio file
            if audio_path and audio_path.exists():
                audio_path.unlink()

    def _extract_audio(self, video_path: Path) -> Optional[Path]:
        """Extract audio from video using FFmpeg"""
        import subprocess
        import tempfile

        output_path = Path(tempfile.mktemp(suffix=".wav"))

        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(self.sample_rate),
                "-ac", "1",
                str(output_path),
                "-loglevel", "error",
            ], check=True, capture_output=True)
            return output_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _analyze_spectrogram(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze audio spectrogram for suspicious patterns"""
        # Compute spectrogram
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)

        # Look for consistent high-energy regions
        mean_magnitude = np.mean(magnitude, axis=1)

        # Check for unusual peaks
        threshold = np.mean(mean_magnitude) + 2 * np.std(mean_magnitude)
        peaks = mean_magnitude > threshold

        peak_ratio = np.sum(peaks) / len(peaks)

        if peak_ratio > 0.1:  # More than 10% of frequency bins have peaks
            return {
                "suspicious": True,
                "severity": "high",
                "peak_ratio": float(peak_ratio),
                "reason": "Unusual frequency peaks detected",
            }
        elif peak_ratio > 0.05:
            return {
                "suspicious": True,
                "severity": "medium",
                "peak_ratio": float(peak_ratio),
                "reason": "Some frequency peaks detected",
            }

        return {"suspicious": False, "peak_ratio": float(peak_ratio)}

    def _analyze_repetitive_patterns(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze audio for repetitive patterns that might indicate watermarks"""
        # Compute autocorrelation
        autocorr = np.correlate(y, y, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        # Normalize
        autocorr = autocorr / (autocorr.max() + 1e-10)

        # Look for periodic peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(autocorr, height=0.3, distance=1000)

        if len(peaks) > 5:
            return {
                "suspicious": True,
                "severity": "high",
                "peak_count": len(peaks),
                "reason": "Strong repetitive patterns detected",
            }
        elif len(peaks) > 2:
            return {
                "suspicious": True,
                "severity": "medium",
                "peak_count": len(peaks),
                "reason": "Some repetitive patterns detected",
            }

        return {"suspicious": False, "peak_count": len(peaks)}

    def _analyze_frequency_domain(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze frequency domain for anomalies"""
        # Compute FFT
        fft = np.fft.fft(y)
        magnitude = np.abs(fft[:len(fft)//2])

        # Look for unusual frequency patterns
        freq_bins = 20
        band_energies = []
        for i in range(freq_bins):
            start = i * len(magnitude) // freq_bins
            end = (i + 1) * len(magnitude) // freq_bins
            band_energies.append(np.mean(magnitude[start:end]))

        # Check for significant variations
        energy_variation = np.std(band_energies) / (np.mean(band_energies) + 1e-10)

        if energy_variation > 2.0:
            return {
                "suspicious": True,
                "severity": "high",
                "energy_variation": float(energy_variation),
                "reason": "High frequency energy variation",
            }
        elif energy_variation > 1.0:
            return {
                "suspicious": True,
                "severity": "medium",
                "energy_variation": float(energy_variation),
                "reason": "Moderate frequency energy variation",
            }

        return {"suspicious": False, "energy_variation": float(energy_variation)}

    def has_audio_track(self, video_path: Path) -> bool:
        """Check if video has audio track"""
        import subprocess
        import json

        try:
            result = subprocess.run([
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "json",
                str(video_path),
            ], capture_output=True, text=True, check=True)

            data = json.loads(result.stdout)
            return "streams" in data and len(data["streams"]) > 0

        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            return False
