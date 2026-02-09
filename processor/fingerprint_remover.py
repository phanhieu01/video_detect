"""Fingerprint remover - re-encode and modify video/image signature"""

from pathlib import Path
import random
from datetime import datetime, timedelta
import tempfile
import shutil
import logging
from utils.ffmpeg_wrapper import FFmpegWrapper

try:
    from PIL import Image, ImageEnhance
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class FingerprintRemover:

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.re_encode = self.config.get("re_encode", True)
        self.change_bitrate = self.config.get("change_bitrate", True)
        self.change_resolution = self.config.get("change_resolution", True)
        self.change_colorspace = self.config.get("change_colorspace", True)
        self.add_noise = self.config.get("add_noise", True)

    def remove_fingerprint(
        self,
        input_path: Path,
        output_path: Path,
    ) -> bool:
        # Detect file type và xử lý phù hợp
        if self._is_image(input_path):
            return self._remove_image_fingerprint(input_path, output_path)
        else:
            return self._remove_video_fingerprint(input_path, output_path)

    def _is_image(self, input_path: Path) -> bool:
        """Check if file is image using imghdr or PIL"""
        import imghdr

        # First try imghdr (built-in)
        img_type = imghdr.what(input_path)
        if img_type:
            return True

        # Fallback: check by extension
        image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]
        if input_path.suffix.lower() in image_extensions:
            return True

        # Final fallback: try opening with PIL
        if PIL_AVAILABLE:
            try:
                with Image.open(input_path):
                    return True
            except Exception:
                pass

        return False

    def _remove_image_fingerprint(
        self,
        input_path: Path,
        output_path: Path,
    ) -> bool:
        """Remove fingerprint from image using PIL"""
        if not PIL_AVAILABLE:
            logger.warning("PIL not available, skipping image fingerprint removal")
            # Just copy the file
            shutil.copy(input_path, output_path)
            return True

        try:
            img = Image.open(input_path)

            # Apply các transform để xóa fingerprint
            if self.change_resolution:
                img = self._resize_image(img)

            if self.change_colorspace:
                img = self._change_image_colorspace(img)

            if self.add_noise:
                img = self._add_image_noise(img)

            # Save với quality thấp hơn để xóa fingerprint
            quality = random.randint(85, 95)
            img.save(output_path, quality=quality)
            return True
        except Exception as e:
            logger.error(f"Image fingerprint removal failed: {e}", exc_info=True)
            return False

    def _remove_video_fingerprint(
        self,
        input_path: Path,
        output_path: Path,
    ) -> bool:
        """Original video fingerprint removal logic"""
        temp_files = []

        try:
            temp_path = input_path

            if self.re_encode:
                new_temp = Path(tempfile.mktemp(suffix=".mp4"))
                temp_files.append(new_temp)
                temp_path = self._apply_re_encode(temp_path, new_temp)

            if self.change_bitrate:
                new_temp = Path(tempfile.mktemp(suffix=".mp4"))
                temp_files.append(new_temp)
                temp_path = self._apply_bitrate_change(temp_path, new_temp)

            if self.change_resolution:
                new_temp = Path(tempfile.mktemp(suffix=".mp4"))
                temp_files.append(new_temp)
                temp_path = self._apply_resolution_change(temp_path, new_temp)

            if self.change_colorspace:
                new_temp = Path(tempfile.mktemp(suffix=".mp4"))
                temp_files.append(new_temp)
                temp_path = self._apply_colorspace_change(temp_path, new_temp)

            # Copy final result to output
            if temp_path != input_path:
                shutil.copy(temp_path, output_path)

            return True
        except Exception as e:
            logger.error(f"Video fingerprint removal failed: {e}", exc_info=True)
            return False
        finally:
            # Cleanup all temp files
            for f in temp_files:
                if f.exists():
                    try:
                        f.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {f}: {e}")
    
    def _apply_re_encode(self, input_path: Path, output_path: Path) -> Path:
        codec = random.choice(["libx264", "libx265", "libvpx-vp9"])
        crf = random.randint(20, 26)
        FFmpegWrapper.re_encode(input_path, output_path, codec=codec, crf=crf)
        return output_path
    
    def _apply_bitrate_change(self, input_path: Path, output_path: Path) -> Path:
        bitrate = random.choice(["3000k", "3500k", "4000k", "4500k", "5000k"])
        FFmpegWrapper.re_encode(input_path, output_path, bitrate=bitrate)
        return output_path
    
    def _apply_resolution_change(self, input_path: Path, output_path: Path) -> Path:
        scale_factors = ["iw-2:ih-2", "iw-4:ih-4", "iw+2:ih+2", "iw-6:ih-6"]
        scale = random.choice(scale_factors)

        try:
            FFmpegWrapper.run_command([
                "-i", str(input_path),
                "-vf", f"scale={scale}",
                "-c:v", "libx264",  # Specify codec for video encoding
                "-map", "0:v",     # Map video stream
                "-map", "0:a?",    # Map audio stream nếu có (? = optional)
                "-c:a", "copy",
                str(output_path),
            ])
        except Exception:
            FFmpegWrapper.re_encode(input_path, output_path)

        return output_path
    
    def _apply_colorspace_change(self, input_path: Path, output_path: Path) -> Path:
        try:
            FFmpegWrapper.run_command([
                "-i", str(input_path),
                "-vf", "eq=contrast=1.02:brightness=0.02:saturation=1.05",
                "-c:v", "libx264",  # Specify codec for video encoding
                "-map", "0:v",     # Map video stream
                "-map", "0:a?",    # Map audio stream nếu có (? = optional)
                "-c:a", "copy",
                str(output_path),
            ])
        except Exception:
            FFmpegWrapper.re_encode(input_path, output_path)

        return output_path

    # ===== Image processing methods =====

    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Resize image slightly to change fingerprint"""
        w, h = img.size

        # Resize factors: +/- 1-3%
        factors = [0.97, 0.98, 0.99, 1.01, 1.02, 1.03]
        factor = random.choice(factors)

        new_w = int(w * factor)
        new_h = int(h * factor)

        # Resize and then resize back to original size
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return resized.resize((w, h), Image.Resampling.LANCZOS)

    def _change_image_colorspace(self, img: Image.Image) -> Image.Image:
        """Change image colorspace slightly"""
        # Random enhancement
        enhancer = ImageEnhance.Color(img)
        factor = random.uniform(0.95, 1.05)
        img = enhancer.enhance(factor)

        # Brightness
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.98, 1.02)
        img = enhancer.enhance(factor)

        # Contrast
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.98, 1.02)
        img = enhancer.enhance(factor)

        return img

    def _add_image_noise(self, img: Image.Image) -> Image.Image:
        """Add slight noise to image"""
        if not PIL_AVAILABLE or np is None:
            return img

        # Only add noise 70% of the time
        if random.random() > 0.7:
            return img

        img_array = np.array(img).astype(float)
        noise = np.random.normal(0, 2, img_array.shape)  # Small noise
        noisy = img_array + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy)
