"""Image transformation for fingerprint removal"""

from pathlib import Path
import random
from PIL import Image, ImageEnhance, ImageOps
try:
    import numpy as np
except ImportError:
    np = None

from utils import ensure_dir


class ImageTransformer:
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
    
    def transform(self, input_path: Path, output_path: Path) -> bool:
        try:
            img = Image.open(input_path)
            
            img = self._apply_color_shift(img)
            img = self._apply_slight_crop(img)
            img = self._apply_noise(img)
            img = self._apply_compression_artifact(img)
            
            ensure_dir(output_path.parent)
            img.save(output_path, quality=random.randint(85, 95))
            return True
        except Exception:
            return False
    
    def _apply_color_shift(self, img: Image.Image) -> Image.Image:
        enhancer = ImageEnhance.Color(img)
        factor = 1.0 + random.uniform(-0.05, 0.05)
        return enhancer.enhance(factor)
    
    def _apply_slight_crop(self, img: Image.Image) -> Image.Image:
        crop_pct = random.randint(1, 3)
        w, h = img.size
        
        left = w * crop_pct // 100
        top = h * crop_pct // 100
        right = w - left
        bottom = h - top
        
        cropped = img.crop((left, top, right, bottom))
        return cropped.resize((w, h), Image.Resampling.LANCZOS)
    
    def _apply_noise(self, img: Image.Image) -> Image.Image:
        if random.random() > 0.7:
            return img
        
        if np is None:
            return img
        
        img_array = np.array(img).astype(float)
        noise = np.random.normal(0, 2, img_array.shape)
        noisy = img_array + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy)
    
    def _apply_compression_artifact(self, img: Image.Image) -> Image.Image:
        return img
