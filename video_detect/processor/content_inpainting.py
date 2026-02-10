"""Content-aware inpainting for better watermark removal"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class ContentAwareInpainter:

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.inpainting_method = self.config.get("inpainting_method", "telea")
        self.patch_size = self.config.get("patch_size", 3)

    def remove_watermark(
        self,
        image_path: Path,
        output_path: Path,
        watermark_region: Dict[str, int],
    ) -> bool:
        """Remove watermark using content-aware inpainting"""
        if not CV2_AVAILABLE:
            logger.error("OpenCV not available")
            return False

        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Cannot read image: {image_path}")
            return False

        # Create mask for watermark region
        mask = self._create_mask(img.shape, watermark_region)

        # Apply content-aware inpainting
        if self.inpainting_method == "telea":
            result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        elif self.inpainting_method == "ns":
            result = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
        else:
            # Use custom patch-based inpainting
            result = self._patch_based_inpaint(img, mask)

        cv2.imwrite(str(output_path), result)
        return True

    def _create_mask(self, shape: Tuple, region: Dict[str, int]) -> np.ndarray:
        """Create binary mask for watermark region"""
        mask = np.zeros(shape[:2], dtype=np.uint8)

        x, y, w, h = region["x"], region["y"], region["width"], region["height"]

        # Add some padding to ensure full coverage
        padding = 5
        y_start = max(0, y - padding)
        y_end = min(shape[0], y + h + padding)
        x_start = max(0, x - padding)
        x_end = min(shape[1], x + w + padding)

        mask[y_start:y_end, x_start:x_end] = 255

        return mask

    def _patch_based_inpaint(
        self,
        img: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Custom patch-based inpainting for better results"""
        result = img.copy()

        # Get masked region
        masked_pixels = mask > 0

        # Find bounding box of masked region
        coords = np.where(masked_pixels)
        if len(coords[0]) == 0:
            return result

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        # Expand region slightly
        y_min = max(0, y_min - 10)
        y_max = min(img.shape[0], y_max + 10)
        x_min = max(0, x_min - 10)
        x_max = min(img.shape[1], x_max + 10)

        # Extract region
        region = img[y_min:y_max, x_min:x_max]
        region_mask = mask[y_min:y_max, x_min:x_max]

        # Find best matching patches
        patch_size = self.patch_size
        for y in range(region.shape[0]):
            for x in range(region.shape[1]):
                if region_mask[y, x] > 0:
                    # Find similar patch from surrounding area
                    patch = self._find_best_patch(region, region_mask, y, x, patch_size)
                    if patch is not None:
                        result[y_min + y, x_min + x] = patch

        return result

    def _find_best_patch(
        self,
        region: np.ndarray,
        mask: np.ndarray,
        y: int,
        x: int,
        patch_size: int,
    ) -> Optional[np.ndarray]:
        """Find best matching patch from unmasked region"""
        h, w = region.shape[:2]
        half_size = patch_size // 2

        # Get target patch (with masked center)
        y_start = max(0, y - half_size)
        y_end = min(h, y + half_size + 1)
        x_start = max(0, x - half_size)
        x_end = min(w, x + half_size + 1)

        target_patch = region[y_start:y_end, x_start:x_end]
        target_mask = mask[y_start:y_end, x_start:x_end]

        # Find candidate patches from unmasked area
        best_patch = None
        best_score = float('inf')

        # Sample candidate patches
        for _ in range(20):  # Try 20 random patches
            # Random position in region
            cy = np.random.randint(h)
            cx = np.random.randint(w)

            cy_start = max(0, cy - half_size)
            cy_end = min(h, cy + half_size + 1)
            cx_start = max(0, cx - half_size)
            cx_end = min(w, cx + half_size + 1)

            candidate_patch = region[cy_start:cy_end, cx_start:cx_end]
            candidate_mask = mask[cy_start:cy_end, cx_start:cx_end]

            # Skip if mostly masked
            if np.mean(candidate_mask) > 0.3:
                continue

            # Calculate similarity
            score = self._patch_similarity(target_patch, candidate_patch, target_mask)

            if score < best_score:
                best_score = score
                best_patch = candidate_patch

        return best_patch

    def _patch_similarity(
        self,
        patch1: np.ndarray,
        patch2: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Calculate similarity between patches, ignoring masked areas"""
        # Create valid mask (where both patches have data)
        valid_mask = (mask == 0) & (patch1.shape == patch2.shape[:2])

        if not np.any(valid_mask):
            return float('inf')

        # Calculate SSD only on valid pixels
        diff = patch1.astype(float) - patch2.astype(float)

        # Apply mask to each channel
        if len(diff.shape) == 3:
            valid_mask_3d = np.stack([valid_mask] * 3, axis=2)
            diff = diff * valid_mask_3d
        else:
            diff = diff * valid_mask

        ssd = np.sum(diff ** 2)
        return ssd

    def apply_blending(
        self,
        original: np.ndarray,
        inpainted: np.ndarray,
        watermark_region: Dict[str, int],
    ) -> np.ndarray:
        """Apply blending at the boundaries for smoother results"""
        result = inpainted.copy()

        x, y, w, h = watermark_region["x"], watermark_region["y"], watermark_region["width"], watermark_region["height"]

        # Feather the edges
        feather_size = 5

        # Top edge
        for i in range(feather_size):
            alpha = i / feather_size
            y_pos = max(0, y - feather_size + i)
            if y_pos >= 0:
                result[y_pos, x:x+w] = (
                    alpha * inpainted[y_pos, x:x+w] +
                    (1 - alpha) * original[y_pos, x:x+w]
                )

        # Bottom edge
        for i in range(feather_size):
            alpha = 1 - i / feather_size
            y_pos = min(original.shape[0] - 1, y + h + i)
            if y_pos < original.shape[0]:
                result[y_pos, x:x+w] = (
                    alpha * inpainted[y_pos, x:x+w] +
                    (1 - alpha) * original[y_pos, x:x+w]
                )

        return result
