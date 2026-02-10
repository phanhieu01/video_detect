"""Quality assessment - compare before/after to verify removal success"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class QualityAssessment:

    @staticmethod
    def assess_removal(
        original_path: Path,
        processed_path: Path,
        watermark_region: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Assess quality of watermark removal by comparing original and processed"""
        if not CV2_AVAILABLE:
            return {"error": "OpenCV not available"}

        original = cv2.imread(str(original_path))
        processed = cv2.imread(str(processed_path))

        if original is None or processed is None:
            return {"error": "Cannot read images"}

        # Ensure same size
        if original.shape != processed.shape:
            processed = cv2.resize(processed, (original.shape[1], original.shape[0]))

        result = {
            "original_size": original.shape,
            "processed_size": processed.shape,
            "metrics": {},
        }

        # Calculate various metrics
        result["metrics"]["mse"] = QualityAssessment._calculate_mse(original, processed)
        result["metrics"]["psnr"] = QualityAssessment._calculate_psnr(original, processed)
        result["metrics"]["ssim"] = QualityAssessment._calculate_ssim(original, processed)

        # Region-specific metrics if watermark region provided
        if watermark_region:
            result["region_metrics"] = QualityAssessment._assess_region(
                original, processed, watermark_region
            )

        # Overall assessment
        result["overall_quality"] = QualityAssessment._overall_assessment(result["metrics"])

        return result

    @staticmethod
    def _calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Mean Squared Error"""
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        return float(mse)

    @staticmethod
    def _calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return float(psnr)

    @staticmethod
    def _calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index (simplified)"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

        # Calculate mean
        mu1 = cv2.GaussianBlur(gray1.astype(float), (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2.astype(float), (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(gray1.astype(float) ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray2.astype(float) ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray1.astype(float) * gray2.astype(float), (11, 11), 1.5) - mu1_mu2

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        return float(np.mean(ssim_map))

    @staticmethod
    def _assess_region(
        img1: np.ndarray,
        img2: np.ndarray,
        region: Dict[str, int],
    ) -> Dict[str, float]:
        """Assess quality specifically in watermark region"""
        x, y, w, h = region["x"], region["y"], region["width"], region["height"]

        # Extract regions
        region1 = img1[y:y+h, x:x+w]
        region2 = img2[y:y+h, x:x+w]

        return {
            "mse": QualityAssessment._calculate_mse(region1, region2),
            "psnr": QualityAssessment._calculate_psnr(region1, region2),
        }

    @staticmethod
    def _overall_assessment(metrics: Dict[str, float]) -> str:
        """Determine overall quality assessment"""
        psnr = metrics.get("psnr", 0)
        ssim = metrics.get("ssim", 0)

        if psnr > 40 and ssim > 0.95:
            return "excellent"
        elif psnr > 30 and ssim > 0.9:
            return "good"
        elif psnr > 20 and ssim > 0.8:
            return "acceptable"
        else:
            return "poor"
