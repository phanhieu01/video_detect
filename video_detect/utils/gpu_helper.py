"""GPU acceleration helper for video processing"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GPUHelper:

    def __init__(self):
        self.cuda_available = self._check_cuda()
        self.opencl_available = self._check_opencl()
        self.gpu_count = 0

        if self.cuda_available:
            self.gpu_count = self._get_cuda_device_count()

    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try PyTorch CUDA
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass

        return False

    def _check_opencl(self) -> bool:
        """Check if OpenCL is available"""
        try:
            import subprocess
            result = subprocess.run(
                ["clinfo"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True
        except FileNotFoundError:
            pass

        # Check through OpenCV
        try:
            import cv2
            # Try to create OpenCL context
            cv2.ocl.setUseOpenCL(True)
            return cv2.ocl.useOpenCL()
        except Exception:
            pass

        return False

    def _get_cuda_device_count(self) -> int:
        """Get number of CUDA devices"""
        try:
            import torch
            return torch.cuda.device_count()
        except ImportError:
            pass

        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass

        return 0

    def get_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        info = {
            "cuda_available": self.cuda_available,
            "opencl_available": self.opencl_available,
            "gpu_count": self.gpu_count,
            "recommended_backend": None,
        }

        if self.cuda_available:
            info["recommended_backend"] = "cuda"
            info["cuda_devices"] = self._get_cuda_device_info()
        elif self.opencl_available:
            info["recommended_backend"] = "opencl"

        return info

    def _get_cuda_device_info(self) -> list:
        """Get detailed CUDA device information"""
        devices = []

        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(', ')
                        if len(parts) >= 2:
                            devices.append({
                                "name": parts[0],
                                "memory": parts[1],
                            })
        except Exception:
            pass

        return devices

    def is_available(self) -> bool:
        """Check if any GPU acceleration is available"""
        return self.cuda_available or self.opencl_available

    def get_ffmpeg_gpu_params(self) -> list:
        """Get FFmpeg parameters for GPU acceleration"""
        params = []

        if self.cuda_available:
            # Use NVIDIA GPU for encoding/decoding
            params.extend([
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
            ])
        elif self.opencl_available:
            # Use OpenCL
            params.extend([
                "-hwaccel", "opencl",
            ])

        return params

    def get_opencv_gpu_info(self) -> Dict[str, Any]:
        """Get OpenCV GPU capabilities"""
        info = {
            "cuda": False,
            "opencl": False,
            "other": [],
        }

        try:
            import cv2

            # Check CUDA
            try:
                cv2.cuda.printCudaDeviceInfo(0)
                info["cuda"] = True
            except Exception:
                pass

            # Check OpenCL
            info["opencl"] = cv2.ocl.useOpenCL()

        except ImportError:
            pass

        return info


# Global GPU helper instance
_gpu_helper: Optional[GPUHelper] = None


def get_gpu_helper() -> GPUHelper:
    """Get global GPU helper instance"""
    global _gpu_helper
    if _gpu_helper is None:
        _gpu_helper = GPUHelper()
    return _gpu_helper


def enable_gpu_acceleration(enable: bool = True) -> None:
    """Enable or disable GPU acceleration globally"""
    helper = get_gpu_helper()

    if not helper.is_available():
        logger.warning("GPU acceleration requested but no GPU available")
        return

    # Configure OpenCV to use GPU
    try:
        import cv2

        if helper.opencl_available:
            cv2.ocl.setUseOpenCL(enable)
            logger.info(f"OpenCL GPU acceleration: {'enabled' if enable else 'disabled'}")

        if helper.cuda_available:
            # Configure PyTorch if available
            try:
                import torch
                if enable:
                    torch.set_grad_enabled(False)  # Disable grad for inference
                logger.info(f"CUDA GPU acceleration: {'enabled' if enable else 'disabled'}")
            except ImportError:
                pass

    except ImportError:
        pass
