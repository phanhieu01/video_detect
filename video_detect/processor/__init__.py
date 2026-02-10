"""Video processor module"""

from .watermark_remover import WatermarkRemover
from .fingerprint_remover import FingerprintRemover
from .video_transform import VideoTransformer
from .image_transform import ImageTransformer
from .metadata_cleaner import MetadataCleaner
from .content_inpainting import ContentAwareInpainter
from .parallel_processor import ParallelProcessor, BatchProcessingPipeline, ProcessingResult
from .pipeline import ProcessingPipeline

__all__ = [
    "WatermarkRemover",
    "FingerprintRemover",
    "VideoTransformer",
    "ImageTransformer",
    "MetadataCleaner",
    "ContentAwareInpainter",
    "ParallelProcessor",
    "BatchProcessingPipeline",
    "ProcessingResult",
    "ProcessingPipeline",
]
