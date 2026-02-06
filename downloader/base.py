"""Base downloader class"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DownloadResult:
    url: str
    file_path: Optional[Path]
    title: Optional[str]
    duration: Optional[float]
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseDownloader(ABC):
    
    platform_name: str
    
    def __init__(self, output_dir: Optional[Path] = None, **kwargs):
        self.output_dir = output_dir or Path("./downloads")
        self.config = kwargs
    
    @abstractmethod
    def download(self, url: str) -> DownloadResult:
        pass
    
    @abstractmethod
    def can_handle(self, url: str) -> bool:
        pass
    
    def get_output_filename(self, url: str, extension: str = "mp4") -> Path:
        from utils import ensure_dir
        
        ensure_dir(self.output_dir)
        filename = f"{self.platform_name}_{hash(url) & 0xffffffff}.{extension}"
        return self.output_dir / filename
