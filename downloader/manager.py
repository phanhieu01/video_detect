"""Download manager - factory for platform-specific downloaders"""

from pathlib import Path
from typing import Optional, List
from .base import BaseDownloader, DownloadResult
from .youtube import YouTubeDownloader
from .tiktok import TikTokDownloader
from .facebook import FacebookDownloader
from .instagram import InstagramDownloader
from .twitter import TwitterDownloader
from .generic import GenericDownloader
from .direct_url import DirectURLDownloader


class DownloadManager:
    
    def __init__(self, output_dir: Optional[Path] = None, **config):
        self.output_dir = output_dir
        self.config = config
        self.downloaders: List[BaseDownloader] = []
        self._init_downloaders()
    
    def _init_downloaders(self):
        self.downloaders = [
            DirectURLDownloader(self.output_dir, **self.config),  # Ưu tiên URL trực tiếp trước
            YouTubeDownloader(self.output_dir, **self.config),
            TikTokDownloader(self.output_dir, **self.config),
            FacebookDownloader(self.output_dir, **self.config),
            InstagramDownloader(self.output_dir, **self.config),
            TwitterDownloader(self.output_dir, **self.config),
            GenericDownloader(self.output_dir, **self.config),
        ]
    
    def get_downloader(self, url: str) -> Optional[BaseDownloader]:
        for downloader in self.downloaders:
            if downloader.can_handle(url):
                return downloader
        return None
    
    def download(self, url: str) -> DownloadResult:
        downloader = self.get_downloader(url)
        if downloader:
            return downloader.download(url)
        return DownloadResult(
            url=url,
            file_path=None,
            title=None,
            duration=None,
            success=False,
            error="Unsupported URL",
        )
    
    def batch_download(self, urls: List[str]) -> List[DownloadResult]:
        results = []
        for url in urls:
            result = self.download(url)
            results.append(result)
        return results
