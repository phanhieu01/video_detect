"""Direct URL downloader for .mp4, .webm, etc. URLs"""

import requests
import urllib.parse
from pathlib import Path
from typing import Optional
from .base import BaseDownloader, DownloadResult


class DirectURLDownloader(BaseDownloader):
    """Downloader for direct video URLs (.mp4, .webm, .mov, etc.)"""

    platform_name = "direct_url"

    def __init__(self, output_dir: Optional[Path] = None, **kwargs):
        super().__init__(output_dir, **kwargs)

    def can_handle(self, url: str) -> bool:
        """Check if URL is a direct video file"""
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.lower()

        # Direct video file extensions
        video_extensions = [
            '.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv',
            '.wmv', '.m4v', '.mpg', '.mpeg', '.3gp', '.ogv'
        ]

        return any(path.endswith(ext) for ext in video_extensions)

    def download(self, url: str) -> DownloadResult:
        try:
            # Extract filename from URL or generate one
            filename = self.get_output_filename(url, "mp4")
            output_path = self.output_dir / filename

            # Download with streaming to support large files
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))

            # Download with progress
            downloaded = 0
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            return DownloadResult(
                url=url,
                file_path=output_path,
                title=filename,
                duration=None,
                success=True,
                metadata={
                    'direct_url': True,
                    'file_size': downloaded,
                    'total_size': total_size
                },
            )
        except Exception as e:
            return DownloadResult(
                url=url,
                file_path=None,
                title=None,
                duration=None,
                success=False,
                error=str(e),
            )
