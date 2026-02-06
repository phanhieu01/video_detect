"""Twitter/X downloader"""

import yt_dlp
from pathlib import Path
from typing import Optional
from .base import BaseDownloader, DownloadResult


class TwitterDownloader(BaseDownloader):
    platform_name = "twitter"
    
    def __init__(self, output_dir: Optional[Path] = None, **kwargs):
        super().__init__(output_dir, **kwargs)
    
    def can_handle(self, url: str) -> bool:
        return "twitter.com" in url.lower() or "x.com" in url.lower()
    
    def download(self, url: str) -> DownloadResult:
        ydl_opts = {
            "format": "best",
            "outtmpl": str(self.output_dir / "%(title)s.%(ext)s"),
            "quiet": False,
            "no_warnings": False,
        }
        
        if self.config.get("cookies_file"):
            ydl_opts["cookiefile"] = self.config["cookies_file"]
        
        if self.config.get("proxy"):
            ydl_opts["proxy"] = self.config["proxy"]
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                file_path = Path(ydl.prepare_filename(info))
                
                return DownloadResult(
                    url=url,
                    file_path=file_path,
                    title=info.get("title") or info.get("description"),
                    duration=info.get("duration"),
                    success=True,
                    metadata=info,
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
