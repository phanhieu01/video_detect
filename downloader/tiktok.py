"""TikTok downloader"""

import yt_dlp
from pathlib import Path
from typing import Optional
from .base import BaseDownloader, DownloadResult


class TikTokDownloader(BaseDownloader):
    platform_name = "tiktok"
    
    def __init__(self, output_dir: Optional[Path] = None, **kwargs):
        super().__init__(output_dir, **kwargs)
        self.no_watermark = kwargs.get("no_watermark", True)
    
    def can_handle(self, url: str) -> bool:
        return "tiktok.com" in url.lower()
    
    def download(self, url: str) -> DownloadResult:
        if self.no_watermark:
            try:
                from .tiktok_nowm import TikTokNoWatermarkAPI
                api = TikTokNoWatermarkAPI()
                
                output_filename = self.get_output_filename(url, "mp4")
                
                if api.download_no_watermark(url, output_filename):
                    api.close()
                    
                    return DownloadResult(
                        url=url,
                        file_path=output_filename,
                        title="TikTok Video (No Watermark)",
                        duration=None,
                        success=True,
                        metadata={"no_watermark": True},
                    )
                
                api.close()
            except Exception:
                pass
        
        ydl_opts = {
            "format": "best",
            "outtmpl": str(self.output_dir / "%(title)s.%(ext)s"),
            "quiet": False,
            "no_warnings": False,
        }
        
        if self.no_watermark:
            ydl_opts["format"] = "best[protocol^=m3u8]/best"
        
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
