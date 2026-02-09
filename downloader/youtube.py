"""YouTube downloader using yt-dlp"""

import yt_dlp
from pathlib import Path
from typing import Optional
from .base import BaseDownloader, DownloadResult
from .utils import get_best_format


class YouTubeDownloader(BaseDownloader):
    platform_name = "youtube"

    def __init__(self, output_dir: Optional[Path] = None, **kwargs):
        super().__init__(output_dir, **kwargs)

    def can_handle(self, url: str) -> bool:
        return any(domain in url.lower() for domain in ["youtube.com", "youtu.be"])

    def download(self, url: str) -> DownloadResult:
        # Sử dụng helper function để lấy format tốt nhất cho YouTube
        format_str = get_best_format(self.config, "youtube")

        ydl_opts = {
            "format": format_str,
            "outtmpl": str(self.output_dir / "%(title)s.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": False,
            "no_warnings": False,
        }

        if self.config.get("proxy"):
            ydl_opts["proxy"] = self.config["proxy"]

        if self.config.get("cookies_file"):
            ydl_opts["cookiefile"] = self.config["cookies_file"]

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                file_path = Path(ydl.prepare_filename(info))

                return DownloadResult(
                    url=url,
                    file_path=file_path,
                    title=info.get("title"),
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
