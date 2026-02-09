"""Generic downloader for yt-dlp supported sites"""

import yt_dlp
from pathlib import Path
from typing import Optional
from .base import BaseDownloader, DownloadResult
from .utils import get_best_format


class GenericDownloader(BaseDownloader):
    platform_name = "generic"

    def __init__(self, output_dir: Optional[Path] = None, **kwargs):
        super().__init__(output_dir, **kwargs)

    def can_handle(self, url: str) -> bool:
        # Generic downloader handles everything as fallback
        return True

    def download(self, url: str) -> DownloadResult:
        # Sử dụng helper function để lấy format tốt nhất
        format_str = get_best_format(self.config, "generic")

        ydl_opts = {
            "format": format_str,
            "outtmpl": str(self.output_dir / "%(title)s.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": False,
            "no_warnings": False,
        }

        # Config từ config.yaml
        if self.config.get("proxy"):
            ydl_opts["proxy"] = self.config["proxy"]
        if self.config.get("cookies_file"):
            ydl_opts["cookiefile"] = self.config["cookies_file"]
        if self.config.get("user_agent"):
            ydl_opts["user_agent"] = self.config["user_agent"]

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                file_path = Path(ydl.prepare_filename(info))

                # Lấy tên nền tảng từ info nếu có
                platform = info.get("extractor", "unknown")
                title = info.get("title")
                duration = info.get("duration")

                return DownloadResult(
                    url=url,
                    file_path=file_path,
                    title=title,
                    duration=duration,
                    success=True,
                    metadata={**info, "platform": platform},
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
