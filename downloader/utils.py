"""Utility functions for downloaders"""

from typing import Dict, Optional


def get_best_format(config: Optional[Dict] = None, platform: str = "generic") -> str:
    """
    Xác định format string tốt nhất cho yt-dlp

    Args:
        config: Config dict từ config.yaml
        platform: Tên nền tảng (youtube, tiktok, facebook, instagram, twitter, generic)

    Returns:
        Format string cho yt-dlp
    """
    if config is None:
        config = {}

    # Lấy config từ quality section nếu có, ngược lại dùng config gốc
    min_height = config.get("quality", {}).get("min_height", 720)
    prefer_4k = config.get("quality", {}).get("prefer_4k", False)

    # Format theo độ ưu tiên
    formats = []

    # Ưu tiên 4K nếu config bật
    if prefer_4k:
        formats.append("bestvideo[height>=2160]+bestaudio/best")  # 4K

    # Theo nền tảng - ưu tiên chất lượng khác nhau
    # Dùng format đơn giản để tránh yt-dlp ưu tiên format all-in-one chất lượng thấp
    if platform == "youtube":
        # YouTube có nhiều format, ưu tiên video+audio riêng để có chất lượng cao nhất
        formats.append("bestvideo+bestaudio/best")
    elif platform == "tiktok":
        # TikTok thường có 720p hoặc 1080p
        formats.append("bestvideo+bestaudio/best")
    elif platform in ("facebook", "instagram", "twitter"):
        # Các mạng xã hội khác
        formats.append("bestvideo+bestaudio/best")
    else:  # generic
        formats.append("bestvideo+bestaudio/best")

    return "/".join(formats)
