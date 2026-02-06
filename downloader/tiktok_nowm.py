"""TikTok no-watermark API fallback - download TikTok videos without watermark using alternative methods"""

from pathlib import Path
from typing import Optional
import httpx
import re

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


class TikTokNoWatermarkAPI:
    
    def __init__(self):
        self.client = httpx.Client(timeout=30, follow_redirects=True)
    
    def get_no_watermark_url(self, tiktok_url: str) -> Optional[str]:
        direct_url = self._try_direct_extract(tiktok_url)
        if direct_url:
            return direct_url
        
        api_url = self._try_snaptik_api(tiktok_url)
        if api_url:
            return api_url
        
        musical_url = self._try_musicaldown_api(tiktok_url)
        if musical_url:
            return musical_url
        
        return None
    
    def _try_direct_extract(self, tiktok_url: str) -> Optional[str]:
        if yt_dlp is None:
            return None
        
        try:
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "format": "best[protocol^=m3u8]/best",
                "extract_flat": False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(tiktok_url, download=False)
                
                if info and "url" in info:
                    url = info["url"]
                    if "wm" not in url.lower() and "watermark" not in url.lower():
                        return url
                
                if info and "formats" in info:
                    for fmt in info["formats"]:
                        url = fmt.get("url", "")
                        if ("wm" not in url.lower() and 
                            "watermark" not in url.lower() and
                            "m3u8" not in url.lower()):
                            return url
        except Exception:
            pass
        
        return None
    
    def _try_snaptik_api(self, tiktok_url: str) -> Optional[str]:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://snaptik.app/",
            }
            
            data = {"url": tiktok_url}
            response = self.client.post(
                "https://snaptik.app/abc2.php",
                headers=headers,
                data=data,
            )
            
            if response.status_code == 200:
                html = response.text
                
                match = re.search(r'href="([^"]+)"\s+class="is download"', html)
                if match:
                    return match.group(1)
        except Exception:
            pass
        
        return None
    
    def _try_musicaldown_api(self, tiktok_url: str) -> Optional[str]:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }
            
            data = {"url": tiktok_url}
            response = self.client.post(
                "https://musicaldown.com/api/ajax/search",
                headers=headers,
                data=data,
            )
            
            if response.status_code == 200:
                json_data = response.json()
                
                if "data" in json_data and json_data["data"]:
                    for item in json_data["data"]:
                        if "url" in item:
                            return item["url"]
        except Exception:
            pass
        
        return None
    
    def download_no_watermark(
        self,
        tiktok_url: str,
        output_path: Path,
    ) -> bool:
        video_url = self.get_no_watermark_url(tiktok_url)
        
        if not video_url:
            return False
        
        try:
            response = self.client.get(video_url)
            if response.status_code == 200:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                return True
        except Exception:
            pass
        
        return False
    
    def close(self):
        if hasattr(self, "client"):
            self.client.close()
