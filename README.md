# Video Downloader Pro

Download and process videos/images from social media platforms with watermark and fingerprint removal.

## Supported Platforms

- YouTube
- TikTok (with no-watermark API)
- Facebook
- Instagram
- Twitter/X

## Installation

```bash
pip install -e .
```

Or install dependencies manually:

```bash
pip install yt-dlp opencv-python ffmpeg-python numpy Pillow typer rich pyyaml instaloader httpx pydantic
```

## Requirements

- Python 3.10+
- FFmpeg (must be installed on system)

### Installing FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH.

## Usage

### Download a video
```bash
python cli.py download "https://youtube.com/watch?v=xxx"
```

### Detect watermarks in a video
```bash
python cli.py detect ./video.mp4
python cli.py detect ./video.mp4 --fingerprint
```

### Process an existing file
```bash
python cli.py process ./video.mp4
```

### Batch download
```bash
echo "https://youtube.com/watch?v=xxx" > urls.txt
python cli.py batch urls.txt
```

### Check dependencies
```bash
python cli.py check
```

### Show supported platforms
```bash
python cli.py info
```

## Configuration

Edit `config.yaml` to customize behavior:

```yaml
download:
  output_dir: "./downloads"
  format: "mp4"
  quality: "best"
  no_watermark: true

watermark:
  enabled: true
  auto_detect: true
  method: "inpaint"
  detection_method: "auto"
  confidence_threshold: 0.7

fingerprint:
  enabled: true
  re_encode: true
  change_bitrate: true
  change_resolution: true
  change_colorspace: true
  add_noise: true

transforms:
  enabled: true
  speed_change: true
  speed_factor: 1.03
  pitch_shift: true
  horizontal_flip: false
  crop_percent: 2
  brightness_shift: 2
  contrast_shift: 2
  saturation_shift: 5
  add_silence: true
  silence_duration: 0.2

metadata:
  strip_metadata: true
  randomize_date: true

detector:
  enabled: true
  analyze_fingerprint: false
```

## Features

### Download Module
- Multi-platform support (YouTube, TikTok, Facebook, Instagram, Twitter)
- TikTok no-watermark API fallback
- Batch download support
- Custom format and quality options
- Proxy and cookies support

### Watermark Removal
- **Template Matching**: Auto-detect common watermarks (TikTok logo, Instagram handle)
- **Edge Detection**: Find watermarks through edge analysis
- **Static Overlay Detection**: Identify static watermarks across frames
- **OpenCV Inpainting**: Natural-looking watermark removal
- **FFmpeg Delogo**: Fast removal for known watermark positions
- **Auto-Detection**: Automatically find watermark regions

### Fingerprint Removal
- **Re-encode**: Change codec (H.264 → H.265)
- **Bitrate Variation**: Randomize bitrate (+/- 10-20%)
- **Resolution Change**: Slight modification (+/- 2-6 pixels)
- **Color Space Manipulation**: BT.709 ↔ BT.601 conversion
- **Noise Injection**: Subtle Gaussian noise addition

### Video Transforms
- **Speed Change**: 1.01x - 1.05x to avoid content ID
- **Audio Pitch Shift**: Slightly modify audio pitch
- **Color Adjustment**: Brightness, contrast, saturation shifts
- **Crop**: 2-5% edge cropping
- **Horizontal Flip**: Optional mirror effect
- **Audio Silence**: Add 0.1-0.5s silence padding

### Metadata Cleaning
- Strip all EXIF data (images)
- Strip all video metadata tags
- Randomize creation dates
- Change file hashes (via re-encode)

### Fingerprint Analysis
- Hash consistency analysis across frames
- LSB (Least Significant Bit) steganography detection
- Static pattern identification
- Severity assessment

## Project Structure

```
video_detect/
├── cli.py                      # CLI entry point
├── config.yaml                 # Configuration file
├── downloader/                 # Download module
│   ├── base.py               # Base downloader class
│   ├── manager.py            # Download factory
│   ├── youtube.py            # YouTube downloader
│   ├── tiktok.py             # TikTok downloader
│   ├── tiktok_nowm.py        # TikTok no-watermark API
│   ├── facebook.py           # Facebook downloader
│   ├── instagram.py          # Instagram downloader
│   └── twitter.py            # Twitter/X downloader
├── processor/                  # Processing module
│   ├── watermark_remover.py   # Remove visible watermarks
│   ├── fingerprint_remover.py # Remove invisible fingerprints
│   ├── video_transform.py    # Transform video signature
│   ├── image_transform.py    # Transform image signature
│   ├── metadata_cleaner.py  # Clean metadata
│   └── pipeline.py          # Coordinate all steps
├── detector/                   # Detection module
│   ├── watermark_detector.py  # Auto watermark detection
│   └── fingerprint_analyzer.py # Fingerprint analysis
└── utils/                     # Utilities
    ├── ffmpeg_wrapper.py      # FFmpeg operations
    └── helpers.py           # Helper functions
```

## Processing Pipeline

```
URL Input → Download → Detect Watermark → Remove Watermark
    ↓
Remove Fingerprint → Apply Transforms → Clean Metadata
    ↓
Output File
```

## Advanced Usage

### Custom watermark region
```yaml
watermark:
  enabled: true
  auto_detect: false
  watermark_region:
    x: 1200
    y: 700
    width: 100
    height: 40
  method: "delogo"
```

### Disable specific processing steps
```yaml
fingerprint:
  enabled: false

transforms:
  enabled: false
```

### Custom processing steps
```python
from processor import ProcessingPipeline
from pathlib import Path

pipeline = ProcessingPipeline(config)
pipeline.process(
    input_path=Path("video.mp4"),
    steps=["watermark_remove", "metadata_clean"]
)
```

## Disclaimer

This tool is for educational and personal use only. Respect copyright and terms of service of platforms. The watermark and fingerprint removal techniques provided are for research purposes only.
