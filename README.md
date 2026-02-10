# Video Detect Pro

Detect and remove watermarks/fingerprints from video and image files.

## Features

- **Watermark Detection**: Auto-detect watermarks using template matching & edge detection
- **Fingerprint Analysis**: Detect invisible fingerprints in videos/images
- **Watermark Removal**: Remove watermarks using inpainting techniques
- **Fingerprint Removal**: Remove fingerprints via re-encoding and transforms
- **Metadata Cleaning**: Strip EXIF data and randomize dates

## Installation

```bash
pip install -e .
```

Or install dependencies manually:

```bash
pip install opencv-python ffmpeg-python numpy Pillow typer rich pyyaml pydantic
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

### Check system dependencies
```bash
vdl-pro check
```

### Show tool information
```bash
vdl-pro info
```

### Detect watermarks in a video/image
```bash
vdl-pro detect ./video.mp4
vdl-pro detect ./image.jpg --fingerprint
```

### Process an existing file
```bash
vdl-pro process ./video.mp4
vdl-pro process ./image.jpg -o ./output.jpg
```

### Process with custom steps
```bash
vdl-pro process ./video.mp4 -s watermark_remove -s metadata_clean
```

## Configuration

Edit `video_detect/config.yaml` to customize behavior:

```yaml
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

output:
  output_dir: "./downloads/processed"
```

## Watermark Removal

- **Template Matching**: Auto-detect common watermarks (TikTok logo, Instagram handle)
- **Edge Detection**: Find watermarks through edge analysis
- **Static Overlay Detection**: Identify static watermarks across frames
- **OpenCV Inpainting**: Natural-looking watermark removal
- **FFmpeg Delogo**: Fast removal for known watermark positions
- **Auto-Detection**: Automatically find watermark regions

## Fingerprint Removal

- **Re-encode**: Change codec (H.264 → H.265)
- **Bitrate Variation**: Randomize bitrate (+/- 10-20%)
- **Resolution Change**: Slight modification (+/- 2-6 pixels)
- **Color Space Manipulation**: BT.709 ↔ BT.601 conversion
- **Noise Injection**: Subtle Gaussian noise addition

## Video Transforms

- **Speed Change**: 1.01x - 1.05x to avoid content ID
- **Audio Pitch Shift**: Slightly modify audio pitch
- **Color Adjustment**: Brightness, contrast, saturation shifts
- **Crop**: 2-5% edge cropping
- **Horizontal Flip**: Optional mirror effect
- **Audio Silence**: Add 0.1-0.5s silence padding

## Metadata Cleaning

- Strip all EXIF data (images)
- Strip all video metadata tags
- Randomize creation dates
- Change file hashes (via re-encode)

## Fingerprint Analysis

- Hash consistency analysis across frames
- LSB (Least Significant Bit) steganography detection
- Static pattern identification
- Severity assessment

## Project Structure

```
video_detect/
├── pyproject.toml               # Project configuration
├── README.md                    # This file
└── video_detect/                # Main package
    ├── __init__.py
    ├── cli.py                   # CLI entry point
    ├── config.yaml              # Configuration file
    ├── detector/                # Detection module
    │   ├── watermark_detector.py    # Auto watermark detection
    │   └── fingerprint_analyzer.py  # Fingerprint analysis
    ├── processor/               # Processing module
    │   ├── watermark_remover.py     # Remove visible watermarks
    │   ├── fingerprint_remover.py   # Remove invisible fingerprints
    │   ├── video_transform.py       # Transform video signature
    │   ├── image_transform.py       # Transform image signature
    │   ├── metadata_cleaner.py      # Clean metadata
    │   └── pipeline.py              # Coordinate all steps
    └── utils/                   # Utilities
        ├── ffmpeg_wrapper.py        # FFmpeg operations
        └── helpers.py               # Helper functions
```

## Processing Pipeline

```
Input File → Detect Watermark → Remove Watermark
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
from video_detect.processor import ProcessingPipeline
from pathlib import Path

pipeline = ProcessingPipeline(config)
pipeline.process(
    input_path=Path("video.mp4"),
    steps=["watermark_remove", "metadata_clean"]
)
```

## License

This tool is for educational and personal use only. Respect copyright and terms of service of platforms. The watermark and fingerprint removal techniques provided are for research purposes only.
