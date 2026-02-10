"""CLI entry point for Video Detect Pro"""

import typer
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table

from .utils import load_config, setup_logging
from .processor import ProcessingPipeline

app = typer.Typer(
    name="vdl-pro",
    help="Detect and remove watermarks/fingerprints from media files",
    add_completion=False,
)

console = Console()


@app.command()
def detect(
    file_path: Path = typer.Argument(
        ...,
        help="Path to video/image file to analyze",
        exists=True,
    ),
    analyze_fingerprint: bool = typer.Option(
        False,
        "--fingerprint",
        "-f",
        help="Also analyze for invisible fingerprints",
    ),
):
    """Detect watermarks and fingerprints in a video/image file"""
    config = load_config()
    setup_logging()

    console.print(f"Analyzing: {file_path.name}", style="cyan")

    from detector import WatermarkDetector, FingerprintAnalyzer
    
    watermark_detector = WatermarkDetector(config.get("watermark", {}))
    result = watermark_detector.detect_watermark(file_path)
    
    table = Table(title="Detection Results")
    table.add_column("Type", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")
    
    if result and result.get("found"):
        table.add_row(
            "Watermark",
            "Detected",
            f"Method: {result['method']}, Region: {result['region']}",
        )
    else:
        table.add_row("Watermark", "Not Detected", "No watermark patterns found")
    
    console.print(table)
    
    if analyze_fingerprint:
        console.print("\nFingerprint analysis...", style="cyan")
        analyzer = FingerprintAnalyzer()
        fp_result = analyzer.analyze(file_path)
        
        if fp_result.get("potential_fingerprints"):
            for fp in fp_result["potential_fingerprints"]:
                console.print(f"⚠ {fp['type']}: {fp['description']} (Severity: {fp['severity']})", style="yellow")
        else:
            console.print("✓ No suspicious fingerprint patterns detected", style="green")


@app.command()
def process(
    file_path: Path = typer.Argument(
        ...,
        help="Path to video/image file to process",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file path",
    ),
    steps: Optional[List[str]] = typer.Option(
        None,
        "-s",
        "--step",
        help="Processing steps to apply",
    ),
):
    """Process an existing video/image file"""
    config = load_config()
    setup_logging()
    
    console.print(f"Processing: {file_path.name}", style="cyan")
    
    pipeline = ProcessingPipeline(config)
    success = pipeline.process(file_path, output, steps)
    
    if success:
        console.print("✓ Processing complete", style="green")
    else:
        console.print("✗ Processing failed", style="red")
        raise typer.Exit(1)


@app.command()
def info():
    """Show information about the tool"""
    table = Table(title="Video Detect Pro")
    table.add_column("Feature", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")

    features = [
        ("Watermark Detection", "Auto-detect watermarks using template matching & edge detection"),
        ("Fingerprint Analysis", "Detect invisible fingerprints in videos/images"),
        ("Watermark Removal", "Remove watermarks using inpainting techniques"),
        ("Fingerprint Removal", "Remove fingerprints via re-encoding and transforms"),
        ("Metadata Cleaning", "Strip EXIF data and randomize dates"),
    ]

    for feature, description in features:
        table.add_row(feature, description)

    console.print(table)


@app.command()
def check():
    """Check system dependencies"""
    console.print("[bold]Checking dependencies...\n", style="cyan")

    try:
        import cv2
        console.print("✓ OpenCV installed", style="green")
    except ImportError:
        console.print("✗ OpenCV not found", style="red")

    try:
        import PIL
        console.print("✓ Pillow installed", style="green")
    except ImportError:
        console.print("✗ Pillow not found", style="red")

    from utils.ffmpeg_wrapper import FFmpegWrapper
    version = FFmpegWrapper.get_version()
    if "FFmpeg not found" not in version:
        console.print(f"✓ {version}", style="green")
    else:
        console.print("✗ FFmpeg not found", style="red")


if __name__ == "__main__":
    app()
