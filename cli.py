"""CLI entry point for Video Downloader Pro"""

import typer
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .utils import load_config, setup_logging, ensure_dir
from .downloader.manager import DownloadManager
from .processor import ProcessingPipeline

app = typer.Typer(
    name="vdl-pro",
    help="Download and process videos from social media",
    add_completion=False,
)

console = Console()


@app.command()
def download(
    url: str = typer.Argument(..., help="URL to download from"),
    output: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="Output directory",
    ),
    format: str = typer.Option(
        "mp4",
        "-f",
        "--format",
        help="Output format (mp4, mkv, webm)",
    ),
    no_process: bool = typer.Option(
        False,
        "--no-process",
        help="Skip processing (watermark/fingerprint removal)",
    ),
):
    """Download a video/image from URL"""
    config = load_config()
    setup_logging()
    
    output_dir = output or Path(config.get("download", {}).get("output_dir", "./downloads"))
    
    with console.status("[bold green]Downloading...", spinner="dots"):
        download_config = config.get("download", {}).copy()
        download_config.pop("output_dir", None)  # Remove to avoid duplicate
        # Thêm quality config vào download_config
        download_config["quality"] = config.get("quality", {})
        manager = DownloadManager(output_dir, **download_config)
        result = manager.download(url)
    
    if result.success:
        console.print(f"✓ Downloaded: {result.title}", style="green")
        console.print(f"  File: {result.file_path}")
        
        if not no_process and result.file_path:
            process_file(result.file_path)
    else:
        console.print(f"✗ Download failed: {result.error}", style="red")
        raise typer.Exit(1)


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

    from .detector import WatermarkDetector, FingerprintAnalyzer
    
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


def process_file(file_path: Path) -> None:
    """Process a downloaded file"""
    config = load_config()
    
    console.print("  Processing...", style="cyan")
    
    pipeline = ProcessingPipeline(config)
    pipeline.process(file_path)
    
    console.print("  ✓ Processed", style="green")


@app.command()
def batch(
    input_file: Path = typer.Argument(
        ...,
        help="Text file containing URLs (one per line)",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="Output directory",
    ),
    no_process: bool = typer.Option(
        False,
        "--no-process",
        help="Skip processing",
    ),
):
    """Batch download multiple URLs"""
    config = load_config()
    setup_logging()
    
    output_dir = output or Path(config.get("download", {}).get("output_dir", "./downloads"))
    ensure_dir(output_dir)
    
    with open(input_file) as f:
        urls = [line.strip() for line in f if line.strip()]
    
    manager = DownloadManager(output_dir, **config.get("download", {}))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Processing URLs...", total=len(urls))
        
        for url in urls:
            progress.update(task, description=f"[green]Downloading: {url[:50]}...")
            result = manager.download(url)
            
            if result.success:
                console.print(f"✓ {result.title[:40]}...", style="green")
                
                if not no_process and result.file_path:
                    process_file(result.file_path)
            else:
                console.print(f"✗ Failed: {result.error[:50]}...", style="red")
            
            progress.advance(task)


@app.command()
def info():
    """Show information about the tool"""
    table = Table(title="Video Downloader Pro")
    table.add_column("Platform", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    
    platforms = [
        ("YouTube", "Supported"),
        ("TikTok", "Supported"),
        ("Facebook", "Supported"),
        ("Instagram", "Supported"),
        ("Twitter/X", "Supported"),
    ]
    
    for platform, status in platforms:
        table.add_row(platform, status)
    
    console.print(table)


@app.command()
def check():
    """Check system dependencies"""
    console.print("[bold]Checking dependencies...\n", style="cyan")

    try:
        import yt_dlp
        console.print("✓ yt-dlp installed", style="green")
    except ImportError:
        console.print("✗ yt-dlp not found", style="red")

    try:
        import cv2
        console.print("✓ OpenCV installed", style="green")
    except ImportError:
        console.print("✗ OpenCV not found", style="red")

    from .utils.ffmpeg_wrapper import FFmpegWrapper
    version = FFmpegWrapper.get_version()
    if "FFmpeg not found" not in version:
        console.print(f"✓ {version}", style="green")
    else:
        console.print("✗ FFmpeg not found", style="red")


if __name__ == "__main__":
    app()
