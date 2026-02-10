"""CLI entry point for Video Detect Pro"""

import typer
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

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

    # Determine number of steps
    if steps is None:
        suffix = file_path.suffix.lower()
        video_extensions = [".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv"]
        image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]

        # Estimate default steps count
        step_count = 2  # metadata_clean at start and end
        if suffix in video_extensions:
            if config.get("fingerprint", {}).get("enabled", True):
                step_count += 1
            if config.get("watermark", {}).get("enabled", True):
                step_count += 1
            if config.get("transforms", {}).get("enabled", True):
                step_count += 1
        elif suffix in image_extensions:
            if config.get("fingerprint", {}).get("enabled", True):
                step_count += 1
            if config.get("transforms", {}).get("enabled", True):
                step_count += 1
    else:
        step_count = len(steps)

    # Process with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing file...", total=100)

        # Update progress during processing
        success = pipeline.process(file_path, output, steps)
        progress.update(task, completed=100)

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

    from .utils.ffmpeg_wrapper import FFmpegWrapper
    version = FFmpegWrapper.get_version()
    if "FFmpeg not found" not in version:
        console.print(f"✓ {version}", style="green")
    else:
        console.print("✗ FFmpeg not found", style="red")


@app.command()
def assess(
    original: Path = typer.Argument(
        ...,
        help="Path to original file",
        exists=True,
    ),
    processed: Path = typer.Argument(
        ...,
        help="Path to processed file",
        exists=True,
    ),
):
    """Assess quality of watermark removal"""
    from .utils import QualityAssessment

    console.print(f"Assessing: {original.name} vs {processed.name}", style="cyan")

    result = QualityAssessment.assess_removal(original, processed)

    if "error" in result:
        console.print(f"✗ Error: {result['error']}", style="red")
        raise typer.Exit(1)

    # Display results
    table = Table(title="Quality Assessment Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("MSE", f"{result['metrics']['mse']:.2f}")
    table.add_row("PSNR", f"{result['metrics']['psnr']:.2f} dB")
    table.add_row("SSIM", f"{result['metrics']['ssim']:.4f}")
    table.add_row("Overall", result['overall_quality'].title())

    console.print(table)

    # Region metrics if available
    if "region_metrics" in result:
        console.print("\n[bold]Watermark Region Quality:[/bold]")
        region_table = Table()
        region_table.add_column("Metric", style="cyan")
        region_table.add_column("Value", style="green")

        region_table.add_row("MSE", f"{result['region_metrics']['mse']:.2f}")
        region_table.add_row("PSNR", f"{result['region_metrics']['psnr']:.2f} dB")

        console.print(region_table)


@app.command()
def batch(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing files to process",
        exists=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="Output directory",
    ),
    pattern: str = typer.Option(
        "*",
        "-p",
        "--pattern",
        help="File pattern to match (e.g., '*.mp4', '*.jpg')",
    ),
):
    """Process multiple files in a directory"""
    config = load_config()
    setup_logging()

    # Find matching files
    files = list(input_dir.glob(pattern))
    files = [f for f in files if f.is_file()]

    if not files:
        console.print("No files found matching pattern", style="yellow")
        raise typer.Exit(1)

    console.print(f"Found {len(files)} files to process", style="cyan")

    # Set output directory
    if output_dir is None:
        output_dir = Path(config.get("output", {}).get("output_dir", "./downloads/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ProcessingPipeline(config)

    # Process files with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Processing files...", total=len(files))

        success_count = 0
        fail_count = 0

        for file_path in files:
            progress.update(task, description=f"[green]Processing: {file_path.name}")

            output_path = output_dir / f"processed_{file_path.name}"
            success = pipeline.process(file_path, output_path)

            if success:
                console.print(f"✓ {file_path.name}", style="green")
                success_count += 1
            else:
                console.print(f"✗ {file_path.name}", style="red")
                fail_count += 1

            progress.advance(task)

    # Summary
    console.print(f"\nComplete: {success_count} succeeded, {fail_count} failed", style="cyan")


if __name__ == "__main__":
    app()
