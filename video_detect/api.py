"""REST API for Video Detect Pro"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import tempfile
import shutil
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Detect Pro API",
    description="REST API for watermark and fingerprint detection/removal",
    version="1.0.0",
)


# Request/Response models
class DetectRequest(BaseModel):
    file_path: str
    analyze_fingerprint: bool = False


class ProcessRequest(BaseModel):
    file_path: str
    output_path: Optional[str] = None
    steps: Optional[List[str]] = None


class BatchProcessRequest(BaseModel):
    input_dir: str
    output_dir: Optional[str] = None
    pattern: str = "*"


class DetectResponse(BaseModel):
    found: bool
    method: str
    confidence: Optional[float] = None
    regions: Optional[List[Dict[str, int]]] = None


class ProcessResponse(BaseModel):
    success: bool
    output_path: Optional[str] = None
    message: Optional[str] = None


class InfoResponse(BaseModel):
    name: str
    version: str
    features: List[Dict[str, str]]


# Global config
config_cache = None


def load_config():
    global config_cache
    if config_cache is None:
        from ..utils import load_config
        config_cache = load_config()
    return config_cache


def setup_logging():
    from ..utils import setup_logging as sl
    sl()


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "video-detect-pro"}


# Info endpoint
@app.get("/info", response_model=InfoResponse)
async def get_info():
    """Get tool information"""
    return InfoResponse(
        name="Video Detect Pro",
        version="1.0.0",
        features=[
            {"feature": "Watermark Detection", "description": "Auto-detect watermarks"},
            {"feature": "Fingerprint Analysis", "description": "Detect invisible fingerprints"},
            {"feature": "Watermark Removal", "description": "Remove watermarks"},
            {"feature": "Fingerprint Removal", "description": "Remove fingerprints"},
        ],
    )


# Detect endpoint
@app.post("/detect", response_model=DetectResponse)
async def detect_watermark(request: DetectRequest):
    """Detect watermarks and fingerprints"""
    setup_logging()
    config = load_config()

    file_path = Path(request.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        from ..detector import WatermarkDetector

        detector = WatermarkDetector(config.get("watermark", {}))
        result = detector.detect_watermark(file_path)

        if result is None:
            return DetectResponse(found=False, method="error")

        return DetectResponse(
            found=result.get("found", False),
            method=result.get("method", "unknown"),
            confidence=result.get("confidence"),
            regions=result.get("regions"),
        )

    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Process endpoint
@app.post("/process", response_model=ProcessResponse)
async def process_file(request: ProcessRequest):
    """Process file to remove watermarks/fingerprints"""
    setup_logging()
    config = load_config()

    input_path = Path(request.file_path)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Input file not found")

    output_path = Path(request.output_path) if request.output_path else None

    try:
        from ..processor import ProcessingPipeline

        pipeline = ProcessingPipeline(config)
        success = pipeline.process(input_path, output_path, request.steps)

        if success and output_path is None:
            output_path = Path(config.get("output", {}).get("output_dir", "./downloads/processed"))
            output_path = output_path / f"processed_{input_path.name}"

        return ProcessResponse(
            success=success,
            output_path=str(output_path) if output_path else None,
            message="Processing completed successfully" if success else "Processing failed",
        )

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Upload and detect endpoint
@app.post("/upload/detect")
async def upload_detect(
    file: UploadFile = File(...),
    analyze_fingerprint: bool = False,
):
    """Upload file and detect watermarks"""
    setup_logging()
    config = load_config()

    # Save uploaded file
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / file.filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Detect
        from ..detector import WatermarkDetector

        detector = WatermarkDetector(config.get("watermark", {}))
        result = detector.detect_watermark(temp_path)

        if result is None:
            return {"found": False, "method": "error"}

        return {
            "found": result.get("found", False),
            "method": result.get("method", "unknown"),
            "confidence": result.get("confidence"),
            "regions": result.get("regions"),
        }

    finally:
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()


# Upload and process endpoint
@app.post("/upload/process")
async def upload_process(
    file: UploadFile = File(...),
    steps: Optional[str] = None,
):
    """Upload file and process it"""
    setup_logging()
    config = load_config()

    # Save uploaded file
    temp_dir = Path(tempfile.gettempdir())
    input_path = temp_dir / file.filename
    output_path = temp_dir / f"processed_{file.filename}"

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process
        from ..processor import ProcessingPipeline

        pipeline = ProcessingPipeline(config)
        step_list = steps.split(",") if steps else None
        success = pipeline.process(input_path, output_path, step_list)

        if success and output_path.exists():
            return FileResponse(
                path=output_path,
                media_type="application/octet-stream",
                filename=output_path.name,
            )
        else:
            raise HTTPException(status_code=500, detail="Processing failed")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup input file
        if input_path.exists():
            input_path.unlink()


# Batch process endpoint
@app.post("/batch")
async def batch_process(request: BatchProcessRequest, background_tasks: BackgroundTasks):
    """Process multiple files in directory"""
    setup_logging()
    config = load_config()

    input_dir = Path(request.input_dir)
    if not input_dir.exists():
        raise HTTPException(status_code=404, detail="Input directory not found")

    output_dir = Path(request.output_dir) if request.output_dir else Path(tempfile.gettempdir())
    output_dir.mkdir(parents=True, exist_ok=True)

    def process_files():
        from ..processor import ProcessingPipeline

        pipeline = ProcessingPipeline(config)
        files = list(input_dir.glob(request.pattern))

        for file_path in files:
            if file_path.is_file():
                output_path = output_dir / f"processed_{file_path.name}"
                try:
                    pipeline.process(file_path, output_path)
                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {e}")

    # Run in background
    background_tasks.add_task(process_files)

    return {
        "message": f"Processing {len(list(input_dir.glob(request.pattern)))} files",
        "output_dir": str(output_dir),
    }


# Plugin endpoints
@app.get("/plugins")
async def list_plugins():
    """List available plugins"""
    from ..plugins import get_plugin_registry

    registry = get_plugin_registry()

    return {
        "detectors": registry.list_detectors(),
        "processors": registry.list_processors(),
        "transformers": registry.list_transformers(),
        "analyzers": registry.list_analyzers(),
    }


# GPU info endpoint
@app.get("/gpu")
async def get_gpu_info():
    """Get GPU information"""
    from ..utils import get_gpu_helper

    helper = get_gpu_helper()
    return helper.get_info()


# Run server with: uvicorn video_detect.api:app --host 0.0.0.0 --port 8000
