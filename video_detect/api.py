"""REST API for Video Detect Pro"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Security, Depends, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader, APIKeyCookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from pathlib import Path
import tempfile
import shutil
import logging
import os
import secrets
import hashlib
from contextlib import contextmanager
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

# Security configuration
API_KEY_NAME = "X-API-Key"
API_KEY_HEADER = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
API_KEY_COOKIE = APIKeyCookie(name=API_KEY_NAME, auto_error=False)

# Rate limiting storage (in production, use Redis or similar)
rate_limit_store = defaultdict(list)
RATE_LIMIT_REQUESTS = 100  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

# File size limits (in bytes)
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}

# Allowed paths for file operations
ALLOWED_BASE_PATHS = [
    Path.home() / "Videos",
    Path.home() / "Pictures",
    Path.home() / "Downloads",
    Path("/tmp"),
    Path.cwd() / "downloads",
]


def get_api_key(
    api_key_header: str = Security(API_KEY_HEADER),
    api_key_cookie: str = Security(API_KEY_COOKIE),
) -> str:
    """Validate API key from header or cookie"""
    api_key = api_key_header or api_key_cookie

    # Get allowed API keys from environment or config
    allowed_keys = os.environ.get("VIDEO_DETECT_API_KEYS", "").split(",")

    # If no keys are configured, allow access (development mode)
    if not allowed_keys or allowed_keys == [""]:
        if os.environ.get("ENVIRONMENT") == "production":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required in production",
            )
        return "dev_mode"

    # Validate API key
    if api_key not in allowed_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    return api_key


def validate_file_path(file_path: Path) -> Path:
    """Validate file path to prevent path traversal attacks"""
    try:
        # Resolve to absolute path and check for symlinks
        resolved_path = file_path.resolve(strict=True)

        # Check if path is within allowed directories
        is_allowed = any(
            str(resolved_path).startswith(str(allowed_path.resolve()))
            for allowed_path in ALLOWED_BASE_PATHS
        )

        if not is_allowed:
            logger.warning(f"Access denied to path: {resolved_path}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access to path not allowed: {file_path}",
            )

        return resolved_path
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found",
        )
    except Exception as e:
        logger.error(f"Path validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path",
        )


def validate_directory_path(dir_path: Path) -> Path:
    """Validate directory path to prevent path traversal attacks"""
    try:
        # Resolve to absolute path
        resolved_path = dir_path.resolve()

        # Check if path is within allowed directories
        is_allowed = any(
            str(resolved_path).startswith(str(allowed_path.resolve()))
            for allowed_path in ALLOWED_BASE_PATHS
        )

        if not is_allowed:
            logger.warning(f"Access denied to directory: {resolved_path}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access to directory not allowed: {dir_path}",
            )

        return resolved_path
    except Exception as e:
        logger.error(f"Directory validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid directory path",
        )


def validate_file_extension(filename: str) -> None:
    """Validate file extension"""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS and ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {ext}. Allowed: {ALLOWED_VIDEO_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS}",
        )


def check_rate_limit(client_id: str) -> None:
    """Check rate limit for client"""
    now = datetime.now()

    # Clean old entries
    rate_limit_store[client_id] = [
        timestamp for timestamp in rate_limit_store[client_id]
        if (now - timestamp).total_seconds() < RATE_LIMIT_WINDOW
    ]

    # Check if limit exceeded
    if len(rate_limit_store[client_id]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds",
        )

    # Add current request
    rate_limit_store[client_id].append(now)


def get_client_id(api_key: str = Depends(get_api_key)) -> str:
    """Get client identifier for rate limiting"""
    # Hash the API key to use as client ID
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


app = FastAPI(
    title="Video Detect Pro API",
    description="REST API for watermark and fingerprint detection/removal",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware in production
if os.environ.get("ENVIRONMENT") == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1").split(","),
    )


# Request/Response models
class DetectRequest(BaseModel):
    file_path: str = Field(..., description="Path to the file to analyze")
    analyze_fingerprint: bool = Field(default=False, description="Also analyze for fingerprints")

    @field_validator("file_path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("file_path cannot be empty")
        # Prevent obvious path traversal attempts
        if ".." in v or v.startswith("/"):
            # Allow absolute paths but log warning
            if ".." in v:
                raise ValueError("Path traversal not allowed")
        return v


class ProcessRequest(BaseModel):
    file_path: str = Field(..., description="Path to the file to process")
    output_path: Optional[str] = Field(None, description="Output file path")
    steps: Optional[List[str]] = Field(None, description="Processing steps to apply")

    @field_validator("file_path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("file_path cannot be empty")
        if ".." in v:
            raise ValueError("Path traversal not allowed")
        return v

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            valid_steps = {
                "metadata_clean", "fingerprint_remove", "watermark_remove",
                "transform_video", "transform_image", "content_inpaint"
            }
            invalid = set(v) - valid_steps
            if invalid:
                raise ValueError(f"Invalid steps: {invalid}. Valid: {valid_steps}")
        return v


class BatchProcessRequest(BaseModel):
    input_dir: str = Field(..., description="Input directory path")
    output_dir: Optional[str] = Field(None, description="Output directory path")
    pattern: str = Field(default="*", description="File pattern to match")

    @field_validator("input_dir", "output_dir")
    @classmethod
    def validate_path(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            return v
        if not v or v.strip() == "":
            raise ValueError(f"{info.field_name} cannot be empty")
        if ".." in v:
            raise ValueError("Path traversal not allowed")
        return v

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: str) -> str:
        # Prevent shell injection in pattern
        dangerous_chars = {"$", ";", "&", "|", "`", "(", ")", "\n", "\r"}
        if any(char in v for char in dangerous_chars):
            raise ValueError("Invalid characters in pattern")
        return v


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
        from video_detect.utils import load_config
        config_cache = load_config()
    return config_cache


def setup_logging():
    from video_detect.utils import setup_logging as sl
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
async def detect_watermark(
    request: DetectRequest,
    api_key: str = Depends(get_api_key),
    client_id: str = Depends(get_client_id),
):
    """Detect watermarks and fingerprints"""
    # Rate limiting
    check_rate_limit(client_id)

    setup_logging()
    config = load_config()

    # Validate and sanitize file path
    try:
        file_path = validate_file_path(Path(request.file_path))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Path validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path",
        )

    # Validate file extension
    validate_file_extension(str(file_path))

    try:
        from video_detect.detector import WatermarkDetector

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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Detection processing failed",
        )


# Process endpoint
@app.post("/process", response_model=ProcessResponse)
async def process_file(
    request: ProcessRequest,
    api_key: str = Depends(get_api_key),
    client_id: str = Depends(get_client_id),
):
    """Process file to remove watermarks/fingerprints"""
    # Rate limiting
    check_rate_limit(client_id)

    setup_logging()
    config = load_config()

    # Validate and sanitize input path
    try:
        input_path = validate_file_path(Path(request.file_path))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Input path validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input file path",
        )

    # Validate output path if provided
    output_path = None
    if request.output_path:
        try:
            # Validate output path is in allowed directories
            out = Path(request.output_path).resolve()
            is_allowed = any(
                str(out.parent).startswith(str(allowed_path.resolve()))
                for allowed_path in ALLOWED_BASE_PATHS
            )
            if not is_allowed:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Output path not in allowed directory",
                )
            output_path = out
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Output path validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid output file path",
            )

    try:
        from video_detect.processor import ProcessingPipeline

        pipeline = ProcessingPipeline(config)
        success = pipeline.process(input_path, output_path, request.steps)

        final_output_path = output_path
        if success and final_output_path is None:
            output_dir = Path(config.get("output", {}).get("output_dir", "./downloads/processed"))
            output_dir.mkdir(parents=True, exist_ok=True)
            final_output_path = output_dir / f"processed_{input_path.name}"

        return ProcessResponse(
            success=success,
            output_path=str(final_output_path) if final_output_path else None,
            message="Processing completed successfully" if success else "Processing failed",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Processing failed",
        )


# Upload and detect endpoint
@app.post("/upload/detect")
async def upload_detect(
    file: UploadFile = File(...),
    analyze_fingerprint: bool = False,
    api_key: str = Depends(get_api_key),
    client_id: str = Depends(get_client_id),
):
    """Upload file and detect watermarks"""
    # Rate limiting
    check_rate_limit(client_id)

    setup_logging()
    config = load_config()

    # Validate filename
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )

    # Sanitize filename to prevent path traversal
    safe_filename = secrets.token_hex(8) + "_" + Path(file.filename).name
    safe_filename = safe_filename.replace("..", "").replace("/", "").replace("\\", "")

    # Validate file extension
    validate_file_extension(safe_filename)

    # Create temp directory
    temp_dir = Path(tempfile.gettempdir()) / "video_detect_upload"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / safe_filename

    try:
        # Check file size
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB",
            )

        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            buffer.write(content)

        # Detect
        from video_detect.detector import WatermarkDetector

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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Detection failed",
        )
    finally:
        # Cleanup
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")


# Upload and process endpoint
@app.post("/upload/process")
async def upload_process(
    file: UploadFile = File(...),
    steps: Optional[str] = None,
    api_key: str = Depends(get_api_key),
    client_id: str = Depends(get_client_id),
):
    """Upload file and process it"""
    # Rate limiting
    check_rate_limit(client_id)

    setup_logging()
    config = load_config()

    # Validate filename
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )

    # Sanitize filename
    safe_filename = secrets.token_hex(8) + "_" + Path(file.filename).name
    safe_filename = safe_filename.replace("..", "").replace("/", "").replace("\\", "")

    # Validate file extension
    validate_file_extension(safe_filename)

    # Create temp directory
    temp_dir = Path(tempfile.gettempdir()) / "video_detect_upload"
    temp_dir.mkdir(parents=True, exist_ok=True)
    input_path = temp_dir / safe_filename
    output_path = temp_dir / f"processed_{safe_filename}"

    try:
        # Check file size
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB",
            )

        # Save uploaded file
        with open(input_path, "wb") as buffer:
            buffer.write(content)

        # Validate and parse steps
        step_list = None
        if steps:
            step_list = steps.split(",")
            valid_steps = {
                "metadata_clean", "fingerprint_remove", "watermark_remove",
                "transform_video", "transform_image", "content_inpaint"
            }
            invalid = set(step_list) - valid_steps
            if invalid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid steps: {invalid}",
                )

        # Process
        from video_detect.processor import ProcessingPipeline

        pipeline = ProcessingPipeline(config)
        success = pipeline.process(input_path, output_path, step_list)

        if success and output_path.exists():
            return FileResponse(
                path=output_path,
                media_type="application/octet-stream",
                filename=f"processed_{file.filename}",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Processing failed",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Processing failed",
        )
    finally:
        # Cleanup input file (output file is sent to user)
        if input_path.exists():
            try:
                input_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {input_path}: {e}")


# Batch process endpoint
@app.post("/batch")
async def batch_process(
    request: BatchProcessRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key),
    client_id: str = Depends(get_client_id),
):
    """Process multiple files in directory"""
    # Rate limiting
    check_rate_limit(client_id)

    setup_logging()
    config = load_config()

    # Validate and sanitize input directory
    try:
        input_dir = validate_directory_path(Path(request.input_dir))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Input directory validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input directory",
        )

    if not input_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Input directory not found",
        )

    if not input_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input path is not a directory",
        )

    # Validate output directory
    output_dir: Path
    if request.output_dir:
        try:
            output_dir = validate_directory_path(Path(request.output_dir))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Output directory validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid output directory",
            )
    else:
        output_dir = Path(tempfile.gettempdir()) / "video_detect_batch_output"
        output_dir.mkdir(parents=True, exist_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    def process_files():
        from video_detect.processor import ProcessingPipeline

        pipeline = ProcessingPipeline(config)
        files = list(input_dir.glob(request.pattern))

        for file_path in files:
            if file_path.is_file():
                output_path = output_dir / f"processed_{file_path.name}"
                try:
                    pipeline.process(file_path, output_path)
                    logger.info(f"Successfully processed {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {e}")

    # Run in background
    background_tasks.add_task(process_files)

    return {
        "message": f"Processing {len(list(input_dir.glob(request.pattern)))} files",
        "output_dir": str(output_dir),
        "pattern": request.pattern,
    }


# Plugin endpoints
@app.get("/plugins")
async def list_plugins(
    api_key: str = Depends(get_api_key),
    client_id: str = Depends(get_client_id),
):
    """List available plugins"""
    # Rate limiting
    check_rate_limit(client_id)

    try:
        from video_detect.plugins import get_plugin_registry

        registry = get_plugin_registry()

        return {
            "detectors": registry.list_detectors(),
            "processors": registry.list_processors(),
            "transformers": registry.list_transformers(),
            "analyzers": registry.list_analyzers(),
        }
    except Exception as e:
        logger.error(f"Failed to list plugins: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve plugin information",
        )


# GPU info endpoint
@app.get("/gpu")
async def get_gpu_info(
    api_key: str = Depends(get_api_key),
    client_id: str = Depends(get_client_id),
):
    """Get GPU information"""
    # Rate limiting
    check_rate_limit(client_id)

    try:
        from video_detect.utils import get_gpu_helper

        helper = get_gpu_helper()
        return helper.get_info()
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve GPU information",
        )


# Run server with: uvicorn video_detect.api:app --host 0.0.0.0 --port 8000
