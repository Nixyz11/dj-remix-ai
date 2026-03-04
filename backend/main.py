"""DJ Remix AI — FastAPI Backend"""

import logging
import os
import sys
import uuid
import shutil
import traceback
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional

from stem_separator import StemSeparator
from audio_analysis import AudioAnalyzer
from remix_engine import RemixEngine
from song_downloader import SongDownloader

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dj-remix-ai")

# Resolve paths relative to project root (one level up from backend/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
MAX_FILE_SIZE_MB = 100

app = FastAPI(title="DJ Remix AI", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job tracking (replace with Redis/DB in production)
jobs: dict = {}

# Check if AI should be enabled (set USE_AI=0 to disable for faster dev)
USE_AI = os.environ.get("USE_AI", "1") != "0"
log.info("AI generation: %s", "ENABLED" if USE_AI else "DISABLED (set USE_AI=1 to enable)")

stem_separator = StemSeparator()
analyzer = AudioAnalyzer()
remix_engine = RemixEngine(use_ai=USE_AI)
downloader = SongDownloader()


# ── Models ──

class RemixRequest(BaseModel):
    url: Optional[str] = None
    remix_style: str = Field(default="party", pattern="^(party|edm|chill|bass_heavy)$")
    bpm_target: Optional[int] = Field(default=None, ge=60, le=200)
    intensity: float = Field(default=0.7, ge=0.0, le=1.0)
    use_ai: Optional[bool] = None  # Override: None=auto, True/False=force


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    result_url: Optional[str] = None
    analysis: Optional[dict] = None
    error: Optional[str] = None


class AnalysisResponse(BaseModel):
    bpm: float
    key: str
    mode: str
    key_full: str
    duration_seconds: float
    average_energy: Optional[float] = None
    sections: Optional[list] = None


# ── Remix Pipeline ──

def _update_job(job_id: str, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)


def process_remix(job_id: str, audio_path: str, request: RemixRequest):
    """Background task that runs the full remix pipeline."""
    try:
        def progress_cb(stage, progress):
            _update_job(job_id, status=stage, progress=0.1 + progress * 0.85)

        _update_job(job_id, status="separating_stems", progress=0.05)
        stems = stem_separator.separate(audio_path, str(OUTPUT_DIR / job_id / "stems"))

        _update_job(job_id, status="analyzing", progress=0.25)
        analysis = analyzer.analyze(audio_path)
        _update_job(job_id, analysis=analysis)

        _update_job(job_id, status="remixing", progress=0.35)
        output_path = str(OUTPUT_DIR / job_id / "remix.wav")
        remix_engine.create_remix(
            stems=stems,
            analysis=analysis,
            style=request.remix_style,
            bpm_target=request.bpm_target,
            intensity=request.intensity,
            output_path=output_path,
            use_ai=request.use_ai,
            progress_callback=progress_cb,
        )

        _update_job(job_id, status="done", progress=1.0,
                    result_url=f"/download/{job_id}")
        log.info("Job %s completed successfully", job_id)

    except Exception as e:
        tb = traceback.format_exc()
        log.error("Job %s failed: %s\n%s", job_id, e, tb)
        _update_job(job_id, status="error", error=str(e))


def _validate_upload(file: UploadFile) -> str:
    """Validate uploaded file and return sanitized filename."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return file.filename


# ── Endpoints ──

@app.post("/remix/upload", response_model=JobStatus)
async def remix_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    remix_style: str = "party",
    bpm_target: Optional[int] = None,
    intensity: float = 0.7,
    use_ai: Optional[bool] = None,
):
    """Upload a song file and start remixing."""
    filename = _validate_upload(file)

    job_id = str(uuid.uuid4())[:8]
    (OUTPUT_DIR / job_id).mkdir(parents=True, exist_ok=True)

    audio_path = str(UPLOAD_DIR / f"{job_id}_{filename}")
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        os.remove(audio_path)
        raise HTTPException(status_code=400, detail=f"File too large ({file_size_mb:.0f}MB). Max: {MAX_FILE_SIZE_MB}MB")

    log.info("Job %s: uploaded %s (%.1fMB), style=%s, intensity=%.1f",
             job_id, filename, file_size_mb, remix_style, intensity)

    request = RemixRequest(
        remix_style=remix_style, bpm_target=bpm_target,
        intensity=intensity, use_ai=use_ai,
    )
    jobs[job_id] = {"status": "pending", "progress": 0.0}
    background_tasks.add_task(process_remix, job_id, audio_path, request)

    return JobStatus(job_id=job_id, status="pending")


@app.post("/remix/url", response_model=JobStatus)
async def remix_from_url(
    background_tasks: BackgroundTasks,
    request: RemixRequest,
):
    """Download a song from URL and start remixing."""
    if not request.url:
        raise HTTPException(status_code=400, detail="URL is required")

    job_id = str(uuid.uuid4())[:8]
    (OUTPUT_DIR / job_id).mkdir(parents=True, exist_ok=True)

    log.info("Job %s: downloading from URL, style=%s", job_id, request.remix_style)

    try:
        audio_path = downloader.download(request.url, str(UPLOAD_DIR), job_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")

    jobs[job_id] = {"status": "pending", "progress": 0.0}
    background_tasks.add_task(process_remix, job_id, audio_path, request)

    return JobStatus(job_id=job_id, status="pending")


@app.post("/analyze/upload", response_model=AnalysisResponse)
async def analyze_upload(file: UploadFile = File(...)):
    """Upload a song and get analysis (BPM, key, etc.) without remixing."""
    filename = _validate_upload(file)

    temp_path = str(UPLOAD_DIR / f"analyze_{uuid.uuid4().hex[:6]}_{filename}")
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        result = analyzer.analyze(temp_path)
        return AnalysisResponse(**result)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Check remix job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0),
        result_url=job.get("result_url"),
        analysis=job.get("analysis"),
        error=job.get("error"),
    )


@app.get("/download/{job_id}")
async def download_remix(job_id: str):
    """Download the finished remix."""
    remix_path = OUTPUT_DIR / job_id / "remix.wav"
    if not remix_path.exists():
        raise HTTPException(status_code=404, detail="Remix not ready or not found")
    return FileResponse(str(remix_path), media_type="audio/wav",
                        filename=f"remix_{job_id}.wav")


@app.get("/download/{job_id}/stems/{stem_name}")
async def download_stem(job_id: str, stem_name: str):
    """Download an individual stem (vocals, drums, bass, other)."""
    if stem_name not in ("vocals", "drums", "bass", "other"):
        raise HTTPException(status_code=400, detail="Invalid stem name")

    stems_dir = OUTPUT_DIR / job_id / "stems"
    if not stems_dir.exists():
        raise HTTPException(status_code=404, detail="Stems not found")

    # Check direct path first (Python API output)
    stem_file = stems_dir / f"{stem_name}.wav"
    if not stem_file.exists():
        # Fallback: check CLI output structure (htdemucs/trackname/stem.wav)
        for candidate in stems_dir.rglob(f"{stem_name}.wav"):
            stem_file = candidate
            break

    if not stem_file.exists():
        raise HTTPException(status_code=404, detail=f"Stem '{stem_name}' not found")

    return FileResponse(str(stem_file), media_type="audio/wav",
                        filename=f"{stem_name}_{job_id}.wav")


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its output files."""
    job_dir = OUTPUT_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    jobs.pop(job_id, None)
    return {"deleted": job_id}


@app.get("/jobs")
async def list_jobs():
    """List all jobs and their statuses."""
    return {
        job_id: {
            "status": job["status"],
            "progress": job.get("progress", 0),
            "has_result": job.get("result_url") is not None,
        }
        for job_id, job in jobs.items()
    }


@app.get("/styles")
async def list_styles():
    """List available remix styles and their settings."""
    return {
        name: {
            "bpm_range": preset["bpm_range"],
            "energy_curve": preset["energy_curve"],
            "ai_elements": preset["ai_elements"],
        }
        for name, preset in RemixEngine.STYLE_PRESETS.items()
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "0.2.0",
        "ai_enabled": USE_AI,
        "device": remix_engine.ai_gen.device,
    }
