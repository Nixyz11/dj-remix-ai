"""DJ Remix AI — FastAPI Backend"""

import os
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from stem_separator import StemSeparator
from audio_analysis import AudioAnalyzer
from remix_engine import RemixEngine
from song_downloader import SongDownloader

app = FastAPI(title="DJ Remix AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("../uploads")
OUTPUT_DIR = Path("../output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job tracking (replace with Redis/DB in production)
jobs: dict = {}

stem_separator = StemSeparator()
analyzer = AudioAnalyzer()
remix_engine = RemixEngine()
downloader = SongDownloader()


class RemixRequest(BaseModel):
    url: Optional[str] = None
    remix_style: str = "party"  # party, chill, bass_heavy, edm
    bpm_target: Optional[int] = None
    intensity: float = 0.7  # 0.0 = subtle, 1.0 = heavy remix


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, separating_stems, analyzing, remixing, generating_ai, done, error
    progress: float = 0.0
    result_url: Optional[str] = None
    analysis: Optional[dict] = None
    error: Optional[str] = None


def process_remix(job_id: str, audio_path: str, request: RemixRequest):
    """Background task that runs the full remix pipeline."""
    try:
        jobs[job_id]["status"] = "separating_stems"
        jobs[job_id]["progress"] = 0.1
        stems = stem_separator.separate(audio_path, OUTPUT_DIR / job_id / "stems")

        jobs[job_id]["status"] = "analyzing"
        jobs[job_id]["progress"] = 0.3
        analysis = analyzer.analyze(audio_path)
        jobs[job_id]["analysis"] = analysis

        jobs[job_id]["status"] = "remixing"
        jobs[job_id]["progress"] = 0.5
        output_path = OUTPUT_DIR / job_id / "remix.wav"
        remix_engine.create_remix(
            stems=stems,
            analysis=analysis,
            style=request.remix_style,
            bpm_target=request.bpm_target,
            intensity=request.intensity,
            output_path=str(output_path),
        )

        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result_url"] = f"/download/{job_id}"

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@app.post("/remix/upload", response_model=JobStatus)
async def remix_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    remix_style: str = "party",
    bpm_target: Optional[int] = None,
    intensity: float = 0.7,
):
    """Upload a song file and start remixing."""
    job_id = str(uuid.uuid4())[:8]
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    audio_path = str(UPLOAD_DIR / f"{job_id}_{file.filename}")
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    request = RemixRequest(
        remix_style=remix_style,
        bpm_target=bpm_target,
        intensity=intensity,
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
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        audio_path = downloader.download(request.url, str(UPLOAD_DIR), job_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download: {e}")

    jobs[job_id] = {"status": "pending", "progress": 0.0}
    background_tasks.add_task(process_remix, job_id, audio_path, request)

    return JobStatus(job_id=job_id, status="pending")


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
    return FileResponse(str(remix_path), media_type="audio/wav", filename=f"remix_{job_id}.wav")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
