"""
api/app.py
━━━━━━━━━━
FastAPI application exposing all pipeline endpoints.

Endpoints:
  GET  /health                    — liveness check
  GET  /scrape/topics             — scrape HF + DevPost, return topic list
  POST /pipeline/run              — start full pipeline (async background job)
  GET  /pipeline/status/{job_id}  — check job status / get results
  GET  /pipeline/jobs             — list all jobs
  POST /youtube/upload            — upload an existing video to YouTube
  GET  /outputs/{filename}        — serve output files (video, audio, images)
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.scraper_agent import ScraperAgent
from agents.youtube_agent import YouTubeAgent
from orchestrator import Orchestrator, PipelineState
from utils.config import BASE_DIR, VIDEO_DIR, AUDIO_DIR, IMAGES_DIR
from utils.logger import get_logger

_log  = get_logger("API")
app   = FastAPI(
    title="🤖 Autonomous AI Content System",
    description=(
        "Multi-agent pipeline: scrape trending topics → generate AI video → upload to YouTube."
    ),
    version="2.0.0",
)

# ── In-memory job store (replace with Redis/DB for production) ────────────────
_JOBS: Dict[str, PipelineState] = {}


# ── Pydantic models ───────────────────────────────────────────────────────────
class PipelineRequest(BaseModel):
    topic: Optional[str] = Field(
        default=None,
        description="Topic to generate content for. Leave empty for autonomous topic discovery.",
        example="The rise of multimodal AI models",
    )
    upload_to_youtube: bool = Field(
        default=True,
        description="Whether to upload the finished video to YouTube.",
    )


class YouTubeUploadRequest(BaseModel):
    video_path:   str            = Field(..., description="Absolute path to the MP4 file")
    title:        str            = Field(..., description="YouTube video title (max 100 chars)")
    description:  str            = Field(default="", description="Video description")
    tags:         List[str]      = Field(default=[], description="YouTube tags")
    privacy:      str            = Field(default="public", description="public | private | unlisted")
    category_id:  str            = Field(default="28", description="YouTube category ID")


class ScrapeResponse(BaseModel):
    total:  int
    topics: List[Dict[str, Any]]


# ── Background task ───────────────────────────────────────────────────────────
def _run_pipeline(job_id: str, topic: Optional[str], upload_to_youtube: bool):
    """Runs the full orchestrator pipeline in a background thread."""
    try:
        orch  = Orchestrator(job_id=job_id)
        state = orch.run(topic=topic, upload_to_youtube=upload_to_youtube)
        _JOBS[job_id] = state
    except Exception as exc:
        _log.error("Background pipeline job %s crashed: %s", job_id, exc)
        if job_id in _JOBS:
            _JOBS[job_id].status = "error"
            _JOBS[job_id].errors.append(str(exc))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Liveness check."""
    return {"status": "ok", "version": "2.0.0"}


@app.get("/scrape/topics", response_model=ScrapeResponse, tags=["Scraper"])
def scrape_topics(
    hf_limit:      int = Query(default=10, ge=1, le=30, description="Max HuggingFace topics"),
    devpost_limit: int = Query(default=10, ge=1, le=30, description="Max DevPost topics"),
):
    """
    Scrape trending AI/ML topics from HuggingFace (Spaces + Papers)
    and DevPost (hackathon projects). Returns ranked topic list.
    """
    _log.info("Scrape request: hf=%d, devpost=%d", hf_limit, devpost_limit)
    topics = ScraperAgent().run(hf_limit=hf_limit, devpost_limit=devpost_limit)
    return ScrapeResponse(
        total=len(topics),
        topics=[t.to_dict() for t in topics],
    )


@app.post("/pipeline/run", tags=["Pipeline"])

def _run_pipeline(job_id: str, topic: Optional[str], upload_to_youtube: bool):
    """Runs the full orchestrator pipeline in a background thread."""
    try:
        orch = Orchestrator(job_id=job_id)
        state = orch.run(topic=topic, upload_to_youtube=upload_to_youtube)
        _JOBS[job_id] = state

    except Exception as exc:
        _log.exception("Background pipeline job %s crashed", job_id)

        state = _JOBS.get(job_id)
        if state is not None:
            state.status = "error"
            state.elapsed_s = round(time.time() - state.started_at, 1)
            state.errors.append(f"{type(exc).__name__}: {exc}")

            try:
                state.snapshot()
            except Exception:
                _log.exception(
                    "Could not persist failed pipeline job %s",
                    job_id,
                )



@app.get("/pipeline/status/{job_id}", tags=["Pipeline"])
def pipeline_status(job_id: str):
    """Retrieve the current status and results of a pipeline job."""
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _JOBS[job_id].to_dict()


@app.get("/pipeline/jobs", tags=["Pipeline"])
def list_jobs():
    """List all pipeline jobs and their statuses."""
    return [
        {"job_id": jid, "status": s.status, "topic": s.topic, "elapsed_s": s.elapsed_s}
        for jid, s in _JOBS.items()
    ]


@app.post("/youtube/upload", tags=["YouTube"])
def youtube_upload(req: YouTubeUploadRequest):
    """
    Upload an existing MP4 file to YouTube directly.

    Useful when you already have a video file and just want to publish it.
    Requires YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN in env.
    """
    video_path = Path(req.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {req.video_path}")

    _log.info("YouTube upload request: '%s'", req.title)
    try:
        url = YouTubeAgent().upload(
            video_path  = video_path,
            title       = req.title,
            description = req.description,
            tags        = req.tags,
            privacy     = req.privacy,
            category_id = req.category_id,
        )
        return {"youtube_url": url, "status": "uploaded"}
    except Exception as exc:
        _log.error("YouTube upload failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/outputs/video/{filename}", tags=["Outputs"])
def serve_video(filename: str):
    """Download a generated video file."""
    path = VIDEO_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(str(path), media_type="video/mp4", filename=filename)


@app.get("/outputs/audio/{filename}", tags=["Outputs"])
def serve_audio(filename: str):
    """Download a generated audio file."""
    path = AUDIO_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(str(path), media_type="audio/mpeg", filename=filename)


@app.get("/outputs/images/{filename}", tags=["Outputs"])
def serve_image(filename: str):
    """Download a generated image file."""
    path = IMAGES_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(str(path), media_type="image/png", filename=filename)
