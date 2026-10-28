"""
orchestrator.py
━━━━━━━━━━━━━━━
Autonomous Orchestrator — coordinates all agents in sequence.

Pipeline:
  1. ScraperAgent  → discover trending topics (HF + DevPost)
  2. ContentAgent  → generate full content package from best topic
  3. ImageAgent    → generate scene images (SDXL)
  4. VoiceAgent    → generate narration (ElevenLabs)
  5. VideoAgent    → build final video (MoviePy + Motif-Video hybrid)
  6. YouTubeAgent  → upload to YouTube (optional, if credentials set)

Fully autonomous: pass topic=None to let ScraperAgent pick the topic.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from agents.content_agent  import ContentAgent
from agents.image_agent    import ImageAgent
from agents.scraper_agent  import ScraperAgent, Topic
from agents.video_agent    import VideoAgent
from agents.voice_agent    import VoiceAgent
from agents.youtube_agent  import YouTubeAgent
from utils.config          import LOGS_DIR, YOUTUBE_REFRESH_TOKEN
from utils.logger          import get_logger

_log = get_logger("Orchestrator")


@dataclass
class PipelineState:
    job_id:       str
    topic:        str                      = ""
    status:       str                      = "pending"   # pending|running|done|error
    content:      Optional[Dict[str, Any]] = None
    image_paths:  List[str]                = field(default_factory=list)
    audio_path:   Optional[str]            = None
    video_path:   Optional[str]            = None
    youtube_url: Optional[str] = None
    youtube_status: str = "not_requested"
    youtube_error: Optional[str] = None
    scraped_topics: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    started_at:   float                    = field(default_factory=time.time)
    elapsed_s:    float                    = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def snapshot(self):
        path = LOGS_DIR / f"{self.job_id}_state.json"
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2, default=str)


class Orchestrator:
    def __init__(self, job_id: str):
        self.state = PipelineState(job_id=job_id)
        _log.info("Orchestrator ready (job_id=%s)", job_id)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _err(self, stage: str, exc: Exception):
        msg = f"[{stage}] {type(exc).__name__}: {exc}"
        _log.error(msg)
        self.state.errors.append(msg)

    def _run_required_stage(self, stage: str, stage_fn: Any) -> bool:
        errors_before = len(self.state.errors)

        try:
            succeeded = bool(stage_fn())
        except Exception as exc:
            self._err(stage, exc)
            return False

        if not succeeded and len(self.state.errors) == errors_before:
            msg = f"[{stage}] Stage returned failure without recording an error."
            _log.error(msg)
            self.state.errors.append(msg)

        return succeeded

    def _free_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
            _log.info("GPU cache cleared. Free VRAM: %.1f GB", free / 1e9)

    # ── Stage 0: Scrape (autonomous topic discovery) ──────────────────────────
    def _stage_scrape(self) -> Optional[str]:
        """Returns the best topic string, or None if scraping failed."""
        _log.info("▶ Stage 0/6 — ScraperAgent (HuggingFace + DevPost)")
        try:
            topics = ScraperAgent().run()
            self.state.scraped_topics = [t.to_dict() for t in topics]
            if not topics:
                _log.warning("No topics scraped — pipeline will need a manual topic.")
                return None
            best = topics[0].as_pipeline_topic()
            _log.info("ScraperAgent ✓  best topic: '%s'", best[:80])
            return best
        except Exception as exc:
            self._err("ScraperAgent", exc)
            return None

    # ── Stage 1: Content generation ───────────────────────────────────────────
    def _stage_content(self, topic: str) -> bool:
        _log.info("▶ Stage 1/6 — ContentAgent (Groq / LLaMA 3.1)")
        try:
            self.state.content = ContentAgent().run(topic)
            self.state.topic   = topic
            _log.info("ContentAgent ✓  title='%s'", self.state.content["title"])
            return True
        except Exception as exc:
            self._err("ContentAgent", exc)
            return False

    # ── Stage 2: Image generation ─────────────────────────────────────────────
    def _stage_images(self) -> bool:
        _log.info("▶ Stage 2/6 — ImageAgent (SDXL via HF InferenceClient)")
        if not self.state.content:
            _log.error("Skipping images — no content.")
            return False
        try:
            prompts = self.state.content.get("image_prompts", [])
            paths   = ImageAgent().run(prompts)
            self.state.image_paths = [str(p) for p in paths]
            self._free_gpu()
            _log.info("ImageAgent ✓  %d images", len(paths))
            return bool(paths)
        except Exception as exc:
            self._err("ImageAgent", exc)
            self._free_gpu()
            return False

    # ── Stage 3: Voice generation ─────────────────────────────────────────────
    def _stage_voice(self) -> bool:
        _log.info("▶ Stage 3/6 — VoiceAgent (ElevenLabs)")
        if not self.state.content:
            _log.error("Skipping voice — no content.")
            return False
        try:
            script = self.state.content.get("voice_script") or self.state.content.get("script", "")
            path   = VoiceAgent().run(script)
            self.state.audio_path = str(path)
            _log.info("VoiceAgent ✓  %s", path.name)
            return True
        except Exception as exc:
            self._err("VoiceAgent", exc)
            return False

    # ── Stage 4: Video production ─────────────────────────────────────────────

    def _stage_video(self) -> bool:
        _log.info("▶ Stage 4/6 — VideoAgent (MoviePy + Motif-Video hybrid)")

        if not self.state.image_paths:
            _log.error("Skipping video — no images.")
            return False

        try:
            title = (self.state.content or {}).get("title", "AI Video")
            images = [Path(p) for p in self.state.image_paths]
            audio = Path(self.state.audio_path) if self.state.audio_path else None

            path = VideoAgent().run(
                images,
                audio,
                title,
                self.state.content,
            )

            if not path:
                _log.error("VideoAgent returned no output path.")
                return False

            video_path = Path(path)

            if not video_path.is_file():
                _log.error(
                    "VideoAgent returned a path that does not exist: %s",
                    video_path,
                )
                return False

            if video_path.stat().st_size == 0:
                _log.error(
                    "VideoAgent produced an empty video file: %s",
                    video_path,
                )
                return False

            self.state.video_path = str(video_path)
            _log.info(
                "VideoAgent ✓ %s (%d bytes)",
                video_path.name,
                video_path.stat().st_size,
            )
            return True

        except Exception as exc:
            self._err("VideoAgent", exc)
            return False


    # ── Stage 5: YouTube upload ───────────────────────────────────────────────
    def _stage_youtube(self) -> bool:
        _log.info("▶ Stage 5/6 — YouTubeAgent")

        if not self.state.video_path:
            message = "Cannot upload: no video path is available."
            self.state.youtube_status = "failed"
            self.state.youtube_error = message
            _log.error(message)
            return False

        if not YOUTUBE_REFRESH_TOKEN:
            self.state.youtube_status = "skipped_no_token"
            self.state.youtube_error = None
            _log.warning("No YOUTUBE_REFRESH_TOKEN set — skipping upload.")
            return False

        try:
            content = self.state.content or {}
            url = YouTubeAgent().upload(
                video_path=Path(self.state.video_path),
                title=content.get("title") or self.state.topic[:100],
                description=content.get("description", ""),
                tags=content.get("tags", []),
                privacy="public",
            )

            if not url:
                raise RuntimeError("YouTubeAgent returned no URL.")

            self.state.youtube_url = url
            self.state.youtube_status = "uploaded"
            self.state.youtube_error = None
            _log.info("YouTubeAgent ✓ %s", url)
            return True

        except Exception as exc:
            self.state.youtube_status = "failed"
            self.state.youtube_error = f"{type(exc).__name__}: {exc}"
            _log.error("[YouTubeAgent] %s", self.state.youtube_error)
            return False

    # ── Main run ──────────────────────────────────────────────────────────────
    def run(
        self,
        topic: Optional[str] = None,
        upload_to_youtube: bool = True,
    ) -> PipelineState:
        self.state.status = "running"
        self.state.youtube_status = (
            "not_attempted" if upload_to_youtube else "not_requested"
        )
        self.state.youtube_error = None
        _log.info("═" * 55)
        _log.info("  PIPELINE START  (job_id=%s)", self.state.job_id)
        _log.info("═" * 55)

        # Step 0: Auto-discover topic if none given
        if not topic:
            topic = self._stage_scrape()
        else:
            _log.info("Using manual topic: '%s'", topic[:80])

        if not topic:
            topic = "Latest AI and Machine Learning Breakthroughs 2025"
            _log.warning("Falling back to default topic: '%s'", topic)

        self.state.snapshot()

                # Steps 1–4: Stop when a required stage fails.
        required_stages = [
            ("ContentAgent", lambda: self._stage_content(topic)),
            ("ImageAgent", self._stage_images),
            ("VoiceAgent", self._stage_voice),
            ("VideoAgent", self._stage_video),
        ]

        for stage_name, stage_fn in required_stages:
            if not self._run_required_stage(stage_name, stage_fn):
                break
            self.state.snapshot()

        # Optional upload is only attempted after required stages succeed.
        if upload_to_youtube and not self.state.errors:
            self._stage_youtube()
            self.state.snapshot()

        # Finalise
        self.state.elapsed_s = round(time.time() - self.state.started_at, 1)
        self.state.status    = "error" if self.state.errors else "done"

        _log.info("═" * 55)
        if self.state.status == "error":
            _log.error(
                "PIPELINE FAILED with %d error(s) in %ss",
                len(self.state.errors),
                self.state.elapsed_s,
            )
        else:
            _log.info("PIPELINE COMPLETE ✓ in %ss", self.state.elapsed_s)
        _log.info("═" * 55)

        self.state.snapshot()
        return self.state
