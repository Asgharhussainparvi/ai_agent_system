"""
agents/youtube_agent.py
━━━━━━━━━━━━━━━━━━━━━━━
YouTubeAgent — uploads the finished video to YouTube via the
YouTube Data API v3 using OAuth2 (refresh-token flow, no browser needed).

Pre-requisites:
  1. Create a Google Cloud project and enable "YouTube Data API v3".
  2. Create OAuth2 credentials (Desktop app type).
  3. Obtain a refresh token (run get_youtube_token.py once, then store it).
  4. Set YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN in .env
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from utils.config import (
    YOUTUBE_CLIENT_ID,
    YOUTUBE_CLIENT_SECRET,
    YOUTUBE_REFRESH_TOKEN,
)
from utils.logger import get_logger

_log = get_logger("YouTubeAgent")

_TOKEN_URL    = "https://oauth2.googleapis.com/token"
_UPLOAD_URL   = "https://www.googleapis.com/upload/youtube/v3/videos"
_CHUNK_SIZE   = 8 * 1024 * 1024   # 8 MB resumable upload chunks


class YouTubeAgent:
    """
    Uploads a local MP4 file to YouTube.

    Returns the YouTube video URL on success, or raises on failure.

    Usage:
        agent = YouTubeAgent()
        url   = agent.upload(
            video_path  = Path("/app/outputs/video/final_video.mp4"),
            title       = "LLMs Revolutionizing Software Engineering",
            description = "...",
            tags        = ["AI", "LLM", "Software Engineering"],
            category_id = "28",    # Science & Technology
            privacy     = "public",
        )
    """

    # ── OAuth token refresh ───────────────────────────────────────────────────
    def _get_access_token(self) -> str:
        _log.info("YouTubeAgent: refreshing OAuth2 access token…")
        resp = requests.post(
            _TOKEN_URL,
            data={
                "client_id":     YOUTUBE_CLIENT_ID,
                "client_secret": YOUTUBE_CLIENT_SECRET,
                "refresh_token": YOUTUBE_REFRESH_TOKEN,
                "grant_type":    "refresh_token",
            },
            timeout=30,
        )
        resp.raise_for_status()
        token = resp.json().get("access_token", "")
        if not token:
            raise RuntimeError(f"No access_token in response: {resp.text[:200]}")
        _log.info("Access token obtained ✓")
        return token

    # ── Resumable upload ──────────────────────────────────────────────────────
    def _start_resumable_session(
        self, access_token: str, metadata: dict, file_size: int
    ) -> str:
        """Initiate a resumable upload session, return upload URL."""
        resp = requests.post(
            _UPLOAD_URL,
            headers={
                "Authorization":  f"Bearer {access_token}",
                "Content-Type":   "application/json; charset=UTF-8",
                "X-Upload-Content-Type": "video/mp4",
                "X-Upload-Content-Length": str(file_size),
            },
            params={"uploadType": "resumable", "part": "snippet,status"},
            json=metadata,
            timeout=30,
        )
        resp.raise_for_status()
        upload_url = resp.headers.get("Location", "")
        if not upload_url:
            raise RuntimeError("No upload URL returned by YouTube API")
        return upload_url

    def _upload_chunks(self, upload_url: str, video_path: Path) -> Dict[str, Any]:
        """Stream the file in chunks to the resumable upload URL."""
        file_size = video_path.stat().st_size
        uploaded  = 0

        with open(video_path, "rb") as fh:
            while uploaded < file_size:
                chunk = fh.read(_CHUNK_SIZE)
                end   = uploaded + len(chunk) - 1

                resp = requests.put(
                    upload_url,
                    headers={
                        "Content-Range": f"bytes {uploaded}-{end}/{file_size}",
                        "Content-Type":  "video/mp4",
                    },
                    data=chunk,
                    timeout=120,
                )

                if resp.status_code in (200, 201):
                    _log.info("Upload complete ✓")
                    return resp.json()

                if resp.status_code == 308:   # Resume incomplete
                    range_header = resp.headers.get("Range", f"bytes=0-{end}")
                    uploaded = int(range_header.split("-")[1]) + 1
                    pct = uploaded / file_size * 100
                    _log.info("Uploaded %.1f%%…", pct)
                    continue

                resp.raise_for_status()

        raise RuntimeError("Upload loop exited without completion")

    # ── Main upload ───────────────────────────────────────────────────────────
    def upload(
        self,
        video_path: Path,
        title: str,
        description: str = "",
        tags: Optional[list] = None,
        category_id: str = "28",      # Science & Technology
        privacy: str = "public",      # "public" | "private" | "unlisted"
        thumbnail_path: Optional[Path] = None,
    ) -> str:
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        file_size = video_path.stat().st_size
        _log.info(
            "YouTubeAgent: uploading '%s' (%.1f MB, privacy=%s)…",
            title[:60], file_size / 1e6, privacy,
        )

        access_token = self._get_access_token()

        metadata = {
            "snippet": {
                "title":       title[:100],
                "description": description[:5000],
                "tags":        (tags or [])[:500],
                "categoryId":  category_id,
            },
            "status": {
                "privacyStatus":          privacy,
                "selfDeclaredMadeForKids": False,
            },
        }

        upload_url = self._start_resumable_session(access_token, metadata, file_size)
        result     = self._upload_chunks(upload_url, video_path)

        video_id  = result.get("id", "")
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        _log.info("YouTubeAgent ✓  Video live at: %s", video_url)

        # Optional: set thumbnail
        if thumbnail_path and thumbnail_path.exists() and video_id:
            self._set_thumbnail(access_token, video_id, thumbnail_path)

        return video_url

    # ── Thumbnail upload ──────────────────────────────────────────────────────
    def _set_thumbnail(self, access_token: str, video_id: str, thumb_path: Path):
        try:
            _log.info("Uploading thumbnail…")
            with open(thumb_path, "rb") as fh:
                resp = requests.post(
                    "https://www.googleapis.com/upload/youtube/v3/thumbnails/set",
                    headers={"Authorization": f"Bearer {access_token}"},
                    params={"videoId": video_id, "uploadType": "media"},
                    data=fh.read(),
                    timeout=60,
                )
            resp.raise_for_status()
            _log.info("Thumbnail uploaded ✓")
        except Exception as exc:
            _log.warning("Thumbnail upload failed (non-critical): %s", exc)
