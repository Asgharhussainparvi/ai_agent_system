
from pathlib import Path
from unittest.mock import patch

import orchestrator
from orchestrator import Orchestrator
from utils.config import VIDEO_DIR

videos = sorted(
    VIDEO_DIR.glob("*.mp4"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
assert videos, "No existing MP4 found; run the video fallback test first."

video_path = videos[0]

# Case A: Missing credentials — upload is skipped.
pipeline_a = Orchestrator(job_id="youtube-missing-token-test")
pipeline_a.state.video_path = str(video_path)
pipeline_a.state.content = {"title": "Controlled test"}

with patch.object(orchestrator, "YOUTUBE_REFRESH_TOKEN", ""):
    result_a = pipeline_a._stage_youtube()

print("\nCASE A — Missing token")
print("Stage result:", result_a)
print("Errors recorded:", pipeline_a.state.errors)

assert result_a is False
assert not pipeline_a.state.errors

# Case B: Credentials appear configured, but upload raises an exception.
pipeline_b = Orchestrator(job_id="youtube-upload-error-test")
pipeline_b.state.video_path = str(video_path)
pipeline_b.state.content = {"title": "Controlled test"}
pipeline_b.state.topic = "Controlled test topic"

with (
    patch.object(orchestrator, "YOUTUBE_REFRESH_TOKEN", "mock-token"),
    patch.object(orchestrator.YouTubeAgent, "upload",
                 side_effect=RuntimeError("Simulated upload failure")),
):
    result_b = pipeline_b._stage_youtube()

print("\nCASE B — Simulated upload exception")
print("Stage result:", result_b)
print("Errors recorded:", pipeline_b.state.errors)

assert result_b is False
assert pipeline_b.state.errors
assert any("YouTubeAgent" in e for e in pipeline_b.state.errors)

print("\nYOUTUBE FAILURE-INJECTION TESTS: PASS")
