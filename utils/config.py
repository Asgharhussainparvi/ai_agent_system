"""
config.py — Central configuration loaded from environment variables.
All agents import from here; no magic strings scattered around.
"""
import os
from pathlib import Path
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(os.getenv("OUTPUT_DIR", "/app/outputs"))
IMAGES_DIR = BASE_DIR / "images"
AUDIO_DIR  = BASE_DIR / "audio"
VIDEO_DIR  = BASE_DIR / "video"
LOGS_DIR   = BASE_DIR / "logs"

for _d in [IMAGES_DIR, AUDIO_DIR, VIDEO_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY        = os.environ["GROQ_API_KEY"]
HF_TOKEN            = os.environ["HF_TOKEN"]
ELEVENLABS_API_KEY  = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

YOUTUBE_CLIENT_ID     = os.getenv("YOUTUBE_CLIENT_ID", "")
YOUTUBE_CLIENT_SECRET = os.getenv("YOUTUBE_CLIENT_SECRET", "")
YOUTUBE_REFRESH_TOKEN = os.getenv("YOUTUBE_REFRESH_TOKEN", "")

# ── Groq / LLM ────────────────────────────────────────────────────────────────
GROQ_API_URL    = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL       = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS  = 1024

# ── Image: SDXL via HF InferenceClient ───────────────────────────────────────
SDXL_MODEL_ID  = "stabilityai/stable-diffusion-xl-base-1.0"
SD_DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SD_DTYPE       = torch.bfloat16 if SD_DEVICE == "cuda" else torch.float32
SD_IMAGE_WIDTH = 1024
SD_IMAGE_HEIGHT= 1024
NUM_IMAGES     = int(os.getenv("NUM_IMAGES", "5"))

# ── Voice: ElevenLabs ─────────────────────────────────────────────────────────
ELEVENLABS_TTS_URL = (
    f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
)
ELEVENLABS_MODEL   = "eleven_multilingual_v2"

# ── Video: Motif-Video-2B ─────────────────────────────────────────────────────
MOTIF_MODEL_ID  = "Motif-Technologies/Motif-Video-2B"
VIDEO_FPS       = int(os.getenv("VIDEO_FPS", "8"))
VIDEO_FRAMES    = int(os.getenv("VIDEO_FRAMES", "16"))
# Key scenes: AI video used on these scene indices (0-based), rest is MoviePy
AI_SCENE_INDICES = {0, 2, 4}   # First, middle, last scene get AI treatment

# ── Scraper ───────────────────────────────────────────────────────────────────
HF_SCRAPE_LIMIT     = int(os.getenv("HF_SCRAPE_LIMIT", "10"))
DEVPOST_SCRAPE_LIMIT= int(os.getenv("DEVPOST_SCRAPE_LIMIT", "10"))
SCRAPE_INTERVAL_HRS = float(os.getenv("SCRAPE_INTERVAL_HOURS", "6"))

# ── Retry ─────────────────────────────────────────────────────────────────────
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = 2.0
