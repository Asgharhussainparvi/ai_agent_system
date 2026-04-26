"""
agents/voice_agent.py
━━━━━━━━━━━━━━━━━━━━━
VoiceAgent — premium narration via ElevenLabs API.

Primary  : ElevenLabs (eleven_multilingual_v2) — high-quality, natural voice
Fallback : gTTS (Google TTS — free, always available)
Last resort: silent WAV (keeps pipeline alive)
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Optional

import requests

from utils.config import (
    AUDIO_DIR,
    ELEVENLABS_API_KEY,
    ELEVENLABS_MODEL,
    ELEVENLABS_TTS_URL,
    ELEVENLABS_VOICE_ID,
)
from utils.logger import get_logger
from utils.retry import with_retry

_log = get_logger("VoiceAgent")


class VoiceAgent:
    """
    Converts narration script to audio.

    Output: MP3 file at AUDIO_DIR/narration.mp3
    """

    OUTPUT_FILE = AUDIO_DIR / "narration.mp3"

    # ── ElevenLabs ────────────────────────────────────────────────────────────
    @with_retry(max_retries=2, delay=5, exceptions=(requests.RequestException, IOError))
    def _elevenlabs(self, text: str) -> Path:
        _log.info("ElevenLabs TTS: generating audio (voice=%s)…", ELEVENLABS_VOICE_ID)

        resp = requests.post(
            ELEVENLABS_TTS_URL,
            headers={
                "xi-api-key":   ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept":       "audio/mpeg",
            },
            json={
                "text":       text,
                "model_id":   ELEVENLABS_MODEL,
                "voice_settings": {
                    "stability":        0.5,
                    "similarity_boost": 0.75,
                    "style":            0.3,
                    "use_speaker_boost": True,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()

        self.OUTPUT_FILE.write_bytes(resp.content)
        size_kb = self.OUTPUT_FILE.stat().st_size // 1024
        _log.info("ElevenLabs TTS ✓  %d KB → %s", size_kb, self.OUTPUT_FILE.name)
        return self.OUTPUT_FILE

    # ── gTTS fallback ─────────────────────────────────────────────────────────
    def _gtts_fallback(self, text: str) -> Optional[Path]:
        try:
            from gtts import gTTS
            _log.warning("ElevenLabs unavailable — falling back to gTTS…")
            tts = gTTS(text=text, lang="en", slow=False)
            tts.save(str(self.OUTPUT_FILE))
            _log.info("gTTS fallback ✓ → %s", self.OUTPUT_FILE.name)
            return self.OUTPUT_FILE
        except Exception as exc:
            _log.warning("gTTS fallback also failed: %s", exc)
            return None

    # ── Silent WAV last resort ────────────────────────────────────────────────
    def _silent_wav(self) -> Path:
        silent = AUDIO_DIR / "narration.wav"
        sr, dur = 22050, 30
        n = sr * dur
        with wave.open(str(silent), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(struct.pack("<" + "h" * n, *([0] * n)))
        _log.warning("Silent WAV last resort → %s", silent.name)
        return silent

    # ── Main run ──────────────────────────────────────────────────────────────
    def run(self, script: str) -> Path:
        # 1. Try ElevenLabs
        try:
            result = self._elevenlabs(script)
            if result.exists() and result.stat().st_size > 0:
                return result
        except Exception as exc:
            _log.warning("ElevenLabs failed: %s", exc)

        # 2. gTTS fallback
        result = self._gtts_fallback(script)
        if result and result.exists() and result.stat().st_size > 0:
            return result

        # 3. Silent placeholder
        return self._silent_wav()
