"""
agents/content_agent.py
━━━━━━━━━━━━━━━━━━━━━━━
ContentAgent — uses Groq / LLaMA 3.1 to turn a topic into a full
YouTube content package (title, script, image prompts, voice script, etc.)
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

import requests

from utils.config import GROQ_API_KEY, GROQ_API_URL, LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE
from utils.logger import get_logger
from utils.retry import with_retry

_log = get_logger("ContentAgent")

_SYSTEM_PROMPT = (
    "You are an expert YouTube content creator specialising in AI and Computer Science. "
    "Always respond with ONLY a valid JSON object — no markdown fences, no extra text, "
    "no newline characters inside string values."
)


def _build_prompt(topic: str) -> str:
    return (
        f'Create a YouTube video content package for: "{topic}". '
        "Return ONLY a JSON object with these exact keys: "
        "title (string, max 70 chars), "
        "description (string, 150 words, SEO-optimised), "
        "script (string, 350-450 words, no newlines inside), "
        "tags (array of 10 YouTube tag strings), "
        "scenes (array of 5 objects each with: scene_number int, heading string, "
        "        description string, duration_seconds int), "
        "image_prompts (array of 5 strings — vivid SDXL prompts, cinematic, tech-themed, "
        "               each must specify lighting, mood, camera angle), "
        "voice_script (string, clean narration 300-350 words, no newlines inside), "
        "thumbnail_prompt (string — single SDXL prompt for YouTube thumbnail, eye-catching). "
        "Important: all string values must be single-line with no embedded newlines."
    )


class ContentAgent:
    @with_retry(
        max_retries=3,
        delay=5,
        exceptions=(requests.RequestException, ValueError, json.JSONDecodeError),
    )
    def run(self, topic: str) -> Dict[str, Any]:
        _log.info("ContentAgent: calling Groq API for topic='%s'…", topic[:80])

        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model":       LLM_MODEL,
                "temperature": LLM_TEMPERATURE,
                "max_tokens":  LLM_MAX_TOKENS,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": _build_prompt(topic)},
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()

        raw = resp.json()["choices"][0]["message"]["content"]

        # Strip markdown fences + control chars
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw.strip())
        raw = re.sub(r"```$", "", raw.strip()).strip()
        raw = re.sub(r"[\x00-\x1f\x7f]", " ", raw)

        result = json.loads(raw)

        required = {"title", "description", "script", "scenes", "image_prompts", "voice_script"}
        missing  = required - result.keys()
        if missing:
            raise ValueError(f"Missing keys in LLM response: {missing}")

        _log.info("ContentAgent ✓  title='%s'", result["title"])
        return result
