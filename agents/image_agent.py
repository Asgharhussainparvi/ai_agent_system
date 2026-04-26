"""
agents/image_agent.py
━━━━━━━━━━━━━━━━━━━━━
ImageAgent — generates high-quality 1024×1024 scene images using
stabilityai/stable-diffusion-xl-base-1.0 via HuggingFace InferenceClient.

Falls back to coloured placeholder images if the API is unreachable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from huggingface_hub import InferenceClient, login

from utils.config import HF_TOKEN, IMAGES_DIR, NUM_IMAGES, SD_IMAGE_HEIGHT, SD_IMAGE_WIDTH, SDXL_MODEL_ID
from utils.logger import get_logger
from utils.retry import with_retry

_log = get_logger("ImageAgent")

_PLACEHOLDER_COLOURS = ["#1a1a2e", "#16213e", "#0f3460", "#533483", "#2b2d42"]


def _make_placeholder(prompt: str, idx: int) -> Path:
    """Create a simple coloured image with text as a fallback."""
    from PIL import Image, ImageDraw

    out = IMAGES_DIR / f"scene_{idx+1:02d}.png"
    colour = _PLACEHOLDER_COLOURS[idx % len(_PLACEHOLDER_COLOURS)]
    img    = Image.new("RGB", (SD_IMAGE_WIDTH, SD_IMAGE_HEIGHT), colour)
    draw   = ImageDraw.Draw(img)

    words, lines, line = prompt.split(), [], ""
    for w in words:
        if len(line + w) < 40:
            line += w + " "
        else:
            lines.append(line.strip())
            line = w + " "
    lines.append(line.strip())

    y = SD_IMAGE_HEIGHT // 2 - len(lines) * 14
    for ln in lines[:8]:
        draw.text((30, y), ln, fill="white")
        y += 26
    draw.text((30, SD_IMAGE_HEIGHT - 40), f"Scene {idx + 1}", fill="#aaaaaa")
    img.save(str(out))
    return out


class ImageAgent:
    """
    Each call to run() generates NUM_IMAGES scene images via SDXL SDXL.
    Images are saved to IMAGES_DIR and paths returned.
    """

    def __init__(self):
        login(token=HF_TOKEN, add_to_git_credential=False)
        self._client = InferenceClient(token=HF_TOKEN)

    @with_retry(max_retries=2, delay=8, exceptions=(Exception,))
    def _generate_one(self, prompt: str, idx: int) -> Path:
        out = IMAGES_DIR / f"scene_{idx+1:02d}.png"
        _log.info("SDXL generating image %d/%d…", idx + 1, NUM_IMAGES)

        enhanced = (
            f"{prompt}, "
            "cinematic lighting, 8k ultra-detailed, professional photography, "
            "sharp focus, vibrant colors, photorealistic"
        )
        image = self._client.text_to_image(enhanced, model=SDXL_MODEL_ID)
        image.save(str(out))
        _log.info("✓ Image %d saved → %s  (%d bytes)", idx + 1, out.name, out.stat().st_size)
        return out

    def run(self, prompts: List[str]) -> List[Path]:
        paths: List[Path] = []
        for idx, prompt in enumerate(prompts[:NUM_IMAGES]):
            try:
                paths.append(self._generate_one(prompt, idx))
            except Exception as exc:
                _log.warning("Image %d failed: %s — using placeholder.", idx + 1, exc)
                paths.append(_make_placeholder(prompt, idx))
        _log.info("ImageAgent ✓  %d images ready.", len(paths))
        return paths
