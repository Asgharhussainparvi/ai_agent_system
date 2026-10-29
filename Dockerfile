# ╔══════════════════════════════════════════════════════════════════╗
# ║  Dockerfile — Autonomous AI Content System                      ║
# ║  Base: nvidia/cuda for GPU support (falls back to CPU)          ║
# ╚══════════════════════════════════════════════════════════════════╝

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ── System packages ───────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        python3.11-venv \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        git \
        curl \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Make python3.11 the default ───────────────────────────────────────────────
RUN useradd -m -u 1000 user \
    && mkdir -p /app/outputs/images \
                 /app/outputs/audio \
                 /app/outputs/video \
                 /app/outputs/logs \
                 /home/user/.cache/huggingface \
    && chown -R user:user /app /home/user/.cache

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ───────────────────────────────────────
COPY requirements.txt .
RUN python -m pip install --user --no-cache-dir --upgrade pip \
 && python -m pip install --user --no-cache-dir -r requirements.txt
# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Environment defaults (overridden by .env / docker-compose) ────────────────
ENV OUTPUT_DIR=/app/outputs \
    API_HOST=0.0.0.0 \
    API_PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

# ── Expose API port ───────────────────────────────────────────────────────────
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Default command: start API server ────────────────────────────────────────
RUN --mount=type=secret,id=GROQ_API_KEY,mode=0444 true
RUN --mount=type=secret,id=HF_TOKEN,mode=0444 true
# etc. for each secret

CMD ["/usr/bin/python3.10", "main.py", "api"]
