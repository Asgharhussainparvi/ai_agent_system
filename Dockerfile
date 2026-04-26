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
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && python3 -m pip install --upgrade pip setuptools wheel

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Output directories ────────────────────────────────────────────────────────
RUN mkdir -p /app/outputs/images \
             /app/outputs/audio \
             /app/outputs/video \
             /app/outputs/logs

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

CMD ["python", "main.py", "api"]
