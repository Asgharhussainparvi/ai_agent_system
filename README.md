---
title: Autonomous Multi-Agent AI Content System
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
---

# 🤖 Autonomous Multi-Agent AI Content System

> Scrapes trending AI topics → generates YouTube-ready videos → uploads automatically

## Architecture

```
ScraperAgent (HuggingFace + DevPost)
       ↓
ContentAgent (Groq / LLaMA 3.1)
       ↓
ImageAgent (SDXL via HF InferenceClient)
       ↓
VoiceAgent (ElevenLabs)
       ↓
VideoAgent (MoviePy story flow + Motif-Video AI key scenes)
       ↓
YouTubeAgent (auto-upload via YouTube Data API v3)
       ↓
Orchestrator (coordinates all agents autonomously)
```

## Quick Start (Docker)

```bash
# 1. Copy and fill env file
cp .env.example .env

# 2. Build and run
docker compose up --build

# 3. API available at
http://localhost:7860/docs
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/scrape/topics` | Scrape trending topics from HF + DevPost |
| POST | `/pipeline/run` | Run full pipeline on a topic |
| GET | `/pipeline/status/{job_id}` | Check pipeline job status |
| POST | `/youtube/upload` | Upload video to YouTube |
| GET | `/health` | Health check |

## Environment Variables

See `.env.example` for all required keys.
