#!/usr/bin/env python3
"""
main.py — Application entry point.

Modes:
  python main.py api        → start FastAPI server (default)
  python main.py run        → run pipeline once (CLI mode)
  python main.py run --topic "Your topic here"
  python main.py scrape     → scrape and print topics only
"""

import argparse
import sys
import uuid

import uvicorn


def run_api():
    """Start the FastAPI server."""
    from utils.config import API_HOST, API_PORT
    uvicorn.run(
        "api.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )


def run_pipeline_cli(topic=None):
    """Run a single pipeline run and print results."""
    from orchestrator import Orchestrator

    job_id = str(uuid.uuid4())[:8]
    orch   = Orchestrator(job_id=job_id)
    state  = orch.run(topic=topic, upload_to_youtube=True)

    d = state.to_dict()
    print()
    print("═" * 65)
    print("  PIPELINE RESULT SUMMARY")
    print("═" * 65)
    print(f"  Job ID    : {d['job_id']}")
    print(f"  Topic     : {d['topic']}")
    print(f"  Title     : {(d.get('content') or {}).get('title', '—')}")
    print(f"  Images    : {len(d['image_paths'])} file(s)")
    print(f"  Audio     : {d['audio_path'] or '—'}")
    print(f"  Video     : {d['video_path'] or '—'}")
    print(f"  YouTube   : {d['youtube_url'] or '—'}")
    print(f"  Elapsed   : {d['elapsed_s']}s")
    print(f"  Errors    : {len(d['errors'])}")
    for err in d["errors"]:
        print(f"    ✗ {err}")
    print("═" * 65)


def run_scrape_cli():
    """Scrape topics and print them."""
    from agents.scraper_agent import ScraperAgent

    topics = ScraperAgent().run()
    print(f"\n{'─'*60}")
    print(f"  Scraped {len(topics)} topics")
    print(f"{'─'*60}")
    for i, t in enumerate(topics, 1):
        print(f"  {i:2d}. [{t.source:12s}] {t.title[:70]}")
        if t.summary:
            print(f"        {t.summary[:80]}…")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous AI Content System")
    parser.add_argument(
        "mode",
        nargs="?",
        default="api",
        choices=["api", "run", "scrape"],
        help="Mode: api (default), run (single pipeline), scrape (topic discovery only)",
    )
    parser.add_argument("--topic", type=str, default=None, help="Topic for pipeline run")
    args = parser.parse_args()

    if args.mode == "api":
        run_api()
    elif args.mode == "run":
        run_pipeline_cli(topic=args.topic)
    elif args.mode == "scrape":
        run_scrape_cli()
