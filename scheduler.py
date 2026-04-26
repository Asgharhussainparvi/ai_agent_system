"""
scheduler.py
━━━━━━━━━━━━
Autonomous scheduler — runs the full pipeline on a configurable interval.

Scrapes fresh topics → picks best → runs full content pipeline → uploads to YouTube.
Runs forever as a background service (used by the Docker scheduler container).
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime

import schedule

from orchestrator import Orchestrator
from utils.config import SCRAPE_INTERVAL_HRS
from utils.logger import get_logger

_log = get_logger("Scheduler")


def _run_once():
    job_id = str(uuid.uuid4())[:8]
    _log.info("=" * 55)
    _log.info("  SCHEDULED RUN  %s  (job_id=%s)", datetime.utcnow().isoformat(), job_id)
    _log.info("=" * 55)

    try:
        orch  = Orchestrator(job_id=job_id)
        state = orch.run(topic=None, upload_to_youtube=True)
        _log.info(
            "Scheduled run %s complete. Status=%s  YouTube=%s",
            job_id, state.status, state.youtube_url or "—",
        )
    except Exception as exc:
        _log.error("Scheduled run %s crashed: %s", job_id, exc)


def main():
    _log.info("Scheduler starting. Interval: every %.1f hour(s).", SCRAPE_INTERVAL_HRS)
    _run_once()
    schedule.every(SCRAPE_INTERVAL_HRS).hours.do(_run_once)
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
