
import unittest
from unittest.mock import patch
from importlib import import_module
from utils.config import LOGS_DIR
from pathlib import Path
import json

api_module = import_module("api.app")
PipelineState = import_module("orchestrator").PipelineState


class ApiBackgroundFailureTests(unittest.TestCase):
    def test_unexpected_exception_marks_and_persists_failed_job(self):
        job_id = "unittest-background-crash"
        api_module._JOBS[job_id] = PipelineState(
            job_id=job_id,
            status="pending",
        )

        try:
            with patch.object(
                api_module,
                "Orchestrator",
                side_effect=RuntimeError("controlled crash"),
            ):
                api_module._run_pipeline(
                    job_id=job_id,
                    topic="Controlled test",
                    upload_to_youtube=False,
                )

            state = api_module._JOBS[job_id]

            self.assertEqual(state.status, "error")
            self.assertTrue(
                any("controlled crash" in error for error in state.errors)
            )
            self.assertGreaterEqual(state.elapsed_s, 0)

            snapshot_path = LOGS_DIR / f"{job_id}_state.json"
            self.assertTrue(
                snapshot_path.exists(),
                "Expected a persisted state snapshot after a background crash.",
            )

            saved = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "error")
            self.assertTrue(
                any("controlled crash" in error for error in saved["errors"])
            )

        finally:
            api_module._JOBS.pop(job_id, None)
            snapshot_path = LOGS_DIR / f"{job_id}_state.json"
            snapshot_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
