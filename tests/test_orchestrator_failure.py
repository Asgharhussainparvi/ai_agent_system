
import unittest
from unittest.mock import patch

from orchestrator import Orchestrator


class OrchestratorFailureTests(unittest.TestCase):
    def test_failed_required_stage_stops_pipeline(self):
        orchestrator = Orchestrator(job_id="unittest-fail-fast")
        orchestrator.state.snapshot = lambda: None

        calls = []

        def content(topic):
            calls.append("content")
            return False

        def images():
            calls.append("images")
            return True

        def voice():
            calls.append("voice")
            return True

        def video():
            calls.append("video")
            return True

        orchestrator._stage_content = content
        orchestrator._stage_images = images
        orchestrator._stage_voice = voice
        orchestrator._stage_video = video

        state = orchestrator.run(
            topic="Controlled fail-fast unit test",
            upload_to_youtube=False,
        )

        self.assertEqual(calls, ["content"])
        self.assertEqual(state.status, "error")
        self.assertEqual(len(state.errors), 1)
        self.assertIsNone(state.video_path)

    def test_failure_log_matches_error_status(self):
        orchestrator = Orchestrator(job_id="unittest-log-test")
        orchestrator.state.snapshot = lambda: None

        orchestrator._stage_content = lambda topic: False
        orchestrator._stage_images = lambda: True
        orchestrator._stage_voice = lambda: True
        orchestrator._stage_video = lambda: True

        with patch("orchestrator._log") as mock_log:
            state = orchestrator.run(
                topic="Controlled logging test",
                upload_to_youtube=False,
            )

        self.assertEqual(state.status, "error")

        logged_messages = [
            str(call.args[0]) % call.args[1:]
            if (
                call.args
                and "%" in str(call.args[0])
                and len(call.args) > 1
            )
            else str(call.args[0])
            for call in mock_log.error.call_args_list
        ]

        self.assertTrue(
            any("PIPELINE FAILED" in message for message in logged_messages),
            f"Expected failure log; got: {logged_messages}",
        )
        self.assertFalse(
            any("PIPELINE COMPLETE" in message for message in logged_messages),
            f"Misleading success wording found: {logged_messages}",
        )


if __name__ == "__main__":
    unittest.main()
