import unittest
from unittest.mock import patch

import orchestrator
from orchestrator import Orchestrator


class YouTubePipelineIntegrationTests(unittest.TestCase):
    def test_upload_failure_does_not_fail_successful_video_pipeline(self):
        pipeline = Orchestrator(job_id="unittest-youtube-integration")
        pipeline.state.video_path = "mock-video.mp4"
        pipeline.state.content = {"title": "Controlled integration test"}
        pipeline.state.topic = "Controlled integration test"

        with (
            patch.object(Orchestrator, "_stage_content", return_value=True),
            patch.object(Orchestrator, "_stage_images", return_value=True),
            patch.object(Orchestrator, "_stage_voice", return_value=True),
            patch.object(Orchestrator, "_stage_video", return_value=True),
            patch.object(orchestrator, "YOUTUBE_REFRESH_TOKEN", "mock-token"),
            patch.object(
                orchestrator.YouTubeAgent,
                "upload",
                side_effect=RuntimeError("Simulated upload failure"),
            ),
        ):
            state = pipeline.run(
                topic="Controlled integration test",
                upload_to_youtube=True,
            )

        self.assertEqual(state.status, "done")
        self.assertEqual(state.youtube_status, "failed")
        self.assertIn("Simulated upload failure", state.youtube_error)
        self.assertEqual(state.errors, [])
        self.assertIsNone(state.youtube_url)

        serialized = state.to_dict()
        self.assertEqual(serialized["status"], "done")
        self.assertEqual(serialized["youtube_status"], "failed")


if __name__ == "__main__":
    unittest.main()
