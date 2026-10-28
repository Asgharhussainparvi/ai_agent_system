import unittest
from unittest.mock import patch

import orchestrator
from orchestrator import Orchestrator


class YouTubeStatusTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = Orchestrator(job_id="unittest-youtube-status")
        self.pipeline.state.video_path = "mock-video.mp4"
        self.pipeline.state.topic = "Controlled test topic"
        self.pipeline.state.content = {"title": "Controlled test"}

    def test_missing_token_is_skipped_without_pipeline_error(self):
        with patch.object(orchestrator, "YOUTUBE_REFRESH_TOKEN", ""):
            result = self.pipeline._stage_youtube()

        self.assertFalse(result)
        self.assertEqual(self.pipeline.state.youtube_status, "skipped_no_token")
        self.assertIsNone(self.pipeline.state.youtube_error)
        self.assertEqual(self.pipeline.state.errors, [])

    def test_upload_exception_is_separate_from_pipeline_errors(self):
        with (
            patch.object(orchestrator, "YOUTUBE_REFRESH_TOKEN", "mock-token"),
            patch.object(
                orchestrator.YouTubeAgent,
                "upload",
                side_effect=RuntimeError("Simulated upload failure"),
            ) as mocked_upload,
        ):
            result = self.pipeline._stage_youtube()

        self.assertFalse(result)
        self.assertEqual(self.pipeline.state.youtube_status, "failed")
        self.assertIn("Simulated upload failure", self.pipeline.state.youtube_error)
        self.assertEqual(self.pipeline.state.errors, [])
        mocked_upload.assert_called_once()

    def test_successful_mocked_upload_records_url(self):
        expected_url = "https://www.youtube.com/watch?v=mock123"

        with (
            patch.object(orchestrator, "YOUTUBE_REFRESH_TOKEN", "mock-token"),
            patch.object(
                orchestrator.YouTubeAgent,
                "upload",
                return_value=expected_url,
            ) as mocked_upload,
        ):
            result = self.pipeline._stage_youtube()

        self.assertTrue(result)
        self.assertEqual(self.pipeline.state.youtube_status, "uploaded")
        self.assertEqual(self.pipeline.state.youtube_url, expected_url)
        self.assertIsNone(self.pipeline.state.youtube_error)
        self.assertEqual(self.pipeline.state.errors, [])
        mocked_upload.assert_called_once()

    def test_empty_upload_url_is_failure(self):
        with (
            patch.object(orchestrator, "YOUTUBE_REFRESH_TOKEN", "mock-token"),
            patch.object(
                orchestrator.YouTubeAgent,
                "upload",
                return_value="",
            ),
        ):
            result = self.pipeline._stage_youtube()

        self.assertFalse(result)
        self.assertEqual(self.pipeline.state.youtube_status, "failed")
        self.assertIn("returned no URL", self.pipeline.state.youtube_error)
        self.assertEqual(self.pipeline.state.errors, [])

    def test_state_serialization_includes_youtube_status(self):
        state = self.pipeline.state.to_dict()

        self.assertIn("youtube_status", state)
        self.assertIn("youtube_error", state)
        self.assertEqual(state["youtube_status"], "not_requested")


if __name__ == "__main__":
    unittest.main()