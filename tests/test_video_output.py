import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from moviepy.editor import VideoFileClip

from agents.video_agent import VideoAgent
from utils.config import VIDEO_DIR


class VideoOutputTests(unittest.TestCase):
    def test_generated_mp4_is_nonempty_and_decodable(self):
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix="video_test_",
            dir=str(VIDEO_DIR),
        ) as temp_dir:
            temp_path = Path(temp_dir)
            image_paths = []

            # Generate fresh local images; no image-generation service is used.
            for index, color in enumerate(("red", "blue")):
                image_path = temp_path / f"scene_{index + 1}.png"
                Image.new("RGB", (320, 180), color=color).save(image_path)
                image_paths.append(image_path)

            output_path = temp_path / "fresh_test_video.mp4"
            agent = VideoAgent()

            # Force the local MoviePy image-clip path if AI animation is selected.
            with (
                patch.object(VideoAgent, "OUTPUT_FILE", output_path),
                patch.object(agent, "_motif_clip", return_value=None),
            ):
                result = agent._build_hybrid(
                    image_paths=image_paths,
                    audio_path=None,
                    content={
                        "scenes": [
                            {"duration_seconds": 1},
                            {"duration_seconds": 1},
                        ],
                        "image_prompts": ["Test scene one", "Test scene two"],
                    },
                )

            self.assertEqual(result, output_path)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

            clip = None
            try:
                clip = VideoFileClip(str(output_path), audio=True)

                self.assertGreater(clip.duration, 0)
                self.assertEqual(tuple(clip.size), (1280, 720))
                self.assertGreater(clip.fps, 0)

                frame = clip.get_frame(min(0.5, clip.duration / 2))
                self.assertGreater(frame.size, 0)

            finally:
                if clip is not None:
                    clip.close()


if __name__ == "__main__":
    unittest.main()
