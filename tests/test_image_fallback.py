import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from agents.image_agent import ImageAgent, _make_placeholder


class ImageFallbackTests(unittest.TestCase):
    def test_placeholder_is_valid_png(self):
        path = Path(_make_placeholder(
            "Automated placeholder test",
            987654,
        ))

        self.assertTrue(path.is_file())

        with Image.open(path) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (1024, 1024))
            image.verify()

    def test_run_uses_placeholder_when_generation_fails(self):
        agent = ImageAgent.__new__(ImageAgent)

        with patch.object(
            ImageAgent,
            "_generate_one",
            side_effect=RuntimeError("Simulated inference failure"),
        ):
            paths = agent.run(["Automated fallback test"])

        self.assertEqual(len(paths), 1)
        self.assertTrue(Path(paths[0]).is_file())

        with Image.open(paths[0]) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (1024, 1024))
            image.verify()


if __name__ == "__main__":
    unittest.main()
