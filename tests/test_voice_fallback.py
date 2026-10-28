
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from agents.voice_agent import VoiceAgent


class VoiceFallbackTests(unittest.TestCase):
    def test_silent_wav_has_expected_audio_properties(self):
        agent = VoiceAgent()
        path = Path(agent._silent_wav())

        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 0)

        with wave.open(str(path), "rb") as audio:
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getsampwidth(), 2)
            self.assertEqual(audio.getframerate(), 22050)
            self.assertEqual(audio.getnframes(), 22050 * 30)

    def test_run_uses_silent_wav_when_both_services_fail(self):
        agent = VoiceAgent()

        with (
            patch.object(
                agent,
                "_elevenlabs",
                side_effect=RuntimeError("Simulated ElevenLabs failure"),
            ),
            patch.object(agent, "_gtts_fallback", return_value=None),
        ):
            path = agent.run("Controlled voice fallback test")

        self.assertTrue(path.is_file())
        self.assertEqual(path.suffix.lower(), ".wav")

        with wave.open(str(path), "rb") as audio:
            self.assertEqual(audio.getnframes(), 22050 * 30)


if __name__ == "__main__":
    unittest.main()
