"""voice-tts-server.py: voice resolution, the length cap and the WAV header (no GPU)."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import struct
import tempfile
import unittest
from unittest import mock


def _load():
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("voice_tts", root / "scripts" / "voice-tts-server.py")
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


tts = _load()
SPEAKERS = ["aiden", "ryan", "serena", "sohee", "vivian"]


class VoiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = pathlib.Path(self.tmp.name) / "voices.json"
        tts.state["speakers"] = SPEAKERS

    def load(self, cfg):
        self.file.write_text(json.dumps(cfg))
        with mock.patch.object(tts, "VOICES_FILE", self.file):
            tts.load_voices()

    def test_named_voice_carries_its_speaker_and_style(self):
        self.load({"default": "Hermes", "voices": {"Hermes": {"speaker": "sohee", "instruct": "calm"}}})
        self.assertEqual(tts.resolve("hermes", None), ("sohee", "calm", "hermes"))
        self.assertEqual(tts.resolve("hermes", "brighter"), ("sohee", "brighter", "hermes"))

    def test_preset_speaker_and_unknown_voice(self):
        self.load({"default": "sohee", "voices": {"sohee": {"speaker": "sohee"}}})
        self.assertEqual(tts.resolve("Vivian", None)[0], "vivian")
        self.assertEqual(tts.resolve("alloy", None), ("sohee", None, "sohee"))
        self.assertEqual(tts.resolve(None, None)[0], "sohee")

    def test_voice_with_missing_speaker_refuses_to_start(self):
        with self.assertRaises(SystemExit):
            self.load({"voices": {"x": {"speaker": "nobody"}}})

    def test_length_cap_scales_with_text(self):
        short, long = tts.max_tokens_for("Okay. Keep going."), tts.max_tokens_for("x" * 230)
        self.assertLess(short / tts.FRAME_HZ, 6)        # a runaway "Okay." stops within 6 s
        self.assertGreater(long / tts.FRAME_HZ, 2 * 14)  # the audition's 14 s sentence fits twice over

    def test_streaming_wav_header(self):
        h = tts.wav_header(24000)
        self.assertEqual(len(h), 44)
        self.assertEqual(h[:4] + h[8:16], b"RIFFWAVEfmt ")
        self.assertEqual(struct.unpack("<HHI", h[20:28]), (1, 1, 24000))


if __name__ == "__main__":
    unittest.main()
