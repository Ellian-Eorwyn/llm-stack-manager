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
        self.dir = pathlib.Path(self.tmp.name)
        self.file = self.dir / "voices.json"
        tts.state["speakers"] = SPEAKERS
        tts.state["models"] = {"preset": object(), "clone": object()}
        (self.dir / "voices" / "deep").mkdir(parents=True)
        (self.dir / "voices" / "deep" / "reference.wav").write_bytes(b"RIFF")
        (self.dir / "voices" / "deep" / "reference.txt").write_text("Take a breath with me.\n")

    def load(self, cfg):
        self.file.write_text(json.dumps(cfg))
        with mock.patch.object(tts, "VOICES_FILE", self.file):
            tts.load_voices()

    def test_named_voice_carries_its_speaker_and_style(self):
        self.load({"default": "Hermes", "voices": {"Hermes": {"speaker": "sohee", "instruct": "calm"}}})
        self.assertEqual(tts.resolve("hermes", None),
                         ("hermes", {"kind": "preset", "speaker": "sohee", "instruct": "calm"}))
        self.assertEqual(tts.resolve("hermes", "brighter")[1]["instruct"], "brighter")

    def test_preset_speaker_and_unknown_voice(self):
        self.load({"default": "sohee", "voices": {"sohee": {"speaker": "sohee"}}})
        self.assertEqual(tts.resolve("Vivian", None)[1]["speaker"], "vivian")
        self.assertEqual(tts.resolve("alloy", None)[0], "sohee")
        self.assertEqual(tts.resolve(None, None)[0], "sohee")
        self.assertEqual(tts.resolve("default", None)[0], "sohee")

    def test_clone_voice_reads_its_clip_and_text_relative_to_the_file(self):
        self.load({"default": "sohee", "voices": {
            "sohee": {"speaker": "sohee"},
            "deep": {"ref_audio": "voices/deep/reference.wav", "ref_text_file": "voices/deep/reference.txt"}}})
        name, spec = tts.resolve("Deep", "ignored for clones")
        self.assertEqual((name, spec["kind"], spec["ref_text"]), ("deep", "clone", "Take a breath with me."))
        self.assertTrue(spec["ref_audio"].endswith("voices/deep/reference.wav"))
        self.assertEqual(tts.kinds_needed(json.loads(self.file.read_text())), {"preset", "clone"})

    def test_bad_voices_are_skipped_not_fatal(self):
        self.load({"default": "gone", "voices": {
            "sohee": {"speaker": "sohee"}, "x": {"speaker": "nobody"},
            "gone": {"ref_audio": "voices/gone/reference.wav", "ref_text": "hi"}}})
        self.assertEqual(sorted(tts.state["skipped"]), ["gone", "x"])
        self.assertEqual(tts.state["default"], "sohee")  # a skipped default falls back to a usable voice

    def test_clone_voice_without_the_clone_model_is_skipped(self):
        tts.state["models"] = {"preset": object()}
        self.load({"voices": {"sohee": {"speaker": "sohee"},
                              "deep": {"ref_audio": "voices/deep/reference.wav", "ref_text": "hi"}}})
        self.assertIn("deep", tts.state["skipped"])

    def test_a_new_default_in_the_file_applies_without_a_restart(self):
        import os
        cfg = {"default": "sohee", "voices": {
            "sohee": {"speaker": "sohee"},
            "deep": {"ref_audio": "voices/deep/reference.wav", "ref_text_file": "voices/deep/reference.txt"}}}
        self.load(cfg)
        with mock.patch.object(tts, "VOICES_FILE", self.file):
            cfg["default"] = "deep"
            self.file.write_text(json.dumps(cfg))
            os.utime(self.file, (1, tts.state["voices_mtime"] + 5))
            self.assertEqual(tts.resolve("default", None)[0], "deep")
            cfg["default"] = "nobody"  # not loaded: keep the current default
            self.file.write_text(json.dumps(cfg))
            os.utime(self.file, (1, tts.state["voices_mtime"] + 5))
            self.assertEqual(tts.resolve(None, None)[0], "deep")

    def test_no_usable_voice_refuses_to_start(self):
        with self.assertRaises(SystemExit):
            self.load({"voices": {"x": {"speaker": "nobody"}}})

    def test_length_cap_scales_with_text(self):
        short, long = tts.max_tokens_for("Okay. Keep going."), tts.max_tokens_for("x" * 230)
        self.assertLess(short / tts.FRAME_HZ, 4)        # a runaway "Okay. Keep going." stops within 4 s
        self.assertLess(tts.max_tokens_for("Check.") / tts.FRAME_HZ, 2.5)  # one word: ~2 s at most
        self.assertGreater(long / tts.FRAME_HZ, 2 * 14)  # the audition's 14 s sentence fits twice over

    def test_long_text_is_spoken_in_sentence_groups(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(60))
        parts = tts.pieces(text)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(p) <= tts.PIECE_CHARS for p in parts))
        self.assertEqual(" ".join(parts), text)  # nothing lost or reordered
        self.assertTrue(all(p.endswith(".") for p in parts))  # cut between sentences
        self.assertEqual(tts.pieces("Short one."), ["Short one."])
        runon = "word " * 200
        self.assertTrue(all(len(p) <= tts.PIECE_CHARS for p in tts.pieces(runon)))
        self.assertEqual(tts.pieces("Line one\n\nLine two"), ["Line one Line two"])

    def test_mp3_is_encoded_with_ffmpeg(self):
        import shutil
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("no ffmpeg here")
        with mock.patch.object(tts, "FFMPEG", ffmpeg):
            audio = tts.encode(b"\0\0" * 24000, 24000, "mp3")
        self.assertTrue(audio[:3] == b"ID3" or audio[0] == 0xFF)  # an MP3 frame or tag

    def test_streaming_wav_header(self):
        h = tts.wav_header(24000)
        self.assertEqual(len(h), 44)
        self.assertEqual(h[:4] + h[8:16], b"RIFFWAVEfmt ")
        self.assertEqual(struct.unpack("<HHI", h[20:28]), (1, 1, 24000))


if __name__ == "__main__":
    unittest.main()
