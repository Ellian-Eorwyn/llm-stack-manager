"""Joining diarization to Parakeet's words: scripts/speaker_attribution.py.

The cases here are the ones the Nemotron 3 Diarization demo clip actually
produced: a turn's first word stamped just before the diarizer hears the
speaker ("To", "Hi,"), a turn's last word just after ("QTI."), and real
crosstalk that must stay unassigned rather than go to the louder voice.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("speaker_attribution",
                                               ROOT / "scripts" / "speaker_attribution.py")
sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sa)

FRAME = 0.1  # coarser than the model's 10 ms, to keep the fixtures readable


def timeline(*spans, seconds=10.0, speakers=3):
    """Per-frame probabilities: 0.9 inside each (speaker, start, end) span."""
    frames = int(round(seconds / FRAME))
    probs = [[0.02] * speakers for _ in range(frames)]
    for speaker, start, end in spans:
        for f in range(int(round(start / FRAME)), int(round(end / FRAME))):
            probs[f][speaker] = 0.9
    return probs


def w(word, start, end):
    return {"word": word, "start": start, "end": end}


class AttributeTests(unittest.TestCase):
    def test_a_word_inside_one_speaker_is_theirs(self):
        out = sa.attribute([w("hello", 1.0, 1.4)], timeline((1, 0.5, 2.0)), FRAME)
        self.assertEqual(out[0]["speaker"], "speaker_1")
        self.assertFalse(out[0]["overlap"])
        self.assertEqual(out[0]["candidates"], ["speaker_1"])

    def test_crosstalk_is_not_given_to_the_louder_voice(self):
        out = sa.attribute([w("both", 1.0, 1.4)],
                           timeline((0, 0.5, 2.0), (2, 0.8, 2.0)), FRAME)
        self.assertIsNone(out[0]["speaker"])
        self.assertTrue(out[0]["overlap"])
        self.assertEqual(out[0]["candidates"], ["speaker_0", "speaker_2"])

    def test_a_word_in_silence_is_unassigned_with_no_candidates(self):
        out = sa.attribute([w("um", 5.0, 5.2)], timeline((0, 0.0, 1.0)), FRAME)
        self.assertIsNone(out[0]["speaker"])
        self.assertEqual(out[0]["candidates"], [])

    def test_a_word_past_the_timeline_is_unassigned_not_an_error(self):
        out = sa.attribute([w("late", 12.0, 12.5)], timeline((0, 0.0, 10.0)), FRAME)
        self.assertIsNone(out[0]["speaker"])

    def test_a_zero_length_word_reads_the_frame_it_was_emitted_in(self):
        out = sa.attribute([w("a", 1.05, 1.05)], timeline((2, 1.0, 1.2)), FRAME)
        self.assertEqual(out[0]["speaker"], "speaker_2")

    def test_input_words_are_not_mutated(self):
        words = [w("hi", 1.0, 1.2)]
        sa.attribute(words, timeline((0, 0.0, 2.0)), FRAME)
        self.assertEqual(words, [w("hi", 1.0, 1.2)])

    def test_nonsense_parameters_are_refused(self):
        with self.assertRaises(ValueError):
            sa.attribute([], [], 0.0)
        with self.assertRaises(ValueError):
            sa.attribute([], [], FRAME, threshold=1.0)


class SmoothTests(unittest.TestCase):
    def _run(self, words, probs):
        return sa.smooth(sa.attribute(words, probs, FRAME))

    def test_a_turns_first_word_stamped_early_joins_the_turn_it_starts(self):
        # Speaker 0 finished a sentence; "Hi," is stamped before speaker 1 is heard.
        probs = timeline((0, 0.0, 2.0), (1, 2.6, 5.0))
        out = self._run([w("done.", 1.5, 1.9), w("Hi,", 2.3, 2.5), w("there", 2.7, 3.0)], probs)
        self.assertEqual([x["speaker"] for x in out], ["speaker_0", "speaker_1", "speaker_1"])
        self.assertTrue(out[1]["speaker_inferred"])
        self.assertNotIn("speaker_inferred", out[0])

    def test_a_turns_last_word_stamped_late_stays_with_its_sentence(self):
        probs = timeline((0, 0.0, 2.0), (1, 3.0, 5.0))
        out = self._run([w("a", 1.0, 1.4), w("voice", 1.5, 1.9), w("from", 1.9, 2.0),
                         w("QTI.", 2.05, 2.3), w("Nice", 3.1, 3.4)], probs)
        self.assertEqual(out[3]["speaker"], "speaker_0")

    def test_the_first_word_of_the_recording_looks_forward(self):
        probs = timeline((2, 0.5, 5.0))
        out = self._run([w("To", 0.1, 0.3), w("celebrate", 0.6, 1.0)], probs)
        self.assertEqual(out[0]["speaker"], "speaker_2")

    def test_the_last_word_of_the_recording_falls_back_to_the_previous_speaker(self):
        probs = timeline((0, 0.0, 3.0), seconds=5.0)
        out = self._run([w("Stay", 2.5, 2.9), w("open.", 3.1, 3.4)], probs)
        self.assertEqual(out[1]["speaker"], "speaker_0")

    def test_crosstalk_is_only_resolved_to_a_speaker_who_was_talking(self):
        # Speakers 1 and 2 overlap; the neighbours are speaker 0. Nobody who
        # was actually talking is a neighbour, so the word stays unassigned.
        probs = timeline((0, 0.0, 1.0), (1, 1.5, 2.5), (2, 1.5, 2.5), (0, 3.0, 4.0))
        out = self._run([w("so", 0.5, 0.8), w("both", 1.8, 2.1), w("then", 3.2, 3.5)], probs)
        self.assertIsNone(out[1]["speaker"])
        self.assertEqual(out[1]["candidates"], ["speaker_1", "speaker_2"])

    def test_crosstalk_goes_to_a_neighbour_who_was_among_the_voices(self):
        probs = timeline((1, 0.0, 2.5), (2, 1.5, 2.5))
        out = self._run([w("I", 0.5, 0.8), w("said", 1.8, 2.1)], probs)
        self.assertEqual(out[1]["speaker"], "speaker_1")
        self.assertTrue(out[1]["overlap"])


class TurnTests(unittest.TestCase):
    def _words(self, *pairs):
        return [{**w(text, i, i + 0.5), "speaker": sp, "overlap": False, "candidates": []}
                for i, (text, sp) in enumerate(pairs)]

    def test_a_sentence_is_split_where_the_speaker_changes(self):
        sentence = self._words(("Let's", "speaker_0"), ("find", "speaker_0"),
                               ("out.", "speaker_0"), ("Even", "speaker_3"))
        segments = sa.turns([sentence])
        self.assertEqual([(s["speaker"], s["text"]) for s in segments],
                         [("speaker_0", "Let's find out."), ("speaker_3", "Even")])
        self.assertEqual([s["id"] for s in segments], [0, 1])
        self.assertEqual(segments[1]["start"], 3)

    def test_sentence_boundaries_survive_within_one_speaker(self):
        a = self._words(("One.", "speaker_0"))
        b = self._words(("Two.", "speaker_0"))
        self.assertEqual(len(sa.turns([a, b])), 2)

    def test_the_summary_is_in_order_of_arrival(self):
        segments = sa.turns([self._words(("hi", "speaker_1"), ("yo", "speaker_0"))])
        summary = sa.speaker_summary(segments)
        self.assertEqual([s["id"] for s in summary], ["speaker_1", "speaker_0"])
        self.assertEqual(summary[0]["words"], 1)

    def test_text_rendering_joins_a_speakers_consecutive_segments(self):
        a = self._words(("One.", "speaker_0"))
        b = self._words(("Two.", "speaker_0"), ("Three.", "speaker_1"))
        self.assertEqual(sa.render_text(sa.turns([a, b])),
                         "speaker_0: One. Two.\n\nspeaker_1: Three.")


class LockTests(unittest.TestCase):
    def test_the_diarization_model_is_pinned_and_fetched_with_transcription(self):
        lock = json.loads((ROOT / "config" / "mlx-models.lock.json").read_text())
        entry = lock["models"]["diarization"]
        self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(entry["local_path"], "models/mlx/Nemotron-3-Diarization")
        installer = (ROOT / "scripts" / "install-mlx-runtime.sh").read_text()
        self.assertIn('"diarization": os.environ.get("WITH_TRANSCRIBE") == "1"', installer)
        example = (ROOT / "config" / "llm-stack.env.example").read_text()
        self.assertIn("MLX_DIARIZATION_MODEL_PATH=@STACK_DIR@/" + entry["local_path"], example)


if __name__ == "__main__":
    unittest.main()
