"""Naming diarized speakers from voice profiles: scripts/speaker_identity.py.

Only the runtime half is tested here — it is pure Python, and it is the part
that decides whose name goes on a transcript. The vectors are toy 3-d voices.
"""

from __future__ import annotations

import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("speaker_identity",
                                               ROOT / "scripts" / "speaker_identity.py")
si = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(si)


def profiles(**voices):
    return {name: {"centroid": si._unit(vec)} for name, vec in voices.items()}


PEOPLE = profiles(Alice=[1, 0, 0], Bob=[0, 1, 0], Carol=[0, 0, 1])


class MatchTests(unittest.TestCase):
    def test_a_clear_voice_gets_its_name(self):
        out = si.match({"speaker_0": [0.95, 0.1, 0.05]}, PEOPLE, threshold=0.5, margin=0.1)
        self.assertEqual(out["speaker_0"]["name"], "Alice")
        self.assertEqual(out["speaker_0"]["runner_up"], "Bob")

    def test_a_stranger_stays_unnamed_but_reports_the_nearest(self):
        out = si.match({"speaker_0": [0.4, 0.4, 0.4]}, PEOPLE, threshold=0.7, margin=0.05)
        self.assertIsNone(out["speaker_0"]["name"])
        self.assertGreater(out["speaker_0"]["score"], 0.5)

    def test_two_similar_candidates_are_not_a_coin_toss(self):
        out = si.match({"speaker_0": [0.7, 0.68, 0]}, PEOPLE, threshold=0.5, margin=0.08)
        self.assertIsNone(out["speaker_0"]["name"])

    def test_one_person_is_never_given_to_two_speakers(self):
        out = si.match({"speaker_0": [1, 0.05, 0], "speaker_1": [0.9, 0.1, 0.1]},
                       PEOPLE, threshold=0.5, margin=0.05)
        names = [v["name"] for v in out.values()]
        self.assertEqual(names.count("Alice"), 1)
        # The closer voice keeps the name.
        self.assertEqual(out["speaker_0"]["name"], "Alice")

    def test_every_cluster_is_answered(self):
        out = si.match({"a": [1, 0, 0], "b": [0, 0, -1]}, PEOPLE)
        self.assertEqual(set(out), {"a", "b"})
        self.assertIsNone(out["b"]["name"])

    def test_no_profiles_means_no_names(self):
        self.assertIsNone(si.match({"a": [1, 0, 0]}, {})["a"]["name"])


class LabelTests(unittest.TestCase):
    DANA = {"labels": {"Acme": "Dana (Acme)"}}

    def test_a_work_call_uses_the_work_name(self):
        self.assertEqual(si.choose_label("Dana", self.DANA, ["Dana", "Alice (Acme)"]),
                         "Dana (Acme)")

    def test_a_personal_call_uses_the_plain_name(self):
        self.assertEqual(si.choose_label("Dana", self.DANA, ["Dana", "Carol"]), "Dana")

    def test_alone_is_the_plain_name(self):
        self.assertEqual(si.choose_label("Dana", self.DANA, ["Dana"]), "Dana")

    def test_a_profile_without_labels_is_its_own_name(self):
        self.assertEqual(si.choose_label("Carol", {}, ["Alice (Acme)"]), "Carol")


class CleanSpanTests(unittest.TestCase):
    def seg(self, speaker, start, end):
        return {"speaker": speaker, "start": start, "end": end}

    def test_a_lone_turn_is_kept_less_its_edges(self):
        spans = si.clean_spans([self.seg("speaker_0", 0.0, 5.0)])
        self.assertEqual(spans, {"speaker_0": [(0.1, 4.9)]})

    def test_crosstalk_is_cut_out_keeping_the_larger_side(self):
        spans = si.clean_spans([self.seg("speaker_0", 0.0, 10.0), self.seg("speaker_1", 2.0, 3.0)])
        self.assertEqual(spans["speaker_0"], [(3.0, 9.9)])
        self.assertNotIn("speaker_1", spans)  # its whole turn was crosstalk

    def test_long_turns_are_capped_and_longest_come_first(self):
        spans = si.clean_spans([self.seg("a", 0, 30), self.seg("a", 40, 42.5)], longest=10)
        self.assertEqual(spans["a"], [(0.1, 10.1), (40.1, 42.4)])

    def test_slivers_are_dropped(self):
        self.assertEqual(si.clean_spans([self.seg("a", 0, 1.0)]), {})


class SimilarPairTests(unittest.TestCase):
    def test_only_alike_pairs_are_reported_most_alike_first(self):
        pairs = si.similar_pairs(profiles(A=[1, 0, 0], A2=[0.95, 0.1, 0], B=[0, 1, 0],
                                          B2=[0.1, 0.9, 0.2]), above=0.6)
        self.assertEqual([(p["a"], p["b"]) for p in pairs], [("A", "A2"), ("B", "B2")])


class AliasTests(unittest.TestCase):
    def test_same_person_specs(self):
        aliases = si.parse_same_person(["Dana=Dana (Acme)"])
        self.assertEqual(aliases, {"Dana": "Dana", "Dana (Acme)": "Dana"})
        self.assertEqual(si.variants_of(aliases), {"Dana": {"Acme": "Dana (Acme)"}})


if __name__ == "__main__":
    unittest.main()
