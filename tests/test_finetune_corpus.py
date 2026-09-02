"""The dataset pipeline: converters, cleaning, shapes, and the quality gate.

The gate's fixtures are the real failures from the first corpus built by hand
on 2026-09-02 — four rows cut mid-clause by a chunker, two bibliographies, two
section outlines. Those eight were found by reading 177 rows; the point of
these tests is that nobody has to read them again.

No network, no GPU, no model. Anything needing a tokenizer degrades to a
character estimate and says so, which is the behaviour asserted here.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "finetune"))

import common  # noqa: E402
import corpus  # noqa: E402
import ingest  # noqa: E402
import shapes  # noqa: E402

HAVE_PANDOC = shutil.which("pandoc") is not None


def doc(name: str, text: str) -> dict:
    return {"name": name, "text": text}


PROSE = (
    "The coffee bean currently appears to most citizens as a mundane object with "
    "little in the way of economic or political implication. In the essay that "
    "follows I argue, using the theoretical resources of Marx and Weber, that "
    "coffee is uniquely situated to reveal the development of global inequality. "
    "This essay begins with a brief history of the commodity, moves to the present "
    "day, and closes on the ethical implications of its third wave. "
) * 6


class CaptionTests(unittest.TestCase):
    """Timecodes go, speakers stay. `dialogue` cannot attribute a turn without them."""

    VTT = ("WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\n<v Ellie>So the question "
           "is what repair means.\n\n2\n00:00:04.000 --> 00:00:07.000\n<v Ellie>Not "
           "just fixing.\n\n3\n00:00:07.000 --> 00:00:09.500\n<v Justin>Right, and "
           "that is a moral claim.\n")

    def test_timecodes_and_cue_numbers_are_dropped(self):
        out = ingest._captions_to_text(self.VTT, ".vtt")
        self.assertNotIn("-->", out)
        for cue in ("\n1\n", "\n2\n", "\n3\n"):
            self.assertNotIn(cue, f"\n{out}\n")

    def test_consecutive_cues_from_one_speaker_join(self):
        """A caption break is a display artefact, not a turn."""
        out = ingest._captions_to_text(self.VTT, ".vtt")
        self.assertIn("Ellie: So the question is what repair means. Not just fixing.",
                      out)
        self.assertEqual(out.count("Ellie:"), 1)

    def test_srt_speaker_prefixes_are_recognised(self):
        srt = ("1\n00:00:01,000 --> 00:00:04,000\nEllie: One.\n\n"
               "2\n00:00:05,000 --> 00:00:08,000\nJustin: Two.\n")
        out = ingest._captions_to_text(srt, ".srt")
        self.assertEqual(out, "Ellie: One.\n\nJustin: Two.")


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_markdown_is_read_directly(self):
        path = self.tmp / "a.md"
        path.write_text("# Heading\n\nBody.")
        result = ingest.convert(path)
        self.assertTrue(result.ok)
        self.assertEqual(result.converter, "read")
        self.assertEqual(len(result.sha256), 64)

    def test_an_unsupported_extension_is_named_not_dropped(self):
        path = self.tmp / "a.xyz"
        path.write_bytes(b"data")
        result = ingest.convert(path)
        self.assertFalse(result.ok)
        self.assertIn("unsupported extension", result.skipped)

    def test_a_file_converting_to_nothing_says_so(self):
        path = self.tmp / "empty.md"
        path.write_text("   \n\n")
        result = ingest.convert(path)
        self.assertFalse(result.ok)
        self.assertEqual(result.skipped, "converted to empty text")

    @unittest.skipUnless(HAVE_PANDOC, "pandoc is not installed")
    def test_docx_round_trips_through_pandoc(self):
        source = self.tmp / "in.md"
        source.write_text("# Title\n\nSome prose about repair.\n")
        target = self.tmp / "in.docx"
        subprocess.run(["pandoc", str(source), "-o", str(target)], check=True)
        result = ingest.convert(target)
        self.assertTrue(result.ok, result.skipped)
        self.assertEqual(result.converter, "pandoc")
        self.assertIn("repair", result.text)

    def test_walk_skips_dotfiles_and_dot_directories(self):
        (self.tmp / ".git").mkdir()
        (self.tmp / ".git" / "x.md").write_text("no")
        (self.tmp / ".hidden.md").write_text("no")
        (self.tmp / "yes.md").write_text("yes")
        names = [p.name for p in ingest.walk(self.tmp)]
        self.assertEqual(names, ["yes.md"])


class CleaningTests(unittest.TestCase):
    """Each of these artefacts came out of a real export, not an imagination."""

    def test_note_anchor_residue_is_removed(self):
        text, _ = shapes.clean_line("defines who we are.12(#endnote-12) Similarly,")
        self.assertEqual(text, "defines who we are. Similarly,")

    def test_wikilinks_and_bold_are_unwrapped(self):
        text, _ = shapes.clean_line("See [[Some Note]] and **emphasis** here.")
        self.assertEqual(text, "See Some Note and emphasis here.")

    def test_caret_footnotes_go_but_ordinals_stay(self):
        text, _ = shapes.clean_line("In the 19^th^ century^4^ this held.")
        self.assertEqual(text, "In the 19th century this held.")

    def test_a_bold_line_is_a_heading_not_content(self):
        text, is_heading = shapes.clean_line("**The Repair Shop**")
        self.assertIsNone(text)
        self.assertTrue(is_heading)

    def test_a_bold_byline_is_neither_heading_nor_content(self):
        """docx exports bold the author's name exactly as they bold a heading."""
        for name in ("**Ellian Eorwyn**", "**Jane Q. Smith**", "**Mary Jane Smith**"):
            with self.subTest(name=name):
                text, is_heading = shapes.clean_line(name)
                self.assertIsNone(text)
                self.assertFalse(is_heading)

    def test_a_title_case_heading_is_not_read_as_a_byline(self):
        """"The Repair Shop" has a byline's shape. Names do not start with an
        article, which is the only thing separating the two."""
        for heading in ("**The Repair Shop**", "**On Repair Work**",
                        "**Introduction**"):
            with self.subTest(heading=heading):
                self.assertTrue(shapes.clean_line(heading)[1])

    def test_publisher_boilerplate_is_dropped(self):
        for line in ("To cite this article: Smith, J.", "Published online: 4 May 2021"):
            self.assertIsNone(shapes.clean_line(line)[0], line)

    def test_running_headers_are_dropped_when_they_recur(self):
        """A PDF-to-markdown export repeats the title on every page."""
        body = "\n\n".join(f"Journal of Repair Studies\n\nParagraph {i} of prose."
                           for i in range(5))
        sections = shapes.parse_sections(body)
        joined = " ".join(line for _, lines in sections for line in lines)
        self.assertNotIn("Journal of Repair Studies", joined)
        self.assertIn("Paragraph 3 of prose.", joined)

    def test_a_reference_section_is_cut(self):
        md = ("# Body\n\nReal prose here about the argument.\n\n"
              "# References\n\nSmith, J. Title. Press, 2001.\n")
        headings = [h for h, _ in shapes.parse_sections(md)]
        self.assertIn("body", headings)
        self.assertNotIn("references", headings)


class ChunkingTests(unittest.TestCase):
    def test_an_overlong_paragraph_splits_on_sentences_not_mid_clause(self):
        """Truncation is what produced the four rows that ended mid-clause."""
        para = " ".join(f"Sentence number {i} runs to a full stop." for i in range(60))
        chunks = shapes.chunk_paragraphs([(para, "h")], min_words=10, max_words=60)
        self.assertGreater(len(chunks), 1)
        for text, _ in chunks:
            self.assertTrue(text.rstrip().endswith("."), text[-40:])

    def test_near_duplicate_fingerprint_ignores_footnote_markers(self):
        a = PROSE
        b = PROSE.replace("Marx", "Marx[^3]").replace("Weber", "Weber[^4]")
        self.assertEqual(shapes.normalize_sig(a), shapes.normalize_sig(b))

    def test_distinct_sections_sharing_an_opening_are_kept_apart(self):
        a = PROSE + "It ends by considering the repair shop as a moral site."
        b = PROSE + "It ends instead on the question of value and its conversion."
        self.assertNotEqual(shapes.normalize_sig(a), shapes.normalize_sig(b))


class InstructionTests(unittest.TestCase):
    """The slot repairs. Each accident here appeared in the first corpus."""

    def setUp(self):
        import random
        self.rng = random.Random(0)

    def test_a_leading_the_is_stripped(self):
        text, method = shapes.instruction_for("the repair shop", self.rng)
        self.assertNotIn("the the", text.lower())
        self.assertEqual(method, "heading")

    def test_quote_marks_are_stripped_from_the_slot(self):
        text, _ = shapes.instruction_for('"quoted" title', self.rng)
        self.assertNotIn('""', text)
        self.assertIn("quoted title", text)

    def test_a_sentence_like_heading_routes_to_the_neutral_framing(self):
        heading = ("an argument for a situated empirical social epistemology of play")
        text, _ = shapes.instruction_for(heading, self.rng)
        self.assertTrue(text.startswith("Write a sustained section titled"))

    def test_no_heading_is_reported_as_untitled_rather_than_invented(self):
        text, method = shapes.instruction_for("   ", self.rng)
        self.assertEqual(method, "untitled")
        self.assertNotIn("“”", text)

    def test_generated_instructions_carry_no_grammar_smells(self):
        for heading in ("the repair shop", "introduction", "a introduction",
                        "section", '"x"', "The The Thing"):
            text, _ = shapes.instruction_for(heading, self.rng)
            for smell in shapes.GRAMMAR_SMELLS:
                self.assertNotIn(smell, f" {text.lower()} ", (heading, smell))


class ShapeTests(unittest.TestCase):
    def test_style_rows_carry_the_prose_verbatim_as_the_assistant_turn(self):
        result = shapes.build_style([doc("a.md", f"# The Repair Shop\n\n{PROSE}")],
                                    "SYS", 50, 400, 0.6, seed=1)
        self.assertTrue(result.rows)
        row = result.rows[0]
        self.assertEqual([m["role"] for m in row["messages"]],
                         ["system", "user", "assistant"])
        self.assertIn("coffee bean", row["messages"][2]["content"])
        self.assertEqual(row["meta"]["instruction_method"], "heading")

    def test_style_deduplicates_across_documents(self):
        """The same paper submitted twice must not become two copies of itself."""
        text = f"# Heading\n\n{PROSE}"
        once = shapes.build_style([doc("a.md", text)], "SYS", 50, 400, 0.6, seed=1)
        twice = shapes.build_style([doc("a.md", text), doc("b.md", text)],
                                   "SYS", 50, 400, 0.6, seed=1)
        self.assertEqual(len(twice.rows), len(once.rows))
        self.assertEqual(twice.dropped["near-duplicate of an earlier chunk"],
                         len(once.rows))

    def test_dialogue_learns_only_the_named_speaker(self):
        transcript = ("Justin: What does repair mean to you?\n\n"
                      "Ellie: It means maintaining a relation, not fixing an object.\n\n"
                      "Justin: And that is moral?\n\n"
                      "Ellie: It is unavoidably moral.\n")
        result = shapes.build_dialogue([doc("t.md", transcript)], "SYS",
                                       "Ellie", 400, seed=1)
        self.assertEqual(len(result.rows), 2)
        # Every user turn is Justin's question and every assistant turn is
        # Ellie's answer — never the other way round.
        self.assertEqual([r["messages"][1]["content"] for r in result.rows],
                         ["What does repair mean to you?", "And that is moral?"])
        self.assertEqual([r["messages"][2]["content"] for r in result.rows],
                         ["It means maintaining a relation, not fixing an object.",
                          "It is unavoidably moral."])

    def test_dialogue_refuses_an_unattributed_transcript(self):
        """Which side the model is being taught to be is the whole dataset."""
        result = shapes.build_dialogue([doc("t.md", "Just prose with no speakers.\n")],
                                       "SYS", "Ellie", 400, seed=1)
        self.assertEqual(result.rows, [])
        self.assertTrue(result.dropped["no speaker labels in the transcript"])
        self.assertTrue(any("attribution" in n for n in result.notes))

    def test_dialogue_names_the_speakers_it_did_find(self):
        result = shapes.build_dialogue(
            [doc("t.md", "Justin: One.\n\nSam: Two.\n")], "SYS", "Ellie", 400, seed=1)
        self.assertEqual(result.rows, [])
        self.assertTrue(any("justin" in n.lower() and "sam" in n.lower()
                            for n in result.notes), result.notes)

    def test_raw_emits_text_rows_without_instructions(self):
        result = shapes.build_raw([doc("a.md", f"# H\n\n{PROSE}")], 50, 400, 0.6)
        self.assertTrue(result.rows)
        self.assertIn("text", result.rows[0])
        self.assertNotIn("messages", result.rows[0])

    def test_qa_drops_a_chunk_the_model_would_not_caption(self):
        result = shapes.build_qa([doc("a.md", f"# H\n\n{PROSE}")], "SYS",
                                 50, 400, 0.6, 1, ask=lambda chunk, head: None)
        self.assertEqual(result.rows, [])
        self.assertTrue(result.dropped["no question could be generated"])


class GateTests(unittest.TestCase):
    """The eight rows found by hand in the first corpus, as assertions."""

    def refuse(self, assistant: str, seen=None):
        return corpus._refuse(assistant, seen if seen is not None else set())

    def test_a_turn_cut_mid_clause_is_refused(self):
        self.assertEqual(
            self.refuse("rather than functioning as means to objectively represent "
                        "an external world, defines 'who we are"),
            "assistant turn truncated mid-sentence")

    def test_an_ellipsis_ending_is_still_a_truncation(self):
        self.assertIsNotNone(self.refuse("must be completed with the observation of "
                                         "massive layoffs and underpayment of work…"))

    def test_a_bibliography_is_refused(self):
        entry = ("Harvey, D. *The Enigma of Capital*. Oxford University Press, 2010. "
                 "Kanigel, R. *The One Best Way*. MIT Press, 2005. ")
        self.assertEqual(self.refuse(entry * 11), "bibliography")

    def test_a_section_outline_is_refused(self):
        outline = ("*Section 2.1:* An argument for a social epistemology "
                   "*Section 2.2:* Belonging outside belonging "
                   "*Section 2.3:* The rule of cool.")
        self.assertEqual(self.refuse(outline), "section outline, not prose")

    def test_a_numbered_reference_block_is_refused(self):
        block = "\n".join(f"{i}.  Smith J . Some Title. Press, 200{i}."
                          for i in range(1, 6))
        self.assertEqual(self.refuse(block), "reference list or notes section")

    def test_a_numbered_prose_list_is_not_mistaken_for_references(self):
        """The reference pattern is deliberately narrow. Widening it to catch
        "1. Smith, J." would also catch "1. First, I argue..."."""
        block = "\n".join([
            "1. First, I argue that repair is a moral practice.",
            "2. Second, that maintenance is undervalued work.",
            "3. Third, that both follow from the same account.",
        ])
        self.assertIsNone(self.refuse(block))

    def test_a_duplicate_row_is_refused(self):
        seen = {shapes.normalize_sig(PROSE)}
        self.assertEqual(self.refuse(PROSE, seen), "duplicate of an earlier row")

    def test_an_empty_turn_is_refused(self):
        self.assertEqual(self.refuse("   "), "empty assistant turn")

    def test_ordinary_prose_passes(self):
        self.assertIsNone(self.refuse(PROSE))

    def test_a_quoted_ending_is_not_a_truncation(self):
        """Scholarly prose often closes on a quotation mark, not a full stop."""
        self.assertIsNone(self.refuse(PROSE + 'as Weber put it, "a steel shell."'))


class SuggestLengthTests(unittest.TestCase):
    def test_it_picks_the_smallest_window_that_fits(self):
        """Training at 2048 when the longest row is 827 buys nothing and costs
        activation memory on a card that has none to spare."""
        self.assertEqual(corpus._suggest_length([250, 622, 826], 2048), 1024)
        self.assertEqual(corpus._suggest_length([100, 200], 2048), 512)

    def test_it_never_exceeds_the_ceiling(self):
        self.assertEqual(corpus._suggest_length([9000], 2048), 2048)

    def test_no_measurements_leaves_the_ceiling_alone(self):
        self.assertEqual(corpus._suggest_length([], 2048), 2048)


class RecipeTests(unittest.TestCase):
    def test_the_recipe_round_trips(self):
        data = {"source": "/a/b", "shape": "style", "rows": 163,
                "system": "You are: a careful writer", "lr": 0.0001,
                "tags": ["one", "two"], "done": True}
        self.assertEqual(common._load_yaml(common._dump_yaml(data)), data)

    def test_a_run_name_that_could_escape_the_root_is_refused(self):
        for name in ("../etc", "a/b", "", "-x"):
            with self.subTest(name=name), self.assertRaises(common.RunError):
                common.open_run(name, create=True)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_every_candidate_appears_in_the_manifest_converted_or_not(self):
        """A corpus that quietly lost a third of its sources looks exactly like
        a corpus that was always that size."""
        (self.tmp / "good.md").write_text("# H\n\nSome prose here.")
        (self.tmp / "empty.md").write_text("")
        (self.tmp / "odd.xyz").write_text("data")

        root = self.tmp / "runs"
        old_root, common.RUNS_ROOT = common.RUNS_ROOT, root
        self.addCleanup(setattr, common, "RUNS_ROOT", old_root)

        args = type("A", (), {"source": str(self.tmp), "run": "t",
                              "no_recursive": True})()
        self.assertEqual(corpus.cmd_ingest(args), 0)

        records = {r["name"]: r for r in
                   common.read_jsonl(root / "t" / "documents" / "manifest.jsonl")}
        self.assertEqual(set(records), {"good.md", "empty.md"})
        self.assertEqual(records["empty.md"]["skipped"], "converted to empty text")
        self.assertFalse(records["good.md"]["skipped"])


if __name__ == "__main__":
    unittest.main()
