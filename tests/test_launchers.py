"""What each launcher actually execs, pinned.

Every backend launcher ends in `exec "${LLAMA_SERVER_BIN}" --model ... `, and
that argv is the whole contract between the configuration surface and the
running backend. Nothing asserted it, so a change to a default, a flag, or a
fallback chain could only be caught by starting the service and reading the
journal -- which costs a multi-GB model reload per attempt.

`tests/launcher-argv.golden.json` is that argv for all nine, captured by running
them against a stub binary that prints its arguments. A diff here is either a
deliberate change to what a backend runs, in which case regenerate the file and
say so in the commit, or a mistake.

This is what makes consolidating the launchers a safe refactor rather than a
hopeful one: the replacement has to produce the same command line.
"""

from __future__ import annotations

import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from launcher_harness import LauncherSandbox

GOLDEN = pathlib.Path(__file__).resolve().parent / "launcher-argv.golden.json"

LAUNCHERS = {
    "chat-backend-dense": "start-chat-backend-dense.sh",
    "chat-backend-moe":   "start-chat-backend-moe.sh",
    "chat-backend":       "start-chat-backend.sh",
    "chat-backend2":      "start-chat-backend2.sh",
    "embed":              "start-embed.sh",
    "rerank":             "start-rerank.sh",
    "task":               "start-task.sh",
    "ocr":                "start-ocr.sh",
}


class LauncherCommandTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sandbox = LauncherSandbox()
        cls.golden = json.loads(GOLDEN.read_text())
        cls.captured = {}
        for slot, script in LAUNCHERS.items():
            argv, said, rc = cls.sandbox.run(script)
            cls.captured[slot] = (cls.sandbox.normalise(argv), said, rc)

    @classmethod
    def tearDownClass(cls):
        cls.sandbox.cleanup()

    def test_every_launcher_reaches_its_exec(self):
        """A launcher that exits before exec starts no backend.

        This is not hypothetical: macOS ships bash 3.2, where expanding an empty
        array under `set -u` is an unbound-variable error, and six of these nine
        died on `"${SPEC_ARGS[@]}"` before reaching llama-server. Nothing caught
        it because nothing had run them on a Mac.
        """
        for slot, (argv, said, rc) in self.captured.items():
            with self.subTest(slot):
                self.assertEqual(rc, 0, f"{slot} exited {rc}: {said[-300:]}")
                self.assertTrue(argv, f"{slot} produced no command")

    def test_the_command_line_matches_the_golden_file(self):
        for slot in LAUNCHERS:
            with self.subTest(slot):
                self.assertEqual(
                    self.captured[slot][0], self.golden[slot],
                    f"{slot}'s command line changed. If deliberate, regenerate "
                    f"{GOLDEN.name} and say why in the commit.")

    def test_the_golden_file_covers_every_launcher(self):
        self.assertEqual(set(self.golden), set(LAUNCHERS))

    def test_every_launcher_names_a_model_and_a_port(self):
        # The two arguments without which a backend cannot be what it claims.
        for slot, (argv, _said, _rc) in self.captured.items():
            with self.subTest(slot):
                self.assertIn("--model", argv)
                self.assertIn("--port", argv)
                self.assertTrue(argv[argv.index("--model") + 1])

    def test_no_flag_is_passed_an_empty_string(self):
        """`--tensor-split ""` is read by llama.cpp as an explicit empty split
        and refused; the same shape of bug can hide behind any flag."""
        for slot, (argv, _said, _rc) in self.captured.items():
            for flag, value in zip(argv, argv[1:]):
                if flag.startswith("--") and not value.startswith("--"):
                    with self.subTest(slot=slot, flag=flag):
                        self.assertNotEqual(value, "", f"{slot} passes {flag} an empty value")


if __name__ == "__main__":
    unittest.main()
