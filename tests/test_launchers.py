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
LOADED = pathlib.Path(__file__).resolve().parent / "launcher-loaded-model.golden.json"

LAUNCHERS = {
    "llm-a": "start-llm-a.sh",
    "llm-b":      "start-llm-b.sh",
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


class LoadedModelTests(unittest.TestCase):
    """The same launchers, on a host where the model file is actually there.

    The argv golden runs against an empty `models/`, so every branch gated on
    `-f "${model}"` is dead in it: `--mmproj` never appears, `add_swa_full_opt`
    short-circuits, and `preflight_report` returns before it says anything. That
    is a third of what a launcher decides, unpinned.

    So this runs them again with the model files present and `budget.py`
    replaced by a stub that records the question. It pins two things the first
    golden cannot see: the `--mmproj` argument, and the backend name and
    settings the launcher reports the memory fit against.
    """

    @classmethod
    def setUpClass(cls):
        cls.sandbox = LauncherSandbox(record_budget=True)
        cls.golden = json.loads(LOADED.read_text())
        cls.captured = {}
        for slot, script in LAUNCHERS.items():
            argv, said, rc = cls.sandbox.run(script)
            cls.captured[slot] = {
                "argv": cls.sandbox.normalise(argv),
                "preflight": cls.sandbox.preflight_call(said),
                "said": said,
                "rc": rc,
            }

    @classmethod
    def tearDownClass(cls):
        cls.sandbox.cleanup()

    def test_every_launcher_still_reaches_its_exec(self):
        for slot, run in self.captured.items():
            with self.subTest(slot):
                self.assertEqual(run["rc"], 0, f"{slot} exited {run['rc']}: {run['said'][-300:]}")

    def test_the_command_line_matches_the_golden_file(self):
        for slot in LAUNCHERS:
            with self.subTest(slot):
                self.assertEqual(self.captured[slot]["argv"], self.golden[slot]["argv"])

    def test_every_launcher_reports_its_memory_fit(self):
        """A launcher that says nothing to `budget.py` starts a backend nobody
        weighed. Silent, and invisible to the argv golden."""
        for slot, run in self.captured.items():
            with self.subTest(slot):
                self.assertTrue(run["preflight"], f"{slot} made no preflight report")

    def test_the_preflight_call_matches_the_golden_file(self):
        """The backend name and the settings, pinned.

        `start-backend.sh` passes five settings where the chat and task
        launchers pass fourteen, and reports the slot's own name where the
        budget model expects `llm-a`. Consolidating one into the other
        without noticing loses both, and nothing else would fail.
        """
        for slot in LAUNCHERS:
            with self.subTest(slot):
                self.assertEqual(
                    self.captured[slot]["preflight"], self.golden[slot]["preflight"],
                    f"{slot}'s preflight report changed. If deliberate, regenerate "
                    f"{LOADED.name} and say why in the commit.")

    def test_the_backend_name_is_one_the_budget_model_knows(self):
        # `budget.py` maps the name to an env prefix; an unknown one is a 400
        # from /api/budget and a report against the wrong slot's settings.
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "web"))
        import budget
        for slot, run in self.captured.items():
            call = run["preflight"]
            with self.subTest(slot):
                self.assertIn(call[call.index("--backend") + 1], budget.BACKEND_PREFIXES)

    def test_the_golden_file_covers_every_launcher(self):
        self.assertEqual(set(self.golden), set(LAUNCHERS))


class ClearedMeansClearedTests(unittest.TestCase):
    """An emptied key is a decision, and it is not the same as an absent one.

    The launchers spell out the difference: required settings resolve with
    `${NEW:-${OLD:-default}}`, so an empty value lands on a working default;
    optional ones -- the paths and tuning knobs where "unset" is a legitimate
    choice -- resolve with `${NEW-${OLD-}}`, so clearing the new key means
    cleared rather than silently inheriting whatever the legacy key still holds.

    That is not decoration. `--fit-ctx` kept being passed alongside `--fit off`
    long after it had been cleared in the UI, because one chain used the wrong
    one. Neither shape is visible in a single command line, so both are pinned
    here as a pair: the cleared case and the inherited case, from the same key.
    """

    MMPROJ = "models/mmproj-legacy.gguf"

    def _argv(self, touch=(), **env):
        """Run the primary launcher against an env built after the sandbox exists.

        `@STACK@` in an override is replaced with the sandbox root, so a test
        can point a key at a file it also creates -- which is the only way to
        exercise a branch gated on the file being there.
        """
        box = LauncherSandbox()
        try:
            for relative in touch:
                (box.root / relative).touch()
            box.write_env({key: (value.replace("@STACK@", str(box.root))
                                 if isinstance(value, str) else value)
                           for key, value in env.items()})
            argv, said, rc = box.run("start-llm-a.sh")
            self.assertEqual(rc, 0, said[-300:])
            return box.normalise(argv)
        finally:
            box.cleanup()

    def test_an_emptied_fit_ctx_is_not_inherited_from_the_legacy_key(self):
        argv = self._argv(CHAT_FIT="on", LLM_A_FIT_CTX="", CHAT_FIT_CTX="8192")
        self.assertNotIn("--fit-ctx", argv)

    def test_an_absent_fit_ctx_still_falls_back_to_the_legacy_key(self):
        # The control for the case above: without it, "no --fit-ctx" would also
        # pass if the flag had simply stopped being emitted at all.
        argv = self._argv(CHAT_FIT="on", LLM_A_FIT_CTX=None, CHAT_FIT_CTX="8192")
        self.assertEqual(argv[argv.index("--fit-ctx") + 1], "8192")

    def test_an_emptied_mmproj_is_not_inherited_from_the_legacy_key(self):
        # The file has to exist, or the launcher would drop `--mmproj` for the
        # wrong reason and this would pass however the chain resolved.
        argv = self._argv(touch=[self.MMPROJ], LLM_A_MMPROJ_PATH="",
                          CHAT_MMPROJ_PATH="@STACK@/" + self.MMPROJ)
        self.assertNotIn("--mmproj", argv)

    def test_an_absent_mmproj_still_falls_back_to_the_legacy_key(self):
        argv = self._argv(touch=[self.MMPROJ], LLM_A_MMPROJ_PATH=None,
                          CHAT_MMPROJ_PATH="@STACK@/" + self.MMPROJ)
        self.assertEqual(argv[argv.index("--mmproj") + 1], "@STACK@/" + self.MMPROJ)

    def test_an_emptied_required_setting_does_fall_through_to_its_default(self):
        # The other half of the rule. A context size is not optional, so an
        # empty one is a mistake to absorb, not an instruction to obey --
        # `--ctx-size ""` would be refused by llama-server.
        argv = self._argv(LLM_A_CTX_SIZE="", CHAT_CTX_SIZE="16384")
        self.assertEqual(argv[argv.index("--ctx-size") + 1], "16384")


if __name__ == "__main__":
    unittest.main()
