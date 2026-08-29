"""The speculative decoding arguments, against the shell that produced them.

Three launchers carried ninety-three identical lines of this each. The
replacement is a Python module, so the check that matters is not "does it look
right" but "does it produce what the shell produced" -- for every method the UI
offers, not just the one the golden file happens to be configured with.

So each case runs the real launcher in a sandbox and compares its `--spec-*`
arguments against `speculative.build` given the same configuration. The shell
is the specification here; this is the diff.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from launcher_harness import LauncherSandbox  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "web"))
from backends import speculative  # noqa: E402
from config_fields import LLAMA_SPEC_METHOD_OPTIONS  # noqa: E402

PREFIXES = ("LLM_A", "CHAT")
DRAFT = "models/draft.gguf"


def env_of(path: pathlib.Path) -> dict:
    """The rendered env file as the shell would have it.

    `start-backend.sh` sources it under `set -a`, so every assignment reaches a
    child process -- including the empty ones, which is the whole point of
    `empty_is_set`.
    """
    out = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key.strip()] = value
    return out


def spec_tail(argv: list[str]) -> list[str]:
    """Everything from the first `--spec-` argument onward.

    The launcher appends SPEC_ARGS after OPTS and before CUSTOM_ARGS, and the
    sandbox's custom args are empty, so the tail is exactly the block under
    test.
    """
    for i, arg in enumerate(argv):
        if arg.startswith("--spec-"):
            return argv[i:]
    return []


class ShellEquivalenceTests(unittest.TestCase):
    """Every method the UI offers, plus the ones it does not."""

    #: The alias no UI ever wrote but configs still carry; a comma pair, which
    #: is what the `SPEC_NGRAM_MOD` hint describes; and a method this build has
    #: never heard of, which the shell passed straight through to --spec-type.
    EXTRA_METHODS = ("mtp", "draft-mtp,ngram-mod", "some-future-method")

    def _compare(self, method, **extra):
        box = LauncherSandbox()
        try:
            (box.root / DRAFT).touch()
            box.write_env({
                "LLM_A_SPEC_METHOD": method,
                "LLM_A_SPEC_DRAFT_MODEL_PATH": str(box.root / DRAFT),
                **extra,
            })
            argv, said, rc = box.run("start-llm-a.sh")
            self.assertEqual(rc, 0, f"{method}: launcher exited {rc}: {said[-400:]}")
            from_shell = box.normalise(spec_tail(argv))
            args, _said = speculative.build(env_of(box.env_file), PREFIXES)
            from_python = box.normalise(args)
        finally:
            box.cleanup()
        self.assertEqual(from_python, from_shell, method)

    def test_every_offered_method_produces_the_same_arguments(self):
        for method in LLAMA_SPEC_METHOD_OPTIONS:
            with self.subTest(method):
                self._compare(method)

    def test_the_methods_the_ui_does_not_offer_agree_too(self):
        for method in self.EXTRA_METHODS:
            with self.subTest(method):
                self._compare(method)

    def test_the_tuning_values_are_read_not_defaulted(self):
        # Every group at a non-default value at once, so a flag reading the
        # wrong key cannot hide behind a default that happens to match.
        self._compare("ngram-mod",
                      LLM_A_SPEC_DRAFT_N_MAX="9",
                      LLM_A_SPEC_DRAFT_N_MIN="2",
                      LLM_A_SPEC_DRAFT_P_MIN="0.5",
                      LLM_A_SPEC_DRAFT_P_SPLIT="0.25",
                      LLM_A_SPEC_DRAFT_TYPE_K="q8_0",
                      LLM_A_SPEC_DRAFT_TYPE_V="q8_0",
                      LLM_A_SPEC_NGRAM_MOD_N_MATCH="7",
                      LLM_A_SPEC_NGRAM_MOD_N_MIN="8",
                      LLM_A_SPEC_NGRAM_MOD_N_MAX="9")

    def test_the_legacy_prefix_is_read_behind_the_slots_own(self):
        # The launcher resolves ${LLM_A_X:-${CHAT_X:-default}}; the
        # module walks the same two prefixes.
        self._compare("ngram-simple", LLM_A_SPEC_NGRAM_SIZE_N=None,
                      CHAT_SPEC_NGRAM_SIZE_N="31")

    def test_a_draft_device_is_appended_only_when_set(self):
        self._compare("draft-model", LLM_A_SPEC_DRAFT_DEVICES="CUDA1")
        self._compare("draft-model")


class DraftModelRefusalTests(unittest.TestCase):
    """Two configurations refuse to build a command at all.

    That is command construction, where the rule is that it must be right or
    not run -- the opposite of the preflight helpers, which degrade to
    permissive so that a helper can never stop a backend from starting.
    """

    def _shell(self, **env):
        box = LauncherSandbox()
        try:
            box.write_env(env)
            _argv, said, rc = box.run("start-llm-a.sh")
            return rc, said, env_of(box.env_file)
        finally:
            box.cleanup()

    def test_an_empty_draft_path_is_refused_by_both(self):
        rc, said, env = self._shell(LLM_A_SPEC_METHOD="draft-model",
                                    LLM_A_SPEC_DRAFT_MODEL_PATH="")
        self.assertEqual(rc, 1)
        self.assertIn("is empty", said)
        with self.assertRaises(SystemExit) as raised:
            speculative.build(env, PREFIXES)
        self.assertIn("is empty", str(raised.exception))

    def test_a_missing_draft_file_is_refused_by_both(self):
        rc, said, env = self._shell(LLM_A_SPEC_METHOD="draft-model",
                                    LLM_A_SPEC_DRAFT_MODEL_PATH="/nope/draft.gguf")
        self.assertEqual(rc, 1)
        self.assertIn("not found", said)
        with self.assertRaises(SystemExit) as raised:
            speculative.build(env, PREFIXES)
        self.assertIn("not found", str(raised.exception))

    def test_draft_mtp_needs_no_draft_model(self):
        # A GGUF carrying its own blk.N.nextn.* head runs MTP with no sidecar;
        # requiring a path would refuse exactly that configuration.
        rc, _said, env = self._shell(LLM_A_SPEC_METHOD="draft-mtp",
                                     LLM_A_SPEC_DRAFT_MODEL_PATH="")
        self.assertEqual(rc, 0)
        args, _ = speculative.build(env, PREFIXES)
        self.assertIn("--spec-type", args)
        self.assertNotIn("--spec-draft-model", args)

    def test_the_three_that_do_need_one_say_so(self):
        for method in speculative.NEEDS_DRAFT_MODEL:
            with self.subTest(method):
                rc, said, env = self._shell(LLM_A_SPEC_METHOD=method,
                                            LLM_A_SPEC_DRAFT_MODEL_PATH="")
                self.assertEqual(rc, 1, said[-300:])
                with self.assertRaises(SystemExit):
                    speculative.build(env, PREFIXES)


class MethodNameTests(unittest.TestCase):

    def test_map_k_does_not_match_map_k4v(self):
        # The shell tested `,${M},` against `*,ngram-map-k,*`, which is a
        # membership test and not a substring one. A substring test would give
        # ngram-map-k4v both groups.
        self.assertEqual(speculative._names("ngram-map-k4v"), ["ngram-map-k4v"])
        self.assertNotIn("ngram-map-k", speculative._names("ngram-map-k4v"))

    def test_mtp_is_the_one_alias(self):
        self.assertEqual(speculative.method({"X_SPEC_METHOD": "mtp"}, ("X",)), "draft-mtp")
        self.assertEqual(speculative.method({}, ("X",)), "off")

    def test_off_produces_nothing_at_all(self):
        self.assertEqual(speculative.build({"X_SPEC_METHOD": "off"}, ("X",)), ([], []))


if __name__ == "__main__":
    unittest.main()
