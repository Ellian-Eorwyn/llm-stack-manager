"""The registry against the launchers it replaces, across configurations.

`tests/launcher-argv.golden.json` pins one command line per launcher. That is
the right shape for catching a change to what ships, and the wrong shape for
this question: whether `backends.build_command` reproduces nine hundred lines of
shell for configurations nobody happened to capture.

So every case here runs the real launcher and the registry against the same env
and compares argv. The shell is the specification. When these three scripts are
deleted, this file is what says the deletion was safe -- and it keeps saying it
afterwards for the two aux slots that were migrated the same way.

The cases are chosen where the shell branches: a cleared optional key, a device
name that a Metal build cannot use, a template id that names no file, an
operator's custom argument that the launcher steps aside for.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from launcher_harness import LauncherSandbox  # noqa: E402
from platform_harness import as_darwin, as_linux  # noqa: E402
from test_speculative import env_of  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "web"))
import backends  # noqa: E402

LAUNCHERS = {
    "llm-a": "start-llm-a.sh",
    "llm-b":      "start-llm-b.sh",
    "task":               "start-task.sh",
    "embed":              "start-embed.sh",
    "rerank":             "start-rerank.sh",
    "ocr":                "start-ocr.sh",
}

LARGE = ("llm-a", "llm-b", "task")

#: Which prefix each large slot's operator-facing keys live under. The cases
#: below are written once and applied to each, which is the point: these three
#: launchers were the same script three times.
PREFIX = {"llm-a": "LLM_A", "llm-b": "LLM_B", "task": "TASK"}

TEMPLATE = "config/chat-templates/demo.jinja"


def QUOTED(value: str) -> str:
    """A JSON value the way the env file carries it.

    `update_env_values` writes `KEY="[]"`, and the quotes are load-bearing:
    unquoted, `source` strips the inner ones and the launcher's JSON parse
    fails silently into an empty list.
    """
    return "'" + value + "'"


class SlotEquivalenceTests(unittest.TestCase):

    def _compare(self, slot, env=None, touch=(), platform="Linux", expect_rc=0):
        script = LAUNCHERS[slot]
        box = LauncherSandbox(platform=platform)
        try:
            for relative in touch:
                path = box.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            box.write_env({key: (value.replace("@STACK@", str(box.root))
                                 if isinstance(value, str) else value)
                           for key, value in (env or {}).items()})
            shell_argv, said, rc = box.run(script)
            values = env_of(box.env_file)
            values["STACK_DIR"] = str(box.root)
            context = as_darwin if platform == "Darwin" else as_linux
            if expect_rc:
                self.assertEqual(rc, expect_rc, f"{slot}: launcher exited {rc}")
                with context(), self.assertRaises(SystemExit):
                    backends.build_command(slot, values)
                return
            self.assertEqual(rc, 0, f"{slot}: launcher exited {rc}: {said[-400:]}")
            with context():
                built = backends.build_command(slot, values, said=[])
            self.assertEqual(box.normalise(built[1:]), box.normalise(shell_argv), slot)
        finally:
            box.cleanup()

    def _for_each_large(self, name, keys, **kwargs):
        for slot in LARGE:
            with self.subTest(case=name, slot=slot):
                self._compare(slot, env={f"{PREFIX[slot]}_{k}": v for k, v in keys.items()},
                              **kwargs)

    # -- the shipped configuration ------------------------------------------

    def test_every_slot_as_shipped(self):
        for slot in LAUNCHERS:
            with self.subTest(slot):
                self._compare(slot)

    # -- the branches ------------------------------------------------------

    def test_a_cleared_device_and_a_set_one(self):
        self._for_each_large("device set", {"DEVICE": "CUDA1"})
        self._for_each_large("device cleared", {"DEVICE": ""})

    def test_a_cuda_device_name_on_a_metal_build(self):
        # The shipped default is CUDA0, and passing it to a Metal build fails
        # at load -- after exec, so it looks like a crash loop.
        self._for_each_large("cuda on metal", {"DEVICE": "CUDA0"}, platform="Darwin")
        self._for_each_large("metal on metal", {"DEVICE": "MTL0"}, platform="Darwin")

    def test_auto_fit_on_with_a_minimum_context(self):
        self._for_each_large("fit on", {"FIT": "on", "FIT_CTX": "4096"})
        self._for_each_large("fit off", {"FIT": "off", "FIT_CTX": "4096"})

    def test_a_full_swa_cache(self):
        self._for_each_large("swa on", {"SWA_FULL": "on"})

    def test_cache_reuse_at_zero_is_the_same_as_unset(self):
        self._for_each_large("reuse 0", {"CACHE_REUSE": "0"})
        self._for_each_large("reuse set", {"CACHE_REUSE": "512"})

    def test_the_offload_flags_both_ways_round(self):
        self._for_each_large("offload off", {"KV_OFFLOAD": "off", "OP_OFFLOAD": "off",
                                             "MMPROJ_OFFLOAD": "off"})

    def test_an_idle_slot_cache_turned_off(self):
        self._for_each_large("idle off", {"CACHE_IDLE_SLOTS": "off"})

    def test_a_fit_target(self):
        self._for_each_large("fit target", {"FIT_TARGET": "20000"})

    # -- thinking and templates --------------------------------------------

    def test_a_reasoning_effort_the_template_would_raise_on(self):
        self._for_each_large("bad effort", {"REASONING_EFFORT": "extreme"})
        self._for_each_large("good effort", {"REASONING_EFFORT": "medium"})

    def test_thinking_off_on_the_task_slot(self):
        # The task slot says enable_thinking where the chat slots say
        # preserve_thinking, and its "off" is a value it passes rather than an
        # argument it omits.
        self._compare("task", env={"TASK_THINKING": "off"})
        self._compare("task", env={"TASK_THINKING": "on", "TASK_REASONING_EFFORT": "xhigh"})

    def test_preserve_thinking_off_on_the_chat_slots(self):
        for slot in ("llm-a", "llm-b"):
            with self.subTest(slot):
                self._compare(slot, env={f"{PREFIX[slot]}_PRESERVE_THINKING": "off"})

    def test_a_chat_template_id_that_resolves(self):
        self._compare("llm-a", env={"LLM_A_TEMPLATE_ID": "demo"},
                      touch=[TEMPLATE])
        self._compare("llm-b", env={"LLM_B_TEMPLATE_ID": "demo"}, touch=[TEMPLATE])
        self._compare("task", env={"TASK_CHAT_TEMPLATE_ID": "demo"}, touch=[TEMPLATE])

    def test_a_chat_template_id_that_names_no_file_is_refused_by_both(self):
        self._compare("llm-a", env={"LLM_A_TEMPLATE_ID": "missing"},
                      expect_rc=1)

    def test_a_chat_template_id_that_could_escape_the_directory_is_refused(self):
        self._compare("llm-a", env={"LLM_A_TEMPLATE_ID": "../../etc/passwd"},
                      expect_rc=1)

    # -- the operator's own arguments --------------------------------------

    def test_the_launcher_steps_aside_for_a_custom_jinja(self):
        self._compare("llm-b", env={"LLM_B_JINJA": "on",
                                            "LLM_B_CUSTOM_ARGS_JSON": QUOTED('["--jinja"]')})

    def test_the_launcher_steps_aside_for_custom_template_kwargs(self):
        self._compare("llm-b", env={
            "LLM_B_CUSTOM_ARGS_JSON": QUOTED('["--chat-template-kwargs {\\"a\\": 1}"]')})

    def test_the_launcher_steps_aside_for_a_custom_chat_template(self):
        self._compare("llm-b", env={"LLM_B_TEMPLATE_ID": "demo",
                                            "LLM_B_CUSTOM_ARGS_JSON": QUOTED('["--chat-template-file /x"]')})

    def test_custom_arguments_come_last(self):
        self._compare("task", env={"TASK_CUSTOM_ARGS_JSON": QUOTED('["--verbose", "--seed 7"]')})

    def test_the_primary_slots_custom_arguments_resolve_through_the_chain(self):
        """Worth pinning, because it only works by a side effect.

        The launcher resolves
        `${LLM_A_CUSTOM_ARGS_JSON:-${CHAT_CUSTOM_ARGS_JSON:-[]}}` into a
        shell variable and then reads `os.environ["CHAT_CUSTOM_ARGS_JSON"]`
        from a Python heredoc -- the unresolved name. It gets the resolved
        value anyway, because `EnvironmentFile=` put that name in the
        environment already and assigning to an exported variable updates what
        children see. Run the script without the unit's environment and the
        primary slot's custom arguments vanish.

        The registry resolves the chain directly, so it does not depend on
        that.
        """
        self._compare("llm-a",
                      env={"LLM_A_CUSTOM_ARGS_JSON": QUOTED('["--verbose"]')})

    def test_the_shared_custom_arguments_still_reach_the_primary_slot(self):
        self._compare("llm-a", env={"LLM_A_CUSTOM_ARGS_JSON": None,
                                                 "CHAT_CUSTOM_ARGS_JSON": QUOTED('["--verbose"]')})

    # -- identity ----------------------------------------------------------

    def test_the_legacy_prefix_is_read_behind_the_primary_one(self):
        # A config written before the rename sets the bare CHAT_* names, and
        # the primary slot has to keep starting from it.
        self._compare("llm-a", env={
            "LLM_A_CTX_SIZE": None, "LLM_A_TEMP": None,
            "CHAT_CTX_SIZE": "16384", "CHAT_TEMP": "0.3"})

    def test_the_dense_spelling_is_read_behind_both(self):
        # Only these five keys were ever spelled CHAT_DENSE_*.
        self._compare("llm-a", env={
            "LLM_A_CTX_SIZE": None, "LLM_A_MODEL_NAME": None,
            "CHAT_DENSE_CTX_SIZE": "24576", "CHAT_DENSE_MODEL_NAME": "old-name"})

    def test_an_mmproj_that_exists(self):
        self._for_each_large("mmproj", {"MMPROJ_PATH": "@STACK@/models/mm.gguf"},
                             touch=["models/mm.gguf"])

    def test_a_cleared_mmproj_is_not_inherited(self):
        self._compare("llm-a",
                      env={"LLM_A_MMPROJ_PATH": "",
                           "CHAT_MMPROJ_PATH": "@STACK@/models/mm.gguf"},
                      touch=["models/mm.gguf"])

    def test_a_port_and_host_that_are_not_the_defaults(self):
        self._compare("llm-a", env={"CHAT_BACKEND_PORT": "9010",
                                                 "CHAT_BACKEND_HOST": "10.0.0.1"})
        self._compare("llm-b", env={"CHAT2_BACKEND_PORT": "9020"})


if __name__ == "__main__":
    unittest.main()
