"""What comes back after a reboot, and who gets to decide it.

`llm-stack-restore.service` runs `restore-active-stack.sh` with no environment,
and that script used to read the boot set from `LLM_STACK_SELECTED_COMPONENTS`
with an else-branch that started *everything* when the variable was unset --
which it always was, on the one code path where it mattered. On the production
host that meant every boot started `llm-b`, a second 27B model, onto a
GPU already holding the primary. `config/service-expectations.json` had recorded
`off` for that unit for months. Nothing on the boot path read the file.

So the rule under test is an ordering of authorities, and most of these tests
are about which source wins when two of them disagree. The regression itself is
`test_a_unit_switched_off_is_not_started_by_a_bare_boot`.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "web"))

import setup_engine  # noqa: E402


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


boot = _load("boot_services", "scripts/lib/boot-services.py")


#: The shape of the production host at the time of the cutover: the primary and
#: its proxy on, the secondary pair deliberately off, the router pooling the
#: four auxiliary models, transcription on.
LLMS_EXPECTATIONS = {
    "llm-a": {"expected": "on"},
    "llm-a-proxy": {"expected": "on"},
    "llm-b": {"expected": "off"},
    "llm-b-proxy": {"expected": "off"},
    "embed": {"expected": "on"},
    "embed2": {"expected": "off"},
    "llama-router": {"expected": "on"},
    "transcript-backend": {"expected": "on"},
    "glmocr-sdk": {"expected": "off"},
    "ocr": {"expected": "off"},
    "task": {"expected": "off"},
}

LLMS_ENV = {
    "MODEL_ROUTER_ENABLED": "on",
    "MODEL_ROUTER_MEMBERS": "EMBED,OCR,RERANK,TASK",
    "TRANSCRIPT_ENABLED": "on",
}


class AuthorityTests(unittest.TestCase):
    """Which source decides, when two of them say different things."""

    def units(self, env=None, expectations=None, **over):
        return boot.boot_units({**LLMS_ENV, **(env or {}), **over},
                               expectations=LLMS_EXPECTATIONS
                               if expectations is None else expectations)

    def test_a_unit_switched_off_is_not_started_by_a_bare_boot(self):
        """The regression. No environment, and the secondary stays down.

        This is the exact call `llm-stack-restore.service` makes: no
        `LLM_STACK_SELECTED_COMPONENTS`, nothing overriding anything.
        """
        units = self.units()
        self.assertNotIn("llm-b", units)
        self.assertNotIn("llm-b-proxy", units)
        self.assertIn("llm-a", units)

    def test_off_outranks_a_component_that_selects_the_unit(self):
        """Selecting `secondary` does not override a deliberate stop."""
        units = self.units(LLM_STACK_SELECTED_COMPONENTS="primary,secondary")
        self.assertNotIn("llm-b", units)

    def test_on_outranks_a_selection_that_never_mentioned_the_unit(self):
        """`transcribe` is not in this host's wizard selection.

        It predates the transcription sidecar, so the only thing that knows the
        operator wants it is the expectation file.
        """
        units = self.units(LLM_STACK_SELECTED_COMPONENTS="primary")
        self.assertIn("transcript-backend", units)

    def test_a_feature_switch_vetoes_a_recorded_on(self):
        """A stale `on` must not outlive the switch that made it possible.

        The sidecar's start script exits 0 without launching anything when
        `TRANSCRIPT_ENABLED=off`, so starting the unit would only produce a
        service that cannot come up.
        """
        self.assertNotIn("transcript-backend", self.units(TRANSCRIPT_ENABLED="off"))
        self.assertIn("transcript-backend", self.units(TRANSCRIPT_ENABLED="on"))

    def test_the_router_unit_answers_to_its_own_switch(self):
        """Starting it against `MODEL_ROUTER_ENABLED=off` gives the pooled
        models two owners, so a recorded `on` cannot bring it up."""
        self.assertNotIn("llama-router", self.units(MODEL_ROUTER_ENABLED="off"))

    def test_a_pooled_model_is_never_started_as_a_unit(self):
        """It is a child of the router and would fight nginx for its port."""
        self.assertNotIn("embed", self.units())

    def test_a_model_the_router_stops_pooling_becomes_a_unit_again(self):
        units = self.units(MODEL_ROUTER_ENABLED="off")
        self.assertIn("embed", units)
        self.assertNotIn("llama-router", units)

    def test_a_backend_starts_before_the_proxy_in_front_of_it(self):
        units = self.units(LLM_STACK_SELECTED_COMPONENTS="primary,secondary",
                           expectations={})
        self.assertLess(units.index("llm-a"), units.index("llm-a-proxy"))
        self.assertLess(units.index("llm-b"), units.index("llm-b-proxy"))

    def test_a_saved_profile_can_name_a_different_primary_unit(self):
        units = boot.boot_units(LLMS_ENV, expectations=LLMS_EXPECTATIONS,
                                chat_backend="chat-backend-moe")
        self.assertIn("chat-backend-moe", units)
        self.assertNotIn("llm-a", units)


class FallbackTests(unittest.TestCase):
    """What answers when the config files do not."""

    def test_a_host_with_no_expectations_falls_back_to_the_selection(self):
        units = boot.boot_units({**LLMS_ENV,
                                 "LLM_STACK_SELECTED_COMPONENTS": "primary,playwright"},
                                expectations={})
        self.assertEqual(units, ["llm-a", "llm-a-proxy",
                                 "llama-router", "playwright-server"])

    def test_the_fallback_matches_the_other_boot_path(self):
        """Two boot paths that disagree about the default is how a host comes
        back from a reboot in a state nobody chose."""
        script = (ROOT / "scripts" / "activate-selected-stack.sh").read_text()
        match = re.search(r"LLM_STACK_SELECTED_COMPONENTS:-([a-z,\-]+)\}", script)
        self.assertIsNotNone(match, "activate-selected-stack.sh changed shape")
        self.assertEqual(match.group(1), boot.FALLBACK_COMPONENTS)

    def test_an_empty_result_is_never_silently_returned_as_success(self):
        """The caller refuses to stop anything on an empty list, so an empty
        list must only happen when it is genuinely correct."""
        units = boot.boot_units({"MODEL_ROUTER_ENABLED": "off"},
                                expectations={}, chat_backend="llm-a")
        self.assertIn("llm-a", units)


class KeySetTests(unittest.TestCase):
    """The tables this derives from, asserted key-for-key."""

    def test_every_unit_a_component_installs_can_be_started(self):
        installable = {unit for units in setup_engine.COMPONENT_SERVICES.values()
                       for unit in units}
        self.assertEqual(installable - set(boot.START_ORDER), set())

    def test_the_start_order_names_only_units_the_shell_knows(self):
        """`stack-services.sh` is the shell's one list; anything started must be
        stoppable by the same script's stop loop."""
        text = (ROOT / "scripts" / "stack-services.sh").read_text()
        known = set(re.findall(r"^\s{2}([a-z0-9-]+)$", text, re.MULTILINE))
        self.assertEqual(set(boot.START_ORDER) - known, set())

    def test_every_pooled_member_maps_to_a_real_router_member_name(self):
        members = {"EMBED", "OCR", "RERANK", "TASK"}
        self.assertEqual(set(boot.ROUTER_MEMBER_BY_UNIT.values()), members)

    def test_every_feature_switch_names_a_unit_in_the_start_order(self):
        self.assertEqual(set(boot.FEATURE_SWITCH) - set(boot.START_ORDER), set())


if __name__ == "__main__":
    unittest.main()
