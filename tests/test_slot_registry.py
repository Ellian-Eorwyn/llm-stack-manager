"""One slot map, and the tables that used to be twelve copies of it.

The same slot -> prefix -> unit relationship was written out independently in
six Python modules and six shell scripts, each in its own vocabulary: the unit
is `chat-backend-dense`, the budget model prices `chat-primary`, the setup
wizard installs `primary`, and the settings live under `CHAT_PRIMARY_`. Any
rename touched all twelve, which is the real reason renaming a slot was
painful.

`web/backends/slots.py` is the one place now, and the rest derive. That trade
is only worth making if the derivations are *proved* rather than trusted, so
every table below is asserted against the literal it replaced. A derivation
that quietly produces a different table is worse than the duplication was: the
duplication was at least visible.

When a slot is deliberately renamed, these literals change with it -- that is
the point, and it is one edit rather than twelve.
"""

from __future__ import annotations

import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "web"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import backends  # noqa: E402
import budget  # noqa: E402
import config_fields  # noqa: E402
import health  # noqa: E402
import public_api  # noqa: E402
import telemetry  # noqa: E402


class DerivedTableTests(unittest.TestCase):
    """Each table, against what it said before it was derived."""

    def test_the_telemetry_targets(self):
        self.assertEqual(telemetry.BACKEND_TARGETS, [
            {"name": "chat-primary",   "label": "Primary Backend", "port_key": "CHAT_BACKEND_PORT",
             "host_key": "CHAT_BACKEND_HOST",   "units": ["chat-backend-dense"]},
            {"name": "chat-secondary", "label": "Secondary Backend", "port_key": "CHAT2_BACKEND_PORT",
             "host_key": "CHAT2_BACKEND_HOST",  "units": ["chat-backend2"]},
            {"name": "embed",          "label": "Embedding",       "port_key": "EMBED_PORT",
             "host_key": "EMBED_BACKEND_HOST",  "units": ["embed"]},
            {"name": "rerank",         "label": "Reranker",        "port_key": "RERANK_PORT",
             "host_key": "RERANK_BACKEND_HOST", "units": ["rerank"]},
            {"name": "task",           "label": "Task Model",      "port_key": "TASK_PORT",
             "host_key": "TASK_BACKEND_HOST",   "units": ["task"]},
            {"name": "ocr",            "label": "OCR Model",       "port_key": "OCR_PORT",
             "host_key": "OCR_BACKEND_HOST",    "units": ["ocr"]},
        ])

    def test_the_secondary_backend_is_probed_where_it_actually_listens(self):
        """`CHAT_BACKEND2_PORT` was written in telemetry.py and nowhere else.

        The config field, the launcher and the proxy all say
        `CHAT2_BACKEND_PORT`, and no alias joined them -- so the panel probed
        the secondary backend on 8020 whatever the operator had configured, and
        looked correct only because the two defaults agreed. Deriving the key
        from the slot is what fixed it, and this is the assertion that keeps it
        fixed.
        """
        target = next(t for t in telemetry.BACKEND_TARGETS if t["name"] == "chat-secondary")
        self.assertEqual(target["port_key"], "CHAT2_BACKEND_PORT")
        self.assertEqual(target["port_key"], backends.SLOTS["chat-backend2"].port_key)
        for key in ("CHAT_BACKEND2_PORT", "CHAT_BACKEND2_HOST"):
            with self.subTest(key):
                self.assertNotIn(key, telemetry.DEFAULT_BACKEND_PORTS)
                self.assertNotIn(key, [t["host_key"] for t in telemetry.BACKEND_TARGETS])

    def test_the_default_ports(self):
        self.assertEqual(telemetry.DEFAULT_BACKEND_PORTS, {
            "CHAT_BACKEND_PORT": "8010", "CHAT2_BACKEND_PORT": "8020",
            "EMBED_PORT": "8005", "RERANK_PORT": "8006",
            "TASK_PORT": "8007", "OCR_PORT": "8009",
        })

    def test_the_budget_prefixes(self):
        self.assertEqual(budget.BACKEND_PREFIXES, {
            "chat-primary": "CHAT_PRIMARY", "chat-secondary": "CHAT2",
            "embed": "EMBED", "rerank": "RERANK", "task": "TASK", "ocr": "OCR",
            # No slot: the audio model is only ever a router child, and it is
            # priced anyway so the pre-flight VRAM total is not silently short.
            "asr": "ASR",
        })

    def test_the_configured_model_keys(self):
        self.assertEqual(public_api.CONFIGURED_MODEL_KEYS, {
            "chat-backend-dense": ("CHAT_PRIMARY_MODEL_PATH", "CHAT_DENSE_MODEL_PATH",
                                   "CHAT_MODEL_PATH"),
            "chat-backend2": ("CHAT2_MODEL_PATH",),
            "embed": ("EMBEDDING_MODEL_PATH",),
            "rerank": ("RERANKER_MODEL_PATH",),
            "task": ("TASK_MODEL_PATH",),
            "ocr": ("OCR_MODEL_PATH",),
        })

    def test_the_shared_chat_restart_list(self):
        self.assertEqual(config_fields.SHARED_CHAT_BACKEND_RESTART,
                         ["chat-backend-dense", "chat-backend2"])

    def test_the_setup_wizards_component_maps(self):
        import setup_engine
        self.assertEqual(setup_engine.COMPONENT_SERVICES, {
            "primary": ["chat-backend-dense", "chat-proxy"],
            "secondary": ["chat-backend2", "chat-proxy2"],
            "embedding": ["embed"], "reranker": ["rerank"],
            "task": ["task"], "ocr": ["ocr"],
            "glmocr-sdk": ["glmocr-sdk"], "playwright": ["playwright-server"],
            "honcho": ["honcho-api", "honcho-deriver"],
            "transcribe": ["transcript-backend"],
        })
        self.assertEqual(setup_engine.MODEL_ENV_KEYS, {
            "primary": ("CHAT_PRIMARY_MODEL_PATH", "CHAT_PRIMARY_MMPROJ_PATH"),
            "secondary": ("CHAT2_MODEL_PATH", "CHAT2_MMPROJ_PATH"),
            "embedding": ("EMBEDDING_MODEL_PATH", ""),
            "reranker": ("RERANKER_MODEL_PATH", ""),
            "task": ("TASK_MODEL_PATH", "TASK_MMPROJ_PATH"),
            "ocr": ("OCR_MODEL_PATH", "OCR_MMPROJ_PATH"),
        })

    def test_the_retired_second_embedding_slot_is_gone_from_the_wizard(self):
        # The slot went in 8f448f3; the checkbox that installed it did not, so
        # selecting it asked for a component with no unit and no launcher.
        import setup_engine
        for table in (setup_engine.ALL_COMPONENTS, setup_engine.MODEL_COMPONENTS,
                      setup_engine.COMPONENT_PORTS, setup_engine.COMPONENT_SERVICES):
            self.assertNotIn("embedding2", table)


class ShellAgreementTests(unittest.TestCase):
    """The shell has one list too, and it must name the same slots.

    Four scripts carried their own copy, and they had drifted -- two still
    named `think` and `nothink` as current services a year after they were
    retired. `scripts/stack-services.sh` is the shell's single source now, and
    this is the arrangement `update.sh` and `deploy.BACKEND_SENSITIVE_PATHS`
    already had: one list, asserted against the other side rather than kept in
    step by hand.
    """

    @classmethod
    def setUpClass(cls):
        cls.text = (ROOT / "scripts" / "stack-services.sh").read_text()

    def _array(self, name):
        block = re.search(rf"{name}=\((.*?)\n\)", self.text, re.S)
        self.assertIsNotNone(block, f"stack-services.sh lost {name}")
        return [line.strip() for line in block.group(1).splitlines()
                if line.strip() and not line.strip().startswith("#")]

    def test_the_shell_names_exactly_the_slots_in_the_registry(self):
        self.assertEqual(self._array("STACK_MODEL_BACKENDS"), list(backends.SLOTS))

    def test_every_script_that_needs_the_list_sources_it(self):
        """One remaining duplication, named rather than hidden.

        `restore-active-stack.sh` and `activate-selected-stack.sh` still map
        components to units themselves -- `has_component embedding && ... embed`
        -- which is `setup_engine.COMPONENT_SERVICES` written in shell, twice.
        Collapsing that needs Python on the install path and is a change of its
        own; what is fixed here is the flat service lists, which is what the
        rename actually touches.
        """
        for name in ("restore-active-stack.sh", "activate-selected-stack.sh",
                     "llm-stack-manager", "../update.sh"):
            with self.subTest(name):
                self.assertIn("stack-services.sh", (ROOT / "scripts" / name).read_text())

    def test_the_retired_units_are_named_as_retired_rather_than_as_current(self):
        retired = self._array("STACK_RETIRED_SERVICES")
        self.assertEqual(sorted(retired),
                         ["chat-backend", "chat-backend-moe", "embed2", "nothink", "think"])
        self.assertEqual([u for u in self._array("STACK_CORE_SERVICES") if u in retired], [])


class OneSourceTests(unittest.TestCase):
    """Nothing may state the relationship a second time."""

    def test_every_service_the_manager_lists_agrees_with_the_registry(self):
        import app
        for entry in app.SERVICES:
            slot = backends.SLOTS.get(entry["name"])
            if slot is None:
                continue
            with self.subTest(entry["name"]):
                self.assertEqual(entry["label"], slot.label)
                self.assertEqual(entry["group"], slot.group)
                self.assertEqual(entry.get("config_section", ""), slot.config_section)

    def test_the_llamacpp_service_list_is_exactly_the_slots(self):
        import app
        self.assertEqual(app.LLAMACPP_MODEL_SERVICES, list(backends.SLOTS))
        self.assertEqual(set(app.SERVICE_ENV_PREFIXES), set(backends.SLOTS))

    def test_health_probes_every_slot_where_the_slot_says_it_listens(self):
        for name, slot in backends.SLOTS.items():
            with self.subTest(name):
                probe = health.SERVICE_PROBES[name]
                self.assertEqual(probe["port_key"], slot.port_key)
                self.assertEqual(probe["default_port"], slot.port_default)

    def test_the_router_members_that_are_slots_name_the_slots_prefix(self):
        for prefix, unit in telemetry.ROUTER_MEMBER_UNITS.items():
            slot = backends.SLOTS.get(unit)
            if slot is None:
                continue  # ASR is a router child with no unit of its own.
            with self.subTest(unit):
                self.assertEqual(slot.prefix, prefix)

    def test_every_slot_has_the_four_names_the_stack_calls_it_by(self):
        for name, slot in backends.SLOTS.items():
            with self.subTest(name):
                self.assertTrue(slot.name and slot.prefix and slot.budget)
                self.assertTrue(slot.component, "a slot the wizard cannot install")
                self.assertTrue(slot.label and slot.config_section)

    def test_the_names_are_unique_in_every_vocabulary(self):
        for attribute in ("name", "prefix", "budget", "component", "port_key"):
            values = [getattr(slot, attribute) for slot in backends.SLOTS.values()]
            with self.subTest(attribute):
                self.assertEqual(len(set(values)), len(values), values)


if __name__ == "__main__":
    unittest.main()
