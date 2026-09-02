"""One slot map, and the tables that used to be twelve copies of it.

The same slot -> prefix -> unit relationship was written out independently in
six Python modules and six shell scripts, each in its own vocabulary: the unit
is `llm-a`, the budget model prices `llm-a`, the setup
wizard installs `primary`, and the settings live under `LLM_A_`. Any
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
import subprocess
import sys
import tempfile
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
            {"name": "llm-a",   "label": "LLM A", "port_key": "CHAT_BACKEND_PORT",
             "host_key": "CHAT_BACKEND_HOST",   "units": ["llm-a"]},
            {"name": "llm-b", "label": "LLM B", "port_key": "CHAT2_BACKEND_PORT",
             "host_key": "CHAT2_BACKEND_HOST",  "units": ["llm-b"]},
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
        target = next(t for t in telemetry.BACKEND_TARGETS if t["name"] == "llm-b")
        self.assertEqual(target["port_key"], "CHAT2_BACKEND_PORT")
        self.assertEqual(target["port_key"], backends.SLOTS["llm-b"].port_key)
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
            "llm-a": "LLM_A", "llm-b": "LLM_B",
            "embed": "EMBED", "rerank": "RERANK", "task": "TASK", "ocr": "OCR",
            # No slot: the audio model is only ever a router child, and it is
            # priced anyway so the pre-flight VRAM total is not silently short.
            "asr": "ASR",
        })

    def test_the_configured_model_keys(self):
        self.assertEqual(public_api.CONFIGURED_MODEL_KEYS, {
            "llm-a": ("LLM_A_MODEL_PATH", "CHAT_PRIMARY_MODEL_PATH",
                      "CHAT_DENSE_MODEL_PATH", "CHAT_MODEL_PATH"),
            "llm-b": ("LLM_B_MODEL_PATH", "CHAT2_MODEL_PATH"),
            "embed": ("EMBEDDING_MODEL_PATH",),
            "rerank": ("RERANKER_MODEL_PATH",),
            "task": ("TASK_MODEL_PATH",),
            "ocr": ("OCR_MODEL_PATH",),
        })

    def test_the_shared_chat_restart_list(self):
        self.assertEqual(config_fields.SHARED_CHAT_BACKEND_RESTART,
                         ["llm-a", "llm-b"])

    def test_the_setup_wizards_component_maps(self):
        import setup_engine
        self.assertEqual(setup_engine.COMPONENT_SERVICES, {
            "llm-a": ["llm-a", "llm-a-proxy"],
            "llm-b": ["llm-b", "llm-b-proxy"],
            "embedding": ["embed"], "reranker": ["rerank"],
            "task": ["task"], "ocr": ["ocr"],
            "glmocr-sdk": ["glmocr-sdk"], "playwright": ["playwright-server"],
            "transcribe": ["transcript-backend"],
        })
        self.assertEqual(setup_engine.MODEL_ENV_KEYS, {
            "llm-a": ("LLM_A_MODEL_PATH", "LLM_A_MMPROJ_PATH"),
            "llm-b": ("LLM_B_MODEL_PATH", "LLM_B_MMPROJ_PATH"),
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
        """The flat service lists, sourced rather than copied.

        `activate-selected-stack.sh` still maps components to units itself --
        `has_component embedding && ... embed` -- which is
        `setup_engine.COMPONENT_SERVICES` written out in shell. The boot path no
        longer does: `restore-active-stack.sh` asks
        `scripts/lib/boot-services.py`, which is the Python on the install path
        that collapsing it needed. The remaining copy is named here rather than
        hidden.
        """
        for name in ("restore-active-stack.sh", "activate-selected-stack.sh",
                     "llm-stack-manager", "../update.sh"):
            with self.subTest(name):
                self.assertIn("stack-services.sh", (ROOT / "scripts" / name).read_text())

    def test_the_boot_path_does_not_spell_out_the_component_map(self):
        """The boot path is the one that runs unattended, so it is the one that
        must not drift. It used to decide the boot set from an environment
        variable `llm-stack-restore.service` never sets, whose unset branch
        started every unit including a second large backend."""
        script = (ROOT / "scripts" / "restore-active-stack.sh").read_text()
        self.assertIn("boot-services.py", script)
        for spelling in ("selected embedding", "selected secondary", "selected reranker"):
            self.assertNotIn(spelling, script)

    def test_the_retired_units_are_named_as_retired_rather_than_as_current(self):
        retired = self._array("STACK_RETIRED_SERVICES")
        self.assertEqual(sorted(retired),
                         ["chat-backend", "chat-backend-moe", "embed2",
                          "honcho-api", "honcho-deriver", "nothink", "think"])
        self.assertEqual([u for u in self._array("STACK_CORE_SERVICES") if u in retired], [])


class SlotRenameTests(unittest.TestCase):
    """`CHAT_PRIMARY_*` / `CHAT2_*` became `LLM_A_*` / `LLM_B_*`.

    The rename is only safe because a config written before it still reads. The
    hub can write to a peer now, so a fleet can span the gap in both directions:
    an older hub sending `CHAT_PRIMARY_CTX_SIZE` to a renamed host has to land,
    which is why the old names go in before the fleet needs them and not after.
    """

    def setUp(self):
        import config_fields
        self.fields = config_fields

    def test_every_renamed_field_answers_to_its_old_name(self):
        renamed = [f["key"] for f in self.fields.CONFIG_FIELDS
                   if f.get("key", "").startswith(("LLM_A_", "LLM_B_"))]
        self.assertTrue(renamed, "no renamed fields found — the map would be vacuous")
        for key in renamed:
            with self.subTest(key):
                aliases = self.fields.NEW_ENV_KEY_LEGACY_ALIASES[key]
                self.assertTrue(
                    any(a.startswith(("CHAT_PRIMARY_", "CHAT2_")) for a in aliases),
                    f"{key} has no pre-rename spelling: {aliases}")

    def test_the_map_is_flat_and_not_a_chain(self):
        """`tests/test_llm_stack_manager.py` asserts the invariant; this says
        why it matters here. `CHAT_DENSE_MODEL_PATH` points straight at
        `LLM_A_MODEL_PATH`, not at `CHAT_PRIMARY_MODEL_PATH`, which is itself
        legacy now -- a chain would make each rename one lookup deeper."""
        self.assertEqual(
            self.fields.LEGACY_ENV_KEY_MAP["CHAT_DENSE_MODEL_PATH"], "LLM_A_MODEL_PATH")
        self.assertEqual(
            self.fields.LEGACY_ENV_KEY_MAP["CHAT_MOE_CTX_SIZE"], "LLM_B_CTX_SIZE")
        self.assertFalse(set(self.fields.LEGACY_ENV_KEY_MAP.values())
                         & set(self.fields.LEGACY_ENV_KEY_MAP))

    def test_the_port_and_host_keys_kept_their_names(self):
        """Section 2.1 freezes the ports, and these keys are the port contract:
        the proxies, telemetry and health all dial them by name."""
        for key in ("CHAT2_BACKEND_PORT", "CHAT2_BACKEND_HOST"):
            with self.subTest(key):
                self.assertNotIn(key, self.fields.LEGACY_ENV_KEY_MAP)

    def test_the_launcher_still_reads_a_config_written_before_the_rename(self):
        """The end-to-end promise: nothing but old keys, and the slot still
        resolves its model and context."""
        env = {"CHAT_PRIMARY_MODEL_PATH": "/models/old.gguf",
               "CHAT_PRIMARY_CTX_SIZE": "131072"}
        slot = backends.SLOTS["llm-a"]
        self.assertEqual(slot.absolute(slot.model_keys)[1], "CHAT_PRIMARY_MODEL_PATH")
        self.assertIn("CHAT_PRIMARY_CTX_SIZE", slot.key_chains["CTX_SIZE"][1])


class ExampleMergeTests(unittest.TestCase):
    """`install.sh` appends example defaults for keys a config lacks.

    A rename inverts the direction that guard has to cover, and getting it wrong
    is not a missing setting -- it is a *shadowed* one. `normalize_env_keys`
    backfills a canonical key from its legacy twin only when the canonical is
    absent, so writing an example default there silently replaces whatever the
    operator had. On the llm-a/llm-b rename it appended 65 keys over a working
    config and pointed the primary backend at a model file that does not exist.
    """

    def setUp(self):
        import config_fields
        self.fields = config_fields
        self.example = (ROOT / "config" / "llm-stack.env.example").read_text()

    def _example_keys(self):
        return [line.split("=", 1)[0] for line in self.example.splitlines()
                if line and not line.startswith("#") and "=" in line]

    #: The old spelling each canonical prefix has to answer to, and which an
    #: existing config is most likely to be holding. Named rather than inferred:
    #: asserting only that *some* alias exists passed throughout the window in
    #: which `LLM_A_TENSOR_SPLIT` knew `CHAT_PRIMARY_TENSOR_SPLIT` and not
    #: `CHAT_TENSOR_SPLIT`, which is the spelling every host actually had.
    LIVE_ALIAS_PREFIX = {"LLM_A_": "CHAT_", "LLM_B_": "CHAT2_"}

    def test_every_renamed_example_key_knows_its_old_spelling(self):
        """The check `install.sh` makes. A key here with no older name is one
        that would be appended over a config still using the old one."""
        checked = 0
        for key in self._example_keys():
            for new_prefix, old_prefix in self.LIVE_ALIAS_PREFIX.items():
                if not key.startswith(new_prefix):
                    continue
                checked += 1
                alias = old_prefix + key[len(new_prefix):]
                with self.subTest(key):
                    self.assertIn(alias, self.fields.legacy_names_for(key),
                                  f"{key} would be written over {alias}")
        self.assertTrue(checked, "no renamed example keys — the check would be vacuous")

    def test_the_helper_covers_keys_no_field_declares(self):
        """`LLM_B_TEMP` is in the example and in no field list, so the declared
        alias map alone does not see it -- which is exactly the gap that let six
        keys through after the first attempt at this fix."""
        self.assertIn("CHAT2_TEMP", self.fields.legacy_names_for("LLM_B_TEMP"))
        self.assertIn("CHAT_PRIMARY_TEMP", self.fields.legacy_names_for("LLM_A_TEMP"))
        self.assertIn("CHAT_TEMP", self.fields.legacy_names_for("LLM_A_TEMP"))

    def test_the_frozen_port_keys_are_not_treated_as_renamed(self):
        for key in ("CHAT2_BACKEND_PORT", "CHAT2_BACKEND_HOST",
                    "CHAT_BACKEND_PORT", "CHAT_BACKEND_HOST"):
            with self.subTest(key):
                self.assertEqual(self.fields.legacy_names_for(key), ())
                self.assertEqual(self.fields.canonical_name_for(key), key)
        # And from the other side: no `LLM_*` key claims them as an old spelling.
        for key in ("LLM_A_BACKEND_PORT", "LLM_B_BACKEND_PORT"):
            with self.subTest(key):
                self.assertEqual(self.fields.legacy_names_for(key), ())

    def test_the_bare_chat_prefix_is_read_but_never_written(self):
        """`CHAT_*` is llm-a's second legacy prefix, so `legacy_names_for` has to
        know it or the example rename shadows live values. It must not reach the
        write side: the bare prefix is shared with `CHAT_BEE_*`, which names no
        slot, and rewriting those would invent `LLM_A_BEE_*` keys."""
        self.assertEqual(self.fields.canonical_name_for("CHAT_TENSOR_SPLIT"),
                         "CHAT_TENSOR_SPLIT")
        self.assertEqual(self.fields.canonical_name_for("CHAT_BEE_LABEL"), "CHAT_BEE_LABEL")
        self.assertNotIn("CHAT_BEE_LABEL", self.fields.LEGACY_ENV_KEY_MAP)

    def _run_merge(self, example_text: str, config_text: str) -> str:
        """`install.sh`'s own merge body, run against a throwaway pair.

        Extracted rather than reimplemented: a copy of the guard would pass
        while the shipped one was broken, which is the failure this whole file
        exists to prevent. The `except Exception` fallback in it degrades to no
        legacy awareness at all, so running the real body is also the only way
        to notice if the import ever stops working.
        """
        body = (ROOT / "install.sh").read_text().split("<<'PYMERGEDEFAULTS'\n", 1)[1]
        body = body.split("\nPYMERGEDEFAULTS", 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            example = pathlib.Path(tmp) / "example.env"
            config = pathlib.Path(tmp) / "llm-stack.env"
            example.write_text(example_text)
            config.write_text(config_text)
            subprocess.run([sys.executable, "-c", body, str(example), str(config),
                            str(ROOT), "llm"], check=True)
            return config.read_text()

    def test_an_example_default_is_never_appended_over_a_live_legacy_value(self):
        """The 65-key incident, as a test. A host on the far side of the rename
        holds `CHAT_TENSOR_SPLIT`; the example holds `LLM_A_TENSOR_SPLIT`. They
        are one setting, so appending the example's default does not fill a gap,
        it shadows the operator's value -- `normalize_env_keys` backfills the
        canonical key only when the canonical is absent."""
        merged = self._run_merge(
            "LLM_A_TENSOR_SPLIT=1,1\nLLM_B_TEMP=0.7\n",
            "CHAT_TENSOR_SPLIT=1,1.25\nCHAT2_TEMP=1.0\n")
        self.assertNotIn("LLM_A_TENSOR_SPLIT", merged)
        self.assertNotIn("LLM_B_TEMP", merged)
        self.assertIn("CHAT_TENSOR_SPLIT=1,1.25", merged)

    def test_a_genuinely_missing_key_is_still_appended(self):
        """Guard the guard: if the suppression matched everything, the test
        above would pass vacuously and installs would stop getting defaults."""
        merged = self._run_merge("LLM_A_TENSOR_SPLIT=1,1\n", "LLM_A_CTX_SIZE=32768\n")
        self.assertIn("LLM_A_TENSOR_SPLIT=1,1", merged)

    def test_canonical_name_for_is_flat_and_not_a_chain(self):
        """The prefix rule restated for the inverse: one lookup reaches the
        current name, from any spelling, for everything on disk anywhere."""
        candidates = set(self._example_keys()) | set(self.fields.LEGACY_ENV_KEY_MAP)
        candidates |= {"CHAT2_TEMP", "CHAT_PRIMARY_TEMP", "CHAT_BEE_LABEL"}
        for key in candidates:
            once = self.fields.canonical_name_for(key)
            with self.subTest(key):
                self.assertEqual(self.fields.canonical_name_for(once), once)


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


class InstallParityTests(unittest.TestCase):
    """The two platform branches of `install.sh` must offer the same choices.

    They did not. `install_mac_service` was called unconditionally for all
    thirteen services where the systemd branch gates each on
    `setup_has_component`, and the retired-service sweep existed only on the
    systemd side. So a Mac got agents for every slot whether or not the
    installer asked for them -- on a machine with no llama.cpp build, that is
    nine agents that fail on start and are reported as broken services rather
    than as services nobody wanted -- and it kept every retired agent forever,
    including four whose launchers were renamed out from under them.

    Asserted against the file rather than by running it: `install.sh` writes
    into `/etc/systemd/system` and `~/Library/LaunchAgents`, so the test that
    could run it is the one nobody would want to.
    """

    @classmethod
    def setUpClass(cls):
        cls.text = (ROOT / "install.sh").read_text()

    #: Component -> the services installing it brings up, from the wizard's own
    #: map. Anything here must be gated on both platforms.
    @property
    def optional(self):
        import setup_engine
        return {component: services
                for component, services in setup_engine.COMPONENT_SERVICES.items()
                if component in setup_engine.OPTIONAL_COMPONENTS
                or component in setup_engine.SLOT_BY_COMPONENT}

    def test_the_component_gate_is_defined_once_for_both_platforms(self):
        self.assertEqual(self.text.count("setup_has_component() {"), 1)

    def test_every_optional_service_is_gated_on_the_launchd_side(self):
        for component, services in self.optional.items():
            for service in services:
                with self.subTest(service=service):
                    call = re.search(
                        r"setup_has_component (\S+) && \\\n\s*install_mac_service "
                        rf'"{re.escape(service)}"', self.text)
                    self.assertIsNotNone(
                        call, f"install_mac_service {service} is not gated on a component")
                    self.assertEqual(call.group(1), component)

    def test_the_manager_itself_is_never_gated(self):
        # It is the thing being installed; a selection cannot exclude it.
        self.assertIn('\n    install_mac_service "llm-manager"', self.text)

    def test_both_platforms_remove_what_was_not_selected(self):
        for helper in ("remove_unselected_units", "remove_unselected_mac_services"):
            with self.subTest(helper):
                self.assertIn(f"{helper}() {{", self.text)
        for component in self.optional:
            with self.subTest(component=component):
                self.assertIn(f"remove_unselected_mac_services {component} ", self.text)

    def test_the_retired_list_is_stated_once_and_swept_on_both_platforms(self):
        self.assertEqual(self.text.count("RETIRED_SERVICES=("), 1)
        self.assertIn('for unit in "${RETIRED_SERVICES[@]}"', self.text)
        self.assertIn('for name in "${RETIRED_SERVICES[@]}"', self.text)

    def test_root_is_required_only_where_it_is_needed(self):
        """The macOS user domain writes nothing root owns.

        The check used to be unconditional, so `install.sh` on a Mac demanded
        sudo -- which would then write root-owned plists into a *user*
        LaunchAgents directory, and launchctl refuses to bootstrap those. Every
        chown on that path targets the operator's own uid and is `|| true`.
        """
        guard = re.search(r'if \[\[ "\$\{EUID\}" -ne 0 \]\](.*?)\nfi', self.text, re.S)
        self.assertIsNotNone(guard, "install.sh lost its root check entirely")
        self.assertIn("is_linux", guard.group(1))
        self.assertIn('LLM_LAUNCHD_DOMAIN:-user}" == "system"', guard.group(1))

    def test_the_renamed_chat_units_are_among_the_retired(self):
        """Their launchers were renamed with them, so an agent or unit left
        behind points at a script that no longer exists."""
        block = re.search(r"RETIRED_SERVICES=\((.*?)\n\)", self.text, re.S).group(1)
        for name in ("chat-backend-dense", "chat-backend2", "chat-proxy", "chat-proxy2"):
            with self.subTest(name):
                self.assertIn(name, block)
