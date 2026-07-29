from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import tempfile
import time
import unittest
from unittest.mock import patch


def _load(name: str, relative: str):
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


ROOT = pathlib.Path(__file__).resolve().parents[1]
scheduling = _load("llm_stack_manager_scheduling", "web/scheduling.py")


def env_for(**overrides) -> dict:
    values = {
        "CHAT_PRIMARY_N_PARALLEL": "2",
        "CHAT_PRIMARY_CTX_SIZE": "262144",
        "CHAT_PRIMARY_CACHE_RAM": "8192",
        "CHAT_PRIMARY_CTX_CHECKPOINTS": "8",
        "CHAT_PRIMARY_CACHE_IDLE_SLOTS": "on",
        "CHAT_PRIMARY_FIT": "off",
    }
    values.update(overrides)
    return values


class ContractParityTests(unittest.TestCase):
    def test_the_python_and_browser_contracts_agree(self):
        """One contract, written down twice, so it is checked twice.

        The browser evaluates the form as you type; this evaluates the running
        backend. If the two constants drift, the panel and the endpoint disagree
        about the same stack and neither is obviously wrong.
        """
        source = (ROOT / "web" / "static" / "cache-aware-scheduling.js").read_text(encoding="utf-8")
        block = re.search(r"const CONTRACT = Object\.freeze\(\{(.*?)\}\);", source, re.S)
        self.assertIsNotNone(block, "CONTRACT block not found in cache-aware-scheduling.js")
        pairs = dict(re.findall(r'(\w+):\s*"([^"]*)"', block.group(1)))
        self.assertEqual(pairs, scheduling.CONTRACT)

    def test_the_per_slot_floor_matches_the_browser(self):
        source = (ROOT / "web" / "static" / "cache-aware-scheduling.js").read_text(encoding="utf-8")
        floor = re.search(r"const MINIMUM_PER_SLOT_CONTEXT = (\d+);", source)
        self.assertEqual(int(floor.group(1)), scheduling.MINIMUM_PER_SLOT_CONTEXT)

    def test_the_stale_window_matches_pi_forge(self):
        # Both pi-forge runtimes hardcode 15 seconds and warn that a mismatch
        # means one of them ignores the other's leases.
        self.assertEqual(scheduling.LEASE_STALE_MS, 15_000)


class ContractCheckTests(unittest.TestCase):
    def test_a_compliant_configuration_passes(self):
        result = scheduling.contract_check(env_for())
        self.assertTrue(result["compatible"])
        self.assertEqual(result["per_slot_context"], 131072)

    def test_one_slot_cannot_pin_interactive_and_background_apart(self):
        result = scheduling.contract_check(env_for(CHAT_PRIMARY_N_PARALLEL="1"))
        self.assertFalse(result["compatible"])
        self.assertTrue(any("2 parallel slots" in issue for issue in result["issues"]))

    def test_context_is_judged_per_slot_not_in_total(self):
        # 65536 across two slots is 32768 each, exactly the floor; one token
        # less of total context is not.
        self.assertTrue(scheduling.contract_check(
            env_for(CHAT_PRIMARY_CTX_SIZE="65536"))["compatible"])
        self.assertFalse(scheduling.contract_check(
            env_for(CHAT_PRIMARY_CTX_SIZE="65534"))["compatible"])

    def test_idle_slot_caching_and_auto_fit_are_required(self):
        self.assertIn("idle-slot caching", " ".join(scheduling.contract_check(
            env_for(CHAT_PRIMARY_CACHE_IDLE_SLOTS="off"))["issues"]))
        self.assertIn("auto-fit", " ".join(scheduling.contract_check(
            env_for(CHAT_PRIMARY_FIT="on"))["issues"]))

    def test_an_unset_slot_count_divides_by_one_rather_than_zero(self):
        result = scheduling.contract_check({"CHAT_PRIMARY_CTX_SIZE": "131072"})
        self.assertEqual(result["per_slot_context"], 131072)
        self.assertTrue(any("2 parallel slots" in issue for issue in result["issues"]))


class LaunchedSettingsTests(unittest.TestCase):
    LIVE = ("/usr/local/bin/llama-server --model /models/qwen.gguf --ctx-size 262144 "
            "--parallel 2 --cache-ram 8192 --ctx-checkpoints 8 --fit off "
            "--cache-idle-slots --cache-reuse 256 --flash-attn on")

    def test_a_live_command_line_is_read_back(self):
        values = scheduling.launched_settings(self.LIVE)
        self.assertEqual(values["CTX_SIZE"], "262144")
        self.assertEqual(values["N_PARALLEL"], "2")
        self.assertEqual(values["CTX_CHECKPOINTS"], "8")
        self.assertEqual(values["FIT"], "off")
        self.assertEqual(values["CACHE_IDLE_SLOTS"], "on")

    def test_short_and_inline_spellings_are_understood(self):
        values = scheduling.launched_settings("llama-server -c=131072 -np 2 -ctxcp 4 -cram 4096")
        self.assertEqual(values["CTX_SIZE"], "131072")
        self.assertEqual(values["N_PARALLEL"], "2")
        self.assertEqual(values["CTX_CHECKPOINTS"], "4")
        self.assertEqual(values["CACHE_RAM"], "4096")

    def test_the_negated_flag_turns_idle_caching_off(self):
        values = scheduling.launched_settings("llama-server --no-cache-idle-slots")
        self.assertEqual(values["CACHE_IDLE_SLOTS"], "off")

    def test_nul_separated_proc_cmdline_parses(self):
        values = scheduling.launched_settings("llama-server\x00--parallel\x002\x00--ctx-size\x0065536")
        self.assertEqual((values["N_PARALLEL"], values["CTX_SIZE"]), ("2", "65536"))

    def test_an_absent_idle_flag_is_treated_as_the_default(self):
        result = scheduling.launched_check("llama-server --parallel 2 --ctx-size 262144 --fit off")
        self.assertTrue(result["compatible"])

    def test_a_process_that_could_not_be_read_reports_unavailable(self):
        result = scheduling.launched_check("")
        self.assertFalse(result["available"])


class DriftTests(unittest.TestCase):
    def test_a_saved_change_that_was_never_restarted_is_reported(self):
        configured = scheduling.contract_check(env_for(CHAT_PRIMARY_CTX_SIZE="131072"))
        launched = scheduling.launched_check(
            "llama-server --ctx-size 262144 --parallel 2 --fit off --cache-idle-slots")
        drift = scheduling.drift_between(configured, launched)
        self.assertEqual(len(drift), 1)
        self.assertIn("131072", drift[0])
        self.assertIn("262144", drift[0])

    def test_a_matching_pair_reports_nothing(self):
        configured = scheduling.contract_check(env_for())
        launched = scheduling.launched_check(
            "llama-server --ctx-size 262144 --parallel 2 --cache-ram 8192 "
            "--ctx-checkpoints 8 --fit off --cache-idle-slots")
        self.assertEqual(scheduling.drift_between(configured, launched), [])

    def test_a_stopped_backend_cannot_drift(self):
        configured = scheduling.contract_check(env_for())
        self.assertEqual(scheduling.drift_between(configured, scheduling.launched_check("")), [])


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = pathlib.Path(self._tmp.name)
        self.now_ms = 1_785_300_000_000

    def write(self, name: str, **fields) -> pathlib.Path:
        path = self.dir / name
        path.write_text(json.dumps(fields), encoding="utf-8")
        return path

    def test_a_fresh_lease_is_left_alone(self):
        import os
        self.write("1234-abcd.json", pid=os.getpid(), kind="interactive", slot=0,
                   updatedAtMs=self.now_ms - 1000)
        entries = scheduling.read_leases(self.dir, self.now_ms)
        self.assertEqual(entries[0]["classification"], "fresh")
        self.assertEqual(scheduling.reapable(entries), [])

    def test_a_dead_writer_past_the_threshold_is_an_orphan(self):
        # The real one: pid gone, 27 hours stale against a 15 second window.
        self.write("background-2485955-128662146551936.json", pid=2485955, kind="background",
                   slot=1, updatedAtMs=self.now_ms - 99_110_691)
        entries = scheduling.read_leases(self.dir, self.now_ms)
        self.assertEqual(entries[0]["classification"], "orphan")
        self.assertEqual(entries[0]["slot"], 1)
        self.assertEqual(len(scheduling.reapable(entries)), 1)

    def test_a_stale_lease_whose_process_lives_is_not_reaped(self):
        import os
        self.write("background-1-2.json", pid=os.getpid(), kind="background", slot=1,
                   updatedAtMs=self.now_ms - 3_600_000)
        entries = scheduling.read_leases(self.dir, self.now_ms)
        self.assertEqual(entries[0]["classification"], "stale")
        self.assertEqual(scheduling.reapable(entries), [])

    def test_both_writers_filename_shapes_are_read(self):
        # `background-<pid>-<clock>` from the JavaScript worker and
        # `background-<pid>-<thread id>` from the Python one. The trailing field
        # means different things, so the age never comes from the name.
        self.write("background-999999-1785198881179.json", pid=999999, kind="background",
                   slot=1, updatedAtMs=self.now_ms - 1000)
        self.write("background-999998-128662146551936.json", pid=999998, kind="background",
                   slot=1, updatedAtMs=self.now_ms - 1000)
        entries = scheduling.read_leases(self.dir, self.now_ms)
        self.assertEqual([e["classification"] for e in entries], ["fresh", "fresh"])

    def test_a_lease_without_a_kind_counts_as_interactive(self):
        # Both pi-forge runtimes default it that way, and an interactive claim
        # is the one that blocks background work.
        self.write("1234-abcd.json", pid=1, slot=0, updatedAtMs=self.now_ms)
        self.assertEqual(scheduling.read_leases(self.dir, self.now_ms)[0]["kind"], "interactive")

    def test_an_unreadable_lease_is_malformed_and_reaped_only_when_old(self):
        path = self.dir / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        entries = scheduling.read_leases(self.dir, self.now_ms)
        self.assertEqual(entries[0]["classification"], "malformed")
        self.assertEqual(scheduling.reapable(entries), [])
        import os
        old = time.time() - scheduling.LEASE_ORPHAN_SECONDS - 60
        os.utime(path, (old, old))
        self.assertEqual(len(scheduling.reapable(scheduling.read_leases(self.dir, self.now_ms))), 1)

    def test_reaping_removes_only_the_orphan(self):
        import os
        self.write("background-2485955-1.json", pid=2485955, kind="background", slot=1,
                   updatedAtMs=self.now_ms - 99_110_691)
        keep = self.write("5678-live.json", pid=os.getpid(), kind="interactive", slot=0,
                          updatedAtMs=time.time() * 1000)
        result = scheduling.reap_leases(self.dir)
        self.assertEqual(len(result["removed"]), 1)
        self.assertTrue(keep.exists())
        self.assertIn("is gone", result["removed"][0]["reason"])

    def test_a_dry_run_removes_nothing(self):
        path = self.write("background-2485955-1.json", pid=2485955, kind="background", slot=1,
                          updatedAtMs=time.time() * 1000 - 99_110_691)
        result = scheduling.reap_leases(self.dir, dry_run=True)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["removed"], [])
        self.assertTrue(path.exists())

    def test_a_missing_directory_is_not_an_error(self):
        self.assertEqual(scheduling.read_leases(self.dir / "nope"), [])
        summary = scheduling.lease_summary(self.dir / "nope")
        self.assertFalse(summary["exists"])
        self.assertEqual(summary["reapable"], 0)


class LeaseDirectoryTests(unittest.TestCase):
    def test_an_explicit_agent_directory_wins(self):
        directory = scheduling.lease_directory({"PI_FORGE_AGENT_DIR": "/srv/forge/agent"})
        self.assertEqual(directory, pathlib.Path("/srv/forge/agent/inference-leases"))

    def test_it_falls_back_to_the_stack_owner_not_the_manager(self):
        # The manager runs as root, so Path.home() would point somewhere the
        # lease directory has never existed.
        with patch.object(scheduling, "owner_home", return_value=pathlib.Path("/home/someone")):
            self.assertEqual(scheduling.lease_directory({}),
                             pathlib.Path("/home/someone/.pi-forge/agent/inference-leases"))


class VerifyTests(unittest.TestCase):
    def verify(self, **kwargs):
        defaults = dict(
            unit="chat-backend-dense", unit_active=True,
            cmdline="llama-server --ctx-size 262144 --parallel 2 --cache-ram 8192 "
                    "--ctx-checkpoints 8 --fit off --cache-idle-slots",
            props={"total_slots": 2, "n_ctx_per_slot": 131072, "n_ctx_total": 262144},
            slots=[{"id": 0}, {"id": 1}],
            stats={"scheduling": {"select_methods": {"id": 62, "lcp": 22},
                                  "select_by_id_slots": {"0": 56, "1": 6}}},
        )
        defaults.update(kwargs)
        with patch.object(scheduling, "lease_directory", return_value=None):
            return scheduling.verify(env_for(), **defaults)

    def test_a_healthy_stack_verifies(self):
        result = self.verify()
        self.assertTrue(result["ok"])
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["drift"], [])

    def test_pinning_to_both_slots_is_the_evidence_the_contract_is_live(self):
        evidence = self.verify()["evidence"]
        self.assertTrue(evidence["interactive_pinned"])
        self.assertTrue(evidence["background_pinned"])
        self.assertTrue(evidence["observed"])

    def test_an_idle_session_is_not_reported_as_a_fault(self):
        result = self.verify(stats={"scheduling": {"select_methods": {}, "select_by_id_slots": {}}})
        self.assertFalse(result["evidence"]["observed"])
        self.assertTrue(result["ok"])

    def test_a_backend_launched_with_one_slot_fails_verification(self):
        result = self.verify(
            cmdline="llama-server --ctx-size 262144 --parallel 1 --fit off --cache-idle-slots",
            props={"total_slots": 1, "n_ctx_per_slot": 262144, "n_ctx_total": 262144})
        self.assertFalse(result["ok"])
        self.assertTrue(any("1 slot" in issue for issue in result["issues"]))

    def test_a_stopped_backend_reports_the_configuration_only(self):
        result = self.verify(unit=None, unit_active=False, cmdline="", props=None, slots=None)
        self.assertFalse(result["launched"]["available"])
        self.assertTrue(result["ok"])

    def test_the_running_context_per_slot_is_reported(self):
        self.assertEqual(self.verify()["runtime"]["n_ctx_per_slot"], 131072)


if __name__ == "__main__":
    unittest.main()
