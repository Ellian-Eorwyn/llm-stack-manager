from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch


def _load(name: str, relative: str):
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


health = _load("llm_stack_manager_health", "web/health.py")
setup_engine = _load("llm_stack_manager_setup_engine", "scripts/setup_engine.py")


class DependencyGraphTests(unittest.TestCase):
    @staticmethod
    def _reachable(service: str) -> set[str]:
        seen, stack = set(), [service]
        while stack:
            for group in health.SERVICE_DEPENDENCIES.get(stack.pop(), []):
                for member in group:
                    if member not in seen:
                        seen.add(member)
                        stack.append(member)
        return seen

    def test_service_graph_agrees_with_the_component_graph(self):
        """The two dependency maps describe the same stack and must not drift.

        `setup_engine` reasons in components, this reasons in units. Every
        cross-component requirement the installer knows about has to be
        reachable here, or a service whose upstream is down renders green.
        Reachable rather than declared: `honcho-deriver` needs the primary
        backend through `honcho-api`, and stating that edge twice would only
        give it two places to rot.
        """
        for component, dependencies in setup_engine.COMPONENT_DEPENDENCIES.items():
            own = set(setup_engine.COMPONENT_SERVICES.get(component, []))
            required = {unit for dependency in dependencies
                        for unit in setup_engine.COMPONENT_SERVICES.get(dependency, [dependency])}
            for service in own:
                external = required - own
                if not external:
                    continue
                reachable = self._reachable(service)
                self.assertTrue(
                    external <= reachable,
                    f"{service} needs {sorted(external - reachable)} per setup_engine, "
                    f"but the service graph cannot reach it")

    def test_glmocr_sdk_declares_its_ocr_upstream(self):
        self.assertEqual(health.SERVICE_DEPENDENCIES["glmocr-sdk"], [["ocr"]])

    def test_dependency_units_includes_backends_with_no_card(self):
        # The panel shows one primary-backend card, but three units can serve
        # it; the two without a card still have to be asked about.
        self.assertIn("chat-backend-moe", health.dependency_units())
        self.assertIn("chat-backend", health.dependency_units())


class ProbeTargetTests(unittest.TestCase):
    def test_backend_probes_come_from_the_telemetry_port_map(self):
        self.assertEqual(health.SERVICE_PROBES["chat-backend-dense"]["path"], "/props")
        self.assertEqual(health.SERVICE_PROBES["chat-backend-dense"]["port_key"], "CHAT_BACKEND_PORT")
        self.assertEqual(health.SERVICE_PROBES["embed"]["port_key"], "EMBED_PORT")

    def test_bind_address_is_rewritten_to_a_reachable_one(self):
        host, port = health.endpoint_for("chat-proxy", {"LISTEN_HOST": "0.0.0.0", "NOTHINK_PORT": "8004"})
        self.assertEqual((host, port), ("127.0.0.1", "8004"))

    def test_services_without_a_probe_have_no_endpoint(self):
        self.assertIsNone(health.endpoint_for("honcho-deriver", {}))
        self.assertNotIn("searxng", health.SERVICE_PROBES)

    def test_defaults_apply_when_the_env_does_not_set_a_port(self):
        host, port = health.endpoint_for("glmocr-sdk", {})
        self.assertEqual((host, port), ("127.0.0.1", "5002"))

    def test_probe_reports_an_http_error_distinctly_from_no_answer(self):
        with patch.object(health.telemetry, "_http_text", return_value=(None, 503)):
            result = health.probe("glmocr-sdk", {})
        self.assertFalse(result["ok"])
        self.assertEqual(result["http_status"], 503)
        self.assertIn("HTTP 503", result["detail"])

        with patch.object(health.telemetry, "_http_text", return_value=(None, None)):
            result = health.probe("glmocr-sdk", {})
        self.assertIn("no answer", result["detail"])

    def test_glmocr_sdk_probe_requires_the_status_field(self):
        with patch.object(health.telemetry, "_http_text", return_value=('{"status":"ok"}', 200)):
            self.assertTrue(health.probe("glmocr-sdk", {})["ok"])
        with patch.object(health.telemetry, "_http_text", return_value=('{"status":"starting"}', 200)):
            result = health.probe("glmocr-sdk", {})
        self.assertFalse(result["ok"])
        self.assertIn("status=ok", result["detail"])


class ExpectationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = pathlib.Path(self._tmp.name) / "service-expectations.json"

    def test_recording_round_trips_and_auto_forgets(self):
        health.record_expectation("ocr", "off", path=self.path)
        self.assertEqual(health.read_expectations(self.path)["ocr"]["expected"], "off")
        health.record_expectation("ocr", "on", path=self.path)
        self.assertEqual(health.read_expectations(self.path)["ocr"]["expected"], "on")
        health.record_expectation("ocr", "auto", path=self.path)
        self.assertNotIn("ocr", health.read_expectations(self.path))

    def test_an_unknown_expectation_is_rejected(self):
        with self.assertRaises(ValueError):
            health.record_expectation("ocr", "maybe", path=self.path)

    def test_an_unreadable_file_reads_as_no_expectations(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(health.read_expectations(self.path), {})

    def test_an_enabled_flag_set_to_off_means_expected_off(self):
        # The start script exits 0 without launching anything, so the unit
        # cannot come up and being down is not a fault.
        self.assertEqual(
            health.expectation_for("glmocr-sdk", {"GLMOCR_SDK_ENABLED": "off"}, {}), "off")

    def test_an_enabled_flag_set_to_on_says_nothing(self):
        # These flags read `on` for services that are deliberately stopped, so
        # `on` is not evidence that a service should be running.
        self.assertEqual(
            health.expectation_for("glmocr-sdk", {"GLMOCR_SDK_ENABLED": "on"}, {}), "unspecified")

    def test_a_recorded_expectation_is_used(self):
        expectations = {"ocr": {"expected": "on"}}
        self.assertEqual(health.expectation_for("ocr", {}, expectations), "on")


class CollectTests(unittest.TestCase):
    def collect(self, statuses, env=None, probes=None, expectations=None):
        return health.collect(env or {}, statuses, probes or {}, expectations or {})

    def test_a_service_whose_upstream_is_down_is_degraded(self):
        entries = self.collect({"glmocr-sdk": "active", "ocr": "inactive"})
        self.assertEqual(entries["glmocr-sdk"]["state"], "degraded")
        self.assertIn("ocr", entries["glmocr-sdk"]["reason"])

    def test_starting_the_upstream_makes_it_active_again(self):
        entries = self.collect({"glmocr-sdk": "active", "ocr": "active"})
        self.assertEqual(entries["glmocr-sdk"]["state"], "active")
        self.assertEqual(entries["glmocr-sdk"]["reason"], "")

    def test_a_failing_probe_degrades_a_running_service(self):
        probes = {"embed": {"ok": False, "detail": "no answer from http://127.0.0.1:8005/props",
                            "target": "", "http_status": None, "checked_at": 1.0}}
        entries = self.collect({"embed": "active"}, probes=probes)
        self.assertEqual(entries["embed"]["state"], "degraded")
        self.assertIn("no answer", entries["embed"]["reason"])

    def test_a_service_with_no_probe_yet_is_judged_on_its_unit_state(self):
        # The first sweep has not landed. Reporting every card as broken until
        # it does would be worse than saying nothing.
        entries = self.collect({"embed": "active"})
        self.assertEqual(entries["embed"]["state"], "active")

    def test_any_of_upstreams_are_satisfied_by_a_unit_without_a_card(self):
        entries = self.collect({"chat-proxy": "active", "chat-backend-dense": "inactive",
                                "chat-backend-moe": "active", "chat-backend": "inactive"})
        self.assertEqual(entries["chat-proxy"]["state"], "active")

    def test_degradation_propagates_along_the_chain(self):
        entries = self.collect({
            "honcho-deriver": "active", "honcho-api": "active",
            "chat-proxy": "active", "embed": "active",
            "chat-backend-dense": "inactive", "chat-backend-moe": "inactive",
            "chat-backend": "inactive",
        })
        self.assertEqual(entries["chat-proxy"]["state"], "degraded")
        self.assertEqual(entries["honcho-api"]["state"], "degraded")
        self.assertEqual(entries["honcho-deriver"]["state"], "degraded")

    def test_a_stopped_service_nobody_asked_for_is_not_a_fault(self):
        entries = self.collect({"rerank": "inactive"})
        self.assertEqual(entries["rerank"]["state"], "stopped")

    def test_a_service_that_was_started_here_and_is_down_is_flagged(self):
        entries = self.collect({"rerank": "inactive"},
                               expectations={"rerank": {"expected": "on"}})
        self.assertEqual(entries["rerank"]["state"], "inactive")
        self.assertIn("expected", entries["rerank"]["reason"])

    def test_a_failed_unit_reports_as_failed(self):
        entries = self.collect({"embed": "failed"})
        self.assertEqual(entries["embed"]["state"], "failed")

    def test_an_uninstalled_unit_stays_unknown(self):
        entries = self.collect({"embed2": "unknown"})
        self.assertEqual(entries["embed2"]["state"], "unknown")

    def test_a_service_mid_launch_is_not_reported_as_down(self):
        entries = self.collect({"ocr": "starting"})
        self.assertEqual(entries["ocr"]["state"], "starting")

    def test_a_service_that_keeps_dying_outranks_whatever_phase_was_sampled(self):
        """Observed live: `ocr` at 32 restarts, unable to allocate on a full GPU.

        Each poll caught a different phase — `activating`, `failed`, briefly
        `active` — so it rendered as starting, or down, or fine, and never as a
        service that could not stay up.
        """
        for sampled in ("starting", "active", "inactive", "failed"):
            entries = health.collect({}, {"ocr": sampled}, {}, {}, flapping={"ocr": 32})
            self.assertEqual(entries["ocr"]["state"], "failed", sampled)
            self.assertIn("32 times", entries["ocr"]["reason"])
            self.assertEqual(entries["ocr"]["restarts"], 32)

    def test_stopped_on_purpose_reads_differently_from_switched_off(self):
        # One is somebody pressing Stop; the other is a flag in the config file.
        pressed = self.collect({"ocr": "inactive"}, expectations={"ocr": {"expected": "off"}})
        self.assertEqual(pressed["ocr"]["reason"], "stopped on purpose")
        flagged = self.collect({"glmocr-sdk": "inactive"}, env={"GLMOCR_SDK_ENABLED": "off"})
        self.assertEqual(flagged["glmocr-sdk"]["reason"], "turned off in the configuration")


class RestartTrackerTests(unittest.TestCase):
    def test_a_first_sample_cannot_prove_anything(self):
        tracker = health.RestartTracker()
        self.assertEqual(tracker.observe({"ocr": 32}), {})

    def test_a_climbing_count_between_polls_is_a_flap(self):
        tracker = health.RestartTracker()
        tracker.observe({"ocr": 30})
        self.assertEqual(tracker.observe({"ocr": 32}), {"ocr": 2})
        self.assertEqual(tracker.observe({"ocr": 33}), {"ocr": 3})

    def test_a_steady_count_clears_it(self):
        # Restarted once and then stayed up. Not flapping; the card goes green.
        tracker = health.RestartTracker()
        tracker.observe({"embed": 0})
        self.assertEqual(tracker.observe({"embed": 1}), {"embed": 1})
        self.assertEqual(tracker.observe({"embed": 1}), {})

    def test_a_stop_resets_the_counter_without_looking_like_a_flap(self):
        # systemd zeroes NRestarts on a clean stop, so the count goes down.
        tracker = health.RestartTracker()
        tracker.observe({"ocr": 32})
        tracker.observe({"ocr": 34})
        self.assertEqual(tracker.observe({"ocr": 0}), {})

    def test_resetting_one_service_leaves_the_others(self):
        tracker = health.RestartTracker()
        tracker.observe({"ocr": 1, "embed": 1})
        tracker.observe({"ocr": 2, "embed": 2})
        tracker.reset("ocr")
        self.assertEqual(set(tracker.flapping()), {"embed"})

    def test_a_service_that_disappears_stops_being_tracked(self):
        tracker = health.RestartTracker()
        tracker.observe({"ocr": 1})
        tracker.observe({"ocr": 2})
        self.assertEqual(tracker.observe({}), {})


class ProberTests(unittest.TestCase):
    def test_only_running_services_are_probed(self):
        prober = health.Prober()
        with patch.object(health, "probe", side_effect=lambda name, env: {"ok": True, "target": name,
                                                                          "http_status": 200,
                                                                          "checked_at": 1.0,
                                                                          "detail": ""}) as probe:
            results = prober.sweep({}, {"embed": "active", "rerank": "inactive",
                                        "honcho-deriver": "active"})
        self.assertEqual(set(results), {"embed"})
        # honcho-deriver is running but has no probe, so it is not asked.
        self.assertEqual([call.args[0] for call in probe.call_args_list], ["embed"])

    def test_a_probe_that_raises_degrades_only_that_service(self):
        prober = health.Prober()
        with patch.object(health, "probe", side_effect=RuntimeError("boom")):
            results = prober.sweep({}, {"embed": "active"})
        self.assertFalse(results["embed"]["ok"])
        self.assertIn("boom", results["embed"]["detail"])

    def test_the_snapshot_is_what_the_last_sweep_found(self):
        prober = health.Prober()
        with patch.object(health, "probe", return_value={"ok": True, "target": "", "http_status": 200,
                                                         "checked_at": 1.0, "detail": ""}):
            prober.sweep({}, {"embed": "active"})
        self.assertEqual(set(prober.snapshot()), {"embed"})


if __name__ == "__main__":
    unittest.main()
