"""Several machines, from one of them.

A fleet is testable without a fleet because this manager touches the network in
exactly one place. `fleet.fetch` is the seam: a spoke here is two Flask
`test_client()`s over the same app factories the real thing serves, spliced in
with one `patch.object`. That is why `fleet` is in
`ModuleBoundaryTests.BEHAVIOUR_MODULES` -- one `from fleet import fetch`
anywhere and every test in this file would silently start testing nothing.

The spoke runs with its own env and its own platform adapter, so a Linux hub
rendering a Darwin peer is a real test rather than two copies of one host.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import threading
import time
import unittest
from contextlib import nullcontext
from unittest.mock import patch
from urllib.parse import urlsplit


def _load_app_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "llm_stack_manager_app_fleet", root / "web" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manager = _load_app_module()
core = sys.modules["core"]
config_env = sys.modules["config_env"]
fleet = sys.modules["fleet"]
public_api = sys.modules["public_api"]
fleet_routes = sys.modules["routes.fleet"]

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import platform_harness  # noqa: E402

#: `unittest.mock.patch` sets a module attribute, so two spoke calls on two
#: collector threads would overlap and one would restore what the other
#: installed -- which leaks a canned env into whatever test runs next. One
#: spoke answers at a time; the concurrency under test is the hub's, not the
#: spoke's.
_SPOKE_LOCK = threading.Lock()

HOST = {
    "id": "studio", "label": "Mac Studio", "host": "studio.tailnet.ts.net",
    "scheme": "http", "read_port": 8078, "control_port": 8079,
    "token": "read-tok", "control_token": "ctl-tok",
    "expected_hostname": "studio", "enabled": True, "control": True,
}


class _Spoke:
    """A second manager, in this process.

    The hub and the spoke are the same code, so a fleet test is two Flask test
    clients and one patched function. The spoke's own env and platform are
    installed for the duration of each call, which is what makes it a different
    machine rather than a second view of this one.
    """

    def __init__(self, env=None, platform=None, token="read-tok", control_token="ctl-tok"):
        # A *factory*, not a context manager instance: one collect makes two
        # calls, and a `@contextmanager` object can only be entered once.
        self.read = manager.create_state_api_app().test_client()
        self.control = manager.create_control_api_app().test_client()
        self.env = env or {}
        self.platform = platform
        self.token = token
        self.control_token = control_token
        self.calls: list[tuple[str, str]] = []
        self.delay = 0.0

    def fetch(self, url, token="", method="GET", payload=None, timeout=None, headers=None):
        parts = urlsplit(url)
        self.calls.append((method, parts.path))
        if self.delay:
            time.sleep(self.delay)
            return 0, {}, f"timed out after {timeout:g}s"
        control = parts.path.startswith("/api/control/")
        expected = self.control_token if control else self.token
        if token != expected:
            return 401, {"error": "unauthorized"}, ""
        client = self.control if control else self.read
        target = parts.path + (f"?{parts.query}" if parts.query else "")
        with _SPOKE_LOCK, \
             (self.platform() if self.platform else nullcontext()), \
             patch.object(config_env, "read_env", return_value=dict(self.env)):
            response = client.open(target, method=method, json=payload,
                                   headers={"Authorization": f"Bearer {token}"})
        return response.status_code, (response.get_json(silent=True) or {}), ""


class _Registry:
    """A `config/fleet.json` in a temp directory, for the duration of a test."""

    def __init__(self, testcase, hosts):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        testcase.addCleanup(self._tmp.cleanup)
        path = pathlib.Path(self._tmp.name) / "fleet.json"
        path.write_text(json.dumps({"version": 1, "hosts": hosts}))
        self._patch = patch.object(core, "FLEET_FILE", path)
        self._patch.start()
        testcase.addCleanup(self._patch.stop)
        self.path = path


class RegistryTests(unittest.TestCase):

    def test_a_missing_file_is_an_empty_fleet_not_an_error(self):
        with patch.object(core, "FLEET_FILE", pathlib.Path("/nope/fleet.json")):
            self.assertEqual(fleet.hosts(), [])

    def test_a_corrupt_file_is_an_empty_fleet_not_an_error(self):
        # The manager is the only way to fix the file, so it must survive it.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "fleet.json"
            path.write_text("{ this is not json")
            with patch.object(core, "FLEET_FILE", path):
                self.assertEqual(fleet.hosts(), [])

    def test_an_id_must_be_a_slug(self):
        for bad in ("Studio", "mac studio", "-leading", "a" * 33, "", "../etc"):
            with self.subTest(bad):
                self.assertTrue(fleet.validate({**HOST, "id": bad}, set()))
        self.assertEqual(fleet.validate(HOST, set()), "")

    def test_an_id_is_not_the_hostname_the_remote_reports(self):
        """`platform.node()` is a value the *remote* controls -- it changes with
        DNS, with mDNS, with a re-image. Keying the registry on it would make
        this hub's idea of which machine is which depend on what the other
        machine says it is called."""
        self.assertIn("expected_hostname", fleet.redacted(HOST))
        self.assertEqual(fleet.redacted(HOST)["id"], "studio")

    def test_a_duplicate_id_is_refused(self):
        self.assertTrue(fleet.validate(HOST, {"studio"}))
        self.assertEqual(fleet.validate(HOST, {"studio"}, replacing="studio"), "")

    def test_a_bad_port_is_refused(self):
        for bad in ("nope", 0, 70000):
            with self.subTest(bad):
                self.assertTrue(fleet.validate({**HOST, "read_port": bad}, set()))

    def test_a_peer_is_never_rendered_with_its_credentials(self):
        rendered = fleet.redacted(HOST)
        self.assertNotIn("read-tok", json.dumps(rendered))
        self.assertNotIn("ctl-tok", json.dumps(rendered))
        self.assertTrue(rendered["has_token"])
        self.assertTrue(rendered["has_control_token"])


class RegistryRouteTests(unittest.TestCase):

    def setUp(self):
        self.registry = _Registry(self, [dict(HOST)])
        self.client = manager.app.test_client()

    def test_listing_never_contains_a_token(self):
        body = self.client.get("/api/fleet/hosts").get_json()
        self.assertNotIn("read-tok", json.dumps(body))
        self.assertNotIn("ctl-tok", json.dumps(body))

    def test_adding_a_host_with_a_bad_id_is_refused(self):
        response = self.client.post("/api/fleet/hosts", json={**HOST, "id": "Nope!"})
        self.assertEqual(response.status_code, 400)

    def test_editing_without_resending_the_token_keeps_it(self):
        # The form never has the current value, because it is never served one.
        self.client.put("/api/fleet/hosts/studio", json={"label": "Studio 2"})
        stored = fleet.get("studio")
        self.assertEqual(stored["label"], "Studio 2")
        self.assertEqual(stored["token"], "read-tok")

    def test_clearing_a_token_is_possible_and_distinct_from_omitting_it(self):
        self.client.put("/api/fleet/hosts/studio", json={"token": ""})
        self.assertEqual(fleet.get("studio")["token"], "")

    def test_an_unknown_host_is_a_404_everywhere(self):
        for path, method in (("/api/fleet/nope/snapshot", "get"),
                             ("/api/fleet/nope/status", "get"),
                             ("/api/fleet/nope/logs", "get"),
                             ("/api/fleet/hosts/nope/test", "post"),
                             ("/api/fleet/hosts/nope", "delete")):
            with self.subTest(path):
                self.assertEqual(getattr(self.client, method)(path).status_code, 404)

    def test_an_entry_pointing_back_at_this_manager_is_refused(self):
        """Otherwise a fleet route calls a fleet route, and the poller spends a
        tick waiting on the thread that is running it."""
        response = self.client.post("/api/fleet/hosts", json={
            **HOST, "id": "myself", "host": "127.0.0.1", "read_port": 8078})
        self.assertEqual(response.status_code, 400)
        self.assertIn("this machine", response.get_json()["error"])

    def test_the_registry_is_not_reachable_from_another_machine(self):
        """A control channel that can be told to trust a new peer is a
        lateral-movement primitive. It is not built."""
        for app_factory in (manager.create_state_api_app, manager.create_control_api_app):
            with self.subTest(app_factory.__name__):
                rules = {str(r) for r in app_factory().url_map.iter_rules()}
                self.assertFalse([r for r in rules if r.startswith("/api/fleet")])


class FetchTests(unittest.TestCase):
    """Nothing here raises. A page showing five machines must not fail because
    one of them is off."""

    def test_an_unreachable_host_is_data_not_an_exception(self):
        status, body, error = fleet.fetch("http://127.0.0.1:1/api/v1/health", timeout=0.2)
        self.assertEqual((status, body), (0, {}))
        self.assertTrue(error)

    def test_a_name_that_does_not_resolve_is_data_too(self):
        status, _body, error = fleet.fetch(
            "http://no-such-host.invalid/api/v1/health", timeout=0.5)
        self.assertEqual(status, 0)
        self.assertIn("unreachable", error)

    def test_a_rejected_token_keeps_its_status(self):
        """A 401 is an answer, not a failure. `core.http_json` raises on 4xx and
        cannot report one, which is why this does not use it."""
        spoke = _Spoke(token="right")
        status, _body, error = spoke.fetch("http://peer/api/v1/schema", "wrong")
        self.assertEqual(status, 401)
        self.assertEqual(error, "")


class CacheTests(unittest.TestCase):

    def setUp(self):
        self.registry = _Registry(self, [dict(HOST)])
        self.cache = fleet.FleetCache(lambda: {}, interval=999)
        self.addCleanup(self.cache.stop)

    def test_the_request_thread_never_collects(self):
        """Same discipline as `deploy.DriftWatcher`: the page reads the cache.

        Asserted by making a collection fail loudly rather than by counting
        fetches -- the route does start the background poller, and that thread
        fetching is the point rather than the problem. The route is given this
        test's own cache, so no real poller outlives the test.
        """
        with patch.object(fleet_routes, "CACHE", self.cache), \
             patch.object(self.cache, "collect",
                          side_effect=AssertionError("collected on the request thread")), \
             patch.object(self.cache, "ensure_running"):
            response = manager.app.test_client().get("/api/fleet")
        self.assertEqual(response.status_code, 200)
        self.assertEqual([h["id"] for h in response.get_json()["hosts"]], ["studio"])

    def test_a_host_that_has_never_answered_still_appears_with_a_reason(self):
        entry = self.cache.snapshot()[0]
        self.assertEqual(entry["id"], "studio")
        self.assertFalse(entry["ok"])
        self.assertTrue(entry["error"])
        self.assertIsNone(entry["payload"])

    def test_a_host_that_stops_answering_goes_stale_rather_than_blank(self):
        spoke = _Spoke(env={"LLM_API_TOKEN": ""})
        with patch.object(fleet, "fetch", spoke.fetch):
            self.cache.collect()
        good = self.cache.snapshot()[0]
        self.assertTrue(good["ok"], good["error"])

        with patch.object(fleet, "fetch",
                          lambda *a, **k: (0, {}, "unreachable: connection refused")):
            self.cache.collect()
        stale = self.cache.snapshot()[0]
        self.assertFalse(stale["ok"])
        self.assertIsNotNone(stale["payload"], "the last good payload is kept")
        self.assertEqual(stale["error"], "unreachable: connection refused")
        self.assertGreaterEqual(stale["stale_for_seconds"], 0)

    def test_one_hung_host_does_not_stop_the_others(self):
        _Registry(self, [dict(HOST), {**HOST, "id": "slow"}, {**HOST, "id": "third"}])
        cache = fleet.FleetCache(lambda: {}, interval=999)
        self.addCleanup(cache.stop)
        good = _Spoke(env={"LLM_API_TOKEN": ""})

        def dispatch(url, token="", **kwargs):
            # One host that never answers, two that do.
            if "slow" in threading.current_thread().name:
                time.sleep(fleet.DEFAULT_TIMEOUT_SECONDS * 3)
                return 0, {}, "timed out"
            return good.fetch(url, token, **kwargs)

        started = time.monotonic()
        with patch.object(fleet, "fetch", dispatch):
            cache.collect()
        elapsed = time.monotonic() - started
        answered = [h for h in cache.snapshot() if h["ok"]]
        self.assertEqual(sorted(h["id"] for h in answered), ["studio", "third"])
        self.assertLess(elapsed, fleet.DEFAULT_TIMEOUT_SECONDS * 3,
                        "the tick waited for the hung host")

    def test_polling_does_not_start_when_there_are_no_peers(self):
        _Registry(self, [])
        cache = fleet.FleetCache(lambda: {}, interval=999)
        cache.ensure_running()
        self.assertIsNone(cache._thread)


class VersionToleranceTests(unittest.TestCase):

    def test_a_major_mismatch_refuses_control_and_keeps_monitoring(self):
        _Registry(self, [dict(HOST)])
        cache = fleet.FleetCache(lambda: {}, interval=999)
        self.addCleanup(cache.stop)
        spoke = _Spoke(env={"LLM_API_TOKEN": ""})

        def dispatch(url, token="", **kwargs):
            if url.endswith("/api/v1/schema"):
                return 200, {"api_version": "2.0", "sections": list(fleet.CARD_SECTIONS)}, ""
            return spoke.fetch(url, token, **kwargs)

        with patch.object(fleet, "fetch", dispatch):
            cache.collect()
        entry = cache.snapshot()[0]
        self.assertTrue(entry["ok"], "monitoring still works")
        self.assertFalse(entry["controllable"])
        self.assertIn("2.0", entry["control_refused"])

    def test_only_the_sections_a_peer_lists_are_requested(self):
        """`resolve_sections` 400s on a section it does not know, which would
        blank the whole card rather than drop one panel."""
        self.assertEqual(fleet.sections_for({"sections": ["stack", "gpus"]}), "stack,gpus")
        self.assertEqual(fleet.sections_for({}), ",".join(fleet.CARD_SECTIONS))
        self.assertEqual(fleet.sections_for({"sections": ["nothing-we-want"]}), "stack")

    def test_compatibility_is_by_major_only(self):
        self.assertTrue(fleet.compatible("1.0", "1.4"))
        self.assertFalse(fleet.compatible("1.0", "2.0"))


class StatusAdapterTests(unittest.TestCase):
    """The one function that lets seventy-eight `fetchJSON` call sites stay as
    they are. Its shape has to match what the page already reads."""

    def _payloads(self):
        snapshot = public_api.snapshot(manager.STATE_API_PROVIDERS)
        local = json.loads(manager.app.test_client().get("/api/status").get_data())
        return fleet.status_from_snapshot(snapshot), local

    def test_the_top_level_keys_match_the_local_endpoint(self):
        adapted, local = self._payloads()
        self.assertEqual(sorted(adapted), sorted(local))

    def test_the_health_entries_have_the_keys_the_panel_reads(self):
        # The snapshot renames `unit` to `unit_state`, because "state" there is
        # the health verdict. The panel reads `unit`; the rename is undone in
        # the adapter rather than in the panel.
        adapted, local = self._payloads()
        adapted_keys = {k for v in adapted["health"].values() for k in v}
        local_keys = {k for v in local["health"].values() for k in v}
        self.assertEqual(adapted_keys, local_keys)

    def test_the_context_entries_have_the_keys_the_panel_reads(self):
        adapted, local = self._payloads()
        adapted_keys = {k for v in adapted["contexts"].values() for k in v}
        local_keys = {k for v in local["contexts"].values() for k in v}
        if adapted_keys:
            self.assertEqual(adapted_keys, local_keys)

    def test_an_empty_snapshot_does_not_raise(self):
        self.assertEqual(fleet.status_from_snapshot({})["services"], {})


class CrossPlatformTests(unittest.TestCase):
    """A payload assembled by one platform, rendered by the other.

    The M5 Ultra is the likely hub once it arrives, so both directions matter.

    `platforms.set_active` is a module global, so this is not two adapters live
    at once -- it is the spoke's snapshot genuinely assembled under its own
    adapter and then read by the hub. The assertion on `unified_memory` is what
    proves the spoke really was the other platform rather than a second view of
    this one.
    """

    def _render(self, hub, spoke_platform):
        _Registry(self, [dict(HOST)])
        cache = fleet.FleetCache(lambda: {}, interval=999)
        self.addCleanup(cache.stop)
        spoke = _Spoke(env={"LLM_API_TOKEN": ""}, platform=spoke_platform)
        with hub(), patch.object(fleet, "fetch", spoke.fetch):
            cache.collect()
        return cache.snapshot()[0]

    def test_a_linux_hub_renders_a_darwin_peer(self):
        entry = self._render(platform_harness.as_linux, platform_harness.as_darwin)
        self.assertTrue(entry["ok"], entry["error"])
        self.assertIn("stack", entry["payload"])
        # Assembled by the Darwin adapter, not by this one: device memory on
        # Apple silicon is host memory, and `gpu_compute_apps` is None rather
        # than [] because there is no per-process accounting to report.
        for gpu in entry["payload"].get("gpus") or []:
            with self.subTest(gpu.get("index")):
                self.assertTrue(gpu.get("unified_memory"))

    def test_a_darwin_hub_renders_a_linux_peer(self):
        entry = self._render(platform_harness.as_darwin, platform_harness.as_linux)
        self.assertTrue(entry["ok"], entry["error"])
        self.assertIn("stack", entry["payload"])
        for gpu in entry["payload"].get("gpus") or []:
            with self.subTest(gpu.get("index")):
                self.assertFalse(gpu.get("unified_memory"))


class ProbeTests(unittest.TestCase):

    def test_a_probe_reports_each_listener_separately(self):
        spoke = _Spoke(env={"LLM_API_TOKEN": "", "LLM_CONTROL_ENABLED": "on",
                            "LLM_CONTROL_TOKEN": "ctl-tok"})
        with patch.object(fleet, "fetch", spoke.fetch):
            result = fleet.probe(HOST)
        self.assertTrue(result["read"]["ok"])
        self.assertTrue(result["control"]["ok"])

    def test_a_wrong_control_token_says_so_rather_than_failing_the_host(self):
        spoke = _Spoke(env={"LLM_API_TOKEN": "", "LLM_CONTROL_ENABLED": "on",
                            "LLM_CONTROL_TOKEN": "ctl-tok"}, control_token="different")
        with patch.object(fleet, "fetch", spoke.fetch):
            result = fleet.probe(HOST)
        self.assertTrue(result["read"]["ok"])
        self.assertFalse(result["control"]["ok"])
        self.assertIn("rejected the token", result["control"]["error"])

    def test_a_hostname_that_stopped_matching_is_reported(self):
        """The guard against pointing a config editor at the wrong machine."""
        spoke = _Spoke(env={"LLM_API_TOKEN": ""})
        with patch.object(fleet, "fetch", spoke.fetch):
            result = fleet.probe({**HOST, "expected_hostname": "something-else"})
        self.assertFalse(result["hostname_matches"])


if __name__ == "__main__":
    unittest.main()
