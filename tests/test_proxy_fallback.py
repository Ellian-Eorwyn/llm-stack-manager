"""The proxy's fallback to another proxy, against real sockets.

A fake backend and a fake fallback proxy run on spare loopback ports, and the
proxy under test sits in front of them. Nothing here touches a model.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock


def _load_proxy_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("llm_chat_proxy_fallback", root / "scripts" / "llm-chat-proxy.py")
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


proxy = _load_proxy_module()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Recorder:
    def __init__(self):
        self.requests: list[dict] = []


def _fake_server(recorder: _Recorder, *, status: int = 200, stream: bool = False, text: str = "hi"):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            recorder.requests.append({"path": self.path, "headers": dict(self.headers), "body": json.loads(body)})
            if status != 200:
                msg = json.dumps({"error": {"code": status, "message": "Loading model"}}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
                return
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for part in (text[:1], text[1:]):
                    event = {"choices": [{"index": 0, "delta": {"content": part}}]}
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                return
            msg = json.dumps({"model": "x", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return server


def _serve_proxy(handler_class):
    server = proxy.ProxyHTTPServer(("127.0.0.1", 0), handler_class)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return server


def _post(port: int, payload: dict, headers: dict | None = None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lease_file = os.path.join(self.tmp.name, "gpu1-lease.json")
        self.fallback_rec = _Recorder()
        self.fallback = _fake_server(self.fallback_rec, text="from fallback")
        self.patches = mock.patch.multiple(
            proxy,
            MEMORY_GATEWAY_ENABLED=False,
            BACKEND_HOST="127.0.0.1",
            BACKEND_PORT=_free_port(),  # nothing listens: refused
            CHAT_FALLBACK_URL=f"http://127.0.0.1:{self.fallback.server_address[1]}",
            CHAT_FALLBACK_TOKEN="sekrit",
            CHAT_FALLBACK_LEASE_FILE=self.lease_file,
        )
        self.patches.start()
        proxy._FALLBACK_STATE["active"] = False
        self.think = _serve_proxy(proxy.make_handler(thinking_enabled=True, model_name="think", port_label="think"))
        self.aggregate = _serve_proxy(proxy.make_aggregate_handler())

    def tearDown(self):
        for server in (self.think, self.aggregate, self.fallback):
            server.shutdown()
            server.server_close()
        self.patches.stop()
        self.tmp.cleanup()

    def _use_backend(self, **kwargs):
        rec = _Recorder()
        server = _fake_server(rec, **kwargs)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        proxy.BACKEND_PORT = server.server_address[1]
        return rec

    def test_refused_backend_is_served_by_the_fallback(self):
        status, headers, body = _post(self.think.server_address[1], {"model": "anything", "messages": [{"role": "user", "content": "q"}]})
        self.assertEqual(status, 200)
        self.assertEqual(headers.get(proxy.SERVED_BY_HEADER), "fallback")
        self.assertIn(b"from fallback", body)
        sent = self.fallback_rec.requests[0]
        # A per-port handler sends its own alias; the client's own body otherwise.
        self.assertEqual(sent["body"]["model"], "think")
        self.assertNotIn("chat_template_kwargs", sent["body"])
        self.assertEqual(sent["headers"].get("Authorization"), "Bearer sekrit")
        self.assertEqual(sent["headers"].get(proxy.FALLBACK_HOP_HEADER), "1")

    def test_aggregate_keeps_the_client_alias(self):
        status, _, _ = _post(self.aggregate.server_address[1], {"model": "chat", "messages": [{"role": "user", "content": "q"}]})
        self.assertEqual(status, 200)
        self.assertEqual(self.fallback_rec.requests[0]["body"]["model"], "chat")

    def test_loading_backend_503_is_served_by_the_fallback(self):
        backend = self._use_backend(status=503)
        status, headers, _ = _post(self.think.server_address[1], {"model": "think", "messages": []})
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(status, 200)
        self.assertEqual(headers.get(proxy.SERVED_BY_HEADER), "fallback")

    def test_a_lease_file_of_off_means_no_lease(self):
        self.assertEqual(proxy._lease_file("off"), "")
        self.assertEqual(proxy._lease_file(" /x/lease.json "), "/x/lease.json")
        self.assertTrue(proxy._lease_file("").endswith("/gpu1-lease.json"))

    def test_lease_file_skips_a_healthy_backend(self):
        backend = self._use_backend(text="from backend")
        pathlib.Path(self.lease_file).write_text("{}")
        status, headers, body = _post(self.think.server_address[1], {"model": "think", "messages": []})
        self.assertEqual(status, 200)
        self.assertEqual(backend.requests, [])
        self.assertIn(b"from fallback", body)

    def test_healthy_backend_is_used_without_the_header(self):
        self._use_backend(text="from backend")
        status, headers, body = _post(self.think.server_address[1], {"model": "think", "messages": []})
        self.assertEqual(status, 200)
        self.assertNotIn(proxy.SERVED_BY_HEADER, headers)
        self.assertIn(b"from backend", body)
        self.assertEqual(self.fallback_rec.requests, [])

    def test_streaming_is_relayed(self):
        self.fallback.shutdown()
        self.fallback.server_close()
        self.fallback = _fake_server(self.fallback_rec, stream=True, text="streamed")
        proxy.CHAT_FALLBACK_URL = f"http://127.0.0.1:{self.fallback.server_address[1]}"
        status, headers, body = _post(self.think.server_address[1], {"model": "think", "stream": True, "messages": []})
        self.assertEqual(status, 200)
        self.assertEqual(headers.get(proxy.SERVED_BY_HEADER), "fallback")
        self.assertIn(b"data: [DONE]", body)
        self.assertIn(b'"treamed"', body)

    def test_a_request_that_already_fell_back_does_not_fall_back_again(self):
        status, _, body = _post(
            self.think.server_address[1],
            {"model": "think", "messages": []},
            {proxy.FALLBACK_HOP_HEADER: "1"},
        )
        self.assertEqual(status, 503)
        self.assertIn(b"backend_unavailable", body)
        self.assertEqual(self.fallback_rec.requests, [])

    def test_both_down_is_one_clear_503(self):
        proxy.CHAT_FALLBACK_URL = f"http://127.0.0.1:{_free_port()}"
        status, _, body = _post(self.think.server_address[1], {"model": "think", "messages": []})
        self.assertEqual(status, 503)
        err = json.loads(body)["error"]
        self.assertEqual(err["type"], "backend_unavailable")
        self.assertIn("failed too", err["message"])

    def test_no_fallback_configured_keeps_the_old_503(self):
        proxy.CHAT_FALLBACK_URL = ""
        status, _, body = _post(self.think.server_address[1], {"model": "think", "messages": []})
        self.assertEqual(status, 503)
        # One JSON body, not the body written twice.
        self.assertEqual(json.loads(body)["error"]["type"], "backend_unavailable")


def _fake_get_server(routes: dict):
    """GET only: routes maps a path to a callable returning (status, content type, body)."""
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_GET(self):
            hits.append(self.path)
            status, ctype, body = routes[self.path]()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.hits = hits
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return server


def _long_payload(tokens: int, *, first_user: str = "first question", max_tokens: int | None = None) -> dict:
    """A chat request whose body is about `tokens` x 3 characters."""
    payload = {
        "model": "think",
        "messages": [
            {"role": "system", "content": "You are a helpful agent."},
            {"role": "user", "content": first_user},
            {"role": "assistant", "content": "x" * (tokens * 3)},
            {"role": "user", "content": "and now?"},
        ],
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def _body(payload: dict) -> bytes:
    return json.dumps(payload).encode()


class GovernorRouteTests(unittest.TestCase):
    """Routing long prompts to the fallback while Splash refuses memory growth."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = [1000.0]
        self.status = {"memory_governor": {"growth_allowed": True}}
        self.metrics = "ninfer_requests_running 0\nninfer_requests_waiting 0\n"
        self.splash = _fake_get_server({"/status": lambda: (200, "application/json", json.dumps(self.status).encode())})
        self.llms = _fake_get_server({"/metrics": lambda: (200, "text/plain", self.metrics.encode())})
        self.lease_file = os.path.join(self.tmp.name, "gpu1-lease.json")
        self.log_dir = os.path.join(self.tmp.name, "routing")
        self.fallback_rec = _Recorder()
        self.fallback = _fake_server(self.fallback_rec, text="from fallback")
        self.backend_rec = _Recorder()
        self.backend = _fake_server(self.backend_rec, text="from backend")
        self.patches = mock.patch.multiple(
            proxy,
            MEMORY_GATEWAY_ENABLED=False,
            LOOP_GUARD=False,
            BACKEND_HOST="127.0.0.1",
            BACKEND_PORT=self.backend.server_address[1],
            CHAT_FALLBACK_URL=f"http://127.0.0.1:{self.fallback.server_address[1]}",
            CHAT_FALLBACK_TOKEN="",
            CHAT_FALLBACK_LEASE_FILE=self.lease_file,
            CHAT_FALLBACK_ON_GOVERNOR="on",
            CHAT_FALLBACK_GOVERNOR_STATUS_URL=f"http://127.0.0.1:{self.splash.server_address[1]}/status",
            CHAT_FALLBACK_METRICS_URL=f"http://127.0.0.1:{self.llms.server_address[1]}/metrics",
            CHAT_FALLBACK_ROUTING_LOG_DIR=self.log_dir,
            _monotonic=lambda: self.clock[0],
        )
        self.patches.start()
        proxy._FALLBACK_STATE["active"] = False
        proxy._ROUTE_STATE.update(pressure=False, denied_since=None, allowed_since=None)
        proxy._PROBE_CACHE.clear()
        proxy._STICKY.clear()
        self.think = _serve_proxy(proxy.make_handler(thinking_enabled=True, model_name="think", port_label="think"))

    def tearDown(self):
        for server in (self.think, self.fallback, self.backend, self.splash, self.llms):
            server.shutdown()
            server.server_close()
        self.patches.stop()
        self.tmp.cleanup()

    def _at(self, t: float):
        self.clock[0] = 1000.0 + t

    def _deny(self, denied: bool = True):
        self.status = {"memory_governor": {"growth_allowed": not denied}}

    def _decide(self, t: float, tokens: int = 30000, **kwargs) -> dict:
        self._at(t)
        return proxy._governor_route(_body(_long_payload(tokens, **kwargs)), "chat")

    def _under_pressure(self):
        self._deny()
        self._decide(0, first_user="warm-up")
        decision = self._decide(31, first_user="warm-up 2")
        self.assertTrue(decision["route"], decision)

    def _log_rows(self, at_least: int = 1) -> list[dict]:
        # The proxy logs after the reply has gone, so the client can be first.
        deadline = time.monotonic() + 5
        while True:
            rows = []
            if os.path.isdir(self.log_dir):
                for name in sorted(os.listdir(self.log_dir)):
                    with open(os.path.join(self.log_dir, name)) as fh:
                        rows.extend(json.loads(line) for line in fh if line.strip())
            if len(rows) >= at_least or time.monotonic() > deadline:
                return rows
            time.sleep(0.02)

    # -- entering and leaving ------------------------------------------------

    def test_enters_only_after_30_seconds_of_refused_growth(self):
        self._deny()
        first = self._decide(0, first_user="a")
        self.assertFalse(first["route"])
        self.assertEqual(first["reason"], "no memory pressure")
        self.assertEqual(first["governor"], "normal")
        self.assertFalse(self._decide(29, first_user="b")["route"])
        routed = self._decide(31, first_user="c")
        self.assertTrue(routed["route"])
        self.assertEqual(routed["reason"], "memory pressure")
        self.assertEqual(routed["governor"], "pressure")

    def test_an_allowed_sample_restarts_the_entry_clock(self):
        self._deny()
        self._decide(0, first_user="a")
        self._deny(False)
        self._decide(20, first_user="b")
        self._deny()
        self.assertFalse(self._decide(40, first_user="c")["route"])  # denied since 40, not 0
        self.assertTrue(self._decide(71, first_user="d")["route"])

    def test_leaves_only_after_300_seconds_allowed(self):
        self._under_pressure()
        self._deny(False)
        self.assertTrue(self._decide(40, first_user="e")["route"])  # allowed since 40
        self.assertTrue(self._decide(339, first_user="f")["route"])
        left = self._decide(341, first_user="g")
        self.assertFalse(left["route"])
        self.assertEqual(left["reason"], "no memory pressure")

    def test_a_refusal_while_leaving_restarts_the_leave_clock(self):
        self._under_pressure()
        self._deny(False)
        self._decide(40, first_user="e")
        self._deny()
        self._decide(200, first_user="f")
        self._deny(False)
        self._decide(210, first_user="g")
        self.assertTrue(self._decide(500, first_user="h")["route"])  # allowed since 210
        self.assertFalse(self._decide(511, first_user="i")["route"])

    def test_status_is_read_at_most_every_two_seconds(self):
        self._decide(0, first_user="a")
        self._decide(1, first_user="b")
        self.assertEqual(len(self.splash.hits), 1)
        self._decide(2.5, first_user="c")
        self.assertEqual(len(self.splash.hits), 2)

    def test_unreadable_status_routes_nothing(self):
        proxy.CHAT_FALLBACK_GOVERNOR_STATUS_URL = f"http://127.0.0.1:{_free_port()}/status"
        decision = self._decide(0)
        self.assertFalse(decision["route"])
        self.assertEqual(decision["reason"], "governor unreadable")

    # -- size ----------------------------------------------------------------

    def test_a_short_prompt_stays(self):
        self._under_pressure()
        decision = self._decide(40, tokens=20000, first_user="short")
        self.assertFalse(decision["route"])
        self.assertEqual(decision["reason"], "small")

    def test_a_prompt_must_leave_room_for_its_reply_and_the_margin(self):
        self._under_pressure()
        # 131072 - 16384 - 8192 = 106496
        too_big = self._decide(40, tokens=107000, max_tokens=16384, first_user="big")
        self.assertFalse(too_big["route"])
        self.assertEqual(too_big["reason"], "too large for the fallback")
        self.assertEqual(too_big["limit_tokens"], 106496)
        fits = self._decide(41, tokens=105000, max_tokens=16384, first_user="fits")
        self.assertTrue(fits["route"], fits)
        no_cap = self._decide(42, tokens=120000, first_user="no cap")  # limit 122880
        self.assertTrue(no_cap["route"], no_cap)

    # -- the fallback's state ------------------------------------------------

    def test_a_lease_refuses(self):
        self._under_pressure()
        pathlib.Path(self.lease_file).write_text("{}")
        decision = self._decide(40, first_user="leased")
        self.assertFalse(decision["route"])
        self.assertEqual(decision["reason"], "lease")

    def test_a_busy_fallback_refuses(self):
        self._under_pressure()
        self.metrics = "ninfer_requests_running 1\nninfer_requests_waiting 0\n"
        self.assertEqual(self._decide(40, first_user="x")["reason"], "fallback busy")
        self.metrics = "ninfer_requests_running 0\nninfer_requests_waiting 2\n"
        self.assertEqual(self._decide(43, first_user="y")["reason"], "fallback busy")
        self.metrics = "llamacpp:requests_processing 1\nllamacpp:requests_deferred 0\n"
        self.assertEqual(self._decide(46, first_user="z")["reason"], "fallback busy")
        self.metrics = "llamacpp:requests_processing 0\nllamacpp:requests_deferred 0\n"
        self.assertTrue(self._decide(49, first_user="w")["route"])

    def test_unreadable_or_unset_metrics_refuse(self):
        self._under_pressure()
        self.metrics = "something_else 1\n"
        self.assertEqual(self._decide(40, first_user="x")["reason"], "fallback metrics unreadable")
        proxy.CHAT_FALLBACK_METRICS_URL = f"http://127.0.0.1:{_free_port()}/metrics"
        self.assertEqual(self._decide(43, first_user="y")["reason"], "fallback metrics unreadable")
        proxy.CHAT_FALLBACK_METRICS_URL = ""
        self.assertEqual(self._decide(46, first_user="z")["reason"], "no fallback metrics url")

    # -- stickiness ----------------------------------------------------------

    def test_a_routed_conversation_stays_after_the_pressure_ends(self):
        self._under_pressure()
        self.assertTrue(self._decide(40, first_user="A")["route"])
        self._deny(False)
        self._decide(50, first_user="other")
        self.assertFalse(self._decide(400, first_user="B")["route"])  # pressure over
        again = self._decide(401, tokens=31000, first_user="A")
        self.assertTrue(again["route"])
        self.assertEqual(again["reason"], "sticky")
        self.assertTrue(again["sticky"])

    def test_stickiness_expires_after_30_idle_minutes(self):
        self._under_pressure()
        self._decide(40, first_user="A")
        self._deny(False)
        self.assertTrue(self._decide(40 + 1799, first_user="A")["route"])  # renews
        self.assertTrue(self._decide(40 + 1799 + 1799, first_user="A")["route"])
        self.assertFalse(self._decide(40 + 1799 * 2 + 1801, first_user="A")["route"])

    def test_a_sticky_conversation_still_needs_an_idle_fallback_and_room(self):
        self._under_pressure()
        self._decide(40, first_user="A")
        self._deny(False)
        self._decide(50, first_user="other")  # allowed since 50: the pressure ends at 350
        self.metrics = "ninfer_requests_running 1\n"
        busy = self._decide(400, first_user="A")
        self.assertEqual(busy["reason"], "fallback busy")
        self.metrics = "ninfer_requests_running 0\n"
        self.assertEqual(self._decide(403, first_user="A")["reason"], "sticky")  # still sticky
        grown = self._decide(406, tokens=125000, max_tokens=16384, first_user="A")
        self.assertEqual(grown["reason"], "too large for the fallback")
        self.assertEqual(self._decide(409, first_user="A")["reason"], "no memory pressure")  # unstuck

    def test_the_conversation_key_is_the_system_prompt_and_first_user_message(self):
        a = _long_payload(10, first_user="same")
        b = _long_payload(20, first_user="same")
        b["messages"].append({"role": "user", "content": "a later turn"})
        c = _long_payload(10, first_user="different")
        self.assertEqual(proxy._conversation_key(a, "chat"), proxy._conversation_key(b, "chat"))
        self.assertNotEqual(proxy._conversation_key(a, "chat"), proxy._conversation_key(c, "chat"))
        self.assertIsNone(proxy._conversation_key({"messages": [{"role": "system", "content": "s"}]}, "chat"))

    # -- end to end ----------------------------------------------------------

    def test_on_mode_serves_a_routed_request_from_the_fallback(self):
        self._under_pressure()
        self._at(40)
        status, headers, body = _post(self.think.server_address[1], _long_payload(30000, first_user="e2e"))
        self.assertEqual(status, 200)
        self.assertEqual(headers.get(proxy.SERVED_BY_HEADER), "fallback")
        self.assertIn(b"from fallback", body)
        self.assertEqual(self.backend_rec.requests, [])
        self.assertEqual(self.fallback_rec.requests[0]["headers"].get(proxy.FALLBACK_HOP_HEADER), "1")
        row = self._log_rows()[-1]
        self.assertEqual((row["route"], row["served_by"], row["reason"], row["port"]),
                         (True, "fallback", "memory pressure", "think"))
        self.assertGreater(row["est_tokens"], 30000)
        self.assertIn("time", row)
        self.assertFalse(proxy._FALLBACK_STATE["active"])  # not the backend-down fallback

    def test_shadow_mode_logs_the_decision_and_serves_here(self):
        proxy.CHAT_FALLBACK_ON_GOVERNOR = "shadow"
        self._under_pressure()
        self._at(40)
        status, headers, body = _post(self.think.server_address[1], _long_payload(30000, first_user="shadow"))
        self.assertEqual(status, 200)
        self.assertNotIn(proxy.SERVED_BY_HEADER, headers)
        self.assertIn(b"from backend", body)
        self.assertEqual(self.fallback_rec.requests, [])
        self.assertEqual(len(self.backend_rec.requests), 1)
        row = self._log_rows()[-1]
        self.assertEqual((row["mode"], row["route"], row["served_by"]), ("shadow", True, "local"))

    def test_a_small_request_is_logged_and_served_here(self):
        status, _, body = _post(self.think.server_address[1], {"model": "think", "messages": [{"role": "user", "content": "q"}]})
        self.assertEqual(status, 200)
        self.assertIn(b"from backend", body)
        row = self._log_rows()[-1]
        self.assertEqual((row["route"], row["reason"], row["served_by"]), (False, "small", "local"))

    def test_a_failing_fallback_is_served_here(self):
        self._under_pressure()
        self._at(40)
        self.fallback.shutdown()
        self.fallback.server_close()
        self.fallback = _fake_server(self.fallback_rec, status=503)
        proxy.CHAT_FALLBACK_URL = f"http://127.0.0.1:{self.fallback.server_address[1]}"
        status, headers, body = _post(self.think.server_address[1], _long_payload(30000, first_user="f503"))
        self.assertEqual(status, 200)
        self.assertIn(b"from backend", body)
        self.assertNotIn(proxy.SERVED_BY_HEADER, headers)
        self.assertEqual(len(self.fallback_rec.requests), 1)
        row = self._log_rows()[-1]
        self.assertEqual((row["served_by"], row.get("fallback_failed")), ("local", True))

        proxy.CHAT_FALLBACK_URL = f"http://127.0.0.1:{_free_port()}"
        status, _, body = _post(self.think.server_address[1], _long_payload(30000, first_user="down"))
        self.assertEqual(status, 200)
        self.assertIn(b"from backend", body)

    def test_off_reads_nothing_and_logs_nothing(self):
        proxy.CHAT_FALLBACK_ON_GOVERNOR = "off"
        proxy._ROUTE_STATE["pressure"] = True
        status, headers, body = _post(self.think.server_address[1], _long_payload(30000))
        self.assertEqual(status, 200)
        self.assertIn(b"from backend", body)
        self.assertEqual((self.splash.hits, self.llms.hits), ([], []))
        self.assertFalse(os.path.exists(self.log_dir))

    def test_a_forwarded_request_is_refused_at_once_while_lent_out(self):
        pathlib.Path(self.lease_file).write_text("{}")
        status, _, body = _post(self.think.server_address[1], {"model": "think", "messages": []},
                                {proxy.FALLBACK_HOP_HEADER: "1"})
        self.assertEqual(status, 503)
        self.assertIn(b"lent out", body)
        self.assertEqual(self.backend_rec.requests, [])
        self.assertEqual(self.fallback_rec.requests, [])


class AuthTests(unittest.TestCase):
    def test_token_is_required_only_off_this_machine(self):
        with mock.patch.object(proxy, "PROXY_AUTH_TOKEN", "t0k"):
            self.assertTrue(proxy._client_authorized("127.0.0.1", None))
            self.assertTrue(proxy._client_authorized("::1", None))
            self.assertFalse(proxy._client_authorized("100.124.56.11", None))
            self.assertFalse(proxy._client_authorized("100.124.56.11", "Bearer nope"))
            self.assertTrue(proxy._client_authorized("100.124.56.11", "Bearer t0k"))
        with mock.patch.object(proxy, "PROXY_AUTH_TOKEN", ""):
            self.assertTrue(proxy._client_authorized("100.124.56.11", None))


if __name__ == "__main__":
    unittest.main()
