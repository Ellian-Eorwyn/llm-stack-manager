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
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _serve_proxy(handler_class):
    server = proxy.ProxyHTTPServer(("127.0.0.1", 0), handler_class)
    threading.Thread(target=server.serve_forever, daemon=True).start()
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
