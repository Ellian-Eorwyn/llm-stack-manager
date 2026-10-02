"""The chat proxy's repetition-loop guard (LOOP_GUARD).

Three parts: the detector on committed text (this repo's own code and docs as
long legitimate output; loops built the way the 2026-09-29 one ran), the
detector on Hermes's recorded replies when its database is on this machine
(nothing from it is copied into the repo), and the relay against a fake
streaming backend over real sockets.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import socket
import sqlite3
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_proxy_module():
    spec = importlib.util.spec_from_file_location("llm_chat_proxy_loop_guard", ROOT / "scripts" / "llm-chat-proxy.py")
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


proxy = _load_proxy_module()

# One paragraph of reasoning, looped the way the 09-29 reply looped: the same
# ~300 characters back to back, with nothing in between.
LOOP_PARAGRAPH = (
    "I'm getting tangled up in the order of the steps here. The scheduler keeps one slot, "
    "but the request before this one may still hold the cache, and I can't tell from the log "
    "which of them finished first. I need to be careful not to assume the order without "
    "checking the timestamps again. "
)
LEAD_IN = (ROOT / "docs" / "splash.md").read_text(encoding="utf-8")[:6000]


def _feed_in_pieces(detector, text: str, size: int = 7) -> int | None:
    """Feed text a few characters at a time, as deltas arrive; returns the
    number of characters fed when the detector tripped, else None."""
    for start in range(0, len(text), size):
        if detector.feed(text[start:start + size]):
            return start + size
    return None


class DetectorTests(unittest.TestCase):
    def test_a_looping_paragraph_trips_within_a_few_repeats(self):
        detector = proxy.RepetitionDetector()
        tripped = _feed_in_pieces(detector, LEAD_IN + LOOP_PARAGRAPH * 60)
        self.assertIsNotNone(tripped)
        period, repeats = detector.loop
        self.assertEqual(period, len(LOOP_PARAGRAPH))
        self.assertGreaterEqual(repeats, proxy.LOOP_GUARD_MIN_REPEATS)
        # Seen within two blocks of the threshold, not at the output cap.
        into_loop = tripped - len(LEAD_IN)
        self.assertLessEqual(into_loop, (proxy.LOOP_GUARD_MIN_REPEATS + 2) * len(LOOP_PARAGRAPH))
        self.assertEqual(len(detector.run_text()), period * repeats)
        self.assertIn(LOOP_PARAGRAPH.strip()[:60], detector.run_text())

    def test_a_cycle_of_several_paragraphs_trips_at_its_full_period(self):
        cycle = "".join(f"Step {n}: " + LOOP_PARAGRAPH for n in range(3))
        detector = proxy.RepetitionDetector()
        self.assertIsNotNone(_feed_in_pieces(detector, LEAD_IN + cycle * 12))
        self.assertEqual(detector.loop[0], len(cycle))

    def test_a_short_line_repeated_trips_once_the_run_is_long(self):
        detector = proxy.RepetitionDetector()
        line = "Wait, let me check the file again.\n"
        tripped = _feed_in_pieces(detector, LEAD_IN + line * 200)
        self.assertIsNotNone(tripped)
        self.assertGreaterEqual(tripped - len(LEAD_IN), proxy.LOOP_GUARD_MIN_RUN_CHARS)

    def test_a_few_repeats_then_new_text_does_not_trip(self):
        detector = proxy.RepetitionDetector()
        text = LEAD_IN + LOOP_PARAGRAPH * (proxy.LOOP_GUARD_MIN_REPEATS - 1) + LEAD_IN
        self.assertIsNone(_feed_in_pieces(detector, text))

    def test_reasoning_that_requotes_one_line_does_not_trip(self):
        # The legitimate near miss in Hermes's history: one source line quoted
        # again and again, with different reasoning between the quotes.
        quote = '"' + LOOP_PARAGRAPH.strip() + '"'
        paragraphs = [p for p in LEAD_IN.split("\n\n") if p.strip()]
        text = "".join(f"Line {n} reads {quote}\n\n{paragraphs[n % len(paragraphs)]}\n\n" for n in range(20))
        self.assertIsNone(_feed_in_pieces(proxy.RepetitionDetector(), text))

    def test_long_code_does_not_trip(self):
        source = (ROOT / "scripts" / "llm-chat-proxy.py").read_text(encoding="utf-8")
        self.assertIsNone(_feed_in_pieces(proxy.RepetitionDetector(), source, size=11))

    def test_the_docs_do_not_trip(self):
        docs = "".join(path.read_text(encoding="utf-8") for path in sorted((ROOT / "docs").glob("*.md")))
        self.assertIsNone(_feed_in_pieces(proxy.RepetitionDetector(), docs, size=11))

    def test_a_long_table_and_json_with_shared_fragments_do_not_trip(self):
        rows = "".join(
            f"| {n:>4} | think | medium | 1.0 | 0.95 | 20 | splash | {n * 37 % 997:>4} ms | ok |\n"
            for n in range(400)
        )
        table = "| n | port | effort | temp | top_p | top_k | engine | ttft | status |\n|---|---|---|---|---|---|---|---|---|\n" + rows
        records = json.dumps(
            [{"type": "function", "status": "completed", "name": "read_file", "arguments": {"path": f"/tmp/file-{n}.txt", "offset": 0, "limit": 2000}} for n in range(300)],
            indent=2,
        )
        self.assertIsNone(_feed_in_pieces(proxy.RepetitionDetector(), table, size=11))
        self.assertIsNone(_feed_in_pieces(proxy.RepetitionDetector(), records, size=11))

    def test_a_block_longer_than_the_span_allows_is_not_a_loop(self):
        detector = proxy.RepetitionDetector(span=4096, min_run=256)
        block = LEAD_IN[:1000]
        self.assertIsNone(_feed_in_pieces(detector, block * 5))


STATE_DB = pathlib.Path(os.environ.get("HERMES_STATE_DB", "/Users/ellie/.hermes/state.db"))
# The 2026-09-29 thinking loop (32,768 tokens at 153 tok/s), and a session of
# long legitimate xhigh thinking from 2026-10-01 that must never trip.
RECORDED_LOOP_ID = 17439
RECORDED_LONG_SESSION = "20261001_134609_56722c"


@unittest.skipUnless(STATE_DB.exists(), "Hermes's state.db is not on this machine")
class RecordedRepliesTests(unittest.TestCase):
    """The detector against Hermes's own replies, read in place."""

    @classmethod
    def setUpClass(cls):
        cls.db = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def _trips(self, text: str) -> bool:
        return _feed_in_pieces(proxy.RepetitionDetector(), text, size=64) is not None

    def test_the_recorded_loop_trips_early(self):
        row = self.db.execute("select reasoning from messages where id=?", (RECORDED_LOOP_ID,)).fetchone()
        if not row or not row[0]:
            self.skipTest("the recorded loop is not in this database")
        detector = proxy.RepetitionDetector()
        tripped = _feed_in_pieces(detector, row[0], size=64)
        self.assertIsNotNone(tripped)
        # The reply ran to 137,681 characters; the guard stops it at a seventh.
        self.assertLess(tripped, 20_000)

    def test_long_legitimate_thinking_does_not_trip(self):
        rows = self.db.execute(
            "select id, coalesce(reasoning,''), coalesce(content,'') from messages "
            "where session_id=? and role='assistant'",
            (RECORDED_LONG_SESSION,),
        ).fetchall()
        if not rows:
            self.skipTest("the recorded session is not in this database")
        for message_id, reasoning, content in rows:
            for text in (reasoning, content):
                self.assertFalse(self._trips(text), f"message {message_id} tripped")

    def test_no_other_recorded_reply_trips(self):
        rows = self.db.execute(
            "select id, coalesce(reasoning,''), coalesce(content,'') from messages "
            "where role='assistant' and id <= 19500 and id != ?",
            (RECORDED_LOOP_ID,),
        ).fetchall()
        tripped = [
            message_id for message_id, reasoning, content in rows
            if (len(reasoning) >= 1500 and self._trips(reasoning)) or (len(content) >= 1500 and self._trips(content))
        ]
        self.assertEqual(tripped, [])


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _pieces(text: str, size: int = 12) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


class _Backend:
    """A fake streaming backend. Each request takes the next plan: a list of
    (field, text) deltas, then a finish reason, or ("status", n, body)."""

    def __init__(self, plans, *, chunked: bool = False, keepalive: bool = False):
        self.plans = list(plans)
        self.requests: list[dict] = []
        self.cut_short: list[bool] = []
        backend = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1" if chunked else "HTTP/1.0"

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                backend.requests.append(body)
                plan = backend.plans.pop(0)
                if plan[0] == "status":
                    msg = json.dumps(plan[2]).encode()
                    self.send_response(plan[1])
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(msg)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(msg)
                    return
                deltas, finish = plan
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                if chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def emit(raw: bytes):
                    if chunked:
                        raw = f"{len(raw):x}\r\n".encode() + raw + b"\r\n"
                    self.wfile.write(raw)
                    self.wfile.flush()

                def chunk(delta, finish_reason=None, **extra):
                    obj = {"id": f"chatcmpl-{len(backend.requests)}", "object": "chat.completion.chunk",
                           "created": 1, "model": "backend-model",
                           "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}], **extra}
                    emit(b"data: " + json.dumps(obj).encode() + b"\n\n")

                try:
                    if keepalive:
                        emit(b": splash-keepalive\n\n")
                    chunk({"role": "assistant", "content": ""})
                    for field, text in deltas:
                        if field == "tool_calls":
                            chunk({"tool_calls": [text]})
                        else:
                            chunk({field: text})
                    chunk({}, finish)
                    if (body.get("stream_options") or {}).get("include_usage"):
                        emit(b"data: " + json.dumps({"id": "u", "object": "chat.completion.chunk", "created": 1,
                                                     "model": "backend-model", "choices": [],
                                                     "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}}).encode() + b"\n\n")
                    emit(b"data: [DONE]\n\n")
                    if chunked:
                        self.wfile.write(b"0\r\n\r\n")
                    backend.cut_short.append(False)
                except (BrokenPipeError, ConnectionResetError):
                    backend.cut_short.append(True)
                self.close_connection = True

            def log_message(self, *args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def port(self) -> int:
        return self.server.server_address[1]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _looping(field: str = "reasoning_content", lead: str = "Let me look at the request first. ") -> list:
    # Enough loop to overflow socket buffers, so cancelling is observable.
    return [(field, piece) for piece in _pieces(lead + LOOP_PARAGRAPH * 3000)]


def _clean() -> list:
    return [("reasoning_content", piece) for piece in _pieces("A short, healthy thought.")] + [
        ("content", piece) for piece in _pieces("The answer.")
    ]


def _post(port: int, payload: dict):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, dict(exc.headers), exc.read()


def _events(body: bytes) -> list:
    events = []
    for line in body.decode().splitlines():
        if line.startswith("data: "):
            data = line[6:]
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


def _joined(events: list, field: str) -> str:
    return "".join(
        choice.get("delta", {}).get(field) or ""
        for event in events if isinstance(event, dict)
        for choice in event.get("choices", [])
    )


def _finish_reasons(events: list) -> list:
    return [
        choice["finish_reason"]
        for event in events if isinstance(event, dict)
        for choice in event.get("choices", []) if choice.get("finish_reason")
    ]


class RelayTests(unittest.TestCase):
    def setUp(self):
        self.patches = mock.patch.multiple(
            proxy,
            LOOP_GUARD=True,
            LOOP_GUARD_RETRIES=1,
            LOOP_GUARD_NONSTREAM=True,
            PROXY_STREAM_PASSTHROUGH=False,
            MEMORY_GATEWAY_ENABLED=False,
            BACKEND_ENGINE="",
            BACKEND_HOST="127.0.0.1",
            CHAT_FALLBACK_URL="",
        )
        self.patches.start()
        self.addCleanup(self.patches.stop)
        self.proxy = proxy.ProxyHTTPServer(("127.0.0.1", 0), proxy.make_handler(
            thinking_enabled=True, model_name="think", port_label="think",
            inject_overrides={"temperature": 1.0},
        ))
        threading.Thread(target=self.proxy.serve_forever, daemon=True).start()
        self.addCleanup(self.proxy.server_close)
        self.addCleanup(self.proxy.shutdown)

    def _backend(self, plans, **kwargs) -> _Backend:
        backend = _Backend(plans, **kwargs)
        self.addCleanup(backend.close)
        proxy.BACKEND_PORT = backend.port
        return backend

    def _ask(self, **extra):
        return _post(self.proxy.server_address[1], {"model": "think", "messages": [{"role": "user", "content": "q"}], **extra})

    def test_a_loop_is_cancelled_and_retried_warmer(self):
        backend = self._backend([(_looping(), "length"), (_clean(), "stop")])
        status, _, body = self._ask(stream=True)
        self.assertEqual(status, 200)
        events = _events(body)
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(backend.cut_short[0], True)
        self.assertAlmostEqual(backend.requests[1]["temperature"], backend.requests[0]["temperature"] + 0.1)
        self.assertEqual(_joined(events, "content"), "The answer.")
        self.assertIn(proxy.LOOP_GUARD_RETRY_MARK, _joined(events, "reasoning_content"))
        self.assertEqual(_finish_reasons(events), ["stop"])
        self.assertEqual(events.count("[DONE]"), 1)
        # One id for the reply, though the backend answered twice.
        self.assertEqual({e["id"] for e in events if isinstance(e, dict)}, {"chatcmpl-1"})
        self.assertTrue(all(e["model"] == "think" for e in events if isinstance(e, dict)))

    def test_passthrough_streams_stay_unrewritten_but_are_still_guarded(self):
        # llms runs PROXY_STREAM_PASSTHROUGH=on; the guard must not quietly
        # turn its streams into rewritten ones.
        proxy.PROXY_STREAM_PASSTHROUGH = True
        self.addCleanup(setattr, proxy, "PROXY_STREAM_PASSTHROUGH", False)
        backend = self._backend([(_looping(), "length"), (_clean(), "stop")])
        status, _, body = self._ask(stream=True)
        events = _events(body)
        self.assertEqual(status, 200)
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(_joined(events, "content"), "The answer.")
        self.assertIn(proxy.LOOP_GUARD_RETRY_MARK, _joined(events, "reasoning_content"))
        # The backend's own model name, as passthrough always relayed it.
        self.assertEqual({e["model"] for e in events if isinstance(e, dict)}, {"backend-model"})
        self.assertEqual(_finish_reasons(events), ["stop"])

    def test_a_client_seed_moves_on_for_the_retry(self):
        backend = self._backend([(_looping(), "length"), (_clean(), "stop")])
        self._ask(stream=True, seed=41)
        self.assertEqual([r["seed"] for r in backend.requests], [41, 42])

    def test_a_second_loop_ends_the_reply_as_a_loop(self):
        backend = self._backend([(_looping(), "length"), (_looping(), "length")])
        status, _, body = self._ask(stream=True)
        events = _events(body)
        self.assertEqual(status, 200)
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(backend.cut_short, [True, True])
        self.assertEqual(_finish_reasons(events), ["length"])
        self.assertEqual(events[-1], "[DONE]")
        # The looping text is the content: what the model did, and what lets
        # Hermes stop with "Repetition Detected" instead of continuing it.
        content = _joined(events, "content")
        self.assertGreaterEqual(content.count(LOOP_PARAGRAPH.strip()[:80]), proxy.LOOP_GUARD_MIN_REPEATS - 1)

    def test_a_loop_after_content_went_out_is_not_retried(self):
        deltas = [("content", piece) for piece in _pieces("Here is the start of the answer. ")] + _looping("content", lead="")
        backend = self._backend([(deltas, "length"), (_clean(), "stop")])
        _, _, body = self._ask(stream=True)
        events = _events(body)
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(_finish_reasons(events), ["length"])
        self.assertTrue(_joined(events, "content").startswith("Here is the start of the answer. "))
        self.assertNotIn(proxy.LOOP_GUARD_RETRY_MARK, _joined(events, "reasoning_content"))

    def test_a_clean_stream_passes_unchanged_and_keepalives_reach_the_client(self):
        backend = self._backend([(_clean(), "stop")], keepalive=True)
        status, _, body = self._ask(stream=True)
        self.assertEqual(status, 200)
        self.assertIn(b": splash-keepalive", body)
        events = _events(body)
        self.assertEqual(_joined(events, "content"), "The answer.")
        self.assertEqual(_joined(events, "reasoning_content"), "A short, healthy thought.")
        self.assertEqual(len(backend.requests), 1)
        self.assertNotIn("stream_options", backend.requests[0])

    def test_a_chunked_backend_is_read_and_reframed(self):
        self._backend([(_looping(), "length"), (_clean(), "stop")], chunked=True)
        status, headers, body = self._ask(stream=True)
        self.assertEqual(status, 200)
        self.assertNotIn("chunked", headers.get("Transfer-Encoding", ""))
        self.assertEqual(_joined(_events(body), "content"), "The answer.")

    def test_a_non_streaming_client_is_guarded_and_gets_one_json(self):
        backend = self._backend([(_looping(), "length"), (_clean(), "stop")])
        status, headers, body = self._ask()
        self.assertEqual(status, 200)
        self.assertTrue(backend.requests[0]["stream"])
        self.assertTrue(backend.requests[0]["stream_options"]["include_usage"])
        reply = json.loads(body)
        self.assertEqual(reply["object"], "chat.completion")
        self.assertEqual(reply["model"], "think")
        message = reply["choices"][0]["message"]
        self.assertEqual(message["content"], "The answer.")
        # Only the attempt that finished: the client never saw the loop.
        self.assertEqual(message["reasoning_content"], "A short, healthy thought.")
        self.assertEqual(reply["choices"][0]["finish_reason"], "stop")
        self.assertEqual(reply["usage"]["completion_tokens"], 7)

    def test_a_non_streaming_reply_with_tool_calls_is_rebuilt(self):
        calls = [
            ("tool_calls", {"index": 0, "id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": ""}}),
            ("tool_calls", {"index": 0, "function": {"arguments": '{"path": '}}),
            ("tool_calls", {"index": 0, "function": {"arguments": '"/tmp/a"}'}}),
        ]
        self._backend([(calls, "tool_calls")])
        _, _, body = self._ask()
        reply = json.loads(body)
        self.assertIsNone(reply["choices"][0]["message"]["content"])
        self.assertEqual(reply["choices"][0]["message"]["tool_calls"], [
            {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "/tmp/a"}'}},
        ])
        self.assertEqual(reply["choices"][0]["finish_reason"], "tool_calls")

    def test_a_non_streaming_double_loop_ends_with_length(self):
        self._backend([(_looping(), "length"), (_looping(), "length")])
        _, _, body = self._ask()
        reply = json.loads(body)
        self.assertEqual(reply["choices"][0]["finish_reason"], "length")
        self.assertIn(LOOP_PARAGRAPH.strip()[:80], reply["choices"][0]["message"]["content"])

    def test_a_backend_error_reaches_the_client_as_is(self):
        self._backend([("status", 400, {"error": {"message": "min_p is not supported", "type": "invalid_request_error"}})])
        status, _, body = self._ask(stream=True)
        self.assertEqual(status, 400)
        self.assertIn(b"min_p is not supported", body)

    def test_a_retry_refused_by_the_backend_ends_the_stream_with_an_error_event(self):
        self._backend([(_looping(), "length"), ("status", 503, {"error": {"message": "loading"}})])
        _, _, body = self._ask(stream=True)
        events = _events(body)
        self.assertTrue(any(isinstance(e, dict) and "error" in e for e in events))
        self.assertEqual(events[-1], "[DONE]")

    def test_a_loading_backend_still_falls_back_with_the_clients_own_body(self):
        self._backend([("status", 503, {"error": {"message": "loading"}})])
        reply = {"id": "x", "object": "chat.completion", "created": 1, "model": "think",
                 "choices": [{"index": 0, "message": {"role": "assistant", "content": "from fallback"}, "finish_reason": "stop"}]}
        fallback = _Backend([("status", 200, reply)])
        self.addCleanup(fallback.close)
        proxy.CHAT_FALLBACK_URL = f"http://127.0.0.1:{fallback.port}"
        proxy._FALLBACK_STATE["active"] = False
        status, headers, body = self._ask()
        self.assertEqual(status, 200)
        self.assertEqual(headers.get(proxy.SERVED_BY_HEADER), "fallback")
        self.assertIn(b"from fallback", body)
        self.assertNotIn("stream", fallback.requests[0])

    def test_off_means_the_old_path(self):
        proxy.LOOP_GUARD = False
        backend = self._backend([(_clean(), "stop")])
        status, _, body = self._ask(stream=True)
        self.assertEqual(status, 200)
        self.assertEqual(_joined(_events(body), "content"), "The answer.")
        self.assertNotIn("stream_options", backend.requests[0])

    def test_only_single_choice_chat_completions_are_guarded(self):
        self.assertTrue(proxy._loop_guard_applies({"stream": True}, "chat"))
        self.assertTrue(proxy._loop_guard_applies({}, "chat"))
        self.assertFalse(proxy._loop_guard_applies({"stream": True}, "responses"))
        self.assertFalse(proxy._loop_guard_applies({"stream": True, "n": 2}, "chat"))
        self.assertFalse(proxy._loop_guard_applies({"logprobs": True}, "chat"))
        with mock.patch.object(proxy, "LOOP_GUARD_NONSTREAM", False):
            self.assertFalse(proxy._loop_guard_applies({}, "chat"))
            self.assertTrue(proxy._loop_guard_applies({"stream": True}, "chat"))


class ChunkedDecoderTests(unittest.TestCase):
    def test_framing_split_anywhere_is_undone(self):
        framed = b"5\r\nhello\r\n7;ext=1\r\n, world\r\n0\r\n\r\n"
        for cut in range(len(framed)):
            decoder = proxy._ChunkedDecoder()
            self.assertEqual(decoder.feed(framed[:cut]) + decoder.feed(framed[cut:]), b"hello, world")


if __name__ == "__main__":
    unittest.main()
