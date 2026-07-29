from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest


def _load_proxy_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    proxy_path = root / "scripts" / "llm-chat-proxy.py"
    spec = importlib.util.spec_from_file_location("llm_chat_proxy", proxy_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


proxy = _load_proxy_module()


class RequestKindTests(unittest.TestCase):
    def test_request_kind_detects_supported_routes(self):
        self.assertEqual(proxy._request_kind("/v1/models"), "models")
        self.assertEqual(proxy._request_kind("/v1/models/chat"), "model")
        self.assertEqual(proxy._request_kind("/v1/chat/completions"), "chat")
        self.assertEqual(proxy._request_kind("/v1/responses"), "responses")
        self.assertEqual(proxy._request_kind("/v1/embeddings"), "embeddings")

    def test_requested_model_id_decodes_path_segment(self):
        self.assertEqual(proxy._requested_model_id_from_path("/v1/models/openclaw%2Fdefault"), "openclaw/default")


class MemoryInjectionTests(unittest.TestCase):
    def test_responses_string_input_becomes_developer_prefixed_messages(self):
        payload = {"input": "hello"}
        injected = proxy._inject_memory(payload, "responses", "remember this")
        self.assertTrue(injected)
        self.assertEqual(
            payload["input"],
            [
                {"role": "developer", "content": "[MEMORY CONTEXT]\nremember this"},
                {"role": "user", "content": "hello"},
            ],
        )

    def test_prefix_user_mode_preserves_string_input_shape(self):
        payload = {"input": "hello"}
        original_mode = proxy.MEMORY_INJECTION_MODE
        try:
            proxy.MEMORY_INJECTION_MODE = "prefix_user"
            injected = proxy._inject_memory(payload, "responses", "remember this")
        finally:
            proxy.MEMORY_INJECTION_MODE = original_mode
        self.assertTrue(injected)
        self.assertEqual(payload["input"], "[MEMORY CONTEXT]\nremember this\n\nhello")


class ResponsesNormalizationTests(unittest.TestCase):
    def test_normalize_responses_payload_merges_instructions_and_message_items(self):
        payload = {
            "instructions": "follow policy",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        }
        normalized = proxy._normalize_responses_payload(payload)
        self.assertEqual(
            normalized["input"],
            [
                {"role": "developer", "content": "follow policy"},
                {"role": "user", "content": "hello"},
            ],
        )
        self.assertNotIn("instructions", normalized)

    def test_normalize_responses_payload_converts_function_call_output(self):
        payload = {
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": {"ok": True},
                }
            ]
        }
        normalized = proxy._normalize_responses_payload(payload)
        self.assertEqual(
            normalized["input"],
            [{"role": "tool", "content": '{"ok":true}', "tool_call_id": "call_123"}],
        )


class MaxTokensOverrideTests(unittest.TestCase):
    def test_chat_override_sets_max_tokens(self):
        payload = {"max_tokens": 40}
        proxy._inject_max_tokens(payload, "chat", 4096)
        self.assertEqual(payload["max_tokens"], 4096)

    def test_chat_override_preserves_completion_token_field_when_present_alone(self):
        payload = {"max_completion_tokens": 40}
        proxy._inject_max_tokens(payload, "chat", 4096)
        self.assertEqual(payload["max_completion_tokens"], 4096)
        self.assertNotIn("max_tokens", payload)

    def test_responses_override_sets_max_output_tokens(self):
        payload = {"max_output_tokens": 40}
        proxy._inject_max_tokens(payload, "responses", 4096)
        self.assertEqual(payload["max_output_tokens"], 4096)

    def test_zero_override_leaves_payload_unchanged(self):
        payload = {"max_tokens": 40}
        proxy._inject_max_tokens(payload, "chat", 0)
        self.assertEqual(payload["max_tokens"], 40)


class ResponseHelpersTests(unittest.TestCase):
    def test_filtered_upstream_headers_removes_hop_by_hop_and_framing_headers(self):
        headers = {
            "Host": "127.0.0.1:8008",
            "Connection": "keep-alive",
            "Content-Length": "123",
            "Transfer-Encoding": "chunked",
            "Expect": "100-continue",
            "Authorization": "Bearer test",
            "Content-Type": "application/json",
        }
        filtered = proxy._filtered_upstream_headers(headers)
        self.assertNotIn("Host", filtered)
        self.assertNotIn("Connection", filtered)
        self.assertNotIn("Content-Length", filtered)
        self.assertNotIn("Transfer-Encoding", filtered)
        self.assertNotIn("Expect", filtered)
        self.assertEqual(filtered["Authorization"], "Bearer test")
        self.assertEqual(filtered["Content-Type"], "application/json")

    def test_extract_assistant_text_from_responses_payload(self):
        raw = json.dumps(
            {
                "output": [
                    {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "scratch"}]},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello world"}],
                    },
                ]
            }
        ).encode("utf-8")
        self.assertEqual(proxy._extract_assistant_text_from_nonstream_response(raw, "responses"), "hello world")

    def test_rewrite_json_response_model_updates_top_level_model(self):
        raw = json.dumps({"model": "chat-moe", "object": "response"}).encode("utf-8")
        rewritten = proxy._rewrite_json_response_model(raw, "chat")
        self.assertEqual(json.loads(rewritten)["model"], "chat")

    def test_prepare_embedding_payload_rewrites_backend_model(self):
        payload, response_model = proxy._prepare_embedding_payload({"model": "chat", "input": ["hello"]}, "chat")
        self.assertEqual(payload["model"], proxy.EMBED_MODEL_NAME)
        self.assertEqual(response_model, "chat")

    def test_sse_model_rewriter_rewrites_streamed_model_field(self):
        rewriter = proxy.SSEModelRewriter("chat")
        chunk = (
            b'data: {"id":"resp_1","model":"chat-moe","type":"response.created"}\n\n'
            b'data: [DONE]\n\n'
        )
        rewritten = rewriter.feed(chunk)
        text = rewritten.decode("utf-8")
        self.assertIn('"model":"chat"', text)
        self.assertNotIn('"model":"chat-moe"', text)
        self.assertIn("data: [DONE]", text)

    def test_sse_event_rewriter_rewrites_model_and_reasoning_visibility(self):
        rewriter = proxy.SSEEventRewriter("think", "content")
        chunk = (
            b'data: {"model":"chat-dense","choices":[{"delta":{"reasoning_content":"step"}}]}\n\n'
            b'data: [DONE]\n\n'
        )
        rewritten = rewriter.feed(chunk).decode("utf-8")
        self.assertIn('"model":"think"', rewritten)
        self.assertIn('"content":"step"', rewritten)
        self.assertNotIn("reasoning_content", rewritten)

    def test_stream_rewrite_is_not_safe_for_chunked_transfer(self):
        self.assertFalse(proxy._stream_rewrite_safe({"transfer-encoding": "chunked"}))
        self.assertFalse(proxy._stream_rewrite_safe({"transfer-encoding": "gzip, chunked"}))
        self.assertTrue(proxy._stream_rewrite_safe({"content-type": "text/event-stream"}))

    def test_stream_passthrough_can_be_forced(self):
        original = proxy.PROXY_STREAM_PASSTHROUGH
        try:
            proxy.PROXY_STREAM_PASSTHROUGH = True
            self.assertTrue(proxy._stream_passthrough_enabled({"content-type": "text/event-stream"}))
        finally:
            proxy.PROXY_STREAM_PASSTHROUGH = original

    def test_stream_passthrough_still_handles_chunked_streams(self):
        original = proxy.PROXY_STREAM_PASSTHROUGH
        try:
            proxy.PROXY_STREAM_PASSTHROUGH = False
            self.assertTrue(proxy._stream_passthrough_enabled({"transfer-encoding": "chunked"}))
        finally:
            proxy.PROXY_STREAM_PASSTHROUGH = original


class AggregateProxyTests(unittest.TestCase):
    def test_aggregate_models_clone_backend_metadata_for_each_alias(self):
        backend_payload = {
            "object": "list",
            "models": [
                {
                    "name": "chat-dense",
                    "model": "chat-dense",
                    "capabilities": ["completion"],
                    "details": {"format": "gguf"},
                }
            ],
            "data": [
                {
                    "id": "chat-dense",
                    "aliases": ["chat-dense"],
                    "object": "model",
                    "created": 123,
                    "owned_by": "llamacpp",
                    "meta": {"n_ctx": 256000, "n_ctx_train": 262144},
                }
            ],
        }
        payload = proxy._aggregate_models_payload(backend_payload)

        ids = [model["id"] for model in payload["data"]]
        self.assertEqual(ids, [proxy.THINK_MODEL_NAME, proxy.NOTHINK_MODEL_NAME, proxy.CODE_MODEL_NAME])
        for model in payload["data"]:
            self.assertEqual(model["meta"]["n_ctx"], 256000)
            self.assertEqual(model["aliases"], [model["id"]])

        self.assertEqual([model["name"] for model in payload["models"]], ids)
        self.assertEqual([model["model"] for model in payload["models"]], ids)
        self.assertEqual(payload["models"][0]["details"]["format"], "gguf")

    def test_aggregate_model_payload_returns_selected_alias_metadata(self):
        backend_payload = {
            "data": [
                {
                    "id": "chat-dense",
                    "object": "model",
                    "created": 123,
                    "owned_by": "llamacpp",
                    "meta": {"n_ctx": 256000},
                }
            ],
        }
        model = proxy._aggregate_model_payload(proxy.CODE_MODEL_NAME, backend_payload)
        self.assertEqual(model["id"], proxy.CODE_MODEL_NAME)
        self.assertEqual(model["meta"]["n_ctx"], 256000)

    def test_profile_for_model_selects_chat_profiles_and_rejects_unknown(self):
        self.assertEqual(proxy._profile_for_model(proxy.THINK_MODEL_NAME)["port_label"], "think")
        self.assertEqual(proxy._profile_for_model(proxy.NOTHINK_MODEL_NAME)["port_label"], "chat")
        self.assertEqual(proxy._profile_for_model(proxy.CODE_MODEL_NAME)["port_label"], "code")
        self.assertIsNone(proxy._profile_for_model("unknown"))


class UpstreamCaptureTests(unittest.TestCase):
    """The capture writes a whole conversation to disk. It used to do that by
    default, to a fixed world-readable path under /tmp."""

    SNAPSHOT = dict(
        path="/v1/chat/completions",
        kind="chat",
        port_label="code",
        public_model_name="code",
        upstream_host="127.0.0.1",
        upstream_port=8010,
        payload={"model": "code", "messages": [{"role": "user", "content": "hello"}]},
        response_body=b'{"error":"bad request"}',
    )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = pathlib.Path(self._tmp.name) / "upstream-400"

        saved = {name: getattr(proxy, name) for name in (
            "UPSTREAM_400_CAPTURE_ENABLED", "UPSTREAM_400_CAPTURE_DIR",
            "UPSTREAM_400_CAPTURE_KEEP", "UPSTREAM_400_CAPTURE_MAX_BYTES")}
        self.addCleanup(lambda: [setattr(proxy, k, v) for k, v in saved.items()])
        proxy.UPSTREAM_400_CAPTURE_ENABLED = True
        proxy.UPSTREAM_400_CAPTURE_DIR = str(self.dir)

    def _captures(self):
        return sorted(self.dir.glob("upstream-400-*.json")) if self.dir.is_dir() else []

    def test_capture_is_off_unless_asked_for(self):
        proxy.UPSTREAM_400_CAPTURE_ENABLED = False
        proxy._capture_upstream_400(**self.SNAPSHOT)
        self.assertFalse(self.dir.exists())

    def test_capture_writes_a_snapshot_when_enabled(self):
        proxy._capture_upstream_400(**self.SNAPSHOT)
        files = self._captures()
        self.assertEqual(len(files), 1)
        snapshot = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(snapshot["path"], "/v1/chat/completions")
        self.assertEqual(snapshot["kind"], "chat")
        self.assertEqual(snapshot["port_label"], "code")
        self.assertEqual(snapshot["public_model_name"], "code")
        self.assertEqual(snapshot["upstream"], "127.0.0.1:8010")
        self.assertEqual(snapshot["payload"]["model"], "code")
        self.assertEqual(snapshot["response_body_text"], '{"error":"bad request"}')

    def test_the_conversation_is_never_world_readable(self):
        proxy._capture_upstream_400(**self.SNAPSHOT)
        self.assertEqual(self._captures()[0].stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.dir.stat().st_mode & 0o777, 0o700)

    def test_captures_rotate_rather_than_accumulate(self):
        proxy.UPSTREAM_400_CAPTURE_KEEP = 2
        for index in range(5):
            proxy._capture_upstream_400(**dict(self.SNAPSHOT, path=f"/v1/{index}"))
        files = self._captures()
        self.assertEqual(len(files), 2)
        # The newest survive, so the most recent failure is the one on disk.
        kept = {json.loads(f.read_text(encoding="utf-8"))["path"] for f in files}
        self.assertEqual(kept, {"/v1/3", "/v1/4"})

    def test_an_oversized_payload_is_capped_not_written_whole(self):
        proxy.UPSTREAM_400_CAPTURE_MAX_BYTES = 4096
        huge = dict(self.SNAPSHOT, payload={"model": "code", "messages": [
            {"role": "user", "content": "x" * 50000}]})
        proxy._capture_upstream_400(**huge)
        files = self._captures()
        self.assertLessEqual(files[0].stat().st_size, 4096)
        snapshot = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertIn("__truncated__", snapshot["payload"])
        # Capped, but it still names the request that failed.
        self.assertEqual(snapshot["path"], "/v1/chat/completions")

    def test_no_directory_means_no_capture_rather_than_a_crash(self):
        proxy.UPSTREAM_400_CAPTURE_DIR = ""
        proxy._capture_upstream_400(**self.SNAPSHOT)
        self.assertFalse(self.dir.exists())


class ResponseWritingTests(unittest.TestCase):
    """A client that has hung up is the normal end of a cancelled generation.
    Only the body write was protected, so the 503 path's header write raised
    BrokenPipeError into the handler thread and filled the journal."""

    class DeadSocket:
        """Every write fails, as a socket to a departed client does."""

        def write(self, _data):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")

    class Handler:
        def __init__(self, wfile):
            self.wfile = wfile
            self.sent = []

        def send_response(self, status):
            self.sent.append(("status", status))
            self.wfile.write(b"HTTP/1.0 %d\r\n" % status)

        def send_header(self, key, value):
            self.sent.append(("header", key, value))
            self.wfile.write(f"{key}: {value}\r\n".encode())

        def end_headers(self):
            self.wfile.write(b"\r\n")

    def test_a_broken_pipe_in_the_headers_does_not_raise(self):
        handler = self.Handler(self.DeadSocket())
        proxy._send_response_safely(
            handler, 503, [("Content-Type", "application/json")], b"{}")
        # It got as far as trying, then gave up quietly.
        self.assertEqual(handler.sent, [("status", 503)])

    def test_a_live_client_receives_the_whole_response(self):
        written = []

        class LiveSocket:
            def write(self, data):
                written.append(data)

            def flush(self):
                pass

        handler = self.Handler(LiveSocket())
        proxy._send_response_safely(handler, 200, [
            ("Content-Type", "application/json"), ("Content-Length", "2")], b"{}")
        self.assertEqual(b"".join(written),
                         b"HTTP/1.0 200\r\nContent-Type: application/json\r\n"
                         b"Content-Length: 2\r\n\r\n{}")


class SlotSchedulingPassthroughTests(unittest.TestCase):
    """`id_slot` has to survive the proxy, and nothing said so.

    pi-forge pins interactive turns to slot 0 and background work to slot 1 by
    putting `id_slot` in the request body. The proxy never mentions the field —
    it works only because every mutation edits the parsed payload in place and
    unknown keys are re-serialised untouched. That is a property, not a
    decision, and a future rewrite that rebuilt the payload from known fields
    would break cooperative scheduling silently: requests would still succeed,
    just on whichever slot llama.cpp picked.
    """

    SCHEDULING_FIELDS = {"id_slot": 1, "cache_prompt": True}

    def _round_trip(self, payload: dict) -> dict:
        raw = json.dumps(payload).encode("utf-8")
        parsed = proxy._safe_json_loads(raw)
        proxy._inject_thinking(parsed, True, True)
        proxy._strip_tool_fields(parsed)
        proxy._inject_overrides(parsed, proxy.CODE_OVERRIDES)
        proxy._inject_max_tokens(parsed, "chat", 4096)
        return json.loads(proxy._body_from_json(parsed, raw))

    def test_the_scheduling_fields_reach_the_backend_unchanged(self):
        sent = self._round_trip({
            "model": "code",
            "messages": [{"role": "user", "content": "hi"}],
            **self.SCHEDULING_FIELDS,
        })
        self.assertEqual(sent["id_slot"], 1)
        self.assertIs(sent["cache_prompt"], True)

    def test_slot_zero_is_not_dropped_as_falsy(self):
        # The interactive slot is 0, so any `if payload.get("id_slot")` guard
        # anywhere in this path would silently unpin every interactive turn.
        sent = self._round_trip({
            "model": "chat",
            "messages": [{"role": "user", "content": "hi"}],
            "id_slot": 0,
        })
        self.assertIn("id_slot", sent)
        self.assertEqual(sent["id_slot"], 0)

    def test_tool_stripping_does_not_take_the_slot_with_it(self):
        sent = self._round_trip({
            "model": "code",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "noop"}}],
            **self.SCHEDULING_FIELDS,
        })
        self.assertNotIn("tools", sent)
        self.assertEqual(sent["id_slot"], 1)

    def test_memory_injection_leaves_the_slot_alone(self):
        payload = {"model": "chat", "messages": [{"role": "user", "content": "hi"}],
                   **self.SCHEDULING_FIELDS}
        proxy._inject_memory(payload, "chat", "remember this")
        self.assertEqual(payload["id_slot"], 1)


class ListenerTests(unittest.TestCase):
    def test_the_accept_queue_is_deeper_than_the_default_five(self):
        """Measured: 300 rapid connections to :8008 with a backlog of 5 gave a
        p90 connect of 1022ms — a dropped SYN and a full retransmit timer."""
        self.assertGreaterEqual(proxy.ProxyHTTPServer.request_queue_size, 128)


if __name__ == "__main__":
    unittest.main()
