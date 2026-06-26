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

    def test_capture_upstream_400_writes_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            capture_path = pathlib.Path(tmpdir) / "payload.json"
            original_path = proxy.UPSTREAM_400_CAPTURE_PATH
            try:
                proxy.UPSTREAM_400_CAPTURE_PATH = str(capture_path)
                proxy._capture_upstream_400(
                    path="/v1/chat/completions",
                    kind="chat",
                    port_label="code",
                    public_model_name="code",
                    upstream_host="127.0.0.1",
                    upstream_port=8010,
                    payload={"model": "code", "messages": [{"role": "user", "content": "hello"}]},
                    response_body=b'{"error":"bad request"}',
                )
            finally:
                proxy.UPSTREAM_400_CAPTURE_PATH = original_path

            snapshot = json.loads(capture_path.read_text(encoding="utf-8"))
            self.assertEqual(snapshot["path"], "/v1/chat/completions")
            self.assertEqual(snapshot["kind"], "chat")
            self.assertEqual(snapshot["port_label"], "code")
            self.assertEqual(snapshot["public_model_name"], "code")
            self.assertEqual(snapshot["upstream"], "127.0.0.1:8010")
            self.assertEqual(snapshot["payload"]["model"], "code")
            self.assertEqual(snapshot["response_body_text"], '{"error":"bad request"}')

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


if __name__ == "__main__":
    unittest.main()
