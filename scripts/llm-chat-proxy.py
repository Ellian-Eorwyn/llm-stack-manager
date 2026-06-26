#!/usr/bin/env python3
"""
llm-chat-proxy.py

OpenAI-compatible multi-port proxy in front of a shared llama-server backend.

Public ports:
  THINK_PORT   (default 8003)
  NOTHINK_PORT (default 8004)
  CODE_PORT    (default 8008)

Backend:
  CHAT_BACKEND_HOST:CHAT_BACKEND_PORT (default 127.0.0.1:8010)

This proxy can transparently enrich chat-completion requests with Graphiti
memory and asynchronously ingest completed conversation turns back into Graphiti,
without requiring tool calls or client integration changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
import socket

BACKEND_HOST = os.environ.get("CHAT_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("CHAT_BACKEND_PORT", "8010"))
THINK_PORT = int(os.environ.get("THINK_PORT", "8003"))
NOTHINK_PORT = int(os.environ.get("NOTHINK_PORT", "8004"))
CODE_PORT = int(os.environ.get("CODE_PORT", "8008"))
LISTEN_HOST = os.environ.get("LISTEN_HOST", "0.0.0.0")
EMBED_BACKEND_HOST = os.environ.get("EMBED_BACKEND_HOST", "127.0.0.1")
EMBED_PORT = int(os.environ.get("EMBED_PORT", "8005"))
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "embed")


def _parse_timeout_env(name: str, default: str, *, allow_disable: bool = False) -> float | None:
    raw = os.environ.get(name, default).strip()
    if allow_disable and raw.lower() in {"0", "off", "none", "false"}:
        return None
    value = float(raw)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {raw!r}")
    return value


BACKEND_CONNECT_TIMEOUT_SEC = _parse_timeout_env("BACKEND_CONNECT_TIMEOUT_SEC", "10")
BACKEND_READ_TIMEOUT_SEC = _parse_timeout_env(
    "BACKEND_READ_TIMEOUT_SEC",
    "off",
    allow_disable=True,
)
UPSTREAM_400_CAPTURE_PATH = os.environ.get(
    "UPSTREAM_400_CAPTURE_PATH",
    "/tmp/openclaw-llamacpp-last-payload.json",
)

THINK_MODEL_NAME = os.environ.get("THINK_MODEL_NAME", "think")
NOTHINK_MODEL_NAME = os.environ.get("NOTHINK_MODEL_NAME", "chat")
CODE_MODEL_NAME = os.environ.get("CODE_MODEL_NAME", "code")

THINK_PRESERVE_THINKING = os.environ.get("THINK_PRESERVE_THINKING", "on").lower() == "on"
THINK_JINJA = os.environ.get("THINK_JINJA", "on").lower() == "on"
THINK_OVERRIDES = {
    "temperature": float(os.environ.get("THINK_TEMP", os.environ.get("CHAT_TEMP", "0.7"))),
    "top_p": float(os.environ.get("THINK_TOP_P", os.environ.get("CHAT_TOP_P", "0.95"))),
    "top_k": int(os.environ.get("THINK_TOP_K", os.environ.get("CHAT_TOP_K", "20"))),
    "min_p": float(os.environ.get("THINK_MIN_P", os.environ.get("CHAT_MIN_P", "0.00"))),
    "presence_penalty": float(os.environ.get("THINK_PRESENCE_PENALTY", "0.00")),
    "repeat_penalty": float(os.environ.get("THINK_REPEAT_PENALTY", "1.00")),
    "reasoning_format": os.environ.get(
        "THINK_REASONING_FORMAT",
        os.environ.get("CHAT_REASONING_FORMAT", "deepseek"),
    ),
}
THINK_MAX_TOKENS = int(os.environ.get("THINK_MAX_TOKENS", "0"))
THINK_REASONING_STREAM_MODE = os.environ.get("THINK_REASONING_STREAM_MODE", "hidden").strip().lower()
NOTHINK_PRESERVE_THINKING = os.environ.get("NOTHINK_PRESERVE_THINKING", "off").lower() == "on"
NOTHINK_JINJA = os.environ.get("NOTHINK_JINJA", "on").lower() == "on"
NOTHINK_OVERRIDES = {
    "temperature": float(os.environ.get("NOTHINK_TEMP", os.environ.get("CHAT_TEMP", "0.7"))),
    "top_p": float(os.environ.get("NOTHINK_TOP_P", os.environ.get("CHAT_TOP_P", "0.95"))),
    "top_k": int(os.environ.get("NOTHINK_TOP_K", os.environ.get("CHAT_TOP_K", "20"))),
    "min_p": float(os.environ.get("NOTHINK_MIN_P", os.environ.get("CHAT_MIN_P", "0.00"))),
    "presence_penalty": float(os.environ.get("NOTHINK_PRESENCE_PENALTY", "0.00")),
    "repeat_penalty": float(os.environ.get("NOTHINK_REPEAT_PENALTY", "1.00")),
    "reasoning_format": os.environ.get(
        "NOTHINK_REASONING_FORMAT",
        os.environ.get("CHAT_REASONING_FORMAT", "deepseek"),
    ),
}
NOTHINK_MAX_TOKENS = int(os.environ.get("NOTHINK_MAX_TOKENS", "0"))
NOTHINK_REASONING_STREAM_MODE = os.environ.get("NOTHINK_REASONING_STREAM_MODE", "hidden").strip().lower()
CODE_THINKING = os.environ.get("CODE_THINKING", "on").lower() == "on"
CODE_PRESERVE_THINKING = os.environ.get("CODE_PRESERVE_THINKING", "on").lower() == "on"
CODE_JINJA = os.environ.get("CODE_JINJA", "on").lower() == "on"
CODE_OVERRIDES = {
    "temperature": float(os.environ.get("CODE_TEMP", os.environ.get("CHAT_TEMP", "0.7"))),
    "top_p": float(os.environ.get("CODE_TOP_P", os.environ.get("CHAT_TOP_P", "0.95"))),
    "top_k": int(os.environ.get("CODE_TOP_K", os.environ.get("CHAT_TOP_K", "20"))),
    "min_p": float(os.environ.get("CODE_MIN_P", os.environ.get("CHAT_MIN_P", "0.00"))),
    "presence_penalty": float(os.environ.get("CODE_PRESENCE_PENALTY", "0.00")),
    "repeat_penalty": float(os.environ.get("CODE_REPEAT_PENALTY", "1.00")),
    "reasoning_format": os.environ.get(
        "CODE_REASONING_FORMAT",
        os.environ.get("CHAT_REASONING_FORMAT", "deepseek"),
    ),
}
CODE_MAX_TOKENS = int(os.environ.get("CODE_MAX_TOKENS", "0"))
CODE_REASONING_STREAM_MODE = os.environ.get("CODE_REASONING_STREAM_MODE", "hidden").strip().lower()

# Graphiti memory gateway settings
MEMORY_GATEWAY_ENABLED = os.environ.get("MEMORY_GATEWAY_ENABLED", "on").lower() == "on"
MEMORY_ENABLE_THINK = os.environ.get("MEMORY_ENABLE_THINK", "on").lower() == "on"
MEMORY_ENABLE_NOTHINK = os.environ.get("MEMORY_ENABLE_NOTHINK", "on").lower() == "on"
MEMORY_ENABLE_CODE = os.environ.get("MEMORY_ENABLE_CODE", "off").lower() == "on"
MEMORY_INJECTION_MODE = os.environ.get("MEMORY_INJECTION_MODE", "developer").strip().lower()
MEMORY_MAX_FACTS = int(os.environ.get("MEMORY_MAX_FACTS", "8"))
MEMORY_MAX_FACT_CHARS = int(os.environ.get("MEMORY_MAX_FACT_CHARS", "220"))
MEMORY_MAX_BLOCK_CHARS = int(os.environ.get("MEMORY_MAX_BLOCK_CHARS", "1800"))
MEMORY_MAX_QUERY_MESSAGES = int(os.environ.get("MEMORY_MAX_QUERY_MESSAGES", "6"))
MEMORY_MAX_INGEST_CHARS = int(os.environ.get("MEMORY_MAX_INGEST_CHARS", "4000"))
MEMORY_INCLUDE_SYSTEM_IN_QUERY = os.environ.get("MEMORY_INCLUDE_SYSTEM_IN_QUERY", "off").lower() == "on"
MEMORY_GROUP_HEADER_PRIORITY = [
    h.strip().lower()
    for h in os.environ.get(
        "MEMORY_GROUP_HEADER_PRIORITY",
        "x-conversation-id,x-session-id,x-chat-id,x-openwebui-chat-id",
    ).split(",")
    if h.strip()
]
MEMORY_GROUP_FALLBACK_SALT = os.environ.get("MEMORY_GROUP_FALLBACK_SALT", "llm-stack")
MEMORY_FAIL_OPEN = os.environ.get("MEMORY_FAIL_OPEN", "on").lower() == "on"

GRAPHITI_BASE_URL = os.environ.get(
    "MEMORY_GRAPHITI_BASE_URL",
    os.environ.get("GRAPHITI_PUBLIC_URL")
    or f"http://127.0.0.1:{os.environ.get('GRAPHITI_PORT', '8070')}",
).rstrip("/")
GRAPHITI_GET_MEMORY_URL = f"{GRAPHITI_BASE_URL}/get-memory"
GRAPHITI_MESSAGES_URL = f"{GRAPHITI_BASE_URL}/messages"
GRAPHITI_TIMEOUT_SEC = float(os.environ.get("MEMORY_GRAPHITI_TIMEOUT_SEC", "1.2"))
GRAPHITI_COOLDOWN_SEC = float(os.environ.get("MEMORY_GRAPHITI_COOLDOWN_SEC", "30"))
_GRAPHITI_DISABLED_UNTIL = 0.0
_GRAPHITI_STATE_LOCK = threading.Lock()


def _log(msg: str):
    print(f"[llm-chat-proxy] {msg}", flush=True)


def _write_response_safely(handler: BaseHTTPRequestHandler, body: bytes):
    try:
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        return


def _filtered_upstream_headers(headers: Any) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in {
            "host",
            "connection",
            "content-length",
            "transfer-encoding",
            "expect",
            "keep-alive",
            "proxy-connection",
            "upgrade",
        }:
            continue
        filtered[key] = value
    return filtered


def _graphiti_is_suspended() -> bool:
    with _GRAPHITI_STATE_LOCK:
        return _GRAPHITI_DISABLED_UNTIL > time.time()


def _graphiti_suspend(reason: str):
    if GRAPHITI_COOLDOWN_SEC <= 0:
        return
    global _GRAPHITI_DISABLED_UNTIL
    now = time.time()
    until = now + GRAPHITI_COOLDOWN_SEC
    should_log = False
    with _GRAPHITI_STATE_LOCK:
        if _GRAPHITI_DISABLED_UNTIL <= now:
            should_log = True
        if until > _GRAPHITI_DISABLED_UNTIL:
            _GRAPHITI_DISABLED_UNTIL = until
    if should_log:
        _log(
            "graphiti-suspend "
            f"cooldown_sec={GRAPHITI_COOLDOWN_SEC:.1f} reason={reason}"
        )


def _graphiti_resume():
    global _GRAPHITI_DISABLED_UNTIL
    with _GRAPHITI_STATE_LOCK:
        _GRAPHITI_DISABLED_UNTIL = 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json_loads(body: bytes) -> dict[str, Any] | None:
    try:
        parsed = json.loads(body)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _body_from_json(data: dict[str, Any], fallback: bytes) -> bytes:
    try:
        return json.dumps(data, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    except Exception:
        return fallback


def _capture_upstream_400(
    *,
    path: str,
    kind: str | None,
    port_label: str,
    public_model_name: str,
    upstream_host: str,
    upstream_port: int,
    payload: dict[str, Any] | None,
    response_body: bytes,
):
    if payload is None:
        return
    try:
        response_text = response_body.decode("utf-8", errors="replace")
    except Exception:
        response_text = repr(response_body[:1024])
    snapshot = {
        "captured_at": _now_iso(),
        "path": path,
        "kind": kind,
        "port_label": port_label,
        "public_model_name": public_model_name,
        "upstream": f"{upstream_host}:{upstream_port}",
        "payload": payload,
        "response_body_text": response_text[:16000],
    }
    tmp_path = f"{UPSTREAM_400_CAPTURE_PATH}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, ensure_ascii=True, indent=2)
            fh.write("\n")
        os.replace(tmp_path, UPSTREAM_400_CAPTURE_PATH)
        _log(f"captured upstream 400 payload -> {UPSTREAM_400_CAPTURE_PATH}")
    except OSError as exc:
        _log(f"failed to capture upstream 400 payload: {exc}")


def _messages_as_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    return [m for m in messages if isinstance(m, dict)]


def _normalized_path(path: str) -> str:
    parsed = urllib.parse.urlsplit(path or "")
    normalized = parsed.path.rstrip("/")
    return normalized or "/"


def _request_kind(path: str) -> str | None:
    normalized = _normalized_path(path)
    if normalized in ("/v1/models", "/models"):
        return "models"
    if normalized.startswith("/v1/models/"):
        return "model"
    if normalized.endswith("/chat/completions"):
        return "chat"
    if normalized.endswith("/responses"):
        return "responses"
    if normalized.endswith("/embeddings"):
        return "embeddings"
    return None


def _is_generation_kind(kind: str | None) -> bool:
    return kind in {"chat", "responses"}


def _message_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = _message_text_content(msg.get("content"))
            if text:
                return text
    return ""


def _responses_input_to_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_input = payload.get("input")
    if isinstance(raw_input, str):
        text = raw_input.strip()
        return [{"role": "user", "content": text}] if text else []
    if not isinstance(raw_input, list):
        return []
    messages: list[dict[str, Any]] = []
    for item in raw_input:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"system", "developer", "user", "assistant"}:
            continue
        messages.append(item)
    return messages


def _payload_messages(payload: dict[str, Any], kind: str | None) -> list[dict[str, Any]]:
    if kind == "chat":
        return _messages_as_list(payload)
    if kind == "responses":
        return _responses_input_to_messages(payload)
    return []


def _content_parts_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"text", "input_text", "output_text", "reasoning_text"} and isinstance(item.get("text"), str):
            parts.append(item["text"])
            continue
        if item_type == "output_text" and isinstance(item.get("content"), str):
            parts.append(item["content"])
    return "\n".join(part for part in parts if part).strip()


def _serialize_tool_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        return str(output)


def _normalize_responses_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    item_type = item.get("type")
    if item_type == "message" and item.get("role") in {"system", "developer", "user", "assistant"}:
        content = _content_parts_to_text(item.get("content"))
        return {"role": item["role"], "content": content or ""}

    if item_type == "function_call_output":
        normalized = {
            "role": "tool",
            "content": _serialize_tool_output(item.get("output")),
        }
        call_id = item.get("call_id")
        if isinstance(call_id, str) and call_id.strip():
            normalized["tool_call_id"] = call_id.strip()
        return normalized

    role = item.get("role")
    if role in {"system", "developer", "user", "assistant", "tool"}:
        normalized = dict(item)
        if "content" in normalized:
            text_content = _content_parts_to_text(normalized.get("content"))
            if text_content:
                normalized["content"] = text_content
        return normalized

    return dict(item)


def _prepend_developer_instruction_to_responses_input(payload: dict[str, Any], instructions: str):
    raw_input = payload.get("input")
    instruction_item = {"role": "developer", "content": instructions}
    if isinstance(raw_input, str):
        payload["input"] = [instruction_item, {"role": "user", "content": raw_input}]
        return
    if isinstance(raw_input, list):
        payload["input"] = [instruction_item, *raw_input]
        return
    payload["input"] = [instruction_item]


def _normalize_responses_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    raw_input = normalized.get("input")
    if isinstance(raw_input, list):
        normalized_items: list[dict[str, Any]] = []
        for item in raw_input:
            normalized_item = _normalize_responses_item(item)
            if normalized_item is not None:
                normalized_items.append(normalized_item)
        normalized["input"] = normalized_items

    instructions = normalized.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        _prepend_developer_instruction_to_responses_input(normalized, instructions.strip())
        normalized.pop("instructions", None)

    return normalized


def _trim_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars < 4:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def _should_include_memory(port_label: str, is_generation_request: bool) -> bool:
    if not MEMORY_GATEWAY_ENABLED or not is_generation_request:
        return False
    if port_label == "think":
        return MEMORY_ENABLE_THINK
    if port_label == "chat":
        return MEMORY_ENABLE_NOTHINK
    if port_label == "code":
        return MEMORY_ENABLE_CODE
    return False


def _header_lookup(headers: Any, key: str) -> str | None:
    value = headers.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _first_nonempty(values: list[str | None]) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _derive_group_id(payload: dict[str, Any], headers: Any, client_ip: str, model_name: str) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

    header_value = None
    for header_name in MEMORY_GROUP_HEADER_PRIORITY:
        header_value = _header_lookup(headers, header_name)
        if header_value:
            break

    explicit = _first_nonempty(
        [
            header_value,
            payload.get("conversation_id") if isinstance(payload.get("conversation_id"), str) else None,
            payload.get("session_id") if isinstance(payload.get("session_id"), str) else None,
            payload.get("chat_id") if isinstance(payload.get("chat_id"), str) else None,
            metadata.get("conversation_id") if isinstance(metadata.get("conversation_id"), str) else None,
            metadata.get("session_id") if isinstance(metadata.get("session_id"), str) else None,
            payload.get("user") if isinstance(payload.get("user"), str) else None,
        ]
    )
    if explicit:
        normalized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in explicit)
        return f"conv_{normalized[:128]}"

    seed = f"{MEMORY_GROUP_FALLBACK_SALT}|{client_ip}|{model_name}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
    return f"fallback_{digest}"


def _graphiti_query_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role")
        if role not in ("system", "user", "assistant"):
            continue
        if role == "system" and not MEMORY_INCLUDE_SYSTEM_IN_QUERY:
            continue
        text = _message_text_content(msg.get("content"))
        if not text:
            continue
        selected.append(
            {
                "role_type": role,
                "role": role,
                "content": _trim_text(text, 1200),
                "name": "",
                "source_description": "chat-completion-query",
            }
        )
    if MEMORY_MAX_QUERY_MESSAGES > 0:
        selected = selected[-MEMORY_MAX_QUERY_MESSAGES:]
    return selected


def _retrieve_memory(group_id: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if _graphiti_is_suspended():
        return []
    query_messages = _graphiti_query_messages(messages)
    if not query_messages:
        return []

    payload = {
        "group_id": group_id,
        "max_facts": max(1, MEMORY_MAX_FACTS),
        "center_node_uuid": None,
        "messages": query_messages,
    }
    request = urllib.request.Request(
        GRAPHITI_GET_MEMORY_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=GRAPHITI_TIMEOUT_SEC) as response:
            body = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        _graphiti_suspend(type(exc).__name__)
        raise
    parsed = _safe_json_loads(body)
    _graphiti_resume()
    if not parsed:
        return []
    facts = parsed.get("facts")
    if not isinstance(facts, list):
        return []
    out: list[dict[str, Any]] = []
    for fact in facts:
        if isinstance(fact, dict) and isinstance(fact.get("fact"), str):
            out.append(fact)
    return out


def _build_memory_block(facts: list[dict[str, Any]]) -> str:
    lines = [
        "Memory context from prior conversation (use only when relevant,",
        "and do not claim certainty beyond these notes):",
    ]
    remaining = MEMORY_MAX_BLOCK_CHARS
    count = 0
    for fact in facts:
        fact_text = _trim_text(fact.get("fact", ""), MEMORY_MAX_FACT_CHARS).strip()
        if not fact_text:
            continue
        line = f"- {fact_text}"
        needed = len(line) + 1
        if remaining - needed < 0:
            break
        lines.append(line)
        remaining -= needed
        count += 1
        if MEMORY_MAX_FACTS > 0 and count >= MEMORY_MAX_FACTS:
            break
    if count == 0:
        return ""
    lines.append("If memory conflicts with the current user message, prioritize the current message.")
    block = "\n".join(lines)
    return _trim_text(block, MEMORY_MAX_BLOCK_CHARS)


def _prepend_to_content(content: Any, prefix: str) -> Any:
    if isinstance(content, str):
        return f"{prefix}\n\n{content}".strip()
    if isinstance(content, list):
        return [{"type": "text", "text": prefix}] + content
    return prefix


def _prepend_to_input_text(text: str, prefix: str) -> str:
    return f"{prefix}\n\n{text}".strip()


def _memory_injection_role() -> str:
    return "developer" if MEMORY_INJECTION_MODE == "developer" else "system"


def _inject_memory_into_messages(messages: list[dict[str, Any]], memory_block: str) -> bool:
    if not messages or not memory_block:
        return False

    if MEMORY_INJECTION_MODE == "prefix_user":
        for msg in reversed(messages):
            if msg.get("role") == "user":
                msg["content"] = _prepend_to_content(msg.get("content"), f"[MEMORY CONTEXT]\n{memory_block}")
                return True
        return False

    role = _memory_injection_role()
    memory_prefix = f"[MEMORY CONTEXT]\n{memory_block}"
    if messages and messages[0].get("role") == role:
        messages[0]["content"] = _prepend_to_content(messages[0].get("content"), memory_prefix)
    else:
        messages.insert(0, {"role": role, "content": memory_prefix})
    return True


def _inject_memory(payload: dict[str, Any], kind: str | None, memory_block: str) -> bool:
    if not memory_block:
        return False

    if kind == "chat":
        messages = _messages_as_list(payload)
        injected = _inject_memory_into_messages(messages, memory_block)
        if injected:
            payload["messages"] = messages
        return injected

    if kind == "responses":
        raw_input = payload.get("input")
        if isinstance(raw_input, str):
            if MEMORY_INJECTION_MODE == "prefix_user":
                payload["input"] = _prepend_to_input_text(raw_input, f"[MEMORY CONTEXT]\n{memory_block}")
            else:
                payload["input"] = [
                    {"role": _memory_injection_role(), "content": f"[MEMORY CONTEXT]\n{memory_block}"},
                    {"role": "user", "content": raw_input},
                ]
            return True

        messages = _responses_input_to_messages(payload)
        injected = _inject_memory_into_messages(messages, memory_block)
        if injected:
            payload["input"] = messages
        return injected

    return False


def _extract_assistant_text_from_response_output_items(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    for item in items:
        if not isinstance(item, dict) or item.get("type") != "message" or item.get("role") != "assistant":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "\n".join(parts).strip()
    return ""


def _extract_assistant_text_from_nonstream_response(raw: bytes, kind: str | None) -> str:
    parsed = _safe_json_loads(raw)
    if not parsed:
        return ""
    if kind == "responses":
        return _extract_assistant_text_from_response_output_items(parsed.get("output"))
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    return _message_text_content(message.get("content"))


def _rewrite_json_response_model(raw: bytes, public_model_name: str) -> bytes:
    parsed = _safe_json_loads(raw)
    if not parsed:
        return raw
    parsed["model"] = public_model_name
    return _body_from_json(parsed, raw)


def _model_obj(model_name: str) -> dict[str, Any]:
    return {
        "id": model_name,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "llamacpp",
    }


def _requested_model_name(payload: dict[str, Any], fallback: str) -> str:
    model = payload.get("model")
    return model if isinstance(model, str) and model.strip() else fallback


def _prepare_embedding_payload(payload: dict[str, Any], fallback_model: str) -> tuple[dict[str, Any], str]:
    response_model = _requested_model_name(payload, fallback_model)
    updated = dict(payload)
    updated["model"] = EMBED_MODEL_NAME
    return updated, response_model


class SSEModelRewriter:
    """Rewrite top-level model ids inside SSE data events without changing framing."""

    def __init__(self, public_model_name: str):
        self._public_model_name = public_model_name
        self._buffer = ""

    def feed(self, chunk: bytes) -> bytes:
        try:
            text = chunk.decode("utf-8", errors="ignore")
        except Exception:
            return chunk
        self._buffer += text
        out: list[str] = []
        while "\n\n" in self._buffer:
            event, self._buffer = self._buffer.split("\n\n", 1)
            out.append(self._rewrite_event(event))
        return ("\n\n".join(out) + ("\n\n" if out else "")).encode("utf-8")

    def flush(self) -> bytes:
        if not self._buffer:
            return b""
        event = self._rewrite_event(self._buffer)
        self._buffer = ""
        return event.encode("utf-8")

    def _rewrite_event(self, event: str) -> str:
        rewritten_lines: list[str] = []
        for line in event.splitlines():
            stripped = line.strip()
            if not stripped.startswith("data:"):
                rewritten_lines.append(line)
                continue
            data = stripped[5:].strip()
            if not data or data == "[DONE]":
                rewritten_lines.append(line)
                continue
            try:
                obj = json.loads(data)
            except Exception:
                rewritten_lines.append(line)
                continue
            if isinstance(obj, dict):
                obj["model"] = self._public_model_name
                rewritten_lines.append(f"data: {json.dumps(obj, ensure_ascii=True, separators=(',', ':'))}")
            else:
                rewritten_lines.append(line)
        return "\n".join(rewritten_lines)


def _normalize_reasoning_stream_mode(mode: str) -> str:
    mode = (mode or "hidden").strip().lower()
    return mode if mode in {"hidden", "mirror", "content"} else "hidden"


def _rewrite_reasoning_delta(delta: dict[str, Any], mode: str):
    reasoning = delta.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning:
        return
    if mode in {"mirror", "content"} and not delta.get("content"):
        delta["content"] = reasoning
    if mode == "content":
        delta.pop("reasoning_content", None)


def _rewrite_json_reasoning_visibility(raw: bytes, mode: str) -> bytes:
    mode = _normalize_reasoning_stream_mode(mode)
    if mode == "hidden":
        return raw
    parsed = _safe_json_loads(raw)
    if not parsed:
        return raw
    choices = parsed.get("choices")
    if not isinstance(choices, list):
        return raw
    changed = False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str) or not reasoning:
            continue
        if mode in {"mirror", "content"} and not message.get("content"):
            message["content"] = reasoning
            changed = True
        if mode == "content":
            message.pop("reasoning_content", None)
            changed = True
    return _body_from_json(parsed, raw) if changed else raw


class SSEEventRewriter:
    """Rewrite streamed model ids and optionally expose reasoning deltas as content."""

    def __init__(self, public_model_name: str, reasoning_mode: str):
        self._public_model_name = public_model_name
        self._reasoning_mode = _normalize_reasoning_stream_mode(reasoning_mode)
        self._buffer = ""

    def feed(self, chunk: bytes) -> bytes:
        try:
            text = chunk.decode("utf-8", errors="ignore")
        except Exception:
            return chunk
        self._buffer += text
        out: list[str] = []
        while "\n\n" in self._buffer:
            event, self._buffer = self._buffer.split("\n\n", 1)
            out.append(self._rewrite_event(event))
        return ("\n\n".join(out) + ("\n\n" if out else "")).encode("utf-8")

    def flush(self) -> bytes:
        if not self._buffer:
            return b""
        event = self._rewrite_event(self._buffer)
        self._buffer = ""
        return event.encode("utf-8")

    def _rewrite_event(self, event: str) -> str:
        rewritten_lines: list[str] = []
        for line in event.splitlines():
            stripped = line.strip()
            if not stripped.startswith("data:"):
                rewritten_lines.append(line)
                continue
            data = stripped[5:].strip()
            if not data or data == "[DONE]":
                rewritten_lines.append(line)
                continue
            try:
                obj = json.loads(data)
            except Exception:
                rewritten_lines.append(line)
                continue
            if not isinstance(obj, dict):
                rewritten_lines.append(line)
                continue
            obj["model"] = self._public_model_name
            choices = obj.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    if isinstance(delta, dict):
                        _rewrite_reasoning_delta(delta, self._reasoning_mode)
            rewritten_lines.append(f"data: {json.dumps(obj, ensure_ascii=True, separators=(',', ':'))}")
        return "\n".join(rewritten_lines)


class SSEAssistantAccumulator:
    """Best-effort extraction of assistant text from OpenAI-style SSE chat chunks."""

    def __init__(self):
        self._buffer = ""
        self._parts: list[str] = []

    def feed(self, chunk: bytes):
        try:
            text = chunk.decode("utf-8", errors="ignore")
        except Exception:
            return
        self._buffer += text
        while "\n\n" in self._buffer:
            event, self._buffer = self._buffer.split("\n\n", 1)
            self._consume_event(event)

    def _consume_event(self, event: str):
        for line in event.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except Exception:
                continue
            choices = obj.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    if isinstance(delta, dict):
                        content = delta.get("content")
                        if isinstance(content, str):
                            self._parts.append(content)
            if obj.get("type") == "response.output_text.delta" and isinstance(obj.get("delta"), str):
                self._parts.append(obj["delta"])
                continue
            if obj.get("type") == "response.output_text.done" and isinstance(obj.get("text"), str):
                self._parts.append(obj["text"])

    def get_text(self) -> str:
        return "".join(self._parts).strip()


def _parse_status_and_headers(raw_header: bytes) -> tuple[int, dict[str, str]]:
    try:
        text = raw_header.decode("iso-8859-1", errors="replace")
    except Exception:
        return 0, {}
    lines = text.split("\r\n")
    status_code = 0
    if lines:
        parts = lines[0].split(" ", 2)
        if len(parts) >= 2 and parts[1].isdigit():
            status_code = int(parts[1])
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    return status_code, headers


def _send_graphiti_ingest(group_id: str, user_text: str, assistant_text: str):
    if _graphiti_is_suspended():
        return
    messages = []
    if user_text:
        messages.append(
            {
                "name": "",
                "content": _trim_text(user_text, MEMORY_MAX_INGEST_CHARS),
                "role_type": "user",
                "role": "user",
                "timestamp": _now_iso(),
                "source_description": "llm-chat-proxy",
            }
        )
    if assistant_text:
        messages.append(
            {
                "name": "",
                "content": _trim_text(assistant_text, MEMORY_MAX_INGEST_CHARS),
                "role_type": "assistant",
                "role": "assistant",
                "timestamp": _now_iso(),
                "source_description": "llm-chat-proxy",
            }
        )
    if not messages:
        return

    payload = {"group_id": group_id, "messages": messages}
    request = urllib.request.Request(
        GRAPHITI_MESSAGES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=GRAPHITI_TIMEOUT_SEC):
            _graphiti_resume()
            return
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        _graphiti_suspend(type(exc).__name__)
        raise


def _enqueue_ingest(group_id: str, user_text: str, assistant_text: str):
    def _task():
        try:
            _send_graphiti_ingest(group_id, user_text, assistant_text)
            _log(f"memory-ingest group={group_id} status=queued")
        except Exception as exc:
            _log(f"memory-ingest group={group_id} status=error detail={exc}")

    thread = threading.Thread(target=_task, daemon=True)
    thread.start()


def _inject_thinking(payload: dict[str, Any], enabled: bool, preserve_thinking: bool | None = None):
    kwargs = payload.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
    kwargs["enable_thinking"] = enabled
    if preserve_thinking is not None:
        kwargs["preserve_thinking"] = preserve_thinking
    payload["chat_template_kwargs"] = kwargs


def _strip_tool_fields(payload: dict[str, Any]):
    payload.pop("tools", None)
    payload.pop("tool_choice", None)


def _inject_overrides(payload: dict[str, Any], overrides: dict[str, Any]):
    for key, value in overrides.items():
        payload[key] = value


def _inject_max_tokens(payload: dict[str, Any], kind: str | None, max_tokens: int):
    if max_tokens <= 0:
        return
    if kind == "responses":
        payload["max_output_tokens"] = max_tokens
        return
    if "max_completion_tokens" in payload and "max_tokens" not in payload:
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens


def _requested_model_id_from_path(path: str) -> str:
    normalized = _normalized_path(path)
    prefix = "/v1/models/"
    if not normalized.startswith(prefix):
        return ""
    return urllib.parse.unquote(normalized[len(prefix) :]).strip()


class ProxyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def get_request(self):  # noqa: ANN001
        request, client_address = super().get_request()
        try:
            request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        return request, client_address


def make_handler(
    *,
    thinking_enabled: bool | None,
    preserve_thinking: bool | None = None,
    model_name: str,
    port_label: str,
    inject_overrides: dict[str, Any] | None = None,
    max_tokens: int = 0,
    strip_tools: bool = False,
    reasoning_stream_mode: str = "hidden",
):
    class ProxyHandler(BaseHTTPRequestHandler):
        _thinking_enabled = thinking_enabled
        _model_name = model_name
        _port_label = port_label
        _overrides = inject_overrides
        _max_tokens = max_tokens
        _strip_tools = strip_tools
        _reasoning_stream_mode = _normalize_reasoning_stream_mode(reasoning_stream_mode)

        def do_GET(self):
            kind = _request_kind(self.path)
            if kind == "models":
                self._serve_models()
            elif kind == "model":
                self._serve_model()
            else:
                self._proxy_raw("GET", b"")

        def do_POST(self):
            self._proxy_raw("POST", self._read_body())

        def do_DELETE(self):
            self._proxy_raw("DELETE", b"")

        def _serve_models(self):
            body = json.dumps(
                {
                    "object": "list",
                    "data": [_model_obj(self._model_name)],
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_model(self):
            requested_id = _requested_model_id_from_path(self.path)
            if requested_id != self._model_name:
                self._gateway_error(404, "model_not_found", f"Model '{requested_id}' was not found")
                return
            body = json.dumps(_model_obj(self._model_name)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> bytes:
            n = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(n) if n > 0 else b""

        def _proxy_raw(self, method: str, body: bytes):
            kind = _request_kind(self.path)
            is_generation_request = method == "POST" and _is_generation_kind(kind)
            is_embeddings_request = method == "POST" and kind == "embeddings"

            payload = _safe_json_loads(body) if (is_generation_request or is_embeddings_request) else None
            request_user_text = ""
            group_id = ""
            memory_injected = False
            memory_facts = 0
            rewrite_response_model = ""
            upstream_host = BACKEND_HOST
            upstream_port = BACKEND_PORT

            if payload is not None:
                if is_embeddings_request:
                    payload, rewrite_response_model = _prepare_embedding_payload(payload, self._model_name)
                    upstream_host = EMBED_BACKEND_HOST
                    upstream_port = EMBED_PORT
                elif kind == "responses":
                    payload = _normalize_responses_payload(payload)
                if self._thinking_enabled is not None and is_generation_request:
                    _inject_thinking(payload, self._thinking_enabled, preserve_thinking)
                if self._strip_tools and is_generation_request:
                    _strip_tool_fields(payload)
                if self._overrides and is_generation_request:
                    _inject_overrides(payload, self._overrides)
                if is_generation_request:
                    _inject_max_tokens(payload, kind, self._max_tokens)

                request_messages = _payload_messages(payload, kind)
                request_user_text = _extract_last_user_text(request_messages)

                if _should_include_memory(self._port_label, is_generation_request):
                    group_id = _derive_group_id(payload, self.headers, self.client_address[0], self._model_name)
                    if _graphiti_is_suspended():
                        _log(
                            f"memory-retrieve port={self._port_label} group={group_id} "
                            "status=skipped detail=graphiti_suspended"
                        )
                    else:
                        try:
                            facts = _retrieve_memory(group_id, request_messages)
                            memory_facts = len(facts)
                            memory_block = _build_memory_block(facts)
                            if memory_block:
                                memory_injected = _inject_memory(payload, kind, memory_block)
                            _log(
                                f"memory-retrieve port={self._port_label} group={group_id} "
                                f"facts={memory_facts} injected={memory_injected}"
                            )
                        except (urllib.error.URLError, TimeoutError, OSError) as exc:
                            _log(
                                f"memory-retrieve port={self._port_label} group={group_id or 'n/a'} "
                                f"status=error detail={exc}"
                            )
                            if not MEMORY_FAIL_OPEN:
                                self._gateway_error(503, "graphiti_unavailable", "Graphiti retrieval failed")
                                return
                        except Exception as exc:
                            _log(
                                f"memory-retrieve port={self._port_label} group={group_id or 'n/a'} "
                                f"status=error detail={exc}"
                            )
                            if not MEMORY_FAIL_OPEN:
                                self._gateway_error(500, "memory_gateway_error", "Memory gateway failed")
                                return

                body = _body_from_json(payload, body)

            headers = _filtered_upstream_headers(self.headers)
            headers["Content-Length"] = str(len(body))
            headers["Connection"] = "close"
            headers["Host"] = f"{upstream_host}:{upstream_port}"

            assistant_text = ""
            response_streaming = False
            backend_phase = "connect"

            try:
                request_line = f"{method} {self.path} HTTP/1.1\r\n"
                header_block = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
                raw_request = (request_line + header_block + "\r\n").encode("utf-8") + body

                with socket.create_connection(
                    (upstream_host, upstream_port),
                    timeout=BACKEND_CONNECT_TIMEOUT_SEC,
                ) as sock:
                    backend_phase = "read"
                    try:
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    except OSError:
                        pass
                    sock.settimeout(BACKEND_READ_TIMEOUT_SEC)
                    sock.sendall(raw_request)

                    seen_header = False
                    header_buf = bytearray()
                    status_code = 0
                    resp_headers: dict[str, str] = {}
                    acc = SSEAssistantAccumulator()
                    nonstream_buf = bytearray()

                    while True:
                        chunk = sock.recv(65536)
                        if not chunk:
                            break

                        out_chunk = chunk
                        body_part = b""

                        if not seen_header:
                            header_buf.extend(chunk)
                            idx = header_buf.find(b"\r\n\r\n")
                            if idx == -1:
                                continue
                            seen_header = True
                            raw_head = bytes(header_buf[:idx])
                            body_part = bytes(header_buf[idx + 4 :])
                            status_code, resp_headers = _parse_status_and_headers(raw_head)
                            response_streaming = (
                                "text/event-stream" in resp_headers.get("content-type", "").lower()
                            )
                            if response_streaming:
                                out_chunk = bytes(header_buf)
                            else:
                                out_chunk = b""
                        else:
                            body_part = chunk

                        if (is_generation_request or is_embeddings_request) and body_part:
                            if response_streaming:
                                acc.feed(body_part)
                            else:
                                nonstream_buf.extend(body_part)

                        if not response_streaming:
                            continue

                        try:
                            self.wfile.write(out_chunk)
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            break

                    if not response_streaming:
                        rewritten_body = _rewrite_json_reasoning_visibility(bytes(nonstream_buf), self._reasoning_stream_mode)
                        if status_code == 400 and is_generation_request:
                            _capture_upstream_400(
                                path=self.path,
                                kind=kind,
                                port_label=self._port_label,
                                public_model_name=self._model_name,
                                upstream_host=upstream_host,
                                upstream_port=upstream_port,
                                payload=payload,
                                response_body=rewritten_body,
                            )
                        if rewrite_response_model:
                            rewritten_body = _rewrite_json_response_model(rewritten_body, rewrite_response_model)
                        elif is_generation_request:
                            rewritten_body = _rewrite_json_response_model(rewritten_body, self._model_name)

                        self.send_response(status_code or 200)
                        for key, value in resp_headers.items():
                            if key in {"content-length", "connection", "transfer-encoding"}:
                                continue
                            self.send_header(key, value)
                        self.send_header("Content-Length", str(len(rewritten_body)))
                        self.end_headers()
                        _write_response_safely(self, rewritten_body)

                    if is_generation_request:
                        if response_streaming:
                            assistant_text = acc.get_text()
                        else:
                            assistant_text = _extract_assistant_text_from_nonstream_response(bytes(nonstream_buf), kind)

                    if is_generation_request and group_id and status_code and status_code < 500:
                        _enqueue_ingest(group_id, request_user_text, assistant_text)
            except ConnectionRefusedError:
                self._backend_unavailable(upstream_host, upstream_port)
            except TimeoutError as exc:
                if backend_phase == "connect":
                    self._backend_unavailable(upstream_host, upstream_port, f"Timed out connecting to backend: {exc}")
                else:
                    self._gateway_error(
                        504,
                        "backend_read_timeout",
                        (
                            f"Backend llama-server timed out while streaming a response from "
                            f"{upstream_host}:{upstream_port}. {exc}"
                        ),
                    )
            except OSError as exc:
                self._backend_unavailable(upstream_host, upstream_port, str(exc))

        def _gateway_error(self, status_code: int, err_type: str, message: str):
            msg = json.dumps({"error": {"message": message, "type": err_type}}).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            _write_response_safely(self, msg)

        def _backend_unavailable(self, host: str, port: int, detail: str = ""):
            msg = json.dumps(
                {
                    "error": {
                        "message": (
                            f"Backend llama-server is not available on {host}:{port}. "
                            + (detail or "Is the target backend service running?")
                        ),
                        "type": "backend_unavailable",
                    }
                }
            ).encode("utf-8")
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            _write_response_safely(self, msg)

        def log_message(self, fmt, *args):  # noqa: ANN001
            return

    return ProxyHandler


def serve(port: int, handler_class, label: str):
    server = ProxyHTTPServer((LISTEN_HOST, port), handler_class)
    _log(f"{label} listening on {LISTEN_HOST}:{port} -> backend {BACKEND_HOST}:{BACKEND_PORT}")
    server.serve_forever()


if __name__ == "__main__":
    think_handler = make_handler(
        thinking_enabled=True,
        preserve_thinking=THINK_PRESERVE_THINKING,
        model_name=THINK_MODEL_NAME,
        port_label="think",
        inject_overrides=THINK_OVERRIDES,
        max_tokens=THINK_MAX_TOKENS,
        strip_tools=not THINK_JINJA,
        reasoning_stream_mode=THINK_REASONING_STREAM_MODE,
    )
    nothink_handler = make_handler(
        thinking_enabled=False,
        preserve_thinking=NOTHINK_PRESERVE_THINKING,
        model_name=NOTHINK_MODEL_NAME,
        port_label="chat",
        inject_overrides=NOTHINK_OVERRIDES,
        max_tokens=NOTHINK_MAX_TOKENS,
        strip_tools=not NOTHINK_JINJA,
        reasoning_stream_mode=NOTHINK_REASONING_STREAM_MODE,
    )
    code_handler = make_handler(
        thinking_enabled=CODE_THINKING,
        preserve_thinking=CODE_PRESERVE_THINKING,
        model_name=CODE_MODEL_NAME,
        port_label="code",
        inject_overrides=CODE_OVERRIDES,
        max_tokens=CODE_MAX_TOKENS,
        strip_tools=not CODE_JINJA,
        reasoning_stream_mode=CODE_REASONING_STREAM_MODE,
    )

    think_thread = threading.Thread(
        target=serve,
        args=(THINK_PORT, think_handler, f"thinking (enable_thinking=true, preserve_thinking={THINK_PRESERVE_THINKING}, reasoning_stream={THINK_REASONING_STREAM_MODE} + overrides: {THINK_OVERRIDES} + optional memory gateway)"),
        daemon=True,
    )
    nothink_thread = threading.Thread(
        target=serve,
        args=(NOTHINK_PORT, nothink_handler, f"instruct (enable_thinking=false, preserve_thinking={NOTHINK_PRESERVE_THINKING}, reasoning_stream={NOTHINK_REASONING_STREAM_MODE} + overrides: {NOTHINK_OVERRIDES} + optional memory gateway)"),
        daemon=True,
    )
    code_thread = threading.Thread(
        target=serve,
        args=(CODE_PORT, code_handler, f"code (inject enable_thinking={CODE_THINKING}, preserve_thinking={CODE_PRESERVE_THINKING}, reasoning_stream={CODE_REASONING_STREAM_MODE} + overrides: {CODE_OVERRIDES} + optional memory gateway)"),
        daemon=True,
    )

    think_thread.start()
    nothink_thread.start()
    code_thread.start()

    _log(
        "All ports active. "
        f"Memory gateway enabled={MEMORY_GATEWAY_ENABLED} graphiti={GRAPHITI_BASE_URL} "
        f"mode={MEMORY_INJECTION_MODE}"
    )

    try:
        think_thread.join()
        nothink_thread.join()
        code_thread.join()
    except KeyboardInterrupt:
        _log("Shutting down.")
        sys.exit(0)
