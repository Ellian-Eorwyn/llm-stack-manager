#!/usr/bin/env python3
"""Turning a chat slot into a `splash serve` command line.

Splash (github.com/incoai/splash) is an Apple-silicon server built for two
models only -- Qwen3.8-27B and Qwen3.6-35B-A3B -- with hand-written Metal
kernels and a trained DFlash2 drafter it pairs with the model by itself. On the
Studio it serves the 27B at 8-bit to 200k tokens in about 48 GiB and keeps
~110 tok/s on code at depth, where MTPLX's Quality pack drops to ~60
(docs/model-evals.md).

A slot moves to it with `{PREFIX}_ENGINE=splash`; `auto` never chooses it,
because its model is a Hugging Face reference (`owner/repo:VARIANT`, e.g.
`unsloth/Qwen3.8-27B-GGUF:Q8_0`), not a file whose shape says which engine it
wants. It downloads into the ordinary Hugging Face cache on first start.

What carries over from the slot: alias, context size, reasoning level, the SSD
offload RAM budget. What does not: llama.cpp placement and cache types, the
draft settings (Splash picks its own drafter), CUSTOM_ARGS_JSON.

Two differences from the other engines a client has to know about. Splash
turns thinking off with `reasoning_effort: "none"`, and ignores the
`chat_template_kwargs.enable_thinking` switch the others take -- so the slot's
default is passed as `--default-reasoning-effort`, and the chat proxy sends
`reasoning_effort` per request. And its KV cache defaults to int8, which is
not lossless; this passes bf16 unless the slot says otherwise.
"""

from __future__ import annotations

import shutil

import platforms

from . import offload
from .llamacpp import custom_args
from .spec import Slot, lookup

#: The chat slots, as for MTPLX: the ones behind the chat proxy.
SLOTS = ("llm-a", "llm-b")

#: Splash takes an API key off loopback, and the chat proxy sends none.
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}

#: `--default-reasoning-effort` values Splash accepts.
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

KV_FORMATS = ("bf16", "int8")

#: Splash's own ceiling for `--max-context`.
MAX_CONTEXT = 256 * 1024

ARGS_KEYS = ("SPLASH_ARGS_JSON",)


def _first(env: dict, keys: tuple[str, ...], prefixes: tuple[str, ...]) -> str:
    return (lookup(env, keys, prefixes) or "").strip()


def binary(env: dict) -> str:
    return str(env.get("SPLASH_BIN") or shutil.which("splash") or "/opt/homebrew/bin/splash")


def build(slot: Slot, env: dict, extra: list[str] | None = None) -> list[str]:
    if platforms.active().name != "darwin":
        raise SystemExit(f"{slot.name}: the splash engine is Apple silicon only")
    if slot.name not in SLOTS:
        raise SystemExit(f"{slot.name}: the splash engine serves the chat slots only "
                         f"({', '.join(SLOTS)})")
    prefixes = slot.prefixes
    model = _first(env, slot.model_keys, prefixes)
    if not model or "/" not in model or model.startswith("/"):
        raise SystemExit(
            f"{slot.name}: the splash engine takes a Hugging Face model reference in "
            f"{slot.prefix}_MODEL_PATH, e.g. unsloth/Qwen3.8-27B-GGUF:Q8_0 (got {model!r})")
    host = _first(env, slot.host_keys, prefixes) or "127.0.0.1"
    if host not in _LOOPBACK:
        raise SystemExit(f"{slot.name}: the splash engine binds loopback only here; the chat "
                         f"proxy sends no API key")
    port = _first(env, slot.port_keys, prefixes) or slot.port_default
    alias = _first(env, slot.alias_keys, prefixes) or slot.alias_default

    argv = [binary(env), "serve", "--model", model, "--host", host, "--port", port,
            "--served-model-name", alias, "--no-webui"]

    revision = _first(env, ("SPLASH_REVISION",), prefixes)
    if revision:
        argv += ["--revision", revision]

    kv = _first(env, ("SPLASH_KV_FORMAT",), prefixes) or "bf16"
    argv += ["--kv-format", kv if kv in KV_FORMATS else "bf16"]

    ctx = _first(env, slot.key_chains.get("CTX_SIZE", ("CTX_SIZE",)), prefixes)
    if ctx.isdigit() and int(ctx) > 0:
        argv += ["--max-context", f"{min(int(ctx), MAX_CONTEXT) // 1024}K"]

    # The slot's thinking level becomes Splash's default for requests that do
    # not say; the chat proxy says, per endpoint.
    effort = (lookup(env, ("REASONING_EFFORT",), prefixes, empty_is_set=True) or "").strip()
    if effort in EFFORTS:
        argv += ["--default-reasoning-effort", effort]

    vision = _first(env, ("SPLASH_VISION",), prefixes) or "off"
    if vision != "on":
        argv.append("--language-only")

    memory = offload.settings(slot, env)
    if memory.active and memory.ram_budget_gb:
        argv += ["--max-memory", f"{int(float(memory.ram_budget_gb))}G"]

    argv += custom_args(env, ARGS_KEYS, prefixes)
    argv += list(extra or [])
    return argv
