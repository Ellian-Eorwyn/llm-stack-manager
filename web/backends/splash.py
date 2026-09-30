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

What carries over from the slot: alias, context size, reasoning level, a memory
cap (below). Vision is on unless `{PREFIX}_SPLASH_VISION=off`; Splash
fetches the model repo's `mmproj` itself. What does not: llama.cpp placement and cache types, the
draft settings (Splash picks its own drafter), CUSTOM_ARGS_JSON.

Two differences from the other engines a client has to know about. Splash
turns thinking off with `reasoning_effort: "none"`, and ignores the
`chat_template_kwargs.enable_thinking` switch the others take -- so the slot's
default is passed as `--default-reasoning-effort`, and the chat proxy sends
`reasoning_effort` per request. And its KV cache defaults to int8, which is
not lossless; this passes bf16 unless the slot says otherwise.

Memory is always capped. Splash's own limit (`--max-memory auto`) is Metal's
recommended working set -- 85% of RAM, about 78 GiB on the 96 GB Studio -- and
it keeps the KV and recurrent state of every finished conversation until that
limit forces an eviction. One conversation at a time never gets there; a day of
Hermes sessions does, and 78 GiB of model on top of macOS and the other
services swaps the Mac to a standstill. So the slot's
`{PREFIX}_SPLASH_MAX_MEMORY_GB` is passed, else the SSD offload RAM budget,
else half of RAM (48 GiB on 96): the 27B's ~33 GiB of weights and buffers and
~15 GiB of KV, one ~230K-token bf16 conversation.

Why not more: every page of KV Splash keeps is locked into RAM while it
serves a request and given back after it, and giving it back stalls every
app's new windows and tabs (docs/splash.md#stalls; the chat proxy keeps it
locked while the stack is in use). Conversations that no longer fit go
to the SSD instead (`--max-cache-disk`, `{PREFIX}_SPLASH_MAX_CACHE_DISK_GB`,
40 GiB unless set; 0 turns it off), so an older session still resumes without
reading its whole prompt again. `auto` hands the memory choice back to Splash.
"""

from __future__ import annotations

import os
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

#: Share of RAM Splash may hold when the slot names no cap.
DEFAULT_MEMORY_FRACTION = 0.5

#: SSD quota, in GiB, for conversations evicted from memory.
DEFAULT_CACHE_DISK_GB = 40


def _first(env: dict, keys: tuple[str, ...], prefixes: tuple[str, ...]) -> str:
    return (lookup(env, keys, prefixes) or "").strip()


def binary(env: dict) -> str:
    return str(env.get("SPLASH_BIN") or shutil.which("splash") or "/opt/homebrew/bin/splash")


def physical_gib() -> float:
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3


def max_memory(slot: Slot, env: dict) -> str | None:
    """`--max-memory` for the slot, or None to leave Splash on its own `auto`."""
    value = _first(env, ("SPLASH_MAX_MEMORY_GB",), slot.prefixes).lower()
    if value == "auto":
        return None
    try:
        gib = int(float(value))
    except ValueError:
        gib = 0
    if gib <= 0:
        memory = offload.settings(slot, env)
        if memory.active and memory.ram_budget_gb:
            gib = int(float(memory.ram_budget_gb))
    if gib <= 0:
        gib = int(physical_gib() * DEFAULT_MEMORY_FRACTION)
    return f"{gib}G"


def max_cache_disk(slot: Slot, env: dict) -> str | None:
    """`--max-cache-disk` for the slot, or None for Splash's default: none."""
    value = _first(env, ("SPLASH_MAX_CACHE_DISK_GB",), slot.prefixes).lower()
    if value in ("off", "0"):
        return None
    try:
        gib = int(float(value)) if value else DEFAULT_CACHE_DISK_GB
    except ValueError:
        gib = DEFAULT_CACHE_DISK_GB
    return f"{gib}G" if gib > 0 else None


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

    # On unless turned off: Qwen3.8-27B sees images and PDFs, and a client that
    # was told the model does (Hermes's supports_vision) would otherwise get a
    # 400 for every image. Off saves the ~1 GB vision tower.
    vision = _first(env, ("SPLASH_VISION",), prefixes) or "on"
    if vision == "off":
        argv.append("--language-only")

    cap = max_memory(slot, env)
    if cap:
        argv += ["--max-memory", cap]
    disk = max_cache_disk(slot, env)
    if disk:
        argv += ["--max-cache-disk", disk]

    argv += custom_args(env, ARGS_KEYS, prefixes)
    argv += list(extra or [])
    return argv
