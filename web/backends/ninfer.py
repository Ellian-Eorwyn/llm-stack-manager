#!/usr/bin/env python3
"""Turning a chat slot into an `ninfer-serve` command line.

NInfer (github.com/ashalliants/ninfer-3090, a fork of Neroued/ninfer for the
RTX 3090) is a CUDA server built for Qwen3.8-27B and Qwen3.6 only. It loads
its own `.ninfer` artifact rather than a GGUF, keeps one model on one card,
and pairs it with MTP or DFlash2 speculation and a paged KV pool
(docs/ninfer.md).

A slot moves to it with `{PREFIX}_ENGINE=ninfer`, or with `auto` and a model
path ending in `.ninfer` -- the file says which engine it wants.

What carries over from the slot: alias (as `--model-id`, the one name the
server accepts), context size (both the per-request ceiling and the KV pool, so
one conversation can use all of it), parallel slots, preserve-thinking. The
card comes from `{PREFIX}_GPU_VISIBLE_DEVICES`, which start-backend.sh exports
as CUDA_VISIBLE_DEVICES; NInfer is then told `--device 0`, the first card it
can see. What does not carry over: llama.cpp placement, cache types, the draft
settings, sampling defaults (the chat proxy sends those per request),
CUSTOM_ARGS_JSON. NInfer's own flags go in `{PREFIX}_NINFER_ARGS_JSON`.

The defaults are the quality-first profile measured on llms: INT8 KV (its
perplexity is bf16's), MTP with three drafts and the draft head, vision loaded
per image rather than held on the card, and none of the fork's lossy
memory trades.
"""

from __future__ import annotations

import os

import platforms

from .llamacpp import custom_args
from .spec import Slot, lookup

#: The chat slots: the ones behind the chat proxy.
SLOTS = ("llm-a", "llm-b")

KV_DTYPES = ("int8", "bf16", "rk8v4", "rk4v4")

#: `--spec` values, and the draft count each takes when the slot names none:
#: three for MTP and seven for DFlash2 are the fork's measured best on the 27B.
SPECS = {"mtp": "3", "dflash2": "7"}

VISION = ("overlay", "resident", "off")

#: `--reasoning-effort` values ninfer-serve takes.
EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")

ARGS_KEYS = ("NINFER_ARGS_JSON",)

#: Where the dependency installer builds it (dependencies.json).
DEFAULT_BIN = "deps/ninfer-3090/build-sm86/apps/ninfer-serve"


def _first(env: dict, keys: tuple[str, ...], prefixes: tuple[str, ...]) -> str:
    return (lookup(env, keys, prefixes) or "").strip()


def binary(env: dict) -> str:
    explicit = str(env.get("NINFER_BIN") or "").strip()
    if explicit:
        return explicit
    return os.path.join(str(env.get("STACK_DIR") or "."), DEFAULT_BIN)


def _positive(value: str) -> int:
    try:
        number = int(float(value))
    except ValueError:
        return 0
    return max(number, 0)


def _device(slot: Slot, env: dict) -> str:
    """The card index NInfer is given. Renumbered space -- the default -- makes
    the slot's visible list the whole choice, so it is always 0; with absolute
    indices nothing is renumbered and the slot's first listed card is named."""
    if str(env.get("LLM_ABSOLUTE_GPU_INDICES") or "off").strip() != "on":
        return "0"
    visible = _first(env, ("GPU_VISIBLE_DEVICES",), slot.prefixes)
    first = visible.split(",")[0].strip()
    return first if first.isdigit() else "0"


def build(slot: Slot, env: dict, extra: list[str] | None = None) -> list[str]:
    if platforms.active().name != "linux":
        raise SystemExit(f"{slot.name}: the ninfer engine needs Linux and an NVIDIA RTX 3090")
    if slot.name not in SLOTS:
        raise SystemExit(f"{slot.name}: the ninfer engine serves the chat slots only "
                         f"({', '.join(SLOTS)})")
    prefixes = slot.prefixes
    model = _first(env, slot.model_keys, prefixes)
    if not model.endswith(".ninfer"):
        raise SystemExit(
            f"{slot.name}: the ninfer engine loads a .ninfer artifact in {slot.prefix}_MODEL_PATH, "
            f"not a GGUF (got {model!r}); docs/ninfer.md says which to download")
    host = _first(env, slot.host_keys, prefixes) or "127.0.0.1"
    port = _first(env, slot.port_keys, prefixes) or slot.port_default
    alias = _first(env, slot.alias_keys, prefixes) or slot.alias_default

    argv = [binary(env), model, "--host", host, "--port", port, "--model-id", alias,
            "--device", _device(slot, env)]

    # One request may use the whole context, so the shared pool is as large as
    # the ceiling unless the slot sizes it separately; `auto` fills the card.
    ctx = _positive(_first(env, slot.key_chains.get("CTX_SIZE", ("CTX_SIZE",)), prefixes))
    if ctx:
        argv += ["--max-context", str(ctx)]
    capacity = _first(env, ("NINFER_KV_CAPACITY",), prefixes).lower()
    if capacity == "auto" or _positive(capacity):
        argv += ["--kv-capacity", capacity if capacity == "auto" else str(_positive(capacity))]
    elif ctx:
        argv += ["--kv-capacity", str(ctx)]

    lanes = _positive(_first(env, ("N_PARALLEL",), prefixes)) or 1
    argv += ["--max-concurrency", str(lanes)]

    kv = _first(env, ("NINFER_KV_DTYPE",), prefixes) or "int8"
    argv += ["--kv-dtype", kv if kv in KV_DTYPES else "int8"]

    spec = _first(env, ("NINFER_SPEC",), prefixes) or "mtp"
    if spec in SPECS:
        drafts = _positive(_first(env, ("NINFER_DRAFT_TOKENS",), prefixes)) or int(SPECS[spec])
        argv += ["--spec", spec, "--draft-tokens", str(drafts), "--lm-head-draft"]

    # Overlay keeps the vision tower in host memory and borrows the card per
    # image, which is what leaves the full context for text.
    vision = _first(env, ("NINFER_VISION",), prefixes) or "overlay"
    if vision in ("overlay", "resident"):
        argv += ["--vision", "--vision-residency", vision]

    if (_first(env, ("PRESERVE_THINKING",), prefixes) or "on") == "on":
        argv.append("--preserve-thinking")

    # The slot's thinking level is the default for a request that names none;
    # the chat proxy names one per endpoint. `none` is left to the requests,
    # because `--no-thinking` would also refuse a request that asks to think.
    effort = (lookup(env, ("REASONING_EFFORT",), prefixes, empty_is_set=True) or "").strip()
    if effort in EFFORTS:
        argv += ["--reasoning-effort", effort]

    # Opt-in: 1.6-1.8x prefill for +0.156% perplexity.
    if _first(env, ("NINFER_PREFILL_CUBLAS",), prefixes) == "on":
        argv.append("--prefill-cublas")

    # Retained conversations: several rotating agent sessions stay cached
    # (8 private continuations, 8 shared prefixes), spilling to pinned host RAM.
    host_kv = _positive(_first(env, ("NINFER_HOST_KV_MIB",), prefixes) or "4096")
    argv += ["--max-private-continuations", "8", "--max-shared-prefixes", "8",
             "--auto-prefix-grid", "--host-kv-mib", str(host_kv)]

    argv += custom_args(env, ARGS_KEYS, prefixes)
    argv += list(extra or [])
    return argv
