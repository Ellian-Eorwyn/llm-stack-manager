#!/usr/bin/env python3
"""Turning a chat slot into an `mtplx serve` command line.

MTPLX (github.com/youssofal/mtplx) is an Apple-silicon server that uses a
model's own multi-token-prediction head as its speculative drafter, in MLX. It
is the same idea as llama.cpp's `draft-mtp`, but in a runtime built for the
hardware, and it serves a *directory* -- an MTPLX pack of safetensors with an
`mtplx_runtime.json` -- rather than a GGUF.

It runs from its own venv, `deps/mtplx-venv`, for the reason the MLX runtime
has its own: its pins (mlx 0.32, mlx-lm 0.31, transformers < 5.15) disagree
with mlx-audio's, and a shared tree would let one upgrade break the other.

The slot's existing settings are read where they mean the same thing, so a
slot moves to this engine by changing `{PREFIX}_ENGINE` and pointing the model
path at a pack. The llama.cpp knobs with no counterpart -- placement, KV cache
types, batch sizes, the draft settings, `--mmproj` -- are not read: MTPLX sizes
its own KV, drafts from the pack's own head, and a pack that sees images
carries its vision tower inside it.
"""

from __future__ import annotations

import platforms

from .llamacpp import custom_args
from .options import REASONING_EFFORTS
from .slots import COMMON_FLAGS
from .spec import MTPLX_PACK_MARKER, Slot, is_mtplx_pack, lookup

#: The slots this engine can serve: the two that hold a conversation behind
#: the chat proxy.
SLOTS = ("llm-a", "llm-b")

#: MTPLX refuses a keyless bind anywhere but loopback, and the chat proxy sends
#: no key, so a wider bind would be a backend nothing in the stack can reach.
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}

#: The operator's own MTPLX arguments. Not `CUSTOM_ARGS_JSON`: that one holds
#: llama-server flags, and MTPLX would refuse to start on them.
ARGS_KEYS = ("MTPLX_ARGS_JSON",)

#: `--paged-kv-quantization` widths. Its own setting, `{PREFIX}_MTPLX_KV_QUANT`,
#: not the GGUF slot's CACHE_TYPE_K/V: those cost llama.cpp little, while
#: MTPLX 2.12 has a fast path for a quantized cache only at short context.
KV_QUANT_MODES = ("q8", "q4")


def _first(env: dict, keys: tuple[str, ...], prefixes: tuple[str, ...]) -> str:
    return lookup(env, keys, prefixes) or ""


def build(slot: Slot, env: dict, extra: list[str] | None = None) -> list[str]:
    """The argv, which is `env ... mtplx serve ...`.

    Refuses rather than guesses, as command construction does: a GGUF path left
    behind by switching the engine alone would otherwise surface as an MTPLX
    traceback in the log instead of the reason.
    """
    if platforms.active().name != "darwin":
        # Said here and not only by the installer: a config copied from a Mac
        # onto a Linux host would otherwise restart-loop on a missing venv.
        raise SystemExit(
            f"{slot.name}: the mtplx engine is Apple silicon only; point "
            f"{slot.prefix}_MODEL_PATH at a GGUF, or set "
            f"{slot.prefix}_ENGINE=llamacpp on this host")
    if slot.name not in SLOTS:
        raise SystemExit(
            f"{slot.name}: the mtplx engine serves the chat slots only "
            f"({', '.join(SLOTS)})")

    prefixes = slot.prefixes
    model = _first(env, slot.model_keys, prefixes)
    if not model:
        raise SystemExit(f"{slot.name}: the mtplx engine needs {slot.prefix}_MODEL_PATH")
    if not is_mtplx_pack(model):
        raise SystemExit(
            f"{slot.name}: {model} is not an MTPLX pack (no {MTPLX_PACK_MARKER}). "
            f"The mtplx engine serves a pack directory, not a GGUF; point "
            f"{slot.prefix}_MODEL_PATH at one, or set {slot.prefix}_ENGINE=auto "
            f"to serve whichever the model is.")

    host = _first(env, slot.host_keys, prefixes) or "127.0.0.1"
    if host not in _LOOPBACK:
        raise SystemExit(
            f"{slot.name}: the mtplx engine binds loopback only, and "
            f"{slot.host_keys[0].lstrip('!')} is {host}. MTPLX requires an API key "
            f"on any other address, which the chat proxy does not send.")
    port = _first(env, slot.port_keys, prefixes) or slot.port_default
    alias = _first(env, slot.alias_keys, prefixes) or slot.alias_default

    stack = str(env.get("STACK_DIR") or ".")
    venv = str(env.get("MTPLX_VENV") or f"{stack}/deps/mtplx-venv")

    # The warm-conversation cache is MTPLX's counterpart of --cache-ram, and
    # left to itself it takes half the memory the weights leave -- about 37 GB
    # beside a 27B on a 96 GB Mac -- which is not its to take on a machine
    # serving other models. Same setting, same units (MiB).
    #
    # Unbuffered because launchd hands it a file, not a terminal, and a
    # block-buffered log shows nothing of a load that is failing.
    argv = ["/usr/bin/env", "PYTHONUNBUFFERED=1"]
    cache_ram = _first(env, ("CACHE_RAM",), prefixes)
    if cache_ram.isdigit() and int(cache_ram) > 0:
        argv.append(f"MTPLX_SESSION_BANK_MAX_BYTES={cache_ram}M")

    argv += [f"{venv}/bin/mtplx", "serve",
             "--model", model,
             "--model-id", alias,
             "--host", host,
             "--port", port,
             # Loopback needs no key, and a key saved in ~/.mtplx would
             # otherwise be demanded of the proxy, which has none to give.
             "--no-auth",
             # Without it MTPLX appends a tokens-per-second footer to the text
             # of every reply.
             "--no-stats-footer"]

    ctx_keys = slot.key_chains.get("CTX_SIZE", ("CTX_SIZE",))
    ctx_size = _first(env, ctx_keys, prefixes)
    if ctx_size.isdigit() and int(ctx_size) > 0:
        argv += ["--context-window", ctx_size]

    # With the defaults llama-server would have been given, so an unset key
    # samples the same way under either engine.
    defaults = {flag.keys[0]: flag.default for flag in COMMON_FLAGS}
    defaults.update(slot.defaults)
    for flag, suffix in (("--default-temperature", "TEMP"),
                         ("--default-top-p", "TOP_P"),
                         ("--default-top-k", "TOP_K")):
        value = _first(env, (suffix,), prefixes) or defaults.get(suffix) or ""
        if value:
            argv += [flag, value]

    effort = (lookup(env, ("REASONING_EFFORT",), prefixes, empty_is_set=True) or "").strip()
    if effort in REASONING_EFFORTS:
        argv += ["--reasoning-effort", effort]
    # The llama.cpp engine asks the template to keep reasoning when this is on
    # and says nothing when it is off; "auto" is MTPLX's saying nothing.
    thinking = (_first(env, ("PRESERVE_THINKING",), prefixes) or "on").strip()
    if thinking == "on":
        argv += ["--preserve-thinking", "on"]

    # Off unless asked for. q8 halves the cache (~16 GiB to ~8 at 262k), but on
    # the Studio it took decode at ~100k context from 67.9 to 22.2 tok/s: past
    # a context threshold MTPLX 2.12 verifies a quantized cache on a slow,
    # uncompiled path. Carrying CACHE_TYPE_K/V over would have made every long
    # session a third of the speed the engine exists for, without a word.
    kv_quant = _first(env, ("MTPLX_KV_QUANT",), prefixes).strip()
    if kv_quant in KV_QUANT_MODES:
        argv += ["--paged-kv-quantization", kv_quant]

    argv += custom_args(env, ARGS_KEYS, prefixes)
    argv += list(extra or [])
    return argv
