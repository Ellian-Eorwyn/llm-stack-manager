#!/usr/bin/env python3
"""SSD offload: serving a model bigger than the RAM it is allowed to use.

One engine-neutral setting set per slot, which each engine turns into its own
arguments. `{PREFIX}_MEMORY_MODE` is the switch. At `resident`, the default,
nothing here emits anything, so every command line is the one it was before
this existed. At `ssd-offload` the model's weights may live partly on the SSD:

  RAM_BUDGET_GB             the most the backend should hold resident
  NGRAM_PREWARM             how much of an n-gram table (Qwen 3.8 Flash-Next's
                            51B-parameter sidecar) to pre-read into the page
                            cache: auto, off, all, or a GiB count
  EXPERT_CACHE_GB           the expert-slot cache, on a build that streams MoE
                            experts from the SSD; blank lets the engine choose
  EXPERT_STREAM_IO_THREADS  that build's reader threads

What each engine can honour differs. MTPLX keeps every expert resident and
streams only the n-gram table; mainline llama.cpp can leave weights mmapped
and pageable; a build with `--moe-stream` pages experts through a cache. So a
setting an engine has no counterpart for is ignored, not refused: the same
slot is meant to move between a GGUF and a pack by choosing the model alone,
and an expert-cache size chosen for one must not stop the other starting.

`{PREFIX}_LLAMA_SERVER_BIN` sits here too, although it is not an offload
setting, because offload is what needs it: expert streaming only exists in
builds other than the pinned one, and it has to serve one slot without
becoming the binary every other slot runs.
"""

from __future__ import annotations

import functools
import subprocess
from dataclasses import dataclass

from .spec import Slot, lookup

RESIDENT = "resident"
SSD_OFFLOAD = "ssd-offload"
MEMORY_MODES = (RESIDENT, SSD_OFFLOAD)

#: The n-gram table's tensor in a Flash-Next GGUF, as an `--override-tensor`
#: pattern: llama.cpp stores it as the per-layer embedding, 26.8 GiB of the
#: 95.5 in the streaming build's checkpoint (read off the file, 2026-09-28).
#: Kept in host memory on an offloaded slot, where the mmap leaves it
#: pageable; in a Metal buffer it would be wired whole. A GGUF without one
#: matches nothing, and llama.cpp ignores a pattern that matches nothing.
NGRAM_TENSOR_PATTERN = "per_layer_token_embd"

#: Reader threads for `--moe-stream-io-threads` when none is set: the value
#: the Flash-Next streaming build was measured with on a 64 GB Mac.
DEFAULT_IO_THREADS = "8"


@dataclass(frozen=True)
class Offload:
    mode: str = RESIDENT
    ram_budget_gb: str = ""
    ngram_prewarm: str = ""
    expert_cache_gb: str = ""
    io_threads: str = ""

    @property
    def active(self) -> bool:
        return self.mode == SSD_OFFLOAD


def _number(value: str) -> str:
    """A positive number as written, or "" -- a typo must not reach a flag."""
    value = (value or "").strip()
    try:
        return value if float(value) > 0 else ""
    except ValueError:
        return ""


def settings(slot: Slot, env: dict) -> Offload:
    prefixes = slot.prefixes

    def read(suffix: str) -> str:
        return (lookup(env, (suffix,), prefixes) or "").strip()

    mode = read("MEMORY_MODE") or RESIDENT
    if mode not in MEMORY_MODES:
        mode = RESIDENT
    prewarm = read("NGRAM_PREWARM").lower()
    if prewarm not in ("", "auto", "off", "all"):
        prewarm = _number(prewarm)
    return Offload(
        mode=mode,
        ram_budget_gb=_number(read("RAM_BUDGET_GB")),
        ngram_prewarm=prewarm,
        expert_cache_gb=_number(read("EXPERT_CACHE_GB")),
        io_threads=_number(read("EXPERT_STREAM_IO_THREADS")),
    )


def server_bin(slot: Slot, env: dict) -> str:
    """The llama-server this slot runs: its own, else the stack's."""
    own = (lookup(env, ("LLAMA_SERVER_BIN",), slot.prefixes) or "").strip()
    return own or str(env.get("LLAMA_SERVER_BIN") or "llama-server")


@functools.lru_cache(maxsize=16)
def _help_text(binary: str) -> str:
    try:
        done = subprocess.run([binary, "--help"], capture_output=True, text=True,
                              timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (done.stdout or "") + (done.stderr or "")


def server_supports(binary: str, flag: str) -> bool:
    """Whether `binary --help` lists `flag`.

    Asked rather than assumed because the builds differ: the pinned one has
    `--load-mode` and no `--moe-stream`, an older fork may have neither, and
    passing a flag a build does not know fails it at start, as a restart loop.
    Unknown -- a binary that will not run -- answers no.
    """
    return flag in _help_text(binary)


def llamacpp_args(offload: Offload, binary: str, custom_has) -> list[str]:
    """What an offloaded llama.cpp slot adds where `--no-mmap`/`--mlock` were.

    mmap without mlock, so the pages macOS finds cold can go back to the SSD;
    the n-gram table kept out of the Metal buffers, which are wired; and, on a
    build that streams experts, the stream and its cache.
    """
    args: list[str] = []
    if server_supports(binary, "--load-mode"):
        args += ["--load-mode", "mmap"]
    if not custom_has("-ot", "--override-tensor"):
        args += ["--override-tensor", f"{NGRAM_TENSOR_PATTERN}=CPU"]
    if server_supports(binary, "--moe-stream") and not custom_has("--moe-stream"):
        args.append("--moe-stream")
        if offload.expert_cache_gb:
            args += ["--moe-stream-cache", offload.expert_cache_gb]
        args += ["--moe-stream-io-threads", offload.io_threads or DEFAULT_IO_THREADS]
        if server_supports(binary, "--moe-stream-direct"):
            args.append("--moe-stream-direct")
    return args


def _gib_bytes(value: str) -> str:
    return f"{int(float(value) * 1024)}M"


def mtplx_env(offload: Offload) -> list[str]:
    """`K=V` for `/usr/bin/env`, ahead of `mtplx serve`.

    MTPLX takes an operator-set allocation limit as its engine budget both
    ways, so one number caps weights, KV and the session bank together. A
    budget below what the pack must hold resident is refused by MTPLX itself,
    at load, with that reason -- it knows the pack's floor and this does not.
    """
    if not offload.active or not offload.ram_budget_gb:
        return []
    return [f"MTPLX_MEMORY_LIMIT_BYTES={_gib_bytes(offload.ram_budget_gb)}"]


def mtplx_args(offload: Offload) -> list[str]:
    if not offload.active or not offload.ngram_prewarm:
        return []
    return ["--ngram-prewarm", offload.ngram_prewarm]
