#!/usr/bin/env python3
"""Turning a slot description into a llama-server command line."""

from __future__ import annotations

import json
import shlex

import platforms

from .options import Context
from .slots import COMMON_FLAGS, COMMON_TOGGLES
from .spec import Slot, lookup


def _first(env: dict, keys: tuple[str, ...], prefixes: tuple[str, ...],
           empty_is_set: bool = False) -> str:
    return lookup(env, keys, prefixes, empty_is_set) or ""


def placement_args(slot: Slot, env: dict) -> list[str]:
    """`--split-mode` and friends.

    Prefers what the launcher already worked out. `resolve_split_opts` in
    `scripts/lib/backend-preflight.sh` is where placement is *vetted* -- it
    refuses `row` outright, vets `tensor` against the model's architecture by
    asking `budget.py` to read the GGUF, and collapses everything to `none` on
    Metal. That vetting has to stay where it is, because it is the thing
    standing between a bad setting and a core dump in a restart loop, and it is
    covered by `SplitModeTests`.

    So the launcher passes its resolved flags through
    `LLM_BACKEND_PLACEMENT_JSON` and this splices them into the right position.
    The fallback below is for callers that build a command without the shell --
    the tests, and anything that wants to know what a slot *would* run.
    """
    passed = env.get("LLM_BACKEND_PLACEMENT_JSON")
    if passed:
        try:
            values = json.loads(passed)
            if isinstance(values, list):
                return [str(v) for v in values]
        except Exception:
            pass

    if platforms.active().unified_memory:
        return ["--split-mode", "none"]

    prefixes = slot.prefixes
    mode = _first(env, ("SPLIT_MODE",), prefixes).strip() or "layer"
    args = ["--split-mode", mode]
    main_gpu = _first(env, ("MAIN_GPU",), prefixes).strip()
    if main_gpu:
        args += ["--main-gpu", main_gpu]
    tensor_split = even_tensor_split(
        _first(env, ("TENSOR_SPLIT",), prefixes).strip(),
        _first(env, ("GPU_VISIBLE_DEVICES",), prefixes).strip())
    # An empty ratio is omitted, never passed as "": llama.cpp reads
    # `--tensor-split ""` as an explicit empty split and refuses it.
    if tensor_split:
        args += ["--tensor-split", tensor_split]
    return args


def even_tensor_split(ratio: str, visible_devices: str) -> str:
    """`auto` becomes an even ratio across the visible devices.

    "1" for one device, "1,1" for two, and so on. The literal string `auto` is
    not something llama.cpp understands, so it has to be expanded before it is
    passed -- and it has to be expanded against the *visible* devices rather
    than every device on the host, because a slot pinned to one card of two
    wants "1", not "1,1".
    """
    if ratio and ratio != "auto":
        return ratio
    if not ratio:
        return ""
    count = len([part for part in visible_devices.replace(" ", "").split(",") if part])
    return ",".join(["1"] * max(count, 1))


def custom_args(env: dict, keys: tuple[str, ...], prefixes: tuple[str, ...]) -> list[str]:
    """Operator-supplied extra arguments, shell-split.

    A JSON list of strings, each split with `shlex` so one entry may carry
    several arguments. Anything unparseable yields nothing rather than raising:
    a malformed custom-args field must not stop a backend from starting.
    """
    if not keys:
        return []
    raw = lookup(env, keys, prefixes) or ""
    if not raw or raw == "[]":
        return []
    try:
        values = json.loads(raw)
    except Exception:
        return []
    out: list[str] = []
    for value in values:
        if isinstance(value, str):
            out.extend(shlex.split(value))
    return out


def build(slot: Slot, env: dict, extra: list[str] | None = None,
          said: list[str] | None = None) -> list[str]:
    """The full argv, binary first.

    `said` collects what the launcher would have echoed -- an ignored device, a
    `--fit-ctx` that auto-fit makes inert. It is a list to append to rather
    than a return value because the argv is what every caller wants and the
    messages are what one caller wants.
    """
    prefixes = slot.prefixes
    argv = [str(env.get("LLAMA_SERVER_BIN") or "llama-server")]

    argv += ["--model", _first(env, slot.model_keys, prefixes)]
    argv += ["--alias", _first(env, slot.alias_keys, prefixes) or slot.alias_default]
    argv += ["--host", _first(env, slot.host_keys, prefixes) or "127.0.0.1"]
    argv += ["--port", _first(env, slot.port_keys, prefixes) or slot.port_default]

    for flag in COMMON_FLAGS:
        if flag.name in slot.omit:
            continue
        suffix = flag.keys[0]
        keys = slot.key_chains.get(suffix, flag.keys)
        default = slot.defaults.get(suffix, flag.default)
        argv += type(flag)(flag.name, keys, default, flag.empty_is_set).resolve(env, prefixes)
        # Placement sits between --n-gpu-layers and --batch-size, where the
        # launchers spliced SPLIT_OPTS.
        if flag.name == "--n-gpu-layers":
            argv += placement_args(slot, env)

    argv += list(slot.literals)

    # The operator's own arguments come last on the command line, but they are
    # parsed first: several of the tail's decisions are "unless they already
    # passed this themselves".
    custom = custom_args(env, slot.custom_args_keys, prefixes)
    ctx = Context(slot=slot, custom=tuple(custom),
                  stack_dir=str(env.get("STACK_DIR") or ""), said=said or [])

    for toggle in COMMON_TOGGLES:
        argv += toggle.resolve(env, prefixes, ctx)
    for option in slot.tail:
        argv += option.resolve(env, prefixes, ctx)

    argv += custom
    argv += list(extra or [])
    return argv
