#!/usr/bin/env python3
"""Turning a slot description into a llama-server command line."""

from __future__ import annotations

import json
import os
import shlex

import platforms

from .slots import COMMON_FLAGS, COMMON_TOGGLES
from .spec import Slot


def _first(env: dict, keys: tuple[str, ...], prefix: str) -> str:
    for key in keys:
        name = key[1:] if key.startswith("!") else f"{prefix}_{key}"
        value = env.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


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

    mode = str(env.get(f"{slot.prefix}_SPLIT_MODE") or "layer").strip() or "layer"
    args = ["--split-mode", mode]
    main_gpu = str(env.get(f"{slot.prefix}_MAIN_GPU") or "").strip()
    if main_gpu:
        args += ["--main-gpu", main_gpu]
    tensor_split = even_tensor_split(
        str(env.get(f"{slot.prefix}_TENSOR_SPLIT") or "").strip(),
        str(env.get(f"{slot.prefix}_GPU_VISIBLE_DEVICES") or "").strip())
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


def custom_args(env: dict, key: str) -> list[str]:
    """Operator-supplied extra arguments, shell-split.

    A JSON list of strings, each split with `shlex` so one entry may carry
    several arguments. Anything unparseable yields nothing rather than raising:
    a malformed custom-args field must not stop a backend from starting.
    """
    if not key:
        return []
    raw = env.get(key) or ""
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


def build(slot: Slot, env: dict, extra: list[str] | None = None) -> list[str]:
    """The full argv, binary first."""
    prefix = slot.prefix
    argv = [str(env.get("LLAMA_SERVER_BIN") or "llama-server")]

    model = _first(env, slot.model_keys, prefix)
    argv += ["--model", model]
    argv += ["--alias", str(env.get(f"{prefix}_MODEL_NAME") or slot.alias_default)]
    argv += ["--host", _first(env, slot.host_keys, prefix) or "127.0.0.1"]
    argv += ["--port", str(env.get(f"{prefix}_PORT") or slot.port_default)]

    for flag in COMMON_FLAGS:
        if flag.name in slot.omit:
            continue
        suffix = flag.keys[0]
        override = slot.defaults.get(suffix)
        resolved = (flag if override is None else
                    type(flag)(flag.name, flag.keys, override)).resolve(env, prefix)
        argv += resolved
        # Placement sits between --n-gpu-layers and --batch-size, where the
        # launchers spliced SPLIT_OPTS.
        if flag.name == "--n-gpu-layers":
            argv += placement_args(slot, env)

    argv += list(slot.literals)

    for toggle in COMMON_TOGGLES + slot.extra_toggles:
        argv += toggle.resolve(env, prefix)

    mmproj = str(env.get(f"{prefix}_MMPROJ_PATH") or "").strip()
    if mmproj and os.path.isfile(mmproj):
        argv += ["--mmproj", mmproj]

    argv += custom_args(env, slot.custom_args_key)
    argv += list(extra or [])
    return argv
