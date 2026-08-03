#!/usr/bin/env python3
"""Render the llama-server router preset from the stack's per-model env keys.

In router mode one `llama-server` owns the auxiliary models instead of one
systemd unit each, and it reads their settings from an INI preset rather than
from a command line. This turns the `EMBED_*`, `OCR_*`, `RERANK_*` and `TASK_*`
keys that `CONFIG_FIELDS` already exposes into that file, so the config UI stays
the single place those models are configured and nothing is duplicated.

The section name is the model id clients must send, taken from `*_MODEL_NAME`.
That is not cosmetic: `server_model_meta::update_args` overwrites the child's
`--alias` with the section name, so the section name *is* the routing key. Note
`RERANK_MODEL_NAME` is `rank`, not `rerank`.

Preset keys are llama.cpp command-line arguments without the leading dashes.
Flag options need no `no-` handling here — `common_preset::to_args` swaps in the
negative form itself when a value reads falsey (`off`/`false`/`0`/`disabled`),
which is exactly the vocabulary the stack's env already uses.

Usage:  render-models-ini.py [output-path]      # env comes from the environment
"""

import os
import shlex
import sys


# Model path keys are not uniform: the embedding model predates the `EMBED_`
# prefix and the reranker is spelled out, so both are named explicitly rather
# than derived. `extra` holds what makes the model that kind of server.
MEMBERS = {
    "EMBED": {
        "model_path_key": "EMBEDDING_MODEL_PATH",
        "default_name": "embed",
        "extra": {"embedding": "true", "pooling": "mean"},
    },
    "EMBED2": {
        "model_path_key": "EMBED2_MODEL_PATH",
        "default_name": "embed2",
        "extra": {"embedding": "true", "pooling": "mean"},
    },
    "RERANK": {
        "model_path_key": "RERANKER_MODEL_PATH",
        "default_name": "rank",
        "extra": {"reranking": "true"},
    },
    "TASK": {
        "model_path_key": "TASK_MODEL_PATH",
        "mmproj_key": "TASK_MMPROJ_PATH",
        "default_name": "task",
        "extra": {},
    },
    "OCR": {
        "model_path_key": "OCR_MODEL_PATH",
        "mmproj_key": "OCR_MMPROJ_PATH",
        "default_name": "ocr",
        "extra": {},
    },
}

# env suffix -> preset key, for options that carry a value.
VALUE_OPTIONS = {
    "CTX_SIZE": "ctx-size",
    "N_GPU_LAYERS": "n-gpu-layers",
    "MAIN_GPU": "main-gpu",
    "DEVICE": "device",
    "SPLIT_MODE": "split-mode",
    "TENSOR_SPLIT": "tensor-split",
    "BATCH_SIZE": "batch-size",
    "UBATCH_SIZE": "ubatch-size",
    "N_PARALLEL": "parallel",
    "THREADS": "threads",
    "THREADS_BATCH": "threads-batch",
    "CACHE_TYPE_K": "cache-type-k",
    "CACHE_TYPE_V": "cache-type-v",
    "CACHE_RAM": "cache-ram",
    "CTX_CHECKPOINTS": "ctx-checkpoints",
    "FLASH_ATTN": "flash-attn",
    "TEMP": "temp",
    "TOP_P": "top-p",
    "TOP_K": "top-k",
    "MIN_P": "min-p",
    "PRESENCE_PENALTY": "presence-penalty",
    "REPEAT_PENALTY": "repeat-penalty",
    "REASONING_FORMAT": "reasoning-format",
    "FIT": "fit",
}

# env suffix -> preset key, for flags. Values pass through verbatim.
FLAG_OPTIONS = {
    "NO_MMAP": "no-mmap",
    "MLOCK": "mlock",
    "METRICS": "metrics",
    "JINJA": "jinja",
    "LOG_PREFIX": "log-prefix",
    "KV_OFFLOAD": "kv-offload",
    "OP_OFFLOAD": "op-offload",
    "MMPROJ_OFFLOAD": "mmproj-offload",
    "SWA_FULL": "swa-full",
}

# The router assigns these to each child; anything written here is discarded or
# overwritten. Emitting them would imply a control the preset does not have.
RESERVED_KEYS = {
    "host", "port", "alias", "api-key", "hf-repo", "hf-repo-file",
    "models-dir", "models-max", "models-preset", "models-autoload",
    "ssl-key-file", "ssl-cert-file",
}


class RenderError(Exception):
    """A member cannot be rendered — the caller decides whether that is fatal."""


def _clean(value) -> str:
    """Env values arrive quoted by the shell often enough to be worth stripping."""
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    return text


def _usable(key: str, value: str) -> bool:
    """Whether a rendered value should be written at all.

    An unset optional key must be omitted rather than written empty: llama.cpp
    would take `tensor-split = ` as an explicit empty split and refuse it. And
    `auto` is a convention of the stack's own start scripts, which expand it in
    bash before exec; llama.cpp has no such value, so it is dropped and the
    default applies.
    """
    if not value:
        return False
    if key == "tensor-split" and value == "auto":
        return False
    # Values run to end-of-line, but the INI grammar starts a comment at ` ;`
    # or ` #`, which would silently truncate.
    return " ;" not in value and " #" not in value


def _custom_args(raw: str) -> dict:
    """Best-effort translation of a *_CUSTOM_ARGS_JSON list into preset keys.

    The start scripts pass these through as raw argv, which a preset cannot
    express in general. Flags and single-valued options map cleanly; anything
    else is reported so the caller can warn rather than drop it silently.
    """
    import json

    try:
        values = json.loads(raw or "[]")
    except (ValueError, TypeError):
        raise RenderError(f"could not parse custom arguments as JSON: {raw!r}")
    if not isinstance(values, list):
        raise RenderError("custom arguments must be a JSON list")

    tokens = []
    for value in values:
        if isinstance(value, str):
            tokens.extend(shlex.split(value))

    options = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("-"):
            raise RenderError(f"custom argument {token!r} has no flag to attach to")
        key = token.lstrip("-")
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
            options[key] = tokens[index + 1]
            index += 2
        else:
            options[key] = "true"
            index += 1
    return options


def render_member(prefix: str, env: dict) -> tuple[str, dict]:
    """(section name, ordered options) for one pooled model."""
    spec = MEMBERS.get(prefix.upper())
    if spec is None:
        raise RenderError(f"unknown model router member {prefix!r}; "
                          f"known members are {', '.join(sorted(MEMBERS))}")
    prefix = prefix.upper()

    model_path = _clean(env.get(spec["model_path_key"]))
    if not model_path:
        raise RenderError(f"{spec['model_path_key']} is not set")

    name = _clean(env.get(f"{prefix}_MODEL_NAME")) or spec["default_name"]

    options = {"model": model_path}

    mmproj_key = spec.get("mmproj_key")
    if mmproj_key:
        mmproj = _clean(env.get(mmproj_key))
        if mmproj:
            options["mmproj"] = mmproj

    for suffix, key in VALUE_OPTIONS.items():
        value = _clean(env.get(f"{prefix}_{suffix}"))
        if _usable(key, value):
            options[key] = value

    for suffix, key in FLAG_OPTIONS.items():
        value = _clean(env.get(f"{prefix}_{suffix}"))
        if value:
            options[key] = value

    options.update(spec["extra"])

    custom = _clean(env.get(f"{prefix}_CUSTOM_ARGS_JSON"))
    if custom and custom != "[]":
        for key, value in _custom_args(custom).items():
            if key not in RESERVED_KEYS:
                options[key] = value

    # Nothing loads until a request asks for it. That is the whole point.
    options["load-on-startup"] = "false"

    for key in options:
        if key in RESERVED_KEYS:
            raise RenderError(f"{prefix} would set router-controlled key {key!r}")

    return name, options


def render(env: dict, members=None, warn=None) -> str:
    """The full preset file.

    A member that cannot be rendered is skipped with a warning rather than
    failing the whole file — one missing model path should not take the other
    three offline.
    """
    warn = warn or (lambda message: None)
    if members is None:
        raw = _clean(env.get("MODEL_ROUTER_MEMBERS")) or "EMBED,OCR,RERANK,TASK"
        members = [m.strip() for m in raw.split(",") if m.strip()]

    # `version` goes inside `[*]` rather than at the top of the file. Top-level
    # keys land in the preset named "default", and the router then advertises
    # `default` on /v1/models as a real, routable model with no model path —
    # visible to anything that enumerates models, and a load error if asked for.
    # Under `[*]` the version is still read and the phantom does not appear.
    lines = [
        "; Generated by scripts/render-models-ini.py — edits here are lost on restart.",
        "; Change these models in the manager's config UI instead; the sections below",
        "; are rendered from the EMBED_*/OCR_*/RERANK_*/TASK_* keys.",
        "[*]",
        "version = 1",
    ]

    seen = {}
    for prefix in members:
        try:
            name, options = render_member(prefix, env)
        except RenderError as exc:
            warn(f"skipping {prefix}: {exc}")
            continue
        if name in seen:
            warn(f"skipping {prefix}: model name {name!r} already used by {seen[name]}")
            continue
        seen[name] = prefix
        lines.append("")
        lines.append(f"[{name}]")
        lines.extend(f"{key} = {value}" for key, value in options.items())

    if not seen:
        raise RenderError("no model router members could be rendered")

    return "\n".join(lines) + "\n"


def main(argv) -> int:
    destination = argv[1] if len(argv) > 1 else ""
    try:
        text = render(os.environ, warn=lambda m: print(f"[models.ini] {m}", file=sys.stderr))
    except RenderError as exc:
        print(f"[models.ini] {exc}", file=sys.stderr)
        return 1
    if destination and destination != "-":
        with open(destination, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"[models.ini] wrote {destination}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
