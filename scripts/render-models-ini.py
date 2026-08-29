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

Most keys are a flat suffix-to-argument mapping, but a few are not a setting
llama.cpp takes directly and have to be assembled the way the start scripts
assemble them: `*_THINKING` becomes chat-template-kwargs, `*_CHAT_TEMPLATE_ID`
is resolved to a file, and `*_FIT_CTX` is dropped when auto-fit is off. Whatever
a start script derives, this has to derive too — a key the preset does not carry
is a setting the config UI claims to control and silently does not.

Usage:  render-models-ini.py [output-path]      # env comes from the environment
"""

import os
import re
import shlex
import sys
from pathlib import Path


# `*_CHAT_TEMPLATE_ID` names a file under the stack's own config directory
# rather than a path, the same indirection the start scripts perform. This
# script lives in `scripts/`, so the stack root is one level up.
STACK_DIR = Path(__file__).resolve().parent.parent


# Model path keys are not uniform: the embedding model predates the `EMBED_`
# prefix and the reranker is spelled out, so both are named explicitly rather
# than derived. `extra` holds what makes the model that kind of server.
MEMBERS = {
    "EMBED": {
        "model_path_key": "EMBEDDING_MODEL_PATH",
        "default_name": "embed",
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
    # An audio LLM, reached through the router's own /v1/audio/transcriptions.
    # `extra` is empty for the same reason TASK and OCR's are: it is a chat model
    # with a projector, not a different kind of server. The mmproj is not
    # optional in practice — llama.cpp refuses transcription unless the model
    # carries an audio encoder and a template it can build an ASR prompt from.
    #
    # Deliberately absent from the default MODEL_ROUTER_MEMBERS string: pooling
    # it is opt-in: it is absent from the default MODEL_ROUTER_MEMBERS string.
    "ASR": {
        "model_path_key": "ASR_MODEL_PATH",
        "mmproj_key": "ASR_MMPROJ_PATH",
        "default_name": "asr",
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
    "FIT_TARGET": "fit-target",
    "FIT_CTX": "fit-ctx",
    "CACHE_REUSE": "cache-reuse",
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
    "CACHE_IDLE_SLOTS": "cache-idle-slots",
}

# Options the start scripts treat as unset when they read zero, rather than
# passing a literal 0 that llama.cpp would act on. `start-task.sh:56,58` guards
# both with `!= "0"`; without this the preset would disagree with the unit.
ZERO_MEANS_UNSET = {"fit-ctx", "cache-reuse"}

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


def _bool_option(value) -> str:
    """An `on`/`off` config value as the `true`/`false` the preset wants.

    The rest of this file passes flags through verbatim because llama.cpp reads
    `on` and `off` for them. `load-on-startup` is not one of those: the router
    parses it as a bool and treats anything it does not recognise as false, so
    an `on` written here would silently mean `off` -- the failure this converts
    away from. Unset is false, which keeps a member nobody configured free.
    """
    return "true" if _clean(value).lower() in {"on", "true", "1", "yes"} else "false"


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
    if key in ZERO_MEANS_UNSET and value == "0":
        return False
    # Values run to end-of-line, but the INI grammar starts a comment at ` ;`
    # or ` #`, which would silently truncate.
    return " ;" not in value and " #" not in value


def _model_is_hybrid(model_path: str):
    """True/False when the model's architecture is known, None when it is not.

    Reuses the GGUF reader the memory model already depends on rather than
    parsing the header a second time.
    """
    try:
        sys.path.insert(0, str(STACK_DIR / "web"))
        from budget import model_geometry, read_gguf_metadata  # noqa: PLC0415
        return bool(model_geometry(read_gguf_metadata(model_path)).get("is_hybrid"))
    except Exception:
        return None


def _resolve_split_mode(model_path: str, requested: str, warn) -> str:
    """The split mode the router can actually give this member.

    Same two refusals as `resolve_split_opts` in
    scripts/lib/backend-preflight.sh, for the same reasons — the router path
    never touches the start scripts, so it needs its own copy of the check or a
    pooled model would core-dump where a standalone unit would not:

      row     removed from the CUDA backend upstream; llama.cpp throws
              "does not support split buffers" before it loads a single tensor.
      tensor  unimplemented for hybrid attention models. The arch gate in
              llama.cpp misses qwen35/qwen35moe/qwen3next, so those reach the
              meta backend and abort during warmup instead of being rejected.

    Falls back to `layer`, which always loads, rather than to the requested
    value — an unreadable model is exactly the case that must not be let
    through.
    """
    if requested == "row":
        warn("split-mode row is not supported by this CUDA build; using layer")
        return "layer"
    if requested != "tensor":
        return requested

    is_hybrid = _model_is_hybrid(model_path)
    if is_hybrid is None:
        warn(f"cannot read the architecture of {os.path.basename(model_path)} "
             "to confirm split-mode tensor supports it; using layer")
        return "layer"
    if is_hybrid:
        warn("split-mode tensor is not implemented for hybrid attention models; using layer")
        return "layer"
    return "tensor"


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


def _chat_template_file(prefix: str, template_id: str) -> str:
    """Resolve a `*_CHAT_TEMPLATE_ID` to the path llama.cpp wants.

    The id is a filename, not a path, so it is validated the same way
    `start-task.sh:115-123` validates it. A missing template is raised rather
    than dropped: silently falling back to the model's built-in template is how
    a model ends up answering correctly-shaped nonsense.
    """
    if not re.fullmatch(r"[A-Za-z0-9._-]+", template_id):
        raise RenderError(f"invalid {prefix}_CHAT_TEMPLATE_ID: {template_id!r}")
    path = STACK_DIR / "config" / "chat-templates" / f"{template_id}.jinja"
    if not path.is_file():
        raise RenderError(f"chat template not found: {path}")
    return str(path)


def render_member(prefix: str, env: dict, warn=None) -> tuple[str, dict]:
    """(section name, ordered options) for one pooled model."""
    warn = warn or (lambda message: None)
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

    # `--fit-ctx` only means anything to a server that is auto-fitting, and
    # `add_fit_ctx_opt` in scripts/lib/backend-preflight.sh drops it otherwise.
    # Emitting it here anyway is how `--fit-ctx` kept surviving `fit off`.
    if options.get("fit") == "off":
        options.pop("fit-ctx", None)

    # Split mode has to be vetted against the model, and `tensor` folds every
    # visible GPU into one device, so the ratio between them and the choice of
    # a main GPU stop meaning anything. The start scripts drop both; a preset
    # that kept them would describe a placement the server does not use.
    split_mode = options.get("split-mode")
    if split_mode:
        effective = _resolve_split_mode(model_path, split_mode,
                                        lambda m: warn(f"{prefix.lower()}: {m}"))
        options["split-mode"] = effective
        if effective == "tensor":
            options.pop("tensor-split", None)
            options.pop("main-gpu", None)

    # Thinking is not a llama.cpp flag — it is a template variable, so it rides
    # in as chat-template-kwargs exactly as start-task.sh:86-91,110-112 sends
    # it. Only members that expose the setting get the key; there is nothing
    # for an embedding or reranking model to think about.
    thinking = _clean(env.get(f"{prefix}_THINKING"))
    if thinking:
        enabled = "true" if thinking == "on" else "false"
        kwargs = f'"enable_thinking":{enabled}'
        # The thinking level only rides along while thinking is on: the template
        # ignores it otherwise, and Qwen 3.8's raises on a level it does not
        # recognize, so an unexpected value is dropped rather than failing every
        # request the router sends to this member.
        effort = _clean(env.get(f"{prefix}_REASONING_EFFORT"))
        if enabled == "true" and effort in {"xhigh", "medium", "low"}:
            kwargs += f', "reasoning_effort": "{effort}"'
        options["chat-template-kwargs"] = f"{{{kwargs}}}"

    template_id = _clean(env.get(f"{prefix}_CHAT_TEMPLATE_ID"))
    if template_id:
        options["chat-template-file"] = _chat_template_file(prefix, template_id)

    options.update(spec["extra"])

    custom = _clean(env.get(f"{prefix}_CUSTOM_ARGS_JSON"))
    if custom and custom != "[]":
        for key, value in _custom_args(custom).items():
            if key not in RESERVED_KEYS:
                options[key] = value

    # Nothing loads until a request asks for it. That is the whole point --
    # with one exception, because "on demand" and "always available" are
    # different requirements and a pool can hold both.
    #
    # An embedding model answering a retrieval path is asked for constantly and
    # in small bursts, so it spends its life being loaded, idling out and being
    # loaded again: on this host `embed` served 14,356 requests in a month and
    # still paid a cold start whenever a gap exceeded the idle window. Loading
    # it with the router costs the VRAM whether or not anyone asks, which is
    # exactly the trade an operator should get to make per member rather than
    # have made for them.
    #
    # Off by default: a member nobody has thought about should still cost
    # nothing until it is used.
    options["load-on-startup"] = _bool_option(env.get(f"{prefix}_LOAD_ON_STARTUP"))

    for key in options:
        if key in RESERVED_KEYS:
            raise RenderError(f"{prefix} would set router-controlled key {key!r}")

    return name, options


def _cap_startup_loads(rendered, env, warn) -> None:
    """Keep `load-on-startup` within `models_max`, loudly.

    The router throws on startup rather than starting without them --
    `server-models.cpp`: "number of models to load on startup (N) exceeds
    models_max (M)". Two settings that are individually reasonable therefore
    combine into a router that will not come up at all, and the failure names
    numbers rather than the keys an operator would go and change.

    So the excess is turned off here and said out loud. Refusing to render would
    reproduce the outage this exists to prevent, and a helper must never stop a
    backend from starting; the members keep their order, so the ones dropped are
    the ones furthest down `MODEL_ROUTER_MEMBERS`.
    """
    try:
        limit = int(_clean(env.get("MODEL_ROUTER_MAX")) or "2")
    except ValueError:
        return
    if limit <= 0:
        return
    eager = [entry for entry in rendered if entry[2].get("load-on-startup") == "true"]
    if len(eager) <= limit:
        return
    for prefix, name, options in eager[limit:]:
        options["load-on-startup"] = "false"
        warn(f"{prefix}: load-on-startup turned off — {len(eager)} members ask to "
             f"load at startup but MODEL_ROUTER_MAX is {limit}, and the router "
             f"refuses to start when more are asked for than it may hold. Raise "
             f"MODEL_ROUTER_MAX or clear {prefix}_LOAD_ON_STARTUP.")


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
        "; are rendered from the EMBED_*/OCR_*/RERANK_*/TASK_*/ASR_* keys.",
        "[*]",
        "version = 1",
    ]

    seen = {}
    rendered = []
    for prefix in members:
        try:
            name, options = render_member(prefix, env, warn)
        except RenderError as exc:
            warn(f"skipping {prefix}: {exc}")
            continue
        if name in seen:
            warn(f"skipping {prefix}: model name {name!r} already used by {seen[name]}")
            continue
        seen[name] = prefix
        rendered.append((prefix, name, options))

    if not seen:
        raise RenderError("no model router members could be rendered")

    _cap_startup_loads(rendered, env, warn)

    for _prefix, name, options in rendered:
        lines.append("")
        lines.append(f"[{name}]")
        lines.extend(f"{key} = {value}" for key, value in options.items())

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
