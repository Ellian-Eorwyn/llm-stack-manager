#!/usr/bin/env python3
"""
The configuration surface: every key the UI can set, and what it costs to change.

This is data, not behaviour — four tables that between them define what the
manager considers configurable:

  * `CONFIG_FIELDS`    - the field registry the config form renders from, and
                         the allow-list writes are checked against.
  * `RESTART_HINTS`    - which services a key's value reaches, so saving one
                         setting does not restart the whole stack. Backend keys
                         are derived by prefix rather than listed, because a
                         hand-maintained list is how a field comes to restart
                         nothing.
  * `LEGACY_ENV_KEY_MAP` and friends - the old names for keys that have been
                         renamed, kept readable so saved profiles written under
                         them still load. See the note above the map for the
                         staged path to removing them.
  * `CODE_TO_CHAT_MIRRORS` - settings the code endpoint shares with the chat
                         backend, because they are the same llama-server.

It lives apart from `app.py` because it is 900 lines of table that nothing
executes, and because the modules that need to validate a key — the config
routes, the saved-config loader, the pre-flight check — should be able to reach
it without importing the whole application.

Primary and secondary backend fields are *generated* from the shared-backend
ones (`_clone_chat_backend_field`) rather than written out three times. That is
why adding a field to the "Shared Backend" section makes it appear on the
primary backend too, and why the identity keys — the model path, the label, the
context size, the ones that genuinely differ per slot — are listed explicitly as
exceptions.
"""

from collections import defaultdict

import backends
import platforms

LLAMA_KV_CACHE_OPTIONS = ["q8_0", "f16", "f32", "bf16", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl"]

# How weights are placed across GPUs. `row` is deliberately not offered: the
# CUDA split-buffer implementation was removed upstream, so llama-server throws
# "device CUDA0 does not support split buffers" before it loads a single
# tensor — there is no configuration that makes it work on this build.
#
# `tensor` additionally needs a non-hybrid architecture, which depends on the
# model file rather than on this setting, so it cannot be gated here. The
# launcher vets it per model and falls back to `layer` with a reason —
# `resolve_split_opts` in scripts/lib/backend-preflight.sh.
LLAMA_SPLIT_MODE_OPTIONS = ["none", "layer", "tensor"]
LLAMA_SPLIT_MODE_HINT = (
    "none=whole model on one GPU; layer=split by layers, the dependable "
    "multi-GPU choice; tensor=experimental tensor+KV parallelism, not "
    "available for hybrid-attention models such as Qwen3.5/3.8. "
    "tensor spreads every tensor across the visible devices and therefore "
    "discards Main GPU and Device \u2014 those controls are inert while it is "
    "selected, which is silent unless something says so."
)
LLAMA_MAIN_GPU_HINT = (
    "GPU index (within visible devices) for split-mode=none. Not used by "
    "layer, and ignored by tensor, which merges the visible GPUs into one device"
)
LLAMA_TENSOR_SPLIT_HINT = (
    "Weight per visible GPU under split-mode=layer, e.g. 1,1 for an even "
    "split or 3,2 to favour the first. Ignored by none and tensor"
)


# Whisper's own model names, used by any engine whose runtime resolves them.
# `faster-whisper` downloads these from its own cache by bare name; the NeMo and
# transformers engines name full repo ids instead, which is why the presets are
# per-engine below rather than one global list.
WHISPER_MODEL_PRESETS = [
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v1", "large-v2", "large-v3",
    "distil-large-v2", "distil-large-v3", "turbo",
]

# The transcription engines the sidecar can host.
#
# An engine is a *runtime* plus the config slot that selects a model for it —
# not a model. The two names this replaced conflated the two: `whisperkit` is an
# Apple-only runtime this host cannot run, and `large-v3` is a model, so the slot
# could never hold turbo or distil without lying about its own name. Naming the
# runtime and letting the model be config is what lets one slot serve every
# Whisper variant, and what makes `hf-asr` a usable "anything I download" slot.
#
# `runtime` is what the sidecar dispatches on and what decides which pip extra
# has to be installed; `install_extra` is the `--engines` token for
# scripts/install-transcribe.sh. `legacy_ids` keeps already-downloaded weights
# reachable after the rename — see `models.transcription_engine_models_dir`.
TRANSCRIPTION_ENGINES = [
    {
        "id": "faster-whisper", "label": "Faster-Whisper", "env_prefix": "FASTER_WHISPER",
        "runtime": "faster-whisper", "install_extra": "faster-whisper", "default_install": True,
        "presets": list(WHISPER_MODEL_PRESETS), "legacy_ids": ["whisperkit-large-v3"],
    },
    {
        "id": "parakeet-v3", "label": "Parakeet TDT 0.6B v3 (NeMo)", "env_prefix": "PARAKEET_V3",
        "runtime": "nemo", "install_extra": "nemo", "default_install": False,
        "presets": ["nvidia/parakeet-tdt-0.6b-v3"], "legacy_ids": [],
    },
    {
        "id": "canary-qwen", "label": "Canary-Qwen 2.5B (NeMo)", "env_prefix": "CANARY_QWEN",
        "runtime": "nemo", "install_extra": "nemo", "default_install": False,
        "presets": ["nvidia/canary-qwen-2.5b"], "legacy_ids": [],
    },
    {
        "id": "hf-asr", "label": "HF Transformers (any ASR model)", "env_prefix": "HF_ASR",
        "runtime": "hf", "install_extra": "hf", "default_install": False,
        "presets": [], "legacy_ids": [],
    },
    # Holds no weights of its own: it forwards to llama-router, where an
    # audio-capable GGUF is pooled with embed/ocr/rank/task and evicted by the
    # same LRU. The only engine that genuinely shares the router's model space.
    {
        "id": "router", "label": "Audio LLM via Model Router", "env_prefix": "ROUTER_ASR",
        "runtime": "router", "install_extra": "", "default_install": True,
        "presets": [], "legacy_ids": [],
    },
]
TRANSCRIPTION_ENGINE_IDS = [item["id"] for item in TRANSCRIPTION_ENGINES]
TRANSCRIPTION_ENGINE_BY_ID = {item["id"]: item for item in TRANSCRIPTION_ENGINES}


def default_transcription_model(engine: dict) -> str:
    """The model ref an engine should hold when nothing has been chosen."""
    presets = engine.get("presets") or []
    if engine.get("runtime") == "router":
        # Served under the models.ini section name, not downloaded from anywhere.
        return "preset:asr"
    return f"preset:{presets[0]}" if presets else ""


def repair_transcription_model(engine: dict, value: str) -> str:
    """Replace a model ref that belongs to a different runtime.

    Every engine's model used to default off one shared
    `TRANSCRIPT_LOCAL_MODEL_SIZE`, so configs written then carry Whisper sizes
    on the NeMo slots — `PARAKEET_V3_LOCAL_MODEL=preset:large-v3`. Those values
    are still sitting in env files, and a Whisper size is not a thing NeMo can
    ever load, so the engine fails on every request with "Model large-v3 was not
    found" rather than anything pointing at the config.

    Only bare preset names are touched. A `local:` path or an explicit repo id
    is a choice someone made, and is left exactly as written.
    """
    raw = (value or "").strip()
    if not raw.startswith("preset:") or engine.get("runtime") == "faster-whisper":
        return raw
    name = raw.split(":", 1)[1]
    if name in WHISPER_MODEL_PRESETS:
        return default_transcription_model(engine)
    return raw

LEGACY_ENV_KEY_MAP = {
    "CHAT_MODEL_27B_PATH": "LLM_A_MODEL_PATH",
    "CHAT_MMPROJ_27B_PATH": "LLM_A_MMPROJ_PATH",
    "CHAT_27B_CTX_SIZE": "LLM_A_CTX_SIZE",
    "CHAT_MODEL_35B_PATH": "LLM_B_MODEL_PATH",
    "CHAT_MMPROJ_35B_PATH": "LLM_B_MMPROJ_PATH",
    "CHAT_35B_CTX_SIZE": "LLM_B_CTX_SIZE",
    "CHAT_DENSE_LABEL": "LLM_A_LABEL",
    "CHAT_DENSE_MODEL_NAME": "LLM_A_MODEL_NAME",
    "CHAT_DENSE_MODEL_PATH": "LLM_A_MODEL_PATH",
    "CHAT_DENSE_MMPROJ_PATH": "LLM_A_MMPROJ_PATH",
    "CHAT_DENSE_CTX_SIZE": "LLM_A_CTX_SIZE",
    # The MoE keys described an *alternative* model for the one shared backend,
    # selected by switch-chat-model.sh and mutually exclusive with the dense
    # one. That slot is gone; what replaced it is a genuinely concurrent second
    # backend on its own port, which is what LLM_B_* configures.
    #
    # Pointing them there rather than dropping them means a host that had a MoE
    # model configured and no second slot gets that model promoted into slot B
    # instead of silently losing it. `normalize_env_keys` only ever backfills --
    # `if new_key not in normalized` -- so a host that already has LLM_B_* set
    # keeps it, and the two cannot collide.
    "CHAT_MOE_LABEL": "LLM_B_LABEL",
    "CHAT_MOE_MODEL_NAME": "LLM_B_MODEL_NAME",
    "CHAT_MOE_MODEL_PATH": "LLM_B_MODEL_PATH",
    "CHAT_MOE_MMPROJ_PATH": "LLM_B_MMPROJ_PATH",
    "CHAT_MOE_CTX_SIZE": "LLM_B_CTX_SIZE",
    "WHISPERKIT_LARGE_V3_BACKEND_TYPE": "FASTER_WHISPER_BACKEND_TYPE",
    "WHISPERKIT_LARGE_V3_LOCAL_MODEL": "FASTER_WHISPER_LOCAL_MODEL",
    "WHISPERKIT_LARGE_V3_UPSTREAM_URL": "FASTER_WHISPER_UPSTREAM_URL",
    "WHISPERKIT_LARGE_V3_MODEL": "FASTER_WHISPER_MODEL",
    "WHISPERKIT_LARGE_V3_API_KEY": "FASTER_WHISPER_API_KEY",
    "WHISPERKIT_LARGE_V3_TRANSCRIBE_PATH": "FASTER_WHISPER_TRANSCRIBE_PATH",
    "WHISPERKIT_LARGE_V3_STREAM_OUTPUT_ENABLED": "FASTER_WHISPER_STREAM_OUTPUT_ENABLED",
    "WHISPERKIT_LARGE_V3_STREAM_OUTPUT_TARGET": "FASTER_WHISPER_STREAM_OUTPUT_TARGET",
    "WHISPERKIT_LARGE_V3_STREAM_OUTPUT_FORMAT": "FASTER_WHISPER_STREAM_OUTPUT_FORMAT",
    "WHISPERKIT_LARGE_V3_SPEAKER_DETECTION": "FASTER_WHISPER_SPEAKER_DETECTION",
    "WHISPERKIT_LARGE_V3_SPEAKER_MODE": "FASTER_WHISPER_SPEAKER_MODE",
    "WHISPERKIT_LARGE_V3_SPEAKER_COUNT": "FASTER_WHISPER_SPEAKER_COUNT",
}
# `CHAT_PRIMARY_*` and `CHAT2_*` were the canonical spellings until the slots
# became peers named `llm-a` and `llm-b`. Generated rather than written out
# because there are 330 of them and a hand-maintained list would go stale the
# first time a field was added.
#
# Filled in after `CONFIG_FIELDS` is defined, at the bottom of this module: the
# canonical names are the field keys, and this map has to be declared before
# them because `normalize_env_keys` imports it.
#
# Flattened, never chained. `CHAT_DENSE_MODEL_PATH` points straight at
# `LLM_A_MODEL_PATH` rather than at `CHAT_PRIMARY_MODEL_PATH`, which is now a
# legacy name itself -- `tests/test_llm_stack_manager.py` asserts no canonical
# key is also a legacy key, and a chain would make every rename a lookup deeper
# than the last.
_RENAMED_SLOT_PREFIXES = (("CHAT_PRIMARY_", "LLM_A_"), ("CHAT2_", "LLM_B_"))

#: The keys that keep their old spelling: they name the ports and hosts the
#: proxies dial, section 2.1 freezes those, and every consumer -- the proxies,
#: telemetry, health -- talks to them by name. Both slots, so the rule is stated
#: once rather than being true of llm-b and merely accidental of llm-a.
_PREFIX_RENAME_EXEMPT = frozenset({
    "CHAT2_BACKEND_PORT", "CHAT2_BACKEND_HOST",
    "CHAT_BACKEND_PORT", "CHAT_BACKEND_HOST",
    # Never a spelling anybody wrote, and listed so the prefix rule cannot
    # invent one in either direction: the port contract has no old name to
    # answer to and no new name to move to.
    "CHAT_PRIMARY_BACKEND_PORT", "CHAT_PRIMARY_BACKEND_HOST",
})

#: Older prefixes a slot answers to but which are never rewritten. `slots.py`
#: gives llm-a `legacy_prefixes=("CHAT_PRIMARY", "CHAT")`, so a bare `CHAT_*` key
#: is live on a host that has no `LLM_A_*` twin -- but the bare prefix is shared
#: with settings that belong to no slot (`CHAT_BEE_*`, `CHAT_SECONDARY_*`) and
#: with the frozen port contract. So it may only ever be *read*: this tier is
#: consulted by `legacy_names_for` and by nothing else. Putting it in
#: `_RENAMED_SLOT_PREFIXES` would reach `normalize_config_updates` and rewrite
#: `CHAT_TEMP` on write, which is not a rename anyone asked for.
_SHADOWED_SLOT_PREFIXES = (("CHAT_", "LLM_A_"),)


def _register_prefix_renames() -> None:
    """Map every `CHAT_PRIMARY_*` / `CHAT2_*` field key to its new name."""
    for field in CONFIG_FIELDS:
        key = field.get("key", "")
        for old_prefix, new_prefix in _RENAMED_SLOT_PREFIXES:
            if not key.startswith(new_prefix):
                continue
            legacy = old_prefix + key[len(new_prefix):]
            if legacy in _PREFIX_RENAME_EXEMPT or legacy in LEGACY_ENV_KEY_MAP:
                continue
            LEGACY_ENV_KEY_MAP[legacy] = key


NEW_ENV_KEY_LEGACY_ALIASES = defaultdict(list)


def legacy_names_for(key: str) -> tuple[str, ...]:
    """Every older spelling of `key`, declared or merely implied by a rename.

    `NEW_ENV_KEY_LEGACY_ALIASES` only knows keys some field declares, and the
    example config carries settings that no field does -- `LLM_B_TEMP` among
    them. A caller asking "has this host got an older name for this setting"
    needs the prefix rule too, or it concludes the setting is absent and writes
    a default over a live value.
    """
    names = list(NEW_ENV_KEY_LEGACY_ALIASES.get(key, ()))
    # Renamed before shadowed: the order decides which line `update_env_values`
    # writes into when a host carries more than one old spelling.
    for old_prefix, new_prefix in _RENAMED_SLOT_PREFIXES + _SHADOWED_SLOT_PREFIXES:
        if not key.startswith(new_prefix):
            continue
        candidate = old_prefix + key[len(new_prefix):]
        if candidate not in names and candidate not in _PREFIX_RENAME_EXEMPT:
            names.append(candidate)
    return tuple(names)


def canonical_name_for(key: str) -> str:
    """The current spelling of `key`, declared or merely implied by a rename.

    The inverse of `legacy_names_for`, and it exists for the same reason: six
    `CHAT2_*` settings that no field declares are live on hosts, so a map-only
    lookup reports 55 legacy keys where 61 are written down -- and a report that
    goes quiet while the keys are still being read is worse than no report.

    Only the renamed prefixes. A bare `CHAT_*` key is read behind `LLM_A_*` but
    is not a spelling of it: rewriting one would sweep up `CHAT_BEE_*`, which
    names no slot at all, and the frozen `CHAT_BACKEND_PORT`.
    """
    if key in LEGACY_ENV_KEY_MAP:
        return LEGACY_ENV_KEY_MAP[key]
    if key in _PREFIX_RENAME_EXEMPT:
        return key
    for old_prefix, new_prefix in _RENAMED_SLOT_PREFIXES:
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix):]
    return key


def _rebuild_legacy_aliases() -> None:
    NEW_ENV_KEY_LEGACY_ALIASES.clear()
    for legacy_key, new_key in LEGACY_ENV_KEY_MAP.items():
        NEW_ENV_KEY_LEGACY_ALIASES[new_key].append(legacy_key)


_rebuild_legacy_aliases()

# Why the dual naming exists, and how it ends.
#
# The backend slots were once named for what they held — CHAT_DENSE_* for a
# dense model, CHAT_MOE_* for a mixture-of-experts one — and before that for the
# specific models themselves (CHAT_MODEL_27B_PATH). Both schemes described the
# contents rather than the slot, so both went stale the moment a slot's model
# changed. LLM_A_* / LLM_B_* name the slot instead, and are the canonical
# form: `normalize_env_keys` backfills the canonical key from its legacy twin on
# read, and `normalize_config_updates` rewrites legacy keys to canonical on
# write. Nothing writes a legacy key any more.
#
# What remains is residue: legacy keys still sitting in llm-stack.env and in
# saved profiles, read every time and written never. They cannot simply be
# deleted — a saved profile that carries only CHAT_DENSE_MODEL_PATH would lose
# its model. The path out, staged so no stage can break a saved config:
#
#   1. (done) Canonical on write, backfilled on read. Legacy keys are inert.
#   2. (here) Report them. `GET /api/config/deprecations` names every legacy key
#      still present, where it lives, and what replaces it.
#   3. (here) Migrate on request. `POST /api/config/deprecations/migrate`
#      rewrites llm-stack.env in place: canonical key written from the legacy
#      value where it is missing, legacy key dropped. Saved profiles are left
#      alone — they are user data, and step 1 keeps reading them correctly.
#   4. (future) Once a report comes back empty on this host, drop the legacy
#      names from `allowed_config_keys` so they stop being writable at all,
#      keeping `LEGACY_ENV_KEY_MAP` for read-side backfill of old profiles.
#
# Step 4 is deliberately not taken here: it is only safe once step 3 has run and
# the report is empty, and that is an operator action, not a code change.
DEPRECATED_ENV_KEY_NOTES = {
    "CHAT_MODEL_27B_PATH": "named for a model size that the slot no longer implies",
    "CHAT_MMPROJ_27B_PATH": "named for a model size that the slot no longer implies",
    "CHAT_27B_CTX_SIZE": "named for a model size that the slot no longer implies",
    "CHAT_MODEL_35B_PATH": "named for a model size that the slot no longer implies",
    "CHAT_MMPROJ_35B_PATH": "named for a model size that the slot no longer implies",
    "CHAT_35B_CTX_SIZE": "named for a model size that the slot no longer implies",
    **{
        _legacy: "named for WhisperKit, an Apple-only runtime this host does not run"
        for _legacy in LEGACY_ENV_KEY_MAP
        if _legacy.startswith("WHISPERKIT_")
    },
}
# Sixteen `HONCHO_*` notes used to sit here, for a memory service `install.sh`
# retired. They were never emitted: this table annotates a *rename*, and the
# report walks keys that have a canonical name to move to. A key belonging to a
# service that no longer exists has no such name, so it needs a different
# mechanism than this one -- and until it has one, a note that cannot be
# rendered is just a claim the report does not make.
DEFAULT_DEPRECATION_NOTE = "named for the model architecture a slot happened to hold"

CORE_CONFIG_SECTIONS = {
    "Chat Templates",
    "LLM A",
    "LLM B",
    "Shared Backend",
    "Task Model",
    "Thinking Endpoint",
    "Instruct Endpoint",
    "Coding Endpoint",
    "Embedding",
    "Reranker",
    "OCR",
    "GLM-OCR SDK",
    "Model Router",
    "SearXNG",
    "Playwright",
    "Transcription",
    "Apple Silicon (MLX)",
    "Ports",
    "State API",
    "Control API",
}

#: The coding endpoint is a proxy persona on `llm-a` (`backends/proxies.py`), not
#: a backend of its own, so a backend-level setting saved under `CODE_*` has to
#: land on the slot that actually serves it.
#:
#: These targeted the bare `CHAT_*` spelling, which was right when one backend
#: was shared by every endpoint. It stopped being right twice over. `CHAT_*` is
#: now only a legacy name read behind `LLM_A_*`, so the mirror wrote a key the
#: launcher no longer prefers -- and `CODE_CTX_SIZE` additionally listed
#: `CHAT_DENSE_CTX_SIZE` *and* `CHAT_MOE_CTX_SIZE`, which normalize onto
#: `LLM_A_CTX_SIZE` and `LLM_B_CTX_SIZE`. Saving the Coding Endpoint section
#: resized both backends, one of which the coding endpoint has never touched.
CODE_TO_CHAT_MIRRORS = {
    "CODE_CTX_SIZE":            "LLM_A_CTX_SIZE",
    "CODE_N_PARALLEL":          "LLM_A_N_PARALLEL",
    "CODE_THREADS":             "LLM_A_THREADS",
    "CODE_THREADS_BATCH":       "LLM_A_THREADS_BATCH",
    "CODE_N_GPU_LAYERS":        "LLM_A_N_GPU_LAYERS",
    "CODE_TENSOR_SPLIT":        "LLM_A_TENSOR_SPLIT",
    "CODE_SPLIT_MODE":          "LLM_A_SPLIT_MODE",
    "CODE_FLASH_ATTN":          "LLM_A_FLASH_ATTN",
    "CODE_CACHE_TYPE_K":        "LLM_A_CACHE_TYPE_K",
    "CODE_CACHE_TYPE_V":        "LLM_A_CACHE_TYPE_V",
    "CODE_BATCH_SIZE":          "LLM_A_BATCH_SIZE",
    "CODE_UBATCH_SIZE":         "LLM_A_UBATCH_SIZE",
    "CODE_NO_MMAP":             "LLM_A_NO_MMAP",
    "CODE_MLOCK":               "LLM_A_MLOCK",
    "CODE_GPU_VISIBLE_DEVICES": "LLM_A_GPU_VISIBLE_DEVICES",
    "CODE_REASONING_FORMAT":    "LLM_A_REASONING_FORMAT",
    "CODE_FIT":                 "LLM_A_FIT",
}

# ---------------------------------------------------------------------------
# Config fields exposed in the UI
# ---------------------------------------------------------------------------
LLAMA_SPEC_METHOD_OPTIONS = ["off", "draft-model", "draft-simple", "draft-eagle3", "draft-mtp", "draft-dflash", "ngram-cache", "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod"]
LLAMA_CACHE_IDLE_OPTIONS = ["on", "off"]
LLAMA_METRICS_OPTIONS = ["on", "off"]
# Qwen 3.8's template accepts exactly these three levels and raises on anything
# else, so the UI offers no free-text path to an invalid one.
LLAMA_REASONING_EFFORT_OPTIONS = [
    {"value": "", "label": "model default"},
    {"value": "xhigh", "label": "xhigh — thorough"},
    {"value": "medium", "label": "medium — unsteered"},
    {"value": "low", "label": "low — brief"},
]

CONFIG_FIELDS = [
    {"section": "Chat Templates", "key": "CHAT_TEMPLATE_MANAGER", "label": "Template Manager", "type": "template_manager", "hint": "Create and edit reusable llama.cpp Jinja chat templates"},

    # LLM B
    {"section": "LLM B", "key": "LLM_B_LABEL",                 "label": "Backend Label",           "type": "text",   "hint": "UI label for the secondary backend slot"},
    {"section": "LLM B", "key": "LLM_B_MODEL_NAME",            "label": "Model Alias",             "type": "text",   "hint": "llama.cpp alias for the secondary backend"},
    {"section": "LLM B", "key": "LLM_B_MODEL_PATH",            "label": "Model Path",              "type": "path"},
    {"section": "LLM B", "key": "LLM_B_MMPROJ_PATH",           "label": "MMProj Path",             "type": "path"},
    {"section": "LLM B", "key": "LLM_B_CTX_SIZE",              "label": "Context Size",            "type": "number"},
    {"section": "LLM B", "key": "CHAT2_BACKEND_PORT",          "label": "Backend Port",            "type": "number"},
    {"section": "LLM B", "key": "THINK2_PORT",                 "label": "Think Port",              "type": "number"},
    {"section": "LLM B", "key": "NOTHINK2_PORT",               "label": "Chat Port",               "type": "number"},
    {"section": "LLM B", "key": "CODE2_PORT",                  "label": "Code Port",               "type": "number"},
    {"section": "LLM B", "key": "AGGREGATE2_ENABLED",          "label": "Aggregate Proxy",         "type": "select", "options": ["on", "off"], "hint": "Single model-routed endpoint exposing think, chat, and code for the secondary backend"},
    {"section": "LLM B", "key": "AGGREGATE2_PORT",             "label": "Aggregate Port",          "type": "number"},
    {"section": "LLM B", "key": "THINK2_MODEL_NAME",           "label": "Think Alias",             "type": "text", "hint": "Advertised model id on the secondary aggregate and think port"},
    {"section": "LLM B", "key": "NOTHINK2_MODEL_NAME",         "label": "Chat Alias",              "type": "text", "hint": "Advertised model id on the secondary aggregate and chat port"},
    {"section": "LLM B", "key": "CODE2_MODEL_NAME",            "label": "Code Alias",              "type": "text", "hint": "Advertised model id on the secondary aggregate and code port"},
    {"section": "LLM B", "key": "LLM_B_N_PARALLEL",            "label": "Parallel Slots",          "type": "number"},
    {"section": "LLM B", "key": "LLM_B_THREADS",               "label": "CPU Threads",             "type": "number", "hint": "llama.cpp --threads for generation; -1 lets llama.cpp choose"},
    {"section": "LLM B", "key": "LLM_B_THREADS_BATCH",         "label": "CPU Batch Threads",       "type": "number", "hint": "llama.cpp --threads-batch for prompt/batch processing; -1 follows --threads"},
    {"section": "LLM B", "key": "LLM_B_N_GPU_LAYERS",          "label": "GPU Layers (−1=all)",     "type": "number"},
    {"section": "LLM B", "key": "LLM_B_MAIN_GPU",              "label": "Main GPU Index",          "type": "number", "hint": LLAMA_MAIN_GPU_HINT},
    {"section": "LLM B", "key": "LLM_B_DEVICE",                "label": "Main/Draft Offload Devices", "type": "text", "hint": "Optional llama.cpp --device override; use --list-devices names like CUDA0,CUDA1 or none"},
    {"section": "LLM B", "key": "LLM_B_TENSOR_SPLIT",          "label": "Tensor Split",            "type": "text",   "hint": LLAMA_TENSOR_SPLIT_HINT},
    {"section": "LLM B", "key": "LLM_B_SPLIT_MODE",            "label": "Split Mode",              "type": "select", "options": LLAMA_SPLIT_MODE_OPTIONS, "hint": LLAMA_SPLIT_MODE_HINT},
    {"section": "LLM B", "key": "LLM_B_KV_OFFLOAD",            "label": "KV Offload",              "type": "select", "options": ["on", "off"], "hint": "Controls --kv-offload / --no-kv-offload"},
    {"section": "LLM B", "key": "LLM_B_OP_OFFLOAD",            "label": "Host Op Offload",         "type": "select", "options": ["on", "off"], "hint": "Controls --op-offload / --no-op-offload for host tensor ops"},
    {"section": "LLM B", "key": "LLM_B_MMPROJ_OFFLOAD",        "label": "MMProj Offload",          "type": "select", "options": ["on", "off"], "hint": "Controls --mmproj-offload / --no-mmproj-offload when an MMProj is loaded"},
    {"section": "LLM B", "key": "LLM_B_FLASH_ATTN",            "label": "Flash Attention",         "type": "select", "options": ["on", "off", "auto"]},
    {"section": "LLM B", "key": "LLM_B_CACHE_TYPE_K",          "label": "KV Cache Key Type",       "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "LLM B", "key": "LLM_B_CACHE_TYPE_V",          "label": "KV Cache Value Type",     "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "LLM B", "key": "LLM_B_CACHE_RAM",             "label": "Prompt Cache RAM",        "type": "number", "hint": "llama.cpp --cache-ram in MiB; 0 disables server prompt-cache storage"},
    {"section": "LLM B", "key": "LLM_B_CTX_CHECKPOINTS",       "label": "Context Checkpoints",     "type": "number", "hint": "llama.cpp --ctx-checkpoints; 0 disables context checkpoint creation"},
    {"section": "LLM B", "key": "LLM_B_SWA_FULL",              "label": "Full SWA KV Cache",       "type": "select", "options": ["off", "on"], "hint": "Adds llama.cpp --swa-full for SWA models; uses more KV VRAM but improves prompt-cache reuse"},
    {"section": "LLM B", "key": "LLM_B_BATCH_SIZE",            "label": "Batch Size",              "type": "number", "hint": "Prefill batch (default 2048)"},
    {"section": "LLM B", "key": "LLM_B_UBATCH_SIZE",           "label": "Micro-Batch Size",        "type": "number", "hint": "Physical sub-batch (default 512)"},
    {"section": "LLM B", "key": "LLM_B_NO_MMAP",               "label": "Disable mmap",            "type": "select", "options": ["false", "true"]},
    {"section": "LLM B", "key": "LLM_B_MLOCK",                 "label": "Lock Memory",             "type": "select", "options": ["false", "true"]},
    {"section": "LLM B", "key": "LLM_B_GPU_VISIBLE_DEVICES",   "label": "GPU Devices",             "type": "text",   "hint": "e.g. 0,1"},
    {"section": "LLM B", "key": "LLM_B_JINJA",                 "label": "Backend Jinja Support",   "type": "select", "options": ["off", "on"], "hint": "Enables --jinja on the secondary backend so proxy ports can expose tool calling"},
    {"section": "LLM B", "key": "LLM_B_PRESERVE_THINKING",     "label": "Preserve Thinking",       "type": "select", "options": ["on", "off"], "hint": "Backend default for chat_template_kwargs.preserve_thinking on the secondary backend"},
    {"section": "LLM B", "key": "LLM_B_REASONING_EFFORT",      "label": "Thinking Level",          "type": "select", "options": LLAMA_REASONING_EFFORT_OPTIONS, "hint": "Backend default for templates that read reasoning_effort (Qwen 3.8+). medium adds no steering instruction — it is the model's unsteered baseline"},
    {"section": "LLM B", "key": "LLM_B_TEMPLATE_ID",           "label": "Effective Chat Template", "type": "chat_template", "hint": "Custom Jinja template file passed to the secondary backend; model default leaves GGUF metadata unchanged"},
    {"section": "LLM B", "key": "LLM_B_FIT",                   "label": "Auto-Fit to VRAM",        "type": "select", "options": ["on", "off"], "hint": "When on, may reduce context size to fit in VRAM"},
    {"section": "LLM B", "key": "LLM_B_FIT_TARGET",            "label": "Fit Target MiB",          "type": "text",   "hint": "llama.cpp --fit-target per-device margin, e.g. 1024 or 1024,2048; empty uses llama.cpp default"},
    {"section": "LLM B", "key": "LLM_B_FIT_CTX",               "label": "Minimum Fit Context",     "type": "number", "hint": "llama.cpp --fit-ctx minimum context when auto-fit adjusts settings"},
    {"section": "LLM B", "key": "LLM_B_CACHE_IDLE_SLOTS",      "label": "Cache Idle Slots",        "type": "select", "options": LLAMA_CACHE_IDLE_OPTIONS, "hint": "Controls --cache-idle-slots / --no-cache-idle-slots"},
    {"section": "LLM B", "key": "LLM_B_METRICS", "label": "Metrics Endpoint", "type": "select", "options": LLAMA_METRICS_OPTIONS, "hint": "Controls --metrics; enables the backend's Prometheus endpoint for the telemetry panel"},
    {"section": "LLM B", "key": "LLM_B_CACHE_REUSE",           "label": "Cache Reuse Chunk",       "type": "number", "hint": "llama.cpp --cache-reuse minimum chunk size; 0 leaves llama.cpp default"},
    {"section": "LLM B", "key": "LLM_B_SPEC_METHOD",           "label": "Speculative Method",      "type": "select", "options": LLAMA_SPEC_METHOD_OPTIONS, "hint": "Base llama.cpp mode. draft-dflash requires an upstream DFlash draft GGUF with general.architecture=dflash;"},
    {"section": "LLM B", "key": "LLM_B_SPEC_NGRAM_MOD",        "label": "N-Gram Mod Assist",       "type": "select", "options": ["off", "on"], "hint": "When on, appends ngram-mod to MTP-style spec types, e.g. draft-mtp,ngram-mod"},
    {"section": "LLM B", "key": "LLM_B_SPEC_DRAFT_MODEL_PATH", "label": "Draft Model Path",        "type": "path",   "hint": "Smaller GGUF used as the speculative draft model"},
    {"section": "LLM B", "key": "LLM_B_SPEC_DRAFT_N_GPU_LAYERS", "label": "Draft GPU Layers",      "type": "text",   "hint": "Draft-model --spec-draft-ngl value: auto, all, or an exact layer count"},
    {"section": "LLM B", "key": "LLM_B_SPEC_DRAFT_DEVICES",    "label": "Draft Devices",           "type": "text",   "hint": "Optional --spec-draft-device override, e.g. 0,1 or none"},
    {"section": "LLM B", "key": "LLM_B_SPEC_DRAFT_TYPE_K",     "label": "Draft KV Key Type",       "type": "select", "options": LLAMA_KV_CACHE_OPTIONS, "hint": "llama.cpp --spec-draft-type-k"},
    {"section": "LLM B", "key": "LLM_B_SPEC_DRAFT_TYPE_V",     "label": "Draft KV Value Type",     "type": "select", "options": LLAMA_KV_CACHE_OPTIONS, "hint": "llama.cpp --spec-draft-type-v"},
    {"section": "LLM B", "key": "LLM_B_SPEC_DRAFT_N_MAX",      "label": "Draft Max Tokens",        "type": "number", "hint": "llama.cpp --spec-draft-n-max (recommended 6 for MTP)"},
    {"section": "LLM B", "key": "LLM_B_SPEC_DRAFT_N_MIN",      "label": "Draft Min Tokens",        "type": "number", "hint": "llama.cpp --spec-draft-n-min (default 0)"},
    {"section": "LLM B", "key": "LLM_B_SPEC_DRAFT_P_MIN",      "label": "Draft Min Probability",   "type": "text",   "hint": "llama.cpp --spec-draft-p-min (default 0.75)"},
    {"section": "LLM B", "key": "LLM_B_SPEC_DRAFT_P_SPLIT",    "label": "Draft Split Probability", "type": "text",   "hint": "llama.cpp --spec-draft-p-split (default 0.10)"},
    {"section": "LLM B", "key": "LLM_B_SPEC_NGRAM_MOD_N_MATCH","label": "N-Gram Match Tokens",     "type": "number", "hint": "llama.cpp --spec-ngram-mod-n-match (default 24)"},
    {"section": "LLM B", "key": "LLM_B_SPEC_NGRAM_MOD_N_MIN",  "label": "N-Gram Min Tokens",       "type": "number", "hint": "llama.cpp --spec-ngram-mod-n-min (default 48)"},
    {"section": "LLM B", "key": "LLM_B_SPEC_NGRAM_MOD_N_MAX",  "label": "N-Gram Max Tokens",       "type": "number", "hint": "llama.cpp --spec-ngram-mod-n-max (default 64)"},
    {"section": "LLM B", "key": "LLM_B_SPEC_NGRAM_SIZE_N",     "label": "N-Gram Lookup Size",      "type": "number", "hint": "llama.cpp --spec-ngram-*-size-n for ngram-simple/map modes"},
    {"section": "LLM B", "key": "LLM_B_SPEC_NGRAM_SIZE_M",     "label": "N-Gram Draft Size",       "type": "number", "hint": "llama.cpp --spec-ngram-*-size-m for ngram-simple/map modes"},
    {"section": "LLM B", "key": "LLM_B_SPEC_NGRAM_MIN_HITS",   "label": "N-Gram Min Hits",         "type": "number", "hint": "llama.cpp --spec-ngram-*-min-hits for ngram-simple/map modes"},
    {"section": "LLM B", "key": "LLM_B_LORA_PATHS",               "label": "LoRA Adapters",        "type": "adapter_path", "hint": "Adapter GGUFs under models/loras/, comma separated. Applied on top of the base model at runtime -- nothing is merged, and swapping between them needs no reload"},
    {"section": "LLM B", "key": "LLM_B_LORA_SCALES",              "label": "LoRA Scales",          "type": "text",   "hint": "One scale per adapter, comma separated, in the same order. Blank means 1.0. With preloading unapplied this is the strength the Services page restores when you switch an adapter on"},
    {"section": "LLM B", "key": "LLM_B_LORA_INIT_WITHOUT_APPLY",  "label": "Preload Unapplied",    "type": "select", "options": ["on", "off"], "hint": "On: every adapter loads at scale 0 and is switched on from the Services page. Off: they all apply at their configured scale from startup, and stack"},
    {"section": "LLM B", "key": "LLM_B_CUSTOM_ARGS_JSON",      "label": "Custom Arguments",        "type": "custom_args", "hint": "Extra llama.cpp flags applied to the secondary backend"},
    # Shared Backend
    {"section": "Shared Backend", "key": "CHAT_DENSE_LABEL",           "label": "Dense Slot Label",       "type": "text",   "hint": "UI label for the dense preset button/card"},
    {"section": "Shared Backend", "key": "CHAT_DENSE_MODEL_NAME",      "label": "Dense Model Alias",      "type": "text",   "hint": "llama.cpp alias for the dense preset"},
    {"section": "Shared Backend", "key": "CHAT_DENSE_MODEL_PATH",      "label": "Dense Model Path",       "type": "path"},
    {"section": "Shared Backend", "key": "CHAT_DENSE_MMPROJ_PATH",     "label": "Dense MMProj Path",      "type": "path"},
    {"section": "Shared Backend", "key": "CHAT_DENSE_CTX_SIZE",        "label": "Dense Context Size",     "type": "number"},
    {"section": "Shared Backend", "key": "CHAT_MOE_LABEL",             "label": "MoE Slot Label",         "type": "text",   "hint": "UI label for the MoE preset button/card"},
    {"section": "Shared Backend", "key": "CHAT_MOE_MODEL_NAME",        "label": "MoE Model Alias",        "type": "text",   "hint": "llama.cpp alias for the MoE preset"},
    {"section": "Shared Backend", "key": "CHAT_MOE_MODEL_PATH",        "label": "MoE Model Path",         "type": "path"},
    {"section": "Shared Backend", "key": "CHAT_MOE_MMPROJ_PATH",       "label": "MoE MMProj Path",        "type": "path"},
    {"section": "Shared Backend", "key": "CHAT_MOE_CTX_SIZE",          "label": "MoE Context Size",       "type": "number"},
    {"section": "Shared Backend", "key": "CHAT_MODEL_NAME",            "label": "Custom Backend Alias",   "type": "text",   "hint": "llama.cpp alias for the generic custom backend"},
    {"section": "Shared Backend", "key": "CHAT_N_PARALLEL",            "label": "Parallel Slots",         "type": "number"},
    {"section": "Shared Backend", "key": "CHAT_THREADS",               "label": "CPU Threads",            "type": "number", "hint": "llama.cpp --threads for generation; -1 lets llama.cpp choose"},
    {"section": "Shared Backend", "key": "CHAT_THREADS_BATCH",         "label": "CPU Batch Threads",      "type": "number", "hint": "llama.cpp --threads-batch for prompt/batch processing; -1 follows --threads"},
    {"section": "Shared Backend", "key": "CHAT_N_GPU_LAYERS",          "label": "GPU Layers (−1=all)",    "type": "number"},
    {"section": "Shared Backend", "key": "CHAT_MAIN_GPU",              "label": "Main GPU Index",         "type": "number", "hint": LLAMA_MAIN_GPU_HINT},
    {"section": "Shared Backend", "key": "CHAT_DEVICE",                "label": "Main/Draft Offload Devices", "type": "text", "hint": "Optional llama.cpp --device override for shared backends; use --list-devices names like CUDA0,CUDA1 or none"},
    {"section": "Shared Backend", "key": "CHAT_TENSOR_SPLIT",          "label": "Tensor Split",           "type": "text",   "hint": LLAMA_TENSOR_SPLIT_HINT},
    {"section": "Shared Backend", "key": "CHAT_SPLIT_MODE",            "label": "Split Mode",             "type": "select", "options": LLAMA_SPLIT_MODE_OPTIONS, "hint": LLAMA_SPLIT_MODE_HINT},
    {"section": "Shared Backend", "key": "CHAT_KV_OFFLOAD",            "label": "KV Offload",             "type": "select", "options": ["on", "off"], "hint": "Controls --kv-offload / --no-kv-offload"},
    {"section": "Shared Backend", "key": "CHAT_OP_OFFLOAD",            "label": "Host Op Offload",        "type": "select", "options": ["on", "off"], "hint": "Controls --op-offload / --no-op-offload for host tensor ops"},
    {"section": "Shared Backend", "key": "CHAT_MMPROJ_OFFLOAD",        "label": "MMProj Offload",         "type": "select", "options": ["on", "off"], "hint": "Controls --mmproj-offload / --no-mmproj-offload when an MMProj is loaded"},
    {"section": "Shared Backend", "key": "CHAT_FLASH_ATTN",            "label": "Flash Attention",        "type": "select", "options": ["on", "off", "auto"]},
    {"section": "Shared Backend", "key": "CHAT_CACHE_TYPE_K",          "label": "KV Cache Key Type",      "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Shared Backend", "key": "CHAT_CACHE_TYPE_V",          "label": "KV Cache Value Type",    "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Shared Backend", "key": "CHAT_CACHE_RAM",             "label": "Prompt Cache RAM",      "type": "number", "hint": "llama.cpp --cache-ram in MiB; 0 disables server prompt-cache storage"},
    {"section": "Shared Backend", "key": "CHAT_CTX_CHECKPOINTS",       "label": "Context Checkpoints",   "type": "number", "hint": "llama.cpp --ctx-checkpoints; 0 disables context checkpoint creation"},
    {"section": "Shared Backend", "key": "CHAT_SWA_FULL",              "label": "Full SWA KV Cache",     "type": "select", "options": ["off", "on"], "hint": "Adds llama.cpp --swa-full for SWA models; uses more KV VRAM but improves prompt-cache reuse"},
    {"section": "Shared Backend", "key": "CHAT_BATCH_SIZE",            "label": "Batch Size",             "type": "number", "hint": "Prefill batch (default 2048)"},
    {"section": "Shared Backend", "key": "CHAT_UBATCH_SIZE",           "label": "Micro-Batch Size",       "type": "number", "hint": "Physical sub-batch (default 512)"},
    {"section": "Shared Backend", "key": "CHAT_NO_MMAP",               "label": "Disable mmap",           "type": "select", "options": ["false", "true"]},
    {"section": "Shared Backend", "key": "CHAT_MLOCK",                 "label": "Lock Memory",            "type": "select", "options": ["false", "true"]},
    {"section": "Shared Backend", "key": "CHAT_GPU_VISIBLE_DEVICES",   "label": "GPU Devices",            "type": "text",   "hint": "e.g. 0,1"},
    {"section": "Shared Backend", "key": "CHAT_JINJA",                 "label": "Backend Jinja Support",  "type": "select", "options": ["off", "on"], "hint": "Enables --jinja on the shared backend so proxy ports can expose tool calling"},
    {"section": "Shared Backend", "key": "CHAT_TEMPLATE_ID",           "label": "Effective Chat Template", "type": "chat_template", "hint": "Custom Jinja template file passed to the shared backend; model default leaves GGUF metadata unchanged"},
    {"section": "Shared Backend", "key": "CHAT_PRESERVE_THINKING",     "label": "Preserve Thinking",       "type": "select", "options": ["on", "off"], "hint": "Backend default for chat_template_kwargs.preserve_thinking; the proxy overrides it per endpoint"},
    {"section": "Shared Backend", "key": "CHAT_REASONING_EFFORT",      "label": "Thinking Level",          "type": "select", "options": LLAMA_REASONING_EFFORT_OPTIONS, "hint": "Backend default for templates that read reasoning_effort (Qwen 3.8+). medium adds no steering instruction — it is the model's unsteered baseline. Ignored where thinking is off, and by templates that do not read it"},
    {"section": "Shared Backend", "key": "CHAT_FIT",                   "label": "Auto-Fit to VRAM",       "type": "select", "options": ["on", "off"], "hint": "When on, may reduce context size to fit in VRAM"},
    {"section": "Shared Backend", "key": "CHAT_FIT_TARGET",            "label": "Fit Target MiB",         "type": "text",   "hint": "llama.cpp --fit-target per-device margin, e.g. 1024 or 1024,2048; empty uses llama.cpp default"},
    {"section": "Shared Backend", "key": "CHAT_FIT_CTX",               "label": "Minimum Fit Context",    "type": "number", "hint": "llama.cpp --fit-ctx minimum context when auto-fit adjusts settings"},
    {"section": "Shared Backend", "key": "CHAT_CACHE_IDLE_SLOTS",      "label": "Cache Idle Slots",       "type": "select", "options": LLAMA_CACHE_IDLE_OPTIONS, "hint": "Controls --cache-idle-slots / --no-cache-idle-slots"},
    {"section": "Shared Backend", "key": "CHAT_METRICS", "label": "Metrics Endpoint", "type": "select", "options": LLAMA_METRICS_OPTIONS, "hint": "Controls --metrics; enables the backend's Prometheus endpoint for the telemetry panel"},
    {"section": "Shared Backend", "key": "CHAT_CACHE_REUSE",           "label": "Cache Reuse Chunk",      "type": "number", "hint": "llama.cpp --cache-reuse minimum chunk size; 0 leaves llama.cpp default"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_METHOD",           "label": "Speculative Method",     "type": "select", "options": LLAMA_SPEC_METHOD_OPTIONS, "hint": "Base llama.cpp mode. draft-dflash requires an upstream DFlash draft GGUF with general.architecture=dflash;"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_NGRAM_MOD",        "label": "N-Gram Mod Assist",      "type": "select", "options": ["off", "on"], "hint": "When on, appends ngram-mod to MTP-style spec types, e.g. draft-mtp,ngram-mod"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_DRAFT_MODEL_PATH", "label": "Draft Model Path",       "type": "path",   "hint": "Smaller GGUF used as the speculative draft model"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_DRAFT_N_GPU_LAYERS", "label": "Draft GPU Layers",     "type": "text",   "hint": "Draft-model --spec-draft-ngl value: auto, all, or an exact layer count"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_DRAFT_DEVICES",    "label": "Draft Devices",          "type": "text",   "hint": "Optional --spec-draft-device override, e.g. 0,1 or none"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_DRAFT_TYPE_K",     "label": "Draft KV Key Type",      "type": "select", "options": LLAMA_KV_CACHE_OPTIONS, "hint": "llama.cpp --spec-draft-type-k"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_DRAFT_TYPE_V",     "label": "Draft KV Value Type",    "type": "select", "options": LLAMA_KV_CACHE_OPTIONS, "hint": "llama.cpp --spec-draft-type-v"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_DRAFT_N_MAX",      "label": "Draft Max Tokens",       "type": "number", "hint": "llama.cpp --spec-draft-n-max (recommended 6 for MTP)"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_DRAFT_N_MIN",      "label": "Draft Min Tokens",       "type": "number", "hint": "llama.cpp --spec-draft-n-min (default 0)"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_DRAFT_P_MIN",      "label": "Draft Min Probability",  "type": "text",   "hint": "llama.cpp --spec-draft-p-min (default 0.75)"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_DRAFT_P_SPLIT",    "label": "Draft Split Probability","type": "text",   "hint": "llama.cpp --spec-draft-p-split (default 0.10)"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_NGRAM_MOD_N_MATCH","label": "N-Gram Match Tokens",    "type": "number", "hint": "llama.cpp --spec-ngram-mod-n-match (default 24)"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_NGRAM_MOD_N_MIN",  "label": "N-Gram Min Tokens",      "type": "number", "hint": "llama.cpp --spec-ngram-mod-n-min (default 48)"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_NGRAM_MOD_N_MAX",  "label": "N-Gram Max Tokens",      "type": "number", "hint": "llama.cpp --spec-ngram-mod-n-max (default 64)"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_NGRAM_SIZE_N",     "label": "N-Gram Lookup Size",     "type": "number", "hint": "llama.cpp --spec-ngram-*-size-n for ngram-simple/map modes"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_NGRAM_SIZE_M",     "label": "N-Gram Draft Size",      "type": "number", "hint": "llama.cpp --spec-ngram-*-size-m for ngram-simple/map modes"},
    {"section": "Shared Backend", "key": "CHAT_SPEC_NGRAM_MIN_HITS",   "label": "N-Gram Min Hits",        "type": "number", "hint": "llama.cpp --spec-ngram-*-min-hits for ngram-simple/map modes"},
    {"section": "Shared Backend", "key": "CHAT_LORA_PATHS",                  "label": "LoRA Adapters",        "type": "adapter_path", "hint": "Adapter GGUFs under models/loras/, comma separated. Applied on top of the base model at runtime -- nothing is merged, and swapping between them needs no reload"},
    {"section": "Shared Backend", "key": "CHAT_LORA_SCALES",                 "label": "LoRA Scales",          "type": "text",   "hint": "One scale per adapter, comma separated, in the same order. Blank means 1.0. With preloading unapplied this is the strength the Services page restores when you switch an adapter on"},
    {"section": "Shared Backend", "key": "CHAT_LORA_INIT_WITHOUT_APPLY",     "label": "Preload Unapplied",    "type": "select", "options": ["on", "off"], "hint": "On: every adapter loads at scale 0 and is switched on from the Services page. Off: they all apply at their configured scale from startup, and stack"},
    {"section": "Shared Backend", "key": "CHAT_CUSTOM_ARGS_JSON",      "label": "Custom Arguments",       "type": "custom_args", "hint": "Extra llama.cpp flags applied to all shared chat backends"},
    # Task Model
    {"section": "Task Model",  "key": "TASK_MODEL_NAME",            "label": "Model Name",           "type": "text",   "hint": "Advertised on /v1/models for the task endpoint"},
    {"section": "Task Model",  "key": "TASK_MODEL_PATH",            "label": "Task Model Path",      "type": "path"},
    {"section": "Task Model",  "key": "TASK_MMPROJ_PATH",           "label": "MMProj Path",          "type": "path"},
    {"section": "Task Model",  "key": "TASK_CTX_SIZE",              "label": "Context Size",         "type": "number"},
    {"section": "Task Model",  "key": "TASK_LOAD_ON_STARTUP",                  "label": "Load At Router Start", "type": "select", "options": ["off", "on"], "hint": "Load this model when the router starts and keep it resident, rather than on the first request. Costs its VRAM whether or not anyone asks; worth it for a model queried constantly in small bursts, which is the case the router's laziness handles worst."},
    {"section": "Task Model",  "key": "TASK_N_PARALLEL",            "label": "Parallel Slots",       "type": "number"},
    {"section": "Task Model",  "key": "TASK_THREADS",               "label": "CPU Threads",          "type": "number", "hint": "llama.cpp --threads for generation; -1 lets llama.cpp choose"},
    {"section": "Task Model",  "key": "TASK_THREADS_BATCH",         "label": "CPU Batch Threads",    "type": "number", "hint": "llama.cpp --threads-batch for prompt/batch processing; -1 follows --threads"},
    {"section": "Task Model",  "key": "TASK_N_GPU_LAYERS",          "label": "GPU Layers (−1=all)",  "type": "number"},
    {"section": "Task Model",  "key": "TASK_MAIN_GPU",              "label": "Main GPU Index",       "type": "number", "hint": LLAMA_MAIN_GPU_HINT},
    {"section": "Task Model",  "key": "TASK_DEVICE",                "label": "Offload Devices",      "type": "text",   "hint": "Optional llama.cpp --device override, e.g. 0,1 or none"},
    {"section": "Task Model",  "key": "TASK_TENSOR_SPLIT",          "label": "Tensor Split",         "type": "text",   "hint": LLAMA_TENSOR_SPLIT_HINT},
    {"section": "Task Model",  "key": "TASK_SPLIT_MODE",            "label": "Split Mode",           "type": "select", "options": LLAMA_SPLIT_MODE_OPTIONS, "hint": LLAMA_SPLIT_MODE_HINT},
    {"section": "Task Model",  "key": "TASK_KV_OFFLOAD",            "label": "KV Offload",           "type": "select", "options": ["on", "off"], "hint": "Controls --kv-offload / --no-kv-offload"},
    {"section": "Task Model",  "key": "TASK_OP_OFFLOAD",            "label": "Host Op Offload",      "type": "select", "options": ["on", "off"], "hint": "Controls --op-offload / --no-op-offload for host tensor ops"},
    {"section": "Task Model",  "key": "TASK_MMPROJ_OFFLOAD",        "label": "MMProj Offload",       "type": "select", "options": ["on", "off"], "hint": "Controls --mmproj-offload / --no-mmproj-offload when an MMProj is loaded"},
    {"section": "Task Model",  "key": "TASK_BATCH_SIZE",             "label": "Batch Size",           "type": "number", "hint": "Prefill batch (default 2048)"},
    {"section": "Task Model",  "key": "TASK_UBATCH_SIZE",            "label": "Micro-Batch Size",     "type": "number", "hint": "Physical sub-batch (default 512)"},
    {"section": "Task Model",  "key": "TASK_NO_MMAP",                "label": "Disable mmap",         "type": "select", "options": ["false", "true"]},
    {"section": "Task Model",  "key": "TASK_MLOCK",                  "label": "Lock Memory",          "type": "select", "options": ["false", "true"]},
    {"section": "Task Model",  "key": "TASK_GPU_VISIBLE_DEVICES",   "label": "GPU Devices",          "type": "text"},
    {"section": "Task Model",  "key": "TASK_FLASH_ATTN",            "label": "Flash Attention",      "type": "select", "options": ["on", "off", "auto"]},
    {"section": "Task Model",  "key": "TASK_CACHE_TYPE_K",          "label": "KV Cache Key Type",    "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Task Model",  "key": "TASK_CACHE_TYPE_V",          "label": "KV Cache Value Type",  "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Task Model",  "key": "TASK_CACHE_RAM",             "label": "Prompt Cache RAM",     "type": "number", "hint": "llama.cpp --cache-ram in MiB; 0 disables server prompt-cache storage"},
    {"section": "Task Model",  "key": "TASK_CTX_CHECKPOINTS",       "label": "Context Checkpoints",  "type": "number", "hint": "llama.cpp --ctx-checkpoints; 0 disables context checkpoint creation"},
    {"section": "Task Model",  "key": "TASK_SWA_FULL",              "label": "Full SWA KV Cache",    "type": "select", "options": ["off", "on"], "hint": "Adds llama.cpp --swa-full for SWA models; uses more KV VRAM but improves prompt-cache reuse"},
    {"section": "Task Model",  "key": "TASK_TEMP",                  "label": "Temperature",          "type": "text",   "hint": "e.g. 1.0"},
    {"section": "Task Model",  "key": "TASK_TOP_P",                 "label": "Top-P",                "type": "text",   "hint": "e.g. 0.95"},
    {"section": "Task Model",  "key": "TASK_TOP_K",                 "label": "Top-K",                "type": "number"},
    {"section": "Task Model",  "key": "TASK_MIN_P",                 "label": "Min-P",                "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Task Model",  "key": "TASK_PRESENCE_PENALTY",      "label": "Presence Penalty",     "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Task Model",  "key": "TASK_REPEAT_PENALTY",        "label": "Repeat Penalty",       "type": "text",   "hint": "e.g. 1.10"},
    {"section": "Task Model",  "key": "TASK_JINJA",                 "label": "Native Tool Calling",  "type": "select", "options": ["off", "on"], "hint": "Enables --jinja for OpenAI-compatible tool/function calling"},
    {"section": "Task Model",  "key": "TASK_CHAT_TEMPLATE_ID",        "label": "Chat Template",        "type": "chat_template", "hint": "Custom Jinja template file passed to the standalone task model"},
    {"section": "Task Model",  "key": "TASK_THINKING",              "label": "Thinking",             "type": "select", "options": ["off", "on"], "hint": "Enable/disable thinking/reasoning for the task model"},
    {"section": "Task Model",  "key": "TASK_REASONING_FORMAT",      "label": "Reasoning Format",     "type": "select", "options": ["none", "deepseek", "deepseek-legacy"], "hint": "How thinking content appears in API responses"},
    {"section": "Task Model",  "key": "TASK_REASONING_EFFORT",      "label": "Thinking Level",       "type": "select", "options": LLAMA_REASONING_EFFORT_OPTIONS, "hint": "Thinking level for templates that read reasoning_effort (Qwen 3.8+). Only applies when Thinking is on"},
    {"section": "Task Model",  "key": "TASK_FIT",                   "label": "Auto-Fit to VRAM",     "type": "select", "options": ["on", "off"], "hint": "When on, may reduce context size to fit in VRAM"},
    {"section": "Task Model",  "key": "TASK_FIT_TARGET",            "label": "Fit Target MiB",       "type": "text",   "hint": "llama.cpp --fit-target per-device margin, e.g. 1024 or 1024,2048; empty uses llama.cpp default"},
    {"section": "Task Model",  "key": "TASK_FIT_CTX",               "label": "Minimum Fit Context",  "type": "number", "hint": "llama.cpp --fit-ctx minimum context when auto-fit adjusts settings"},
    {"section": "Task Model",  "key": "TASK_CACHE_IDLE_SLOTS",      "label": "Cache Idle Slots",     "type": "select", "options": LLAMA_CACHE_IDLE_OPTIONS, "hint": "Controls --cache-idle-slots / --no-cache-idle-slots"},
    {"section": "Task Model", "key": "TASK_METRICS", "label": "Metrics Endpoint", "type": "select", "options": LLAMA_METRICS_OPTIONS, "hint": "Controls --metrics; enables the backend's Prometheus endpoint for the telemetry panel"},
    {"section": "Task Model",  "key": "TASK_CACHE_REUSE",           "label": "Cache Reuse Chunk",    "type": "number", "hint": "llama.cpp --cache-reuse minimum chunk size; 0 leaves llama.cpp default"},
    {"section": "Task Model", "key": "TASK_LORA_PATHS",                  "label": "LoRA Adapters",        "type": "adapter_path", "hint": "Adapter GGUFs under models/loras/, comma separated. Applied on top of the base model at runtime -- nothing is merged, and swapping between them needs no reload"},
    {"section": "Task Model", "key": "TASK_LORA_SCALES",                 "label": "LoRA Scales",          "type": "text",   "hint": "One scale per adapter, comma separated, in the same order. Blank means 1.0. With preloading unapplied this is the strength the Services page restores when you switch an adapter on"},
    {"section": "Task Model", "key": "TASK_LORA_INIT_WITHOUT_APPLY",     "label": "Preload Unapplied",    "type": "select", "options": ["on", "off"], "hint": "On: every adapter loads at scale 0 and is switched on from the Services page. Off: they all apply at their configured scale from startup, and stack"},
    {"section": "Task Model",  "key": "TASK_CUSTOM_ARGS_JSON",      "label": "Custom Arguments",     "type": "custom_args", "hint": "Extra llama.cpp flags applied to the task model launcher"},
    {"section": "Task Model",  "key": "TASK_SPEC_METHOD",           "label": "Speculative Method",     "type": "select", "options": LLAMA_SPEC_METHOD_OPTIONS, "hint": "Base llama.cpp mode. draft-dflash requires an upstream DFlash draft GGUF with general.architecture=dflash;"},
    {"section": "Task Model",  "key": "TASK_SPEC_NGRAM_MOD",        "label": "N-Gram Mod Assist",      "type": "select", "options": ["off", "on"], "hint": "When on, appends ngram-mod to MTP-style spec types"},
    {"section": "Task Model",  "key": "TASK_SPEC_DRAFT_MODEL_PATH", "label": "Draft Model Path",       "type": "path",   "hint": "Smaller GGUF used as the speculative draft model"},
    {"section": "Task Model",  "key": "TASK_SPEC_DRAFT_N_GPU_LAYERS", "label": "Draft GPU Layers",     "type": "text",   "hint": "Draft-model --spec-draft-ngl value: auto, all, or an exact layer count"},
    {"section": "Task Model",  "key": "TASK_SPEC_DRAFT_DEVICES",    "label": "Draft Devices",          "type": "text",   "hint": "Optional --spec-draft-device override, e.g. 0,1 or none"},
    {"section": "Task Model",  "key": "TASK_SPEC_DRAFT_TYPE_K",     "label": "Draft KV Key Type",      "type": "select", "options": LLAMA_KV_CACHE_OPTIONS, "hint": "llama.cpp --spec-draft-type-k"},
    {"section": "Task Model",  "key": "TASK_SPEC_DRAFT_TYPE_V",     "label": "Draft KV Value Type",    "type": "select", "options": LLAMA_KV_CACHE_OPTIONS, "hint": "llama.cpp --spec-draft-type-v"},
    {"section": "Task Model",  "key": "TASK_SPEC_DRAFT_N_MAX",      "label": "Draft Max Tokens",       "type": "number", "hint": "llama.cpp --spec-draft-n-max (recommended 6 for MTP)"},
    {"section": "Task Model",  "key": "TASK_SPEC_DRAFT_N_MIN",      "label": "Draft Min Tokens",       "type": "number", "hint": "llama.cpp --spec-draft-n-min (default 0)"},
    {"section": "Task Model",  "key": "TASK_SPEC_DRAFT_P_MIN",      "label": "Draft Min Probability",  "type": "text",   "hint": "llama.cpp --spec-draft-p-min (default 0.75)"},
    {"section": "Task Model",  "key": "TASK_SPEC_DRAFT_P_SPLIT",    "label": "Draft Split Probability","type": "text",   "hint": "llama.cpp --spec-draft-p-split (default 0.10)"},
    {"section": "Task Model",  "key": "TASK_SPEC_NGRAM_MOD_N_MATCH","label": "N-Gram Match Tokens",    "type": "number", "hint": "llama.cpp --spec-ngram-mod-n-match (default 24)"},
    {"section": "Task Model",  "key": "TASK_SPEC_NGRAM_MOD_N_MIN",  "label": "N-Gram Min Tokens",      "type": "number", "hint": "llama.cpp --spec-ngram-mod-n-min (default 48)"},
    {"section": "Task Model",  "key": "TASK_SPEC_NGRAM_MOD_N_MAX",  "label": "N-Gram Max Tokens",      "type": "number", "hint": "llama.cpp --spec-ngram-mod-n-max (default 64)"},
    {"section": "Task Model",  "key": "TASK_SPEC_NGRAM_SIZE_N",     "label": "N-Gram Lookup Size",     "type": "number", "hint": "llama.cpp --spec-ngram-*-size-n for ngram-simple/map modes"},
    {"section": "Task Model",  "key": "TASK_SPEC_NGRAM_SIZE_M",     "label": "N-Gram Draft Size",      "type": "number", "hint": "llama.cpp --spec-ngram-*-size-m for ngram-simple/map modes"},
    {"section": "Task Model",  "key": "TASK_SPEC_NGRAM_MIN_HITS",   "label": "N-Gram Min Hits",        "type": "number", "hint": "llama.cpp --spec-ngram-*-min-hits for ngram-simple/map modes"},
    # Thinking Endpoint (proxied request-time overrides)
    {"section": "Thinking Endpoint", "key": "THINK_MODEL_NAME",          "label": "Thinking Model Name",   "type": "text",   "hint": "Advertised on /v1/models for the thinking endpoint"},
    {"section": "Thinking Endpoint", "key": "PROXY_STREAM_PASSTHROUGH",  "label": "Raw Stream Passthrough", "type": "select", "options": ["off", "on"], "hint": "When on, SSE responses bypass proxy JSON rewriting after request shaping"},
    {"section": "Thinking Endpoint", "key": "UPSTREAM_400_CAPTURE_ENABLED", "label": "Capture Upstream 400s", "type": "select", "options": ["off", "on"], "hint": "Diagnostic only: writes the request payload behind an upstream 400 — a whole conversation — to logs/upstream-400 at mode 0600, rotated"},
    {"section": "Thinking Endpoint", "key": "THINK_PRESERVE_THINKING",   "label": "Preserve Thinking",     "type": "select", "options": ["on", "off"], "hint": "Injects chat_template_kwargs.preserve_thinking into thinking requests"},
    {"section": "Thinking Endpoint", "key": "THINK_REASONING_EFFORT",   "label": "Thinking Level",        "type": "select", "options": LLAMA_REASONING_EFFORT_OPTIONS, "hint": "Default level for templates that read reasoning_effort (Qwen 3.8+). medium adds no steering instruction — it is the model's unsteered baseline. A request carrying its own reasoning_effort overrides this, and 'none' turns thinking off for that request; an OpenAI level (high, minimal) is mapped onto the nearest of these rather than failing"},
    {"section": "Thinking Endpoint", "key": "THINK_REASONING_STREAM_MODE", "label": "Reasoning Stream", "type": "select", "options": ["hidden", "content", "mirror"], "hint": "hidden keeps thinking in reasoning_content; content streams it into the answer for clients that ignore that field"},
    {"section": "Thinking Endpoint", "key": "THINK_JINJA",               "label": "Expose Tool Calling",   "type": "select", "options": ["on", "off"], "hint": "When off, strips tools/tool_choice from thinking requests"},
    {"section": "Thinking Endpoint", "key": "THINK_TEMP",                "label": "Temperature",           "type": "text",   "hint": "e.g. 0.7"},
    {"section": "Thinking Endpoint", "key": "THINK_MAX_TOKENS",          "label": "Max Tokens",            "type": "number", "hint": "Overrides client max_tokens; 0 leaves client setting unchanged"},
    {"section": "Thinking Endpoint", "key": "THINK_TOP_P",               "label": "Top-P",                 "type": "text",   "hint": "e.g. 0.95"},
    {"section": "Thinking Endpoint", "key": "THINK_TOP_K",               "label": "Top-K",                 "type": "number"},
    {"section": "Thinking Endpoint", "key": "THINK_MIN_P",               "label": "Min-P",                 "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Thinking Endpoint", "key": "THINK_PRESENCE_PENALTY",    "label": "Presence Penalty",      "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Thinking Endpoint", "key": "THINK_REPEAT_PENALTY",      "label": "Repeat Penalty",        "type": "text",   "hint": "e.g. 1.10"},
    {"section": "Thinking Endpoint", "key": "THINK_REASONING_FORMAT",    "label": "Reasoning Format",      "type": "select", "options": ["none", "deepseek", "deepseek-legacy"], "hint": "Injected per request for the thinking endpoint"},
    # Instruct Endpoint (proxied request-time overrides)
    {"section": "Instruct Endpoint", "key": "NOTHINK_MODEL_NAME",        "label": "Instruct Model Name",   "type": "text",   "hint": "Advertised on /v1/models for the non-thinking instruct endpoint"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_PRESERVE_THINKING", "label": "Preserve Thinking",     "type": "select", "options": ["on", "off"], "hint": "Injects chat_template_kwargs.preserve_thinking into instruct requests"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_REASONING_STREAM_MODE", "label": "Reasoning Stream", "type": "select", "options": ["hidden", "content", "mirror"], "hint": "Usually hidden because this endpoint disables thinking"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_JINJA",             "label": "Expose Tool Calling",   "type": "select", "options": ["on", "off"], "hint": "When off, strips tools/tool_choice from instruct requests"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_TEMP",              "label": "Temperature",           "type": "text",   "hint": "e.g. 0.7"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_MAX_TOKENS",        "label": "Max Tokens",            "type": "number", "hint": "Overrides client max_tokens; 0 leaves client setting unchanged"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_TOP_P",             "label": "Top-P",                 "type": "text",   "hint": "e.g. 0.95"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_TOP_K",             "label": "Top-K",                 "type": "number"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_MIN_P",             "label": "Min-P",                 "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_PRESENCE_PENALTY",  "label": "Presence Penalty",      "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_REPEAT_PENALTY",    "label": "Repeat Penalty",        "type": "text",   "hint": "e.g. 1.10"},
    {"section": "Instruct Endpoint", "key": "NOTHINK_REASONING_FORMAT",  "label": "Reasoning Format",      "type": "select", "options": ["none", "deepseek", "deepseek-legacy"], "hint": "Injected per request for the instruct endpoint"},
    # Coding Endpoint (proxied request-time overrides)
    {"section": "Coding Endpoint", "key": "CODE_MODEL_NAME",          "label": "Coding Model Name",     "type": "text",   "hint": "Advertised on /v1/models for the code endpoint"},
    {"section": "Coding Endpoint", "key": "CODE_THINKING",            "label": "Thinking",              "type": "select", "options": ["on", "off"], "hint": "Enable/disable thinking for the code endpoint"},
    {"section": "Coding Endpoint", "key": "CODE_PRESERVE_THINKING",   "label": "Preserve Thinking",     "type": "select", "options": ["on", "off"], "hint": "Injects chat_template_kwargs.preserve_thinking into code requests"},
    {"section": "Coding Endpoint", "key": "CODE_REASONING_EFFORT",     "label": "Thinking Level",        "type": "select", "options": LLAMA_REASONING_EFFORT_OPTIONS, "hint": "Default level for templates that read reasoning_effort (Qwen 3.8+). Only applies while Thinking is on. A request carrying its own reasoning_effort overrides this, and 'none' turns thinking off for that request"},
    {"section": "Coding Endpoint", "key": "CODE_REASONING_STREAM_MODE", "label": "Reasoning Stream", "type": "select", "options": ["hidden", "content", "mirror"], "hint": "hidden keeps thinking in reasoning_content; content streams it into the answer for clients that ignore that field"},
    {"section": "Coding Endpoint", "key": "CODE_JINJA",               "label": "Expose Tool Calling",   "type": "select", "options": ["on", "off"], "hint": "When off, strips tools/tool_choice from code requests"},
    {"section": "Coding Endpoint", "key": "CODE_TEMP",                "label": "Temperature",           "type": "text",   "hint": "e.g. 0.7"},
    {"section": "Coding Endpoint", "key": "CODE_MAX_TOKENS",          "label": "Max Tokens",            "type": "number", "hint": "Overrides client max_tokens; 0 leaves client setting unchanged"},
    {"section": "Coding Endpoint", "key": "CODE_TOP_P",               "label": "Top-P",                 "type": "text",   "hint": "e.g. 0.95"},
    {"section": "Coding Endpoint", "key": "CODE_TOP_K",               "label": "Top-K",                 "type": "number"},
    {"section": "Coding Endpoint", "key": "CODE_MIN_P",               "label": "Min-P",                 "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Coding Endpoint", "key": "CODE_PRESENCE_PENALTY",    "label": "Presence Penalty",      "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Coding Endpoint", "key": "CODE_REPEAT_PENALTY",      "label": "Repeat Penalty",        "type": "text",   "hint": "e.g. 1.10"},
    {"section": "Coding Endpoint", "key": "CODE_REASONING_FORMAT",    "label": "Reasoning Format",      "type": "select", "options": ["none", "deepseek", "deepseek-legacy"], "hint": "Injected per request for the coding endpoint"},
    # Embedding
    {"section": "Embedding",   "key": "EMBED_MODEL_NAME",           "label": "Model Name",           "type": "text",   "hint": "Advertised on /v1/models for the embedding endpoint"},
    {"section": "Embedding",   "key": "EMBEDDING_MODEL_PATH",       "label": "Model Path",           "type": "path"},
    {"section": "Embedding",   "key": "EMBED_CTX_SIZE",             "label": "Context Size",         "type": "number"},
    {"section": "Embedding",   "key": "EMBED_LOAD_ON_STARTUP",                 "label": "Load At Router Start", "type": "select", "options": ["off", "on"], "hint": "Load this model when the router starts and keep it resident, rather than on the first request. Costs its VRAM whether or not anyone asks; worth it for a model queried constantly in small bursts, which is the case the router's laziness handles worst."},
    {"section": "Embedding",   "key": "EMBED_N_PARALLEL",           "label": "Parallel Slots",       "type": "number"},
    {"section": "Embedding",   "key": "EMBED_THREADS",              "label": "CPU Threads",          "type": "number", "hint": "llama.cpp --threads for generation; -1 lets llama.cpp choose"},
    {"section": "Embedding",   "key": "EMBED_THREADS_BATCH",        "label": "CPU Batch Threads",    "type": "number", "hint": "llama.cpp --threads-batch for prompt/batch processing; -1 follows --threads"},
    {"section": "Embedding",   "key": "EMBED_N_GPU_LAYERS",         "label": "GPU Layers (−1=all)",  "type": "number"},
    {"section": "Embedding",   "key": "EMBED_TENSOR_SPLIT",         "label": "Tensor Split",         "type": "text",   "hint": LLAMA_TENSOR_SPLIT_HINT},
    {"section": "Embedding",   "key": "EMBED_SPLIT_MODE",           "label": "Split Mode",           "type": "select", "options": LLAMA_SPLIT_MODE_OPTIONS, "hint": LLAMA_SPLIT_MODE_HINT},
    {"section": "Embedding",   "key": "EMBED_FLASH_ATTN",           "label": "Flash Attention",      "type": "select", "options": ["on", "off", "auto"]},
    {"section": "Embedding",   "key": "EMBED_CACHE_TYPE_K",         "label": "KV Cache Key Type",    "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Embedding",   "key": "EMBED_CACHE_TYPE_V",         "label": "KV Cache Value Type",  "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Embedding",   "key": "EMBED_BATCH_SIZE",           "label": "Batch Size",           "type": "number"},
    {"section": "Embedding",   "key": "EMBED_UBATCH_SIZE",          "label": "Micro-Batch Size",     "type": "number"},
    {"section": "Embedding", "key": "EMBED_METRICS", "label": "Metrics Endpoint", "type": "select", "options": LLAMA_METRICS_OPTIONS, "hint": "Controls --metrics; enables the backend's Prometheus endpoint for the telemetry panel"},
    {"section": "Embedding",   "key": "EMBED_NO_MMAP",              "label": "Disable mmap",         "type": "select", "options": ["false", "true"]},
    {"section": "Embedding",   "key": "EMBED_MLOCK",                "label": "Lock Memory",          "type": "select", "options": ["false", "true"]},
    {"section": "Embedding",   "key": "EMBED_GPU_VISIBLE_DEVICES",  "label": "GPU Devices",          "type": "text"},
    {"section": "Embedding",   "key": "EMBED_TEMP",                 "label": "Temperature",          "type": "text",   "hint": "e.g. 1.0"},
    {"section": "Embedding",   "key": "EMBED_TOP_P",                "label": "Top-P",                "type": "text",   "hint": "e.g. 0.95"},
    {"section": "Embedding",   "key": "EMBED_TOP_K",                "label": "Top-K",                "type": "number"},
    {"section": "Embedding",   "key": "EMBED_MIN_P",                "label": "Min-P",                "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Embedding",   "key": "EMBED_JINJA",                "label": "Native Tool Calling",  "type": "select", "options": ["off", "on"]},
    {"section": "Embedding",   "key": "EMBED_REASONING_FORMAT",     "label": "Reasoning Format",     "type": "select", "options": ["none", "deepseek", "deepseek-legacy"]},
    {"section": "Embedding",   "key": "EMBED_FIT",                  "label": "Auto-Fit to VRAM",     "type": "select", "options": ["on", "off"]},
    # Reranker
    {"section": "Reranker",    "key": "RERANK_MODEL_NAME",          "label": "Model Name",           "type": "text",   "hint": "Advertised on /v1/models for the reranker endpoint"},
    {"section": "Reranker",    "key": "RERANKER_MODEL_PATH",        "label": "Model Path",           "type": "path"},
    {"section": "Reranker",    "key": "RERANK_CTX_SIZE",            "label": "Context Size",         "type": "number"},
    {"section": "Reranker",    "key": "RERANK_LOAD_ON_STARTUP",                "label": "Load At Router Start", "type": "select", "options": ["off", "on"], "hint": "Load this model when the router starts and keep it resident, rather than on the first request. Costs its VRAM whether or not anyone asks; worth it for a model queried constantly in small bursts, which is the case the router's laziness handles worst."},
    {"section": "Reranker",    "key": "RERANK_N_PARALLEL",          "label": "Parallel Slots",       "type": "number"},
    {"section": "Reranker",    "key": "RERANK_THREADS",             "label": "CPU Threads",          "type": "number", "hint": "llama.cpp --threads for generation; -1 lets llama.cpp choose"},
    {"section": "Reranker",    "key": "RERANK_THREADS_BATCH",       "label": "CPU Batch Threads",    "type": "number", "hint": "llama.cpp --threads-batch for prompt/batch processing; -1 follows --threads"},
    {"section": "Reranker",    "key": "RERANK_N_GPU_LAYERS",        "label": "GPU Layers (−1=all)",  "type": "number"},
    {"section": "Reranker",    "key": "RERANK_TENSOR_SPLIT",        "label": "Tensor Split",         "type": "text",   "hint": LLAMA_TENSOR_SPLIT_HINT},
    {"section": "Reranker",    "key": "RERANK_SPLIT_MODE",          "label": "Split Mode",           "type": "select", "options": LLAMA_SPLIT_MODE_OPTIONS, "hint": LLAMA_SPLIT_MODE_HINT},
    {"section": "Reranker",    "key": "RERANK_FLASH_ATTN",          "label": "Flash Attention",      "type": "select", "options": ["on", "off", "auto"]},
    {"section": "Reranker",    "key": "RERANK_CACHE_TYPE_K",        "label": "KV Cache Key Type",    "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Reranker",    "key": "RERANK_CACHE_TYPE_V",        "label": "KV Cache Value Type",  "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "Reranker",    "key": "RERANK_BATCH_SIZE",          "label": "Batch Size",           "type": "number"},
    {"section": "Reranker",    "key": "RERANK_UBATCH_SIZE",         "label": "Micro-Batch Size",     "type": "number"},
    {"section": "Reranker", "key": "RERANK_METRICS", "label": "Metrics Endpoint", "type": "select", "options": LLAMA_METRICS_OPTIONS, "hint": "Controls --metrics; enables the backend's Prometheus endpoint for the telemetry panel"},
    {"section": "Reranker",    "key": "RERANK_NO_MMAP",             "label": "Disable mmap",         "type": "select", "options": ["false", "true"]},
    {"section": "Reranker",    "key": "RERANK_MLOCK",               "label": "Lock Memory",          "type": "select", "options": ["false", "true"]},
    {"section": "Reranker",    "key": "RERANK_GPU_VISIBLE_DEVICES", "label": "GPU Devices",          "type": "text"},
    {"section": "Reranker",    "key": "RERANK_TEMP",                "label": "Temperature",          "type": "text",   "hint": "e.g. 1.0"},
    {"section": "Reranker",    "key": "RERANK_TOP_P",               "label": "Top-P",                "type": "text",   "hint": "e.g. 0.95"},
    {"section": "Reranker",    "key": "RERANK_TOP_K",               "label": "Top-K",                "type": "number"},
    {"section": "Reranker",    "key": "RERANK_MIN_P",               "label": "Min-P",                "type": "text",   "hint": "e.g. 0.00"},
    {"section": "Reranker",    "key": "RERANK_JINJA",               "label": "Native Tool Calling",  "type": "select", "options": ["off", "on"]},
    {"section": "Reranker",    "key": "RERANK_REASONING_FORMAT",    "label": "Reasoning Format",     "type": "select", "options": ["none", "deepseek", "deepseek-legacy"]},
    {"section": "Reranker",    "key": "RERANK_FIT",                 "label": "Auto-Fit to VRAM",     "type": "select", "options": ["on", "off"]},
    # OCR
    {"section": "OCR",        "key": "OCR_MODEL_NAME",           "label": "Model Name",           "type": "text",   "hint": "Advertised on /v1/models for the OCR endpoint"},
    {"section": "OCR",        "key": "OCR_MODEL_PATH",           "label": "GLM-OCR Model Path",   "type": "path"},
    {"section": "OCR",        "key": "OCR_MMPROJ_PATH",          "label": "MMProj Path",          "type": "path",   "hint": "Optional multimodal projector if your GGUF build requires a separate file"},
    {"section": "OCR",        "key": "OCR_HOST",                 "label": "Listen Host",          "type": "text"},
    {"section": "OCR",        "key": "OCR_PORT",                 "label": "Port",                 "type": "number"},
    {"section": "OCR",        "key": "OCR_CTX_SIZE",             "label": "Context Size",         "type": "number"},
    {"section": "OCR",         "key": "OCR_LOAD_ON_STARTUP",                   "label": "Load At Router Start", "type": "select", "options": ["off", "on"], "hint": "Load this model when the router starts and keep it resident, rather than on the first request. Costs its VRAM whether or not anyone asks; worth it for a model queried constantly in small bursts, which is the case the router's laziness handles worst."},
    {"section": "OCR",        "key": "OCR_N_PARALLEL",           "label": "Parallel Slots",       "type": "number"},
    {"section": "OCR",        "key": "OCR_THREADS",              "label": "CPU Threads",          "type": "number", "hint": "llama.cpp --threads for generation; -1 lets llama.cpp choose"},
    {"section": "OCR",        "key": "OCR_THREADS_BATCH",        "label": "CPU Batch Threads",    "type": "number"},
    {"section": "OCR",        "key": "OCR_N_GPU_LAYERS",         "label": "GPU Layers (-1=all)",  "type": "number"},
    {"section": "OCR",        "key": "OCR_MAIN_GPU",             "label": "Main GPU Index",       "type": "number", "hint": "GPU index within OCR GPU Devices; use 0 for the first visible GPU, 1 for the second"},
    {"section": "OCR",        "key": "OCR_DEVICE",               "label": "Offload Devices",      "type": "text",   "hint": "Optional llama.cpp --device override for OCR, e.g. CUDA0,CUDA1 or none"},
    {"section": "OCR",        "key": "OCR_TENSOR_SPLIT",         "label": "Tensor Split",         "type": "text",   "hint": "auto expands to one weight per visible OCR GPU, e.g. GPU Devices 0,1 -> 1,1; set 2,1 to bias GPU 0. Ignored by split-mode none and tensor"},
    {"section": "OCR",        "key": "OCR_SPLIT_MODE",           "label": "Split Mode",           "type": "select", "options": LLAMA_SPLIT_MODE_OPTIONS, "hint": LLAMA_SPLIT_MODE_HINT},
    {"section": "OCR",        "key": "OCR_KV_OFFLOAD",           "label": "KV Offload",           "type": "select", "options": ["on", "off"]},
    {"section": "OCR",        "key": "OCR_OP_OFFLOAD",           "label": "Host Op Offload",      "type": "select", "options": ["on", "off"]},
    {"section": "OCR",        "key": "OCR_MMPROJ_OFFLOAD",       "label": "MMProj Offload",       "type": "select", "options": ["on", "off"]},
    {"section": "OCR",        "key": "OCR_BATCH_SIZE",           "label": "Batch Size",           "type": "number"},
    {"section": "OCR",        "key": "OCR_UBATCH_SIZE",          "label": "Micro-Batch Size",     "type": "number"},
    {"section": "OCR", "key": "OCR_METRICS", "label": "Metrics Endpoint", "type": "select", "options": LLAMA_METRICS_OPTIONS, "hint": "Controls --metrics; enables the backend's Prometheus endpoint for the telemetry panel"},
    {"section": "OCR",        "key": "OCR_FLASH_ATTN",           "label": "Flash Attention",      "type": "select", "options": ["on", "off", "auto"]},
    {"section": "OCR",        "key": "OCR_CACHE_TYPE_K",         "label": "KV Cache Key Type",    "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "OCR",        "key": "OCR_CACHE_TYPE_V",         "label": "KV Cache Value Type",  "type": "select", "options": LLAMA_KV_CACHE_OPTIONS},
    {"section": "OCR",        "key": "OCR_NO_MMAP",              "label": "Disable mmap",         "type": "select", "options": ["false", "true"]},
    {"section": "OCR",        "key": "OCR_MLOCK",                "label": "Lock Memory",          "type": "select", "options": ["false", "true"]},
    {"section": "OCR",        "key": "OCR_GPU_VISIBLE_DEVICES",  "label": "OCR GPU Devices",      "type": "text",   "hint": "CUDA_VISIBLE_DEVICES for OCR. Use 0 or 1 for one GPU, 0,1 for both GPUs."},
    {"section": "OCR",        "key": "OCR_PROMPT",               "label": "Default OCR Prompt",   "type": "text",   "hint": "Used by /api/ocr/extract when a call does not provide a prompt"},
    {"section": "OCR",        "key": "OCR_TIMEOUT_SECONDS",      "label": "Extract Timeout",      "type": "number", "hint": "How long /api/ocr/extract waits for the backend. Must cover a cold model load when the router is on."},
    {"section": "OCR",        "key": "OCR_TEMP",                 "label": "Temperature",          "type": "text",   "hint": "Low values are best for OCR"},
    {"section": "OCR",        "key": "OCR_TOP_P",                "label": "Top-P",                "type": "text"},
    {"section": "OCR",        "key": "OCR_TOP_K",                "label": "Top-K",                "type": "number"},
    {"section": "OCR",        "key": "OCR_MIN_P",                "label": "Min-P",                "type": "text"},
    {"section": "OCR",        "key": "OCR_FIT",                  "label": "Auto-Fit to VRAM",     "type": "select", "options": ["on", "off"]},
    {"section": "OCR",        "key": "OCR_CUSTOM_ARGS_JSON",     "label": "Custom Arguments",     "type": "custom_args", "hint": "Extra llama.cpp flags applied to the OCR backend"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_SDK_ENABLED",        "label": "SDK Server Enabled",   "type": "select", "options": ["on", "off"], "hint": "Runs the local self-hosted GLM-OCR SDK parser; no MaaS/cloud OCR calls"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_SDK_HOST",           "label": "SDK Listen Host",      "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_SDK_PORT",           "label": "SDK Port",             "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PUBLIC_URL",         "label": "Public OCR URL",       "type": "text", "hint": "Stable URL for other apps; defaults to the SDK server"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_SDK_LOG_LEVEL",      "label": "Log Level",            "type": "select", "options": ["DEBUG", "INFO", "WARNING", "ERROR"]},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_OCR_API_MODE",       "label": "OCR API Mode",         "type": "select", "options": ["openai", "ollama_generate"], "hint": "How the SDK calls the local OCR model backend"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_OCR_API_URL",        "label": "OCR API URL Override", "type": "text", "hint": "Optional full local OCR backend URL; leave empty to use OCR_HOST/OCR_PORT"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_OCR_REQUEST_TIMEOUT", "label": "OCR Request Timeout",  "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_OCR_CONNECT_TIMEOUT", "label": "OCR Connect Timeout",  "type": "number", "hint": "How long the SDK's startup probe keeps retrying. It is a real inference request, so with the model router this must outlast a cold model load or the SDK exits on boot."},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_OCR_RETRY_MAX_ATTEMPTS", "label": "OCR Retry Attempts", "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_OCR_RETRY_BACKOFF_BASE_SECONDS", "label": "Retry Backoff Base", "type": "text", "hint": "Read by the SDK config generator; was not previously editable here"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_OCR_RETRY_BACKOFF_MAX_SECONDS",  "label": "Retry Backoff Cap",  "type": "text", "hint": "Raise alongside retry attempts so a retry can outlast a cold model load"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_OCR_CONNECTION_POOL_SIZE", "label": "Connection Pool", "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_MAX_WORKERS",        "label": "OCR Workers",          "type": "number", "hint": "Concurrent region OCR requests to the local OCR model"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PAGE_MAXSIZE",       "label": "Page Queue Size",      "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_REGION_MAXSIZE",     "label": "Region Queue Size",    "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PAGE_MAX_TOKENS",    "label": "Max Output Tokens",    "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PAGE_TEMPERATURE",   "label": "Temperature",          "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PAGE_TOP_P",         "label": "Top-P",                "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PAGE_TOP_K",         "label": "Top-K",                "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PAGE_REPETITION_PENALTY", "label": "Repeat Penalty", "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_IMAGE_FORMAT",       "label": "Region Image Format",  "type": "select", "options": ["JPEG", "PNG", "WEBP"]},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_MIN_PIXELS",         "label": "Minimum Pixels",       "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_MAX_PIXELS",         "label": "Maximum Pixels",       "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PDF_DPI",            "label": "PDF DPI",              "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PDF_MAX_PAGES",      "label": "PDF Max Pages",        "type": "number", "hint": "Empty means no SDK-side page cap"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_LAYOUT_MODEL_DIR",   "label": "Layout Model",         "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_LAYOUT_DEVICE",      "label": "Layout Device",        "type": "text", "hint": "cpu, cuda, cuda:0, or empty for auto"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES", "label": "Layout GPUs", "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_LAYOUT_THRESHOLD",   "label": "Layout Threshold",     "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_LAYOUT_BATCH_SIZE",  "label": "Layout Batch Size",    "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_LAYOUT_WORKERS",     "label": "Layout Workers",       "type": "number"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_LAYOUT_USE_POLYGON", "label": "Polygon Crops",        "type": "select", "options": ["off", "on"]},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_OUTPUT_FORMAT",      "label": "Output Format",        "type": "select", "options": ["both", "markdown", "json"]},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_MERGE_FORMULA_NUMBERS", "label": "Merge Formula Numbers", "type": "select", "options": ["on", "off"]},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_MERGE_TEXT_BLOCKS",  "label": "Merge Text Blocks",    "type": "select", "options": ["on", "off"]},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_FORMAT_BULLET_POINTS", "label": "Format Bullets",     "type": "select", "options": ["on", "off"]},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PROMPT_TEXT",        "label": "Text Prompt",          "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PROMPT_TABLE",       "label": "Table Prompt",         "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_PROMPT_FORMULA",     "label": "Formula Prompt",       "type": "text"},
    {"section": "GLM-OCR SDK", "key": "GLMOCR_ADVANCED_CONFIG_JSON", "label": "Advanced SDK Config", "type": "text", "hint": "JSON object deep-merged into the generated SDK config"},
    # Graphiti
    {"section": "Graphiti",    "key": "GRAPHITI_PUBLIC_URL",        "label": "Public URL",           "type": "text",   "hint": "URL used by external tools (OpenWebUI, OpenClaw, etc.)"},
    {"section": "Graphiti",    "key": "GRAPHITI_HOST",              "label": "Listen Host",          "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_PORT",              "label": "Port",                 "type": "number"},
    {"section": "Graphiti",    "key": "GRAPHITI_LLM_BASE_URL",      "label": "LLM Base URL",         "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_LLM_MODEL",         "label": "LLM Model",            "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_EMBED_BASE_URL",    "label": "Embedding Base URL",   "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_EMBED_MODEL",       "label": "Embedding Model",      "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_RERANKER_PROVIDER", "label": "Reranker Provider",    "type": "select", "options": ["llamacpp", "openai"]},
    {"section": "Graphiti",    "key": "GRAPHITI_RERANKER_BASE_URL", "label": "Reranker Base URL",    "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_RERANKER_MODEL",    "label": "Reranker Model",       "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_MEMORY_MAX_QUERY_CHARS", "label": "Memory Query Max Chars", "type": "number", "hint": "Trim retrieval query text before embedding"},
    {"section": "Graphiti",    "key": "GRAPHITI_MEMORY_MAX_MESSAGES",    "label": "Memory Query Max Messages", "type": "number", "hint": "Number of latest messages used to compose retrieval query"},
    {"section": "Graphiti",    "key": "GRAPHITI_MEMORY_MAX_FACTS",       "label": "Memory Max Facts", "type": "number", "hint": "Server-side cap for /search and /get-memory facts"},
    {"section": "Graphiti",    "key": "GRAPHITI_NEO4J_URI",         "label": "Neo4j URI",            "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_NEO4J_USER",        "label": "Neo4j User",           "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_NEO4J_PASSWORD",    "label": "Neo4j Password",       "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_NEO4J_DATABASE",    "label": "Neo4j Database",       "type": "text"},
    {"section": "Graphiti",    "key": "GRAPHITI_NEO4J_BOLT_PORT",   "label": "Neo4j Bolt Port",      "type": "number"},
    {"section": "Graphiti",    "key": "GRAPHITI_NEO4J_HTTP_PORT",   "label": "Neo4j HTTP Port",      "type": "number"},
    # SearXNG
    {"section": "SearXNG",     "key": "SEARXNG_ENABLED",            "label": "Install On Stack Setup", "type": "select", "options": ["on", "off"]},
    {"section": "SearXNG",     "key": "SEARXNG_PUBLIC_URL",         "label": "Public URL",             "type": "text", "hint": "URL clients and the manager should use"},
    {"section": "SearXNG",     "key": "SEARXNG_BASE_URL",           "label": "Base URL",               "type": "text", "hint": "URL passed through to SearXNG when needed"},
    {"section": "SearXNG",     "key": "SEARXNG_URL_PATH",           "label": "Nginx URL Path",         "type": "text", "hint": "Path mounted into the default nginx server block"},
    {"section": "SearXNG",     "key": "SEARXNG_INSTANCE_NAME",      "label": "Instance Name",          "type": "text"},
    {"section": "SearXNG",     "key": "SEARXNG_SAFE_SEARCH",        "label": "Safe Search",            "type": "select", "options": ["0", "1", "2"]},
    {"section": "SearXNG",     "key": "SEARXNG_AUTOCOMPLETE",       "label": "Autocomplete",           "type": "text"},
    {"section": "SearXNG",     "key": "SEARXNG_FORMATS",            "label": "Search Formats",         "type": "text", "hint": "Comma-separated: html,json,csv,rss"},
    {"section": "SearXNG",     "key": "SEARXNG_LIMITER",            "label": "Limiter",                "type": "select", "options": ["false", "true"]},
    {"section": "SearXNG",     "key": "SEARXNG_IMAGE_PROXY",        "label": "Image Proxy",            "type": "select", "options": ["true", "false"]},
    {"section": "SearXNG",     "key": "SEARXNG_VALKEY_URL",         "label": "Valkey URL",             "type": "text"},
    {"section": "SearXNG",     "key": "SEARXNG_HOME",               "label": "Install Directory",      "type": "path"},
    {"section": "SearXNG",     "key": "SEARXNG_SETTINGS_PATH",      "label": "Settings File",          "type": "path"},
    {"section": "SearXNG",     "key": "SEARXNG_UWSGI_INI",          "label": "uWSGI Config",           "type": "path"},
    {"section": "SearXNG",     "key": "SEARXNG_UWSGI_SOCKET",       "label": "uWSGI Socket",           "type": "path"},
    {"section": "SearXNG",     "key": "SEARXNG_NGINX_CONF",         "label": "Nginx Config",           "type": "path"},
    # Playwright
    {"section": "Playwright",  "key": "PLAYWRIGHT_ENABLED",         "label": "Install On Stack Setup", "type": "select", "options": ["on", "off"]},
    {"section": "Playwright",  "key": "PLAYWRIGHT_PUBLIC_WS_URL",   "label": "Public WS URL",          "type": "text", "hint": "Use with playwright.chromium.connect(...)"},
    {"section": "Playwright",  "key": "PLAYWRIGHT_PUBLIC_HTTP_URL", "label": "Public HTTP URL",        "type": "text", "hint": "Same listener exposed as HTTP/WebSocket"},
    {"section": "Playwright",  "key": "PLAYWRIGHT_URL_PATH",        "label": "Nginx URL Path",         "type": "text", "hint": "Path mounted into the default nginx server block"},
    {"section": "Playwright",  "key": "PLAYWRIGHT_HOST",            "label": "Listen Host",            "type": "text"},
    {"section": "Playwright",  "key": "PLAYWRIGHT_PORT",            "label": "Port",                   "type": "number"},
    {"section": "Playwright",  "key": "PLAYWRIGHT_UPSTREAM_PORT",   "label": "Internal Upstream Port", "type": "number", "hint": "Loopback-only Playwright run-server port used behind the public wrapper"},
    {"section": "Playwright",  "key": "PLAYWRIGHT_BROWSER",         "label": "Browser",                "type": "select", "options": ["chromium", "firefox", "webkit"]},
    {"section": "Playwright",  "key": "PLAYWRIGHT_INSTALL_BROWSERS", "label": "Install Browser Binaries", "type": "select", "options": ["on", "off"]},
    {"section": "Playwright",  "key": "PLAYWRIGHT_BROWSERS_PATH",   "label": "Browser Cache Path",     "type": "path"},
    {"section": "Playwright",  "key": "PLAYWRIGHT_NODE_ENV",        "label": "Node Environment",       "type": "text"},
    {"section": "Playwright",  "key": "PLAYWRIGHT_NGINX_CONF",      "label": "Nginx Config",           "type": "path"},
    # Model Router. One llama-server in router mode owns the auxiliary models
    # instead of one systemd unit each, so they load on demand and evict each
    # other rather than each holding VRAM permanently. Per-model settings still
    # come from the sections above; `scripts/render-models-ini.py` turns them
    # into the preset file the router reads.
    {"section": "Model Router", "key": "MODEL_ROUTER_ENABLED",       "label": "Router Enabled",       "type": "select", "options": ["off", "on"], "hint": "Replaces the embed/ocr/rerank/task units with one on-demand router. Off leaves every service exactly as it is today."},
    {"section": "Model Router", "key": "MODEL_ROUTER_PORT",          "label": "Router Port",          "type": "number", "hint": "The router's own listener. Callers keep using the per-model ports, which nginx fronts onto this one."},
    {"section": "Model Router", "key": "MODEL_ROUTER_HOST",          "label": "Router Listen Host",   "type": "text",   "hint": "Loopback by default. nginx reaches it from this host, so exposing it on the LAN only adds a second unauthenticated way in."},
    {"section": "Model Router", "key": "MODEL_ROUTER_MAX",           "label": "Models Resident",      "type": "number", "hint": "How many models may be loaded at once. A count, not a memory budget — the router evicts least-recently-used, but does not know how large the survivors are. Use 1 for strict one-at-a-time."},
    {"section": "Model Router", "key": "MODEL_ROUTER_MEMBERS",       "label": "Pooled Models",        "type": "text",   "hint": "Derived from the per-model \u201cPooled By Router\u201d switches when any is set; otherwise read directly. Comma-separated env prefixes, e.g. EMBED,OCR,RERANK,TASK"},
    {"section": "Model Router", "key": "MODEL_ROUTER_SLEEP_IDLE_SECONDS", "label": "Idle Unload (s)", "type": "number", "hint": "Unload a resident model's weights and KV after this much idleness; the next request reloads it. -1 disables."},
    {"section": "Model Router", "key": "LLM_ABSOLUTE_GPU_INDICES", "label": "Absolute GPU Indices", "type": "select", "options": ["off", "on"], "hint": "off: each slot\u2019s Main GPU indexes into its own Visible GPUs list, so the same number means a different card under the router than as a dedicated unit. on: every index is the machine\u2019s. Switching moves models unless the stored values move too \u2014 run scripts/lib/gpu-indices.py first, which reports what would move and prints the values that keep it put."},
    {"section": "Model Router", "key": "MODEL_ROUTER_GPU_VISIBLE_DEVICES", "label": "GPUs The Router May Use", "type": "text",   "hint": "The superset of devices the router and every model it pools may touch \u2014 not a per-model choice. CUDA_VISIBLE_DEVICES is process-level and the router spawns its own children, so it cannot be set per member; place individual models with their own Main GPU / Device / Tensor Split, which use real indices within this set."},
    # The pooled audio model, served on the router's own /v1/audio/transcriptions
    # and reached by the transcription sidecar's `router` engine. Configured
    # here rather than under Transcription because it is a router child: it is
    # loaded and evicted by the same LRU as embed, ocr, rank and task, and it
    # only exists at all when ASR is listed in Pooled Models above.
    {"section": "Model Router", "key": "ASR_MODEL_NAME",         "label": "Audio Model Alias",    "type": "text",   "hint": "The models.ini section name the router serves it under, and the model string the sidecar sends. Default: asr"},
    {"section": "Model Router", "key": "ASR_MODEL_PATH",         "label": "Audio Model Path",     "type": "path",   "hint": "An audio-capable GGUF, e.g. Voxtral, Qwen3-Audio or Granite-Speech. Whisper and Parakeet are not llama.cpp models — use a local engine for those."},
    {"section": "Model Router", "key": "ASR_MMPROJ_PATH",        "label": "Audio MMProj Path",    "type": "path",   "hint": "Required, not optional: llama.cpp refuses transcription unless the model carries an audio projector"},
    {"section": "Model Router", "key": "ASR_CTX_SIZE",           "label": "Audio Context Size",   "type": "number"},
    {"section": "Model Router", "key": "ASR_N_GPU_LAYERS",       "label": "Audio GPU Layers",     "type": "number"},
    {"section": "Model Router", "key": "ASR_MAIN_GPU",           "label": "Audio Main GPU",       "type": "number"},
    {"section": "Model Router", "key": "ASR_TENSOR_SPLIT",       "label": "Audio Tensor Split",   "type": "text"},
    {"section": "Model Router", "key": "ASR_SPLIT_MODE",         "label": "Audio Split Mode",     "type": "select", "options": LLAMA_SPLIT_MODE_OPTIONS, "hint": LLAMA_SPLIT_MODE_HINT},
    {"section": "Model Router", "key": "ASR_FLASH_ATTN",         "label": "Audio Flash Attention","type": "select", "options": ["on", "off", "auto"]},
    {"section": "Model Router", "key": "ASR_MMPROJ_OFFLOAD",     "label": "Audio MMProj Offload", "type": "select", "options": ["on", "off"]},
    {"section": "Model Router", "key": "ASR_JINJA",              "label": "Audio Native Templates","type": "select","options": ["on", "off"], "hint": "Leave on: llama.cpp builds the ASR prompt from the model's chat template"},
    {"section": "Model Router", "key": "ASR_GPU_VISIBLE_DEVICES","label": "Audio GPU Devices",    "type": "text"},
    {"section": "Model Router", "key": "ASR_CUSTOM_ARGS_JSON",   "label": "Audio Custom Arguments","type": "custom_args"},
    # Ports
    {"section": "Ports",       "key": "THINK_PORT",                 "label": "Thinking Port",        "type": "number"},
    {"section": "Ports",       "key": "NOTHINK_PORT",               "label": "Chat Port",            "type": "number"},
    {"section": "Ports",       "key": "CODE_PORT",                  "label": "Code Port",            "type": "number"},
    {"section": "Ports",       "key": "AGGREGATE_ENABLED",          "label": "Aggregate Proxy",      "type": "select", "options": ["on", "off"], "hint": "Single model-routed endpoint exposing think, chat, and code"},
    {"section": "Ports",       "key": "AGGREGATE_PORT",             "label": "Aggregate Port",       "type": "number"},
    {"section": "Ports",       "key": "EMBED_PORT",                 "label": "Embedding Port",       "type": "number"},
    {"section": "Ports",       "key": "RERANK_PORT",                "label": "Reranker Port",        "type": "number"},
    {"section": "Ports",       "key": "TASK_PORT",                  "label": "Task Port",            "type": "number"},
    {"section": "Ports",       "key": "CHAT_BACKEND_PORT",          "label": "Backend Port",         "type": "number"},
    {"section": "Ports",       "key": "CHAT_BACKEND_HOST",          "label": "Backend Host",         "type": "text"},
    {"section": "Ports",       "key": "LISTEN_HOST",                "label": "Listen Host",          "type": "select", "options": ["0.0.0.0", "127.0.0.1"]},
    # Read-only state API. Its own section rather than more Ports entries,
    # because the host and token together decide who can read the stack's state
    # and that is a decision worth presenting as one.
    {"section": "State API",   "key": "LLM_API_ENABLED",            "label": "Enable State API",     "type": "select", "options": ["on", "off"], "hint": "Read-only live state for other applications: GPU, VRAM per model, context, health. Takes effect when the manager restarts"},
    {"section": "State API",   "key": "LLM_API_HOST",               "label": "Listen Host",          "type": "text",   "hint": "127.0.0.1 keeps it on this box; 0.0.0.0 for the LAN; a Tailscale IP for the tailnet only. Takes effect when the manager restarts"},
    {"section": "State API",   "key": "LLM_API_PORT",               "label": "Port",                 "type": "number", "hint": "Separate from the manager's own port, which is not safe to expose. Takes effect when the manager restarts"},
    {"section": "State API",   "key": "LLM_API_TOKEN",              "label": "Access Token",         "type": "text",   "hint": "Blank means no authentication. Set it and requests need Authorization: Bearer <token>"},
    {"section": "State API",   "key": "LLM_API_ALLOW_ORIGINS",      "label": "CORS Origins",         "type": "text",   "hint": "Comma-separated origins allowed to read this from a browser page; blank sends no CORS headers"},
    {"section": "State API",   "key": "LLM_API_STREAM_INTERVAL",    "label": "Stream Interval",      "type": "number", "hint": "Seconds between event-stream collections (1-60). One collector serves every connected client"},
    {"section": "State API",   "key": "LLM_API_WEBHOOK_URL",        "label": "Webhook URL",          "type": "text",   "hint": "Optional: POST service-state and alert transitions here instead of polling"},
    {"section": "State API",   "key": "LLM_API_WEBHOOK_EVENTS",     "label": "Webhook Events",       "type": "text",   "hint": "Comma-separated: service_state, alert"},
    # The write half, on its own port and its own credential. Two tokens rather
    # than one because reading the stack's state and stopping its backends are
    # different blast radii and must be independently rotatable.
    {"section": "Control API", "key": "LLM_CONTROL_ENABLED",        "label": "Enable Control API",   "type": "select", "options": ["off", "on"], "hint": "Lets another machine edit this host's configuration and start or stop its services. Off unless you want that. Takes effect when the manager restarts"},
    {"section": "Control API", "key": "LLM_CONTROL_HOST",           "label": "Listen Host",          "type": "text",   "hint": "127.0.0.1 keeps it on this box; a Tailscale IP exposes it to the tailnet. It refuses to bind off-box with no token set"},
    {"section": "Control API", "key": "LLM_CONTROL_PORT",           "label": "Port",                 "type": "number", "hint": "Separate from both the manager and the state API. Takes effect when the manager restarts"},
    {"section": "Control API", "key": "LLM_CONTROL_TOKEN",          "label": "Access Token",         "type": "text",   "hint": "Required. Every request must send Authorization: Bearer <token>; blank means every request is refused. Use a different value from the State API token"},
    {"section": "Control API", "key": "LLM_CONTROL_ALLOW_SECRETS",  "label": "Allow Writing Secrets", "type": "select", "options": ["off", "on"], "hint": "Whether a remote controller may set API keys and tokens on this host. Secrets are never readable back either way"},
    # pi-forge integration. Slot scheduling is coordinated through lease files
    # in pi-forge's own agent directory; the manager reads them to verify the
    # contract, and only writes there when explicitly allowed to.
    {"section": "LLM A", "key": "PI_FORGE_AGENT_DIR",     "label": "pi-forge Agent Directory", "type": "text",   "hint": "Holds inference-leases/. Blank uses the stack owner's ~/.pi-forge/agent"},
    {"section": "LLM A", "key": "PI_FORGE_LEASE_REAP",    "label": "Reap Orphaned Leases",     "type": "select", "options": ["off", "on"], "hint": "Periodically delete leases whose writing process is gone; the panel's Reap button does this on request either way"},
    # TTS Gateway
    {"section": "TTS Gateway", "key": "TTS_PUBLIC_URL",             "label": "Public TTS URL",       "type": "text",   "hint": "Stable URL clients should use"},
    {"section": "TTS Gateway", "key": "TTS_GATEWAY_HOST",           "label": "Gateway Listen Host",  "type": "text"},
    {"section": "TTS Gateway", "key": "TTS_GATEWAY_PORT",           "label": "Gateway Port",         "type": "number"},
    {"section": "TTS Gateway", "key": "TTS_SINGLE_ACTIVE",          "label": "Single Active Backend","type": "select", "options": ["on", "off"], "hint": "When on, activating one backend stops the others"},
    {"section": "TTS Gateway", "key": "TTS_DEFAULT_FORMAT",         "label": "Default Audio Format", "type": "select", "options": ["mp3", "wav", "flac", "opus", "aac", "pcm"]},
    {"section": "TTS Backends","key": "KOKORO_UPSTREAM_URL",        "label": "Kokoro Upstream URL",  "type": "text",   "hint": "Local HTTP runtime exposing /v1/audio/speech"},
    {"section": "TTS Backends","key": "KOKORO_LAUNCH_CMD",          "label": "Kokoro Launch Command","type": "text",   "hint": "Optional command for a local Kokoro runtime"},
    {"section": "TTS Backends","key": "KOKORO_VOICES",              "label": "Kokoro Voices",        "type": "text",   "hint": "Comma-separated voice ids"},
    {"section": "TTS Backends","key": "CHATTERBOX_UPSTREAM_URL",    "label": "Chatterbox Upstream URL","type": "text", "hint": "Local HTTP runtime exposing /v1/audio/speech"},
    {"section": "TTS Backends","key": "CHATTERBOX_LAUNCH_CMD",      "label": "Chatterbox Launch Command","type": "text","hint": "Optional command for a local Chatterbox runtime"},
    {"section": "TTS Backends","key": "CHATTERBOX_VOICES",          "label": "Chatterbox Voices",    "type": "text",   "hint": "Comma-separated voice ids"},
    {"section": "TTS Backends","key": "VIBEVOICE_UPSTREAM_URL",     "label": "VibeVoice Upstream URL","type": "text",  "hint": "Local HTTP runtime exposing /v1/audio/speech"},
    {"section": "TTS Backends","key": "VIBEVOICE_LAUNCH_CMD",       "label": "VibeVoice Launch Command","type": "text", "hint": "Optional command for a local VibeVoice runtime"},
    {"section": "TTS Backends","key": "VIBEVOICE_VOICES",           "label": "VibeVoice Voices",     "type": "text",   "hint": "Comma-separated voice ids"},
    {"section": "TTS Backends","key": "VIBEVOICE_MODEL_PATH",       "label": "VibeVoice Model Path", "type": "text",   "hint": "HF model id or local model directory"},
    {"section": "TTS Backends","key": "VIBEVOICE_DEVICE",           "label": "VibeVoice Device",     "type": "select", "options": ["cuda", "cpu", "mps"]},
    {"section": "TTS Backends","key": "VIBEVOICE_RUNTIME_HOST",     "label": "VibeVoice Runtime Host","type": "text"},
    {"section": "TTS Backends","key": "VIBEVOICE_RUNTIME_PORT",     "label": "VibeVoice Runtime Port","type": "number"},
    {"section": "TTS Backends","key": "VIBEVOICE_CFG_SCALE",        "label": "VibeVoice CFG Scale",  "type": "text"},
    {"section": "TTS Backends","key": "VIBEVOICE_DDPM_STEPS",       "label": "VibeVoice DDPM Steps", "type": "number"},
    # Transcription. The gateway keys are written out; the per-engine ones are
    # generated from TRANSCRIPTION_ENGINES below, because five engines times
    # twelve keys is sixty fields that differ only in their prefix.
    {"section": "Transcription", "key": "TRANSCRIPT_ENABLED",           "label": "Enable Transcription",   "type": "select", "options": ["off", "on"], "hint": "Off means the start script exits without launching, so the unit being down is not a fault"},
    {"section": "Transcription", "key": "TRANSCRIPT_PUBLIC_URL",        "label": "Public Transcript URL",   "type": "text",   "hint": "Stable URL clients should use"},
    {"section": "Transcription", "key": "TRANSCRIPT_HOST",              "label": "Listen Host",            "type": "text",   "hint": "A Tailscale IP keeps it on the tailnet; 0.0.0.0 exposes it to the LAN"},
    {"section": "Transcription", "key": "TRANSCRIPT_PORT",              "label": "Port",                   "type": "number"},
    {"section": "Transcription", "key": "TRANSCRIPT_API_TOKEN",         "label": "Access Token",           "type": "text",   "hint": "Blank means no authentication. Set it and requests need Authorization: Bearer <token>"},
    {"section": "Transcription", "key": "TRANSCRIPT_ACTIVE_ENGINE",     "label": "Default Engine",         "type": "select", "options": list(TRANSCRIPTION_ENGINE_IDS), "hint": "Used when a request names no engine"},
    {"section": "Transcription", "key": "TRANSCRIPT_ENGINES",           "label": "Installed Engines",      "type": "text",   "hint": "Comma-separated --engines tokens for scripts/install-transcribe.sh"},
    {"section": "Apple Silicon (MLX)", "key": "EMBED_ENGINE",      "label": "Embedding Server",   "type": "select", "options": ["llamacpp", "mlx"], "hint": "Which process serves the embedding slot. mlx is Apple silicon only"},
    {"section": "Apple Silicon (MLX)", "key": "TRANSCRIPT_ENGINE",  "label": "Transcription Server", "type": "select", "options": ["sidecar", "parakeet-mlx"], "hint": "Which server runs. Distinct from Active Engine above, which picks the runtime *inside* the sidecar"},
    {"section": "Apple Silicon (MLX)", "key": "MLX_RUNTIME_VENV",   "label": "Runtime Venv",       "type": "path",   "hint": "Kept apart from the manager's own venv, which depends on nothing but Flask"},
    {"section": "Apple Silicon (MLX)", "key": "MLX_RUNTIME_PYTHON", "label": "Python",             "type": "text",   "hint": "Interpreter used to create the runtime venv"},
    {"section": "Apple Silicon (MLX)", "key": "MLX_HF_HOME",        "label": "HuggingFace Cache",  "type": "path",   "hint": "HF_HOME for the MLX services"},
    {"section": "Apple Silicon (MLX)", "key": "MLX_EMBED_MODEL_PATH",    "label": "Embedding Model",   "type": "path", "hint": "Local MLX model directory; revision pinned in config/mlx-models.lock.json"},
    {"section": "Apple Silicon (MLX)", "key": "MLX_PARAKEET_MODEL_PATH", "label": "Parakeet Model",    "type": "path", "hint": "Local MLX model directory; revision pinned in config/mlx-models.lock.json"},
    {"section": "Apple Silicon (MLX)", "key": "MLX_PARAKEET_MODEL_NAME", "label": "Parakeet Alias",    "type": "text", "hint": "The `model` value this endpoint answers to"},
    {"section": "Apple Silicon (MLX)", "key": "MLX_PARAKEET_CHUNK_SECONDS",   "label": "Chunk Seconds",   "type": "number", "hint": "Chunk length; keeps peak memory predictable on a small unified-memory machine"},
    {"section": "Apple Silicon (MLX)", "key": "MLX_PARAKEET_OVERLAP_SECONDS", "label": "Overlap Seconds", "type": "number", "hint": "Overlap between chunks, so a word spanning a boundary is not lost"},
    {"section": "Transcription", "key": "TRANSCRIPT_TIMEOUT_SECONDS",   "label": "Request Timeout (sec)",  "type": "number", "hint": "Must cover a cold model load, not just the decode"},
    {"section": "Transcription", "key": "TRANSCRIPT_LOCAL_DEVICE",      "label": "Local Device",           "type": "select", "options": ["cuda", "cpu"]},
    {"section": "Transcription", "key": "TRANSCRIPT_LOCAL_COMPUTE_TYPE","label": "Local Compute Type",     "type": "select", "options": ["float16", "int8", "int8_float16", "float32"]},
    {"section": "Transcription", "key": "TRANSCRIPT_IDLE_UNLOAD_SECONDS","label": "Idle Unload (sec)",     "type": "number", "hint": "Free the model's VRAM after this long unused. 0 or -1 keeps it resident"},
    {"section": "Transcription", "key": "TRANSCRIPT_ROUTER_YIELD",      "label": "Yield to Model Router",  "type": "select", "options": ["asr", "all", "off"], "hint": "Unload router models before loading locally so the two never stack. asr drops only the pooled audio model; all drops every resident one"},
    {"section": "Transcription", "key": "TRANSCRIPT_ROUTER_ALLOW_DEGRADED", "label": "Allow Degraded Router Output", "type": "select", "options": ["off", "on"], "hint": "The router engine returns no timestamps. Off refuses srt/vtt/verbose_json rather than emitting a fabricated timeline"},
    {"section": "Transcription", "key": "TRANSCRIPT_MAX_CONCURRENCY",   "label": "Concurrent Decodes",     "type": "number", "hint": "1 is correct for a single resident model; NeMo is not thread-safe"},
    {"section": "Transcription", "key": "TRANSCRIPT_NEMO_CHUNK_SECONDS", "label": "NeMo Window (sec)",     "type": "number", "hint": "NeMo buffers whatever it is handed in host RAM, so long audio is decoded in windows this size. Shorter windows use less VRAM. 0 disables windowing"},
    {"section": "Transcription", "key": "TRANSCRIPT_MAX_VRAM_MB",       "label": "VRAM Budget (MB)",      "type": "number", "hint": "Hard ceiling on this process's GPU memory, enforced by the allocator: exceeding it fails the request rather than taking VRAM the router needs. Includes ~300MB of CUDA context. 0 means no limit"},
    {"section": "Transcription", "key": "TRANSCRIPT_MAX_UPLOAD_MB",     "label": "Max Upload (MB)",        "type": "number"},
    {"section": "Transcription", "key": "TRANSCRIPT_ASYNC_THRESHOLD_SECONDS", "label": "Async Threshold (sec)", "type": "number", "hint": "Audio longer than this returns a job id from /transcribe instead of blocking"},
    {"section": "Transcription", "key": "TRANSCRIPT_JOB_TTL_SECONDS",   "label": "Job Retention (sec)",    "type": "number"},
    {"section": "Transcription", "key": "TRANSCRIPT_URL_ALLOW_HOSTS",   "label": "URL Fetch Allow-List",   "type": "text",   "hint": "Comma-separated hosts /transcribe may fetch audio from. Blank denies all, because this process can reach the whole tailnet"},
    {"section": "Transcription", "key": "TRANSCRIPT_DEFAULT_FORMAT",    "label": "Default Response Format","type": "select", "options": ["json", "verbose_json", "text", "srt", "vtt", "markdown"]},
    {"section": "Transcription", "key": "TRANSCRIPT_LOG_LEVEL",         "label": "Log Level",              "type": "select", "options": ["INFO", "DEBUG", "WARNING", "ERROR"]},
]


#: The placement controls every llama.cpp member takes, and the section each one
#: is filed under. A table rather than a per-member copy, and rather than
#: `_clone_chat_backend_field`, whose `secondary` branch is dead code that would
#: emit `CHAT_SECONDARY_*` where the rest of the tree says `LLM_B_*`.
#:
#: `render-models-ini.py` already maps MAIN_GPU and DEVICE per member. The
#: config surface did not: EMBED and RERANK had neither field, and ASR had
#: MAIN_GPU without DEVICE -- so the router could place those models and nobody
#: could say where. Three gaps in one table, which is how a table earns its keep.
MEMBER_PLACEMENT_SECTIONS = {
    "EMBED": ("Embedding", "Embedding"),
    "RERANK": ("Reranker", "Reranker"),
    "ASR": ("Model Router", "Audio"),
}


def _member_pooling_fields() -> list[dict]:
    """`<MEMBER>_POOLED`, the question an operator actually has.

    "Should embed hold VRAM permanently or load on demand" beats editing a
    comma-separated list and remembering that ASR is in the member table but not
    in the default string. `MODEL_ROUTER_MEMBERS` remains readable forever and
    is what the string-shaped consumers still see.
    """
    from backends import router  # noqa: PLC0415
    sections = {"EMBED": "Embedding", "RERANK": "Reranker", "TASK": "Task Model",
                "OCR": "OCR", "ASR": "Model Router"}
    labels = {"EMBED": "Embedding", "RERANK": "Reranker", "TASK": "Task",
              "OCR": "OCR", "ASR": "Audio"}
    fields = []
    for prefix in router.MEMBER_PREFIXES:
        fields.append({
            "section": sections[prefix], "key": router.pooled_key(prefix),
            "label": f"{labels[prefix]} Pooled By Router", "type": "select",
            "options": ["inherit", "on", "off"],
            "hint": ("inherit: follow MODEL_ROUTER_MEMBERS, which is what every "
                     "host did before these existed. "
                     "on: the router owns this model and loads it on demand, "
                     "sharing one VRAM budget with the other pooled models. "
                     "off: it runs as its own unit and holds VRAM from the "
                     "moment it starts. Setting any of these switches makes "
                     "them the source for MODEL_ROUTER_MEMBERS."),
        })
    return fields


def _member_placement_fields() -> list[dict]:
    """The `_MAIN_GPU` and `_DEVICE` controls members were missing.

    Only the two suffixes that were absent. TENSOR_SPLIT and SPLIT_MODE are
    already declared per member with their own hints, and moving them here would
    reorder the config page for no gain.
    """
    fields = []
    for prefix, (section, label) in MEMBER_PLACEMENT_SECTIONS.items():
        existing = {f.get("key") for f in CONFIG_FIELDS}
        for suffix, kind, hint in (
            ("MAIN_GPU", "number",
             "Which GPU index holds this model. Ignored when Split Mode is "
             "`tensor`, which spreads every tensor across the devices instead."),
            ("DEVICE", "text",
             "Explicit llama.cpp device list, e.g. CUDA0,CUDA1. Overrides Main "
             "GPU; leave empty to let Main GPU and Tensor Split decide."),
        ):
            key = f"{prefix}_{suffix}"
            if key in existing:
                continue
            fields.append({
                "section": section, "key": key,
                "label": f"{label} {'Main GPU' if suffix == 'MAIN_GPU' else 'Device'}",
                "type": kind, "hint": hint,
            })
    return fields


def _transcription_engine_fields() -> list[dict]:
    """One identical block of config per engine, keyed on its env prefix.

    Written as a generator for the same reason `_clone_chat_backend_field`
    exists: the per-engine keys differ only in their prefix, and a hand-written
    copy per engine is how one of them comes to be missing a field.
    """
    fields = []
    for engine in TRANSCRIPTION_ENGINES:
        prefix, label = engine["env_prefix"], engine["label"]
        is_router = engine["runtime"] == "router"
        fields.extend([
            {"section": "Transcription", "key": f"{prefix}_BACKEND_TYPE", "label": f"{label} Backend Type",
             "type": "select", "options": ["local", "upstream"],
             "hint": "local runs the model here; upstream forwards to another OpenAI-compatible server"},
            {"section": "Transcription", "key": f"{prefix}_LOCAL_MODEL", "label": f"{label} Local Model",
             "type": "transcript_model", "engine_id": engine["id"],
             "hint": ("The model alias the router serves, normally asr" if is_router
                      else f"Model used when {label} backend type is local")},
            {"section": "Transcription", "key": f"{prefix}_UPSTREAM_URL", "label": f"{label} Upstream URL",
             "type": "text", "hint": "Upstream OpenAI-compatible transcription endpoint host"},
            {"section": "Transcription", "key": f"{prefix}_MODEL", "label": f"{label} Model Name",
             "type": "text", "hint": "Model string sent upstream"},
            {"section": "Transcription", "key": f"{prefix}_API_KEY", "label": f"{label} API Key", "type": "text"},
            {"section": "Transcription", "key": f"{prefix}_TRANSCRIBE_PATH", "label": f"{label} Path",
             "type": "text", "hint": "Default: /v1/audio/transcriptions"},
            {"section": "Transcription", "key": f"{prefix}_STREAM_OUTPUT_ENABLED", "label": f"{label} Streaming Output",
             "type": "select", "options": ["off", "on"]},
            {"section": "Transcription", "key": f"{prefix}_STREAM_OUTPUT_TARGET", "label": f"{label} Stream Target",
             "type": "text", "hint": "Future scaffold: webhook/SSE/WebSocket destination"},
            {"section": "Transcription", "key": f"{prefix}_STREAM_OUTPUT_FORMAT", "label": f"{label} Stream Format",
             "type": "select", "options": ["webhook", "sse", "websocket"]},
            {"section": "Transcription", "key": f"{prefix}_SPEAKER_DETECTION", "label": f"{label} Speaker Detection",
             "type": "select", "options": ["off", "on"]},
            {"section": "Transcription", "key": f"{prefix}_SPEAKER_MODE", "label": f"{label} Speaker Mode",
             "type": "select", "options": ["auto", "fixed"]},
            {"section": "Transcription", "key": f"{prefix}_SPEAKER_COUNT", "label": f"{label} Speaker Count",
             "type": "number"},
        ])
    return fields


CONFIG_FIELDS.extend(_member_pooling_fields())
CONFIG_FIELDS.extend(_member_placement_fields())
CONFIG_FIELDS.extend(_transcription_engine_fields())

CHAT_BACKEND_IDENTITY_KEYS = {
    "primary": {
        "CHAT_DENSE_LABEL": "LLM_A_LABEL",
        "CHAT_DENSE_MODEL_NAME": "LLM_A_MODEL_NAME",
        "CHAT_DENSE_MODEL_PATH": "LLM_A_MODEL_PATH",
        "CHAT_DENSE_MMPROJ_PATH": "LLM_A_MMPROJ_PATH",
        "CHAT_DENSE_CTX_SIZE": "LLM_A_CTX_SIZE",
    },
}
CHAT_BACKEND_GENERIC_SKIP_KEYS = {
    "CHAT_MODEL_NAME",
    "CHAT_DENSE_LABEL",
    "CHAT_DENSE_MODEL_NAME",
    "CHAT_DENSE_MODEL_PATH",
    "CHAT_DENSE_MMPROJ_PATH",
    "CHAT_DENSE_CTX_SIZE",
    "CHAT_MOE_LABEL",
    "CHAT_MOE_MODEL_NAME",
    "CHAT_MOE_MODEL_PATH",
    "CHAT_MOE_MMPROJ_PATH",
    "CHAT_MOE_CTX_SIZE",
}


def _clone_chat_backend_field(field: dict, variant: str) -> dict | None:
    key = field.get("key", "")
    identity_key = CHAT_BACKEND_IDENTITY_KEYS[variant].get(key)
    if identity_key:
        cloned = dict(field)
        cloned["key"] = identity_key
        if variant == "primary":
            cloned["section"] = "LLM A"
            cloned["label"] = cloned.get("label", "").replace("Dense", "Primary").replace("Slot", "Backend")
            cloned["hint"] = cloned.get("hint", "").replace("dense preset", "primary backend")
        else:
            cloned["section"] = "LLM B"
            cloned["label"] = cloned.get("label", "").replace("MoE", "Secondary").replace("Slot", "Backend")
            cloned["hint"] = cloned.get("hint", "").replace("MoE preset", "secondary backend")
        return cloned
    if not key.startswith("CHAT_") or key in CHAT_BACKEND_GENERIC_SKIP_KEYS:
        return None
    cloned = dict(field)
    cloned["section"] = "LLM A" if variant == "primary" else "LLM B"
    cloned["key"] = ("LLM_A" if variant == "primary" else "CHAT_SECONDARY") + key[len("CHAT"):]
    return cloned


_shared_backend_fields = [field for field in CONFIG_FIELDS if field.get("section") == "Shared Backend"]
_generated_backend_fields = []
for _variant in ("primary",):
    for _field in _shared_backend_fields:
        _cloned = _clone_chat_backend_field(_field, _variant)
        if _cloned is not None:
            _generated_backend_fields.append(_cloned)

_rebuilt_config_fields = []
_inserted_backend_fields = False
for _field in CONFIG_FIELDS:
    if _field.get("section") == "Shared Backend":
        if not _inserted_backend_fields:
            _rebuilt_config_fields.extend(_generated_backend_fields)
            _inserted_backend_fields = True
    else:
        _rebuilt_config_fields.append(_field)
CONFIG_FIELDS = _rebuilt_config_fields

# One backend is shared by the think/chat/code endpoints, so a change to a
# shared setting reaches every unit that might be hosting it.
SHARED_CHAT_BACKEND_RESTART = [slot.name for slot in backends.SLOTS.values()
                               if slot.group == "chat"]


# Which platform capability a setting needs in order to do anything. Matched on
# the key's suffix, because the same setting exists under eight prefixes and
# listing twenty-seven keys would be a list to forget to update.
#
# `*_DEVICE` is deliberately absent. It is not inert on a Metal build -- `MTL0`
# is a real answer there -- it is only the shipped `CUDA0` default that is
# wrong, and that is a different problem from a control with nothing behind it.
CAPABILITY_BY_SUFFIX = {
    "_GPU_VISIBLE_DEVICES": "gpu_visible_devices",
    "_MAIN_GPU": "gpu_indices",
    "_TENSOR_SPLIT": "gpu_indices",
    "_SPLIT_MODE": "split_modes",
}


def applicable_fields(inert: dict[str, str] | None = None,
                      fields: list[dict] | None = None) -> tuple[list[dict], dict[str, str]]:
    """(the fields this host can act on, {omitted key: why}).

    The Mac renders `LLM_A_GPU_VISIBLE_DEVICES`, `_MAIN_GPU`,
    `_TENSOR_SPLIT` and `_SPLIT_MODE` today -- twenty-seven controls across
    seven sections -- and every one of them does nothing:
    `CUDA_VISIBLE_DEVICES` is not read by a Metal build, and
    `resolve_split_opts` collapses the placement flags before they reach
    llama-server. The operator sets one, the UI says saved, the backend starts,
    and the setting is not in the command line.

    This filters what is *rendered*. It deliberately does not filter what is
    *written*: `filter_config_updates` still accepts these keys, so a config
    file carrying them keeps them and a host that can act on them still reads
    them. Hiding a control is a statement about this machine; deleting the
    value would be a statement about every machine that shares the file.
    """
    if inert is None:
        inert = platforms.active().inert_config_capabilities
    if fields is None:
        fields = CONFIG_FIELDS
    if not inert:
        return list(fields), {}

    keep, omitted = [], {}
    for field in fields:
        key = field.get("key", "")
        reason = next((inert[capability]
                       for suffix, capability in CAPABILITY_BY_SUFFIX.items()
                       if key.endswith(suffix) and capability in inert), "")
        if reason:
            omitted[key] = reason
        else:
            keep.append(field)
    return keep, omitted

# Which services should be restarted after changing a given config key
RESTART_HINTS = {
    # MLX. The engine keys change which *launcher* a service runs, so they
    # need the installer re-run to regenerate the unit, not just a restart --
    # reported as a restart of the affected service so the UI at least names it.
    "EMBED_ENGINE":                 ["embed"],
    "TRANSCRIPT_ENGINE":            ["transcript-backend"],
    "MLX_RUNTIME_VENV":             ["embed", "transcript-backend"],
    "MLX_RUNTIME_PYTHON":           ["embed", "transcript-backend"],
    "MLX_HF_HOME":                  ["embed", "transcript-backend"],
    "MLX_EMBED_MODEL_PATH":         ["embed"],
    "MLX_PARAKEET_MODEL_PATH":      ["transcript-backend"],
    "MLX_PARAKEET_MODEL_NAME":      ["transcript-backend"],
    "MLX_PARAKEET_CHUNK_SECONDS":   ["transcript-backend"],
    "MLX_PARAKEET_OVERLAP_SECONDS": ["transcript-backend"],
    "CHAT_MODEL_NAME":           ["llm-a"],
    "CHAT_N_PARALLEL":           ["llm-a"],
    "CHAT_THREADS":              ["llm-a"],
    "CHAT_THREADS_BATCH":        ["llm-a"],
    "CHAT_N_GPU_LAYERS":         ["llm-a"],
    "CHAT_MAIN_GPU":             ["llm-a"],
    "CHAT_DEVICE":               ["llm-a"],
    "CHAT_TENSOR_SPLIT":         ["llm-a"],
    "CHAT_SPLIT_MODE":           ["llm-a"],
    "CHAT_KV_OFFLOAD":           ["llm-a"],
    "CHAT_OP_OFFLOAD":           ["llm-a"],
    "CHAT_MMPROJ_OFFLOAD":       ["llm-a"],
    "CHAT_FLASH_ATTN":           ["llm-a"],
    "CHAT_CACHE_TYPE_K":         ["llm-a"],
    "CHAT_CACHE_TYPE_V":         ["llm-a"],
    "CHAT_BATCH_SIZE":           ["llm-a"],
    "CHAT_UBATCH_SIZE":          ["llm-a"],
    "CHAT_METRICS":              ["llm-a"],
    "CHAT_NO_MMAP":              ["llm-a"],
    "CHAT_MLOCK":                ["llm-a"],
    "CHAT_GPU_VISIBLE_DEVICES":  ["llm-a"],
    "CHAT_TEMP":                 ["llm-a"],
    "CHAT_TOP_P":                ["llm-a"],
    "CHAT_TOP_K":                ["llm-a"],
    "CHAT_MIN_P":                ["llm-a"],
    "CHAT_PRESERVE_THINKING":    ["llm-a"],
    "CHAT_REASONING_EFFORT":     ["llm-a"],
    "CHAT_JINJA":                ["llm-a"],
    "CHAT_REASONING_FORMAT":     ["llm-a"],
    "CHAT_FIT":                  ["llm-a"],
    "CHAT_SPEC_METHOD":          ["llm-a"],
    "CHAT_SPEC_NGRAM_MOD":       ["llm-a"],
    "CHAT_SPEC_DRAFT_MODEL_PATH": ["llm-a"],
    "CHAT_SPEC_DRAFT_N_GPU_LAYERS": ["llm-a"],
    "CHAT_SPEC_DRAFT_DEVICES":   ["llm-a"],
    "CHAT_SPEC_DRAFT_N_MAX":     ["llm-a"],
    "CHAT_SPEC_DRAFT_N_MIN":     ["llm-a"],
    "CHAT_SPEC_DRAFT_P_MIN":     ["llm-a"],
    "CHAT_SPEC_DRAFT_P_SPLIT":   ["llm-a"],
    "CHAT_SPEC_NGRAM_MOD_N_MATCH": ["llm-a"],
    "CHAT_SPEC_NGRAM_MOD_N_MIN": ["llm-a"],
    "CHAT_SPEC_NGRAM_MOD_N_MAX": ["llm-a"],
    "CHAT_CACHE_RAM":            ["llm-a"],
    "CHAT_CTX_CHECKPOINTS":      ["llm-a"],
    "CHAT_SWA_FULL":             ["llm-a"],
    "CHAT_CUSTOM_ARGS_JSON":     ["llm-a"],
    "CHAT_TEMPLATE_ID":           ["llm-a"],
    "CHAT_BACKEND_HOST":         ["llm-a-proxy"],
    "CHAT_BACKEND_PORT":         ["llm-a-proxy"],
    "PROXY_STREAM_PASSTHROUGH":  ["llm-a-proxy"],
    "UPSTREAM_400_CAPTURE_ENABLED": ["llm-a-proxy"],
    "LLM_B_CACHE_RAM":           ["llm-b"],
    "LLM_B_CTX_CHECKPOINTS":     ["llm-b"],
    "LLM_B_SWA_FULL":            ["llm-b"],
    "LLM_B_CUSTOM_ARGS_JSON":    ["llm-b"],
    "CODE_THINKING":             ["llm-a-proxy"],
    "CODE_PRESERVE_THINKING":    ["llm-a-proxy"],
    "CODE_REASONING_EFFORT":     ["llm-a-proxy"],
    "CODE_REASONING_STREAM_MODE": ["llm-a-proxy"],
    "CODE_JINJA":                ["llm-a-proxy"],
    "CODE_CTX_SIZE":             ["llm-a-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_N_PARALLEL":           ["llm-a-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_THREADS":              ["llm-a-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_THREADS_BATCH":        ["llm-a-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_N_GPU_LAYERS":         ["llm-a-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_TENSOR_SPLIT":         ["llm-a-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_SPLIT_MODE":           ["llm-a-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_FLASH_ATTN":           ["llm-a-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_CACHE_TYPE_K":         ["llm-a-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_CACHE_TYPE_V":         ["llm-a-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_BATCH_SIZE":           ["llm-a-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_UBATCH_SIZE":          ["llm-a-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_NO_MMAP":              ["llm-a-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_MLOCK":                ["llm-a-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_GPU_VISIBLE_DEVICES":  ["llm-a-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_TEMP":                 ["llm-a-proxy"],
    "CODE_MAX_TOKENS":           ["llm-a-proxy"],
    "CODE_TOP_P":                ["llm-a-proxy"],
    "CODE_TOP_K":                ["llm-a-proxy"],
    "CODE_MIN_P":                ["llm-a-proxy"],
    "CODE_PRESENCE_PENALTY":     ["llm-a-proxy"],
    "CODE_REPEAT_PENALTY":       ["llm-a-proxy"],
    "CODE_REASONING_FORMAT":     ["llm-a-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "CODE_FIT":                  ["llm-a-proxy"] + SHARED_CHAT_BACKEND_RESTART,
    "THINK_MODEL_NAME":          ["llm-a-proxy"],
    "THINK_PRESERVE_THINKING":   ["llm-a-proxy"],
    "THINK_REASONING_EFFORT":    ["llm-a-proxy"],
    "THINK_REASONING_STREAM_MODE": ["llm-a-proxy"],
    "THINK_JINJA":               ["llm-a-proxy"],
    "THINK_TEMP":                ["llm-a-proxy"],
    "THINK_MAX_TOKENS":          ["llm-a-proxy"],
    "THINK_TOP_P":               ["llm-a-proxy"],
    "THINK_TOP_K":               ["llm-a-proxy"],
    "THINK_MIN_P":               ["llm-a-proxy"],
    "THINK_PRESENCE_PENALTY":    ["llm-a-proxy"],
    "THINK_REPEAT_PENALTY":      ["llm-a-proxy"],
    "THINK_REASONING_FORMAT":    ["llm-a-proxy"],
    "NOTHINK_MODEL_NAME":        ["llm-a-proxy"],
    "NOTHINK_PRESERVE_THINKING": ["llm-a-proxy"],
    "NOTHINK_REASONING_STREAM_MODE": ["llm-a-proxy"],
    "NOTHINK_JINJA":             ["llm-a-proxy"],
    "NOTHINK_TEMP":              ["llm-a-proxy"],
    "NOTHINK_TOP_P":             ["llm-a-proxy"],
    "NOTHINK_TOP_K":             ["llm-a-proxy"],
    "NOTHINK_MIN_P":             ["llm-a-proxy"],
    "NOTHINK_PRESENCE_PENALTY":  ["llm-a-proxy"],
    "NOTHINK_REPEAT_PENALTY":    ["llm-a-proxy"],
    "NOTHINK_REASONING_FORMAT":  ["llm-a-proxy"],
    "CODE_MODEL_NAME":           ["llm-a-proxy"],
    "TASK_MODEL_NAME":           ["task"],
    "TASK_MODEL_PATH":           ["task"],
    "TASK_MMPROJ_PATH":          ["task"],
    "TASK_CTX_SIZE":             ["task"],
    "TASK_LOAD_ON_STARTUP":      ["task"],
    "TASK_N_PARALLEL":           ["task"],
    "TASK_THREADS":              ["task"],
    "TASK_THREADS_BATCH":        ["task"],
    "TASK_N_GPU_LAYERS":         ["task"],
    "TASK_MAIN_GPU":             ["task"],
    "TASK_DEVICE":               ["task"],
    "TASK_TENSOR_SPLIT":         ["task"],
    "TASK_SPLIT_MODE":           ["task"],
    "TASK_KV_OFFLOAD":           ["task"],
    "TASK_OP_OFFLOAD":           ["task"],
    "TASK_MMPROJ_OFFLOAD":       ["task"],
    "TASK_BATCH_SIZE":           ["task"],
    "TASK_UBATCH_SIZE":          ["task"],
    "TASK_METRICS":              ["task"],
    "TASK_NO_MMAP":              ["task"],
    "TASK_MLOCK":                ["task"],
    "TASK_GPU_VISIBLE_DEVICES":  ["task"],
    "TASK_FLASH_ATTN":           ["task"],
    "TASK_CACHE_TYPE_K":         ["task"],
    "TASK_CACHE_TYPE_V":         ["task"],
    "TASK_CACHE_RAM":            ["task"],
    "TASK_CTX_CHECKPOINTS":      ["task"],
    "TASK_SWA_FULL":             ["task"],
    "TASK_TEMP":                 ["task"],
    "TASK_TOP_P":                ["task"],
    "TASK_TOP_K":                ["task"],
    "TASK_MIN_P":                ["task"],
    "TASK_PRESENCE_PENALTY":     ["task"],
    "TASK_REPEAT_PENALTY":       ["task"],
    "TASK_JINJA":                ["task"],
    "TASK_THINKING":             ["task"],
    "TASK_REASONING_FORMAT":     ["task"],
    "TASK_REASONING_EFFORT":     ["task"],
    "TASK_FIT":                  ["task"],
    "TASK_CUSTOM_ARGS_JSON":     ["task"],
    "TASK_LORA_PATHS":           ["task"],
    "TASK_LORA_SCALES":          ["task"],
    "TASK_LORA_INIT_WITHOUT_APPLY": ["task"],
    "TASK_CHAT_TEMPLATE_ID":      ["task"],
    "TASK_SPEC_METHOD":             ["task"],
    "TASK_SPEC_NGRAM_MOD":          ["task"],
    "TASK_SPEC_DRAFT_MODEL_PATH":   ["task"],
    "TASK_SPEC_DRAFT_N_GPU_LAYERS": ["task"],
    "TASK_SPEC_DRAFT_DEVICES":      ["task"],
    "TASK_SPEC_DRAFT_N_MAX":        ["task"],
    "TASK_SPEC_DRAFT_N_MIN":        ["task"],
    "TASK_SPEC_DRAFT_P_MIN":        ["task"],
    "TASK_SPEC_DRAFT_P_SPLIT":      ["task"],
    "TASK_SPEC_NGRAM_MOD_N_MATCH":  ["task"],
    "TASK_SPEC_NGRAM_MOD_N_MIN":    ["task"],
    "TASK_SPEC_NGRAM_MOD_N_MAX":    ["task"],
    "EMBED_MODEL_NAME":          ["embed"],
    "EMBEDDING_MODEL_PATH":      ["embed"],
    "EMBED_CTX_SIZE":            ["embed"],
    "EMBED_LOAD_ON_STARTUP":     ["embed"],
    "EMBED_N_PARALLEL":          ["embed"],
    "EMBED_THREADS":             ["embed"],
    "EMBED_THREADS_BATCH":       ["embed"],
    "EMBED_N_GPU_LAYERS":        ["embed"],
    "EMBED_MAIN_GPU":            ["embed"],
    "EMBED_DEVICE":              ["embed"],
    "EMBED_TENSOR_SPLIT":        ["embed"],
    "EMBED_SPLIT_MODE":          ["embed"],
    "EMBED_FLASH_ATTN":          ["embed"],
    "EMBED_CACHE_TYPE_K":        ["embed"],
    "EMBED_CACHE_TYPE_V":        ["embed"],
    "EMBED_BATCH_SIZE":          ["embed"],
    "EMBED_UBATCH_SIZE":         ["embed"],
    "EMBED_METRICS":             ["embed"],
    "EMBED_NO_MMAP":             ["embed"],
    "EMBED_MLOCK":               ["embed"],
    "EMBED_GPU_VISIBLE_DEVICES": ["embed"],
    "EMBED_TEMP":                ["embed"],
    "EMBED_TOP_P":               ["embed"],
    "EMBED_TOP_K":               ["embed"],
    "EMBED_MIN_P":               ["embed"],
    "EMBED_JINJA":               ["embed"],
    "EMBED_REASONING_FORMAT":    ["embed"],
    "EMBED_FIT":                 ["embed"],
    "RERANK_MODEL_NAME":         ["rerank"],
    "RERANKER_MODEL_PATH":       ["rerank"],
    "RERANK_CTX_SIZE":           ["rerank"],
    "RERANK_LOAD_ON_STARTUP":    ["rerank"],
    "RERANK_N_PARALLEL":         ["rerank"],
    "RERANK_THREADS":            ["rerank"],
    "RERANK_THREADS_BATCH":      ["rerank"],
    "RERANK_N_GPU_LAYERS":       ["rerank"],
    "RERANK_MAIN_GPU":           ["rerank"],
    "RERANK_DEVICE":             ["rerank"],
    "RERANK_TENSOR_SPLIT":       ["rerank"],
    "RERANK_SPLIT_MODE":         ["rerank"],
    "RERANK_FLASH_ATTN":         ["rerank"],
    "RERANK_CACHE_TYPE_K":       ["rerank"],
    "RERANK_CACHE_TYPE_V":       ["rerank"],
    "RERANK_BATCH_SIZE":         ["rerank"],
    "RERANK_UBATCH_SIZE":        ["rerank"],
    "RERANK_METRICS":            ["rerank"],
    "RERANK_NO_MMAP":            ["rerank"],
    "RERANK_MLOCK":              ["rerank"],
    "RERANK_GPU_VISIBLE_DEVICES":["rerank"],
    "RERANK_TEMP":               ["rerank"],
    "RERANK_TOP_P":              ["rerank"],
    "RERANK_TOP_K":              ["rerank"],
    "RERANK_MIN_P":              ["rerank"],
    "RERANK_JINJA":              ["rerank"],
    "RERANK_REASONING_FORMAT":   ["rerank"],
    "RERANK_FIT":                ["rerank"],
    "GRAPHITI_PUBLIC_URL":       ["graphiti"],
    "GRAPHITI_HOST":             ["graphiti"],
    "GRAPHITI_PORT":             ["graphiti"],
    "GRAPHITI_LLM_BASE_URL":     ["graphiti"],
    "GRAPHITI_LLM_MODEL":        ["graphiti"],
    "GRAPHITI_EMBED_BASE_URL":   ["graphiti"],
    "GRAPHITI_EMBED_MODEL":      ["graphiti"],
    "GRAPHITI_RERANKER_PROVIDER":["graphiti"],
    "GRAPHITI_RERANKER_BASE_URL":["graphiti"],
    "GRAPHITI_RERANKER_MODEL":   ["graphiti"],
    "GRAPHITI_MEMORY_MAX_QUERY_CHARS": ["graphiti"],
    "GRAPHITI_MEMORY_MAX_MESSAGES":    ["graphiti"],
    "GRAPHITI_MEMORY_MAX_FACTS":       ["graphiti"],
    "GRAPHITI_NEO4J_URI":        ["graphiti"],
    "GRAPHITI_NEO4J_USER":       ["graphiti"],
    "GRAPHITI_NEO4J_PASSWORD":   ["graphiti"],
    "GRAPHITI_NEO4J_DATABASE":   ["graphiti"],
    "GRAPHITI_NEO4J_BOLT_PORT":  ["graphiti"],
    "GRAPHITI_NEO4J_HTTP_PORT":  ["graphiti"],
    "THINK_PORT":                ["llm-a-proxy"],
    "NOTHINK_PORT":              ["llm-a-proxy"],
    "CODE_PORT":                 ["llm-a-proxy"],
    "AGGREGATE_ENABLED":         ["llm-a-proxy"],
    "AGGREGATE_PORT":            ["llm-a-proxy"],
    "THINK2_PORT":               ["llm-b-proxy"],
    "NOTHINK2_PORT":             ["llm-b-proxy"],
    "CODE2_PORT":                ["llm-b-proxy"],
    "AGGREGATE2_ENABLED":        ["llm-b-proxy"],
    "AGGREGATE2_PORT":           ["llm-b-proxy"],
    "THINK2_MODEL_NAME":         ["llm-b-proxy"],
    "NOTHINK2_MODEL_NAME":       ["llm-b-proxy"],
    "CODE2_MODEL_NAME":          ["llm-b-proxy"],
    "EMBED_PORT":                ["embed"],
    "RERANK_PORT":               ["rerank"],
    "TASK_PORT":                 ["task"],
    "LISTEN_HOST":               ["llm-a-proxy", "embed", "rerank", "task"],
    "CHAT_MODEL_PATH":           ["llm-a"],
    "CHAT_MMPROJ_PATH":          ["llm-a"],
    "CHAT_CTX_SIZE":             ["llm-a"],
    "TTS_PUBLIC_URL":            ["tts-gateway"],
    "TTS_GATEWAY_HOST":          ["tts-gateway"],
    "TTS_GATEWAY_PORT":          ["tts-gateway"],
    "TTS_SINGLE_ACTIVE":         ["tts-gateway"],
    "TTS_DEFAULT_FORMAT":        ["tts-gateway"],
    "KOKORO_UPSTREAM_URL":       ["tts-backend-kokoro"],
    "KOKORO_LAUNCH_CMD":         ["tts-backend-kokoro"],
    "KOKORO_VOICES":             ["tts-backend-kokoro", "tts-gateway"],
    "CHATTERBOX_UPSTREAM_URL":   ["tts-backend-chatterbox"],
    "CHATTERBOX_LAUNCH_CMD":     ["tts-backend-chatterbox"],
    "CHATTERBOX_VOICES":         ["tts-backend-chatterbox", "tts-gateway"],
    "VIBEVOICE_UPSTREAM_URL":    ["tts-backend-vibevoice"],
    "VIBEVOICE_LAUNCH_CMD":      ["tts-backend-vibevoice"],
    "VIBEVOICE_VOICES":          ["tts-backend-vibevoice", "tts-gateway"],
    "VIBEVOICE_MODEL_PATH":      ["tts-backend-vibevoice"],
    "VIBEVOICE_DEVICE":          ["tts-backend-vibevoice"],
    "VIBEVOICE_RUNTIME_HOST":    ["tts-backend-vibevoice"],
    "VIBEVOICE_RUNTIME_PORT":    ["tts-backend-vibevoice"],
    "VIBEVOICE_CFG_SCALE":       ["tts-backend-vibevoice"],
    "VIBEVOICE_DDPM_STEPS":      ["tts-backend-vibevoice"],
    # Read-side legacy only: a backfill source for TRANSCRIPT_LOCAL_MODEL with no
    # CONFIG_FIELDS entry, so it needs naming here to stay writable at all. Every
    # other transcription key is covered by the prefix rules below.
    "TRANSCRIPT_LOCAL_MODEL_SIZE":  ["transcript-backend"],
}

for _field in CONFIG_FIELDS:
    _key = _field.get("key", "")
    if _key.startswith("LLM_A_"):
        RESTART_HINTS.setdefault(_key, ["llm-a"])
    if _key.startswith("LLM_B_"):
        RESTART_HINTS.setdefault(_key, ["llm-b"])
    if _key.startswith("OCR_"):
        RESTART_HINTS.setdefault(_key, ["ocr"])
    if _key.startswith("GLMOCR_"):
        RESTART_HINTS.setdefault(_key, ["glmocr-sdk"])
    if _key.startswith("MODEL_ROUTER_"):
        RESTART_HINTS.setdefault(_key, ["llama-router"])
    if _key.startswith("SEARXNG_"):
        RESTART_HINTS.setdefault(_key, ["searxng"])
    if _key.startswith("PLAYWRIGHT_"):
        RESTART_HINTS.setdefault(_key, ["playwright-server"])
    if _key.startswith("TRANSCRIPT_") or any(
        _key.startswith(_engine["env_prefix"] + "_") for _engine in TRANSCRIPTION_ENGINES
    ):
        RESTART_HINTS.setdefault(_key, ["transcript-backend"])
    # The pooled audio model is a router child, not a unit, so this cannot go
    # through `apply_router_restart_hints` — that only redirects pooled *units*.
    if _key.startswith("ASR_"):
        RESTART_HINTS.setdefault(_key, ["llama-router"])


# Last, deliberately: `CONFIG_FIELDS` is extended three times after its literal
# definition -- the transcription engines, the cloned chat-backend variants --
# and a rename map built before those ran would cover only the fields that
# happened to exist yet. It did, the first time this was written, and silently
# produced half a map.
_register_prefix_renames()
_rebuild_legacy_aliases()
