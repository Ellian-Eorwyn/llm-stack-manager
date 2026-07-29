#!/usr/bin/env python3
"""
Memory budget model for the LLM Stack Manager.

The manager has never been able to answer "will this configuration fit?" before
launching it. `setup_engine.estimate_model_mib` guesses `file_size x 1.10 +
context_mib`, which ignores KV geometry entirely, and the config form accepts
anything. The cost of that showed up on this box as an eviction storm: a hybrid
attention model was configured with 32 context checkpoints across 2 slots, and
because each checkpoint of a hybrid model carries a fixed ~150 MiB of recurrent
state regardless of token count, the prompt cache needed several times the RAM
it was given and evicted multi-GiB entries on most requests.

Nothing in that chain was visible until someone read the journal. This module
makes it computable instead, from three inputs the manager already has: the
model's own GGUF metadata, the launcher settings, and detected host memory.

Confidence is tiered on purpose, because not all of it is equally knowable:

  * `exact`     - weights and projector (file size on disk), KV cache (layer
                  geometry x context x quant), recurrent state, and the host
                  prompt-cache requirement. These are the terms the operator
                  actually controls, and they are the ones that caused the
                  storm.
  * `estimated` - compute buffers, CUDA contexts and speculative-decode
                  overhead, carried with an explicit uncertainty band.

Verdicts are computed against the upper bound of the estimate, so "fits" means
fits. `--validate` against a running backend reports predicted vs. observed so
the estimate can be checked rather than trusted.

The recurrent-state formula was validated against this box: 149.62 MiB
predicted vs 149.6 MiB observed in `erasing old context checkpoint` journal
lines for Qwen3.6-27B.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

MIB = 1024 * 1024

# GGUF metadata value types (gguf_metadata_value_type in the spec).
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL = range(8)
_STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_FORMAT = {
    _UINT8: "<B", _INT8: "<b", _UINT16: "<H", _INT16: "<h",
    _UINT32: "<I", _INT32: "<i", _FLOAT32: "<f", _BOOL: "<?",
    _UINT64: "<Q", _INT64: "<q", _FLOAT64: "<d",
}
_SCALAR_SIZE = {t: struct.calcsize(f) for t, f in _SCALAR_FORMAT.items()}

# Arrays this long are almost certainly tokenizer vocabulary. We still have to
# walk them to find where the next key starts, but we never materialise them.
_MAX_ARRAY_ITEMS = 512

# Bytes per element for every KV cache type llama.cpp accepts for
# --cache-type-k/v. Quantised types are block-encoded: q8_0 stores 32 values in
# 34 bytes, hence the fractional sizes.
KV_TYPE_BYTES = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
    "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32,
}

# Per-GPU allocation that exists before a single tensor is placed: the CUDA
# context, cuBLAS workspaces and the allocator's own bookkeeping.
CUDA_CONTEXT_MIB = 400
# Compute buffers scale with micro-batch and hidden size, but also carry a
# vocabulary-sized output tensor and, for vision models, an image encoder whose
# working set dwarfs both. This band is deliberately wide; `--validate` reports
# where a given host actually lands.
COMPUTE_UNCERTAINTY = 0.35


# llama_ftype values, for turning `general.file_type` into the quant name the
# operator recognises from the filename.
FILE_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
}


class GGUFError(Exception):
    """Raised when a file is not readable as GGUF metadata."""


# --------------------------------------------------------------------------
# GGUF metadata
# --------------------------------------------------------------------------

def _read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise GGUFError("unexpected end of file while reading metadata")
    return data


def _read_scalar(handle, value_type: int):
    fmt = _SCALAR_FORMAT.get(value_type)
    if fmt is None:
        raise GGUFError(f"unknown GGUF value type {value_type}")
    return struct.unpack(fmt, _read_exact(handle, _SCALAR_SIZE[value_type]))[0]


def _read_string(handle) -> str:
    length = struct.unpack("<Q", _read_exact(handle, 8))[0]
    return _read_exact(handle, length).decode("utf-8", errors="replace")


def _skip_string(handle):
    length = struct.unpack("<Q", _read_exact(handle, 8))[0]
    handle.seek(length, 1)


def _read_value(handle, value_type: int):
    """Read one metadata value, seeking past oversized arrays rather than
    building them. Returns None for values we deliberately skip."""
    if value_type == _STRING:
        return _read_string(handle)
    if value_type != _ARRAY:
        return _read_scalar(handle, value_type)

    item_type = struct.unpack("<I", _read_exact(handle, 4))[0]
    count = struct.unpack("<Q", _read_exact(handle, 8))[0]
    if item_type == _ARRAY:
        raise GGUFError("nested GGUF arrays are not supported")
    if item_type == _STRING:
        # Strings are variable-length, so skipping still means walking them.
        for _ in range(count):
            _skip_string(handle)
        # The length is worth keeping even when the contents are not: the
        # tokenizer's token array is how we learn the vocabulary size.
        return {"__len__": count}
    if count > _MAX_ARRAY_ITEMS:
        handle.seek(_SCALAR_SIZE[item_type] * count, 1)
        return {"__len__": count}
    return [_read_scalar(handle, item_type) for _ in range(count)]


def read_gguf_metadata(path) -> dict:
    """Parse a GGUF file's key/value metadata block.

    The block sits at the head of the file, so this reads a few megabytes of a
    model that may be tens of gigabytes. Tensor data is never touched.
    """
    path = Path(path)
    with path.open("rb") as handle:
        if _read_exact(handle, 4) != b"GGUF":
            raise GGUFError("not a GGUF file")
        version = struct.unpack("<I", _read_exact(handle, 4))[0]
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        tensor_count = struct.unpack("<Q", _read_exact(handle, 8))[0]
        kv_count = struct.unpack("<Q", _read_exact(handle, 8))[0]

        metadata: dict = {}
        for _ in range(kv_count):
            key = _read_string(handle)
            value_type = struct.unpack("<I", _read_exact(handle, 4))[0]
            value = _read_value(handle, value_type)
            if value is not None:
                metadata[key] = value

    metadata["__version__"] = version
    metadata["__tensor_count__"] = tensor_count
    metadata["__file_size__"] = path.stat().st_size
    return metadata


def _first(metadata: dict, arch: str, *suffixes, default=None):
    for suffix in suffixes:
        value = metadata.get(f"{arch}.{suffix}")
        if value is not None:
            return value
    return default


def _array_length(value) -> int:
    """Item count of an array value, whether it was materialised or skipped."""
    if isinstance(value, dict):
        return int(value.get("__len__") or 0)
    if isinstance(value, list):
        return len(value)
    return 0


def model_geometry(metadata: dict) -> dict:
    """Reduce raw GGUF metadata to the shape the budget math needs.

    Layer counting is the fiddly part. A model's `block_count` includes any
    multi-token-prediction head, which holds no KV cache of its own, and hybrid
    models split the remaining layers between full attention and a recurrent
    state-space mechanism. Only full-attention layers grow with context; the
    recurrent ones cost a fixed amount per sequence. Conflating the two is what
    makes checkpoint budgets on a hybrid model so surprising.
    """
    arch = metadata.get("general.architecture") or ""
    block_count = int(_first(metadata, arch, "block_count", default=0) or 0)
    nextn = int(_first(metadata, arch, "nextn_predict_layers", default=0) or 0)
    layers = max(0, block_count - nextn)

    head_count = int(_first(metadata, arch, "attention.head_count", default=0) or 0)
    head_count_kv = _first(metadata, arch, "attention.head_count_kv", default=head_count)
    # Some architectures publish this per layer; the maximum is the safe budget.
    if isinstance(head_count_kv, list):
        head_count_kv = max(head_count_kv) if head_count_kv else head_count
    elif isinstance(head_count_kv, dict):
        head_count_kv = head_count
    head_count_kv = int(head_count_kv or head_count or 0)

    embedding_length = int(_first(metadata, arch, "embedding_length", default=0) or 0)
    default_head_dim = embedding_length // head_count if head_count else 0
    key_length = int(_first(metadata, arch, "attention.key_length", default=default_head_dim) or 0)
    value_length = int(_first(metadata, arch, "attention.value_length", default=default_head_dim) or 0)

    # Which layers actually hold a KV cache.
    recurrent_flags = _first(metadata, arch, "recurrent_layer_arr")
    interval = int(_first(metadata, arch, "full_attention_interval", default=0) or 0)
    if isinstance(recurrent_flags, list) and recurrent_flags:
        full_attention_layers = sum(1 for flag in recurrent_flags if not flag)
        recurrent_layers = len(recurrent_flags) - full_attention_layers
    elif interval > 1:
        full_attention_layers = layers // interval
        recurrent_layers = layers - full_attention_layers
    else:
        full_attention_layers = layers
        recurrent_layers = 0

    d_inner = int(_first(metadata, arch, "ssm.inner_size", default=0) or 0)
    d_state = int(_first(metadata, arch, "ssm.state_size", default=0) or 0)
    d_conv = int(_first(metadata, arch, "ssm.conv_kernel", default=0) or 0)
    n_group = int(_first(metadata, arch, "ssm.group_count", default=0) or 0)

    sliding_window = int(_first(metadata, arch, "attention.sliding_window", default=0) or 0)
    vocab_size = int(_first(metadata, arch, "vocab_size", default=0) or 0) \
        or _array_length(metadata.get("tokenizer.ggml.tokens"))

    return {
        "architecture": arch,
        "name": metadata.get("general.name") or "",
        "file_type": FILE_TYPE_NAMES.get(metadata.get("general.file_type"),
                                         metadata.get("general.file_type")),
        "vocab_size": vocab_size,
        "file_size_mib": round((metadata.get("__file_size__") or 0) / MIB),
        "block_count": block_count,
        "nextn_predict_layers": nextn,
        "layers": layers,
        "full_attention_layers": full_attention_layers,
        "recurrent_layers": recurrent_layers,
        "full_attention_interval": interval,
        "head_count": head_count,
        "head_count_kv": head_count_kv,
        "key_length": key_length,
        "value_length": value_length,
        "embedding_length": embedding_length,
        "train_context_length": int(_first(metadata, arch, "context_length", default=0) or 0),
        "ssm": {"d_inner": d_inner, "d_state": d_state, "d_conv": d_conv, "n_group": n_group},
        "sliding_window": sliding_window,
        # `--swa-full` only means anything to a model with sliding-window
        # attention. llama-server logs "swa_full is not supported by this model"
        # and carries on, so the flag looks accepted when it is inert.
        "supports_swa": sliding_window > 0,
        "is_hybrid": recurrent_layers > 0,
    }


# --------------------------------------------------------------------------
# memory terms
# --------------------------------------------------------------------------

def kv_bytes_per_token(geometry: dict, type_k: str = "q8_0", type_v: str = "q8_0") -> float:
    """KV cache bytes for one token across every full-attention layer."""
    bytes_k = KV_TYPE_BYTES.get(str(type_k).lower(), KV_TYPE_BYTES["f16"])
    bytes_v = KV_TYPE_BYTES.get(str(type_v).lower(), KV_TYPE_BYTES["f16"])
    per_layer = geometry["head_count_kv"] * (
        geometry["key_length"] * bytes_k + geometry["value_length"] * bytes_v
    )
    return per_layer * geometry["full_attention_layers"]


def recurrent_state_bytes(geometry: dict) -> float:
    """Fixed recurrent state held for one sequence, independent of length.

    This is the term that makes hybrid models expensive to checkpoint: a
    checkpoint taken at token 1 costs the same as one taken at token 100000.
    Verified at 149.62 MiB predicted against 149.6 MiB observed for
    Qwen3.6-27B.
    """
    ssm = geometry["ssm"]
    if not geometry["recurrent_layers"] or not ssm["d_state"] or not ssm["d_inner"]:
        return 0.0
    conv_dim = ssm["d_inner"] + 2 * ssm["n_group"] * ssm["d_state"]
    conv_bytes = max(0, ssm["d_conv"] - 1) * conv_dim * 4
    ssm_bytes = ssm["d_inner"] * ssm["d_state"] * 4
    return geometry["recurrent_layers"] * (conv_bytes + ssm_bytes)


def checkpoint_bytes(geometry: dict, ctx_size: int, type_k: str, type_v: str) -> tuple[float, float]:
    """(fixed, per_token) bytes for one context checkpoint.

    A checkpoint snapshots recurrent state in full, plus one full-attention
    layer's worth of KV per token. On a pure-attention model the fixed term
    vanishes and the whole cache is the per-token term, which is why the same
    `--ctx-checkpoints` value behaves so differently across models.

    Both terms are validated against `erasing old context checkpoint` journal
    lines on this box: 149.6 MiB fixed and ~0.002 MiB/token observed, against
    149.62 and 0.00208 predicted.
    """
    fixed = recurrent_state_bytes(geometry)
    per_token = kv_bytes_per_token(geometry, type_k, type_v)
    if geometry["is_hybrid"]:
        per_token /= max(1, geometry["full_attention_layers"])
    return fixed, per_token


# A vision projector's working set is several times its weights: the image
# encoder runs at its own resolution with its own activations. Calibrated
# against an 885 MiB projector holding roughly 4.4 GiB at run time.
_PROJECTOR_WORKING_SET = 5.0
# Activation tensors live per graph node; llama.cpp keeps on the order of two
# dozen ubatch-sized tensors alive at once.
_GRAPH_TENSORS = 24


def compute_buffer_mib(geometry: dict, ubatch: int, devices: int, projector_mib: float) -> float:
    """Coarse estimate of compute buffers across all devices.

    Three terms that at least scale correctly: graph activations with
    micro-batch and hidden size, the output tensor with vocabulary, and a
    vision projector's encoder with its own weights. Estimated, not exact -
    see COMPUTE_UNCERTAINTY and the `--validate` path, which reports where a
    given host actually lands.
    """
    hidden = max(geometry["embedding_length"], 1)
    graph = ubatch * hidden * 4 * _GRAPH_TENSORS / MIB * max(1, devices)
    logits = ubatch * max(geometry.get("vocab_size") or 0, 0) * 4 / MIB
    return graph + logits + projector_mib * _PROJECTOR_WORKING_SET


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------

def _split_weights(total_mib: float, tensor_split: str, devices: int) -> list[float]:
    """Distribute weight MiB across devices using llama.cpp's --tensor-split."""
    weights = []
    for part in str(tensor_split or "").replace(" ", "").split(","):
        try:
            weights.append(max(0.0, float(part)))
        except ValueError:
            continue
    if len(weights) < devices:
        weights += [0.0] * (devices - len(weights))
    weights = weights[:devices] if devices else weights
    total_weight = sum(weights)
    if total_weight <= 0:
        return [total_mib / max(1, devices)] * max(1, devices)
    return [total_mib * w / total_weight for w in weights]


def predict(geometry: dict, settings: dict) -> dict:
    """Predict the memory footprint of one backend configuration.

    `settings` mirrors the launcher's own vocabulary: ctx_size, parallel,
    ubatch, cache_type_k/v, ctx_checkpoints, cache_ram, tensor_split, devices,
    projector_mib, spec_method.
    """
    ctx_size = max(0, int(settings.get("ctx_size") or 0))
    parallel = max(1, int(settings.get("parallel") or 1))
    ubatch = max(1, int(settings.get("ubatch") or 512))
    type_k = settings.get("cache_type_k") or "f16"
    type_v = settings.get("cache_type_v") or "f16"
    checkpoints = max(0, int(settings.get("ctx_checkpoints") or 0))
    cache_ram_mib = max(0, int(settings.get("cache_ram") or 0))
    devices = max(1, int(settings.get("devices") or 1))
    projector_mib = float(settings.get("projector_mib") or 0)
    weights_mib = float(settings.get("weights_mib") or geometry["file_size_mib"])

    per_slot_ctx = ctx_size // parallel if parallel else ctx_size

    kv_mib = kv_bytes_per_token(geometry, type_k, type_v) * ctx_size / MIB
    recurrent_mib = recurrent_state_bytes(geometry) * parallel / MIB

    # Multi-token-prediction reuses the main model's weights but keeps its own
    # small cache for the extra head.
    spec_method = str(settings.get("spec_method") or "off").lower()
    draft_mib = 0.0
    if spec_method in {"mtp", "draft-mtp"} and geometry["nextn_predict_layers"]:
        draft_geometry = dict(geometry, full_attention_layers=geometry["nextn_predict_layers"])
        draft_mib = kv_bytes_per_token(
            draft_geometry,
            settings.get("spec_draft_type_k") or type_k,
            settings.get("spec_draft_type_v") or type_v,
        ) * ctx_size / MIB

    compute_mib = compute_buffer_mib(geometry, ubatch, devices, projector_mib)
    overhead_mib = CUDA_CONTEXT_MIB * devices

    exact_vram = weights_mib + projector_mib + kv_mib + recurrent_mib + draft_mib
    estimated_vram = compute_mib + overhead_mib

    # Host prompt cache. Each slot keeps up to `checkpoints` context
    # checkpoints, and every one of them pays the fixed recurrent cost.
    fixed_b, per_token_b = checkpoint_bytes(geometry, ctx_size, type_k, type_v)
    checkpoint_mib = (fixed_b + per_token_b * per_slot_ctx) / MIB
    checkpoint_total_mib = checkpoint_mib * checkpoints * parallel

    device_weights = _split_weights(weights_mib + projector_mib, settings.get("tensor_split"), devices)
    device_kv = (kv_mib + recurrent_mib + draft_mib) / devices
    per_device = [
        {
            "device": index,
            "weights_mib": round(share),
            "kv_mib": round(device_kv),
            "compute_mib": round(compute_mib / devices + CUDA_CONTEXT_MIB),
            "total_mib": round(share + device_kv + compute_mib / devices + CUDA_CONTEXT_MIB),
            "upper_mib": round(share + device_kv
                               + (compute_mib / devices + CUDA_CONTEXT_MIB) * (1 + COMPUTE_UNCERTAINTY)),
        }
        for index, share in enumerate(device_weights)
    ]

    return {
        "per_slot_context": per_slot_ctx,
        "total_context": ctx_size,
        "slots": parallel,
        "vram": {
            "weights_mib": round(weights_mib),
            "projector_mib": round(projector_mib),
            "kv_mib": round(kv_mib),
            "recurrent_mib": round(recurrent_mib),
            "draft_mib": round(draft_mib),
            "compute_mib": round(compute_mib),
            "overhead_mib": round(overhead_mib),
            "exact_mib": round(exact_vram),
            "estimated_mib": round(estimated_vram),
            "total_mib": round(exact_vram + estimated_vram),
            "upper_mib": round(exact_vram + estimated_vram * (1 + COMPUTE_UNCERTAINTY)),
            "per_device": per_device,
        },
        "host": {
            "checkpoint_fixed_mib": round(fixed_b / MIB, 1),
            "checkpoint_per_token_mib": round(per_token_b / MIB, 6),
            "checkpoint_each_mib": round(checkpoint_mib, 1),
            "checkpoints_per_slot": checkpoints,
            "checkpoint_total_mib": round(checkpoint_total_mib),
            "cache_ram_mib": cache_ram_mib,
            "cache_ram_shortfall_mib": round(max(0.0, checkpoint_total_mib - cache_ram_mib)),
        },
        "kv_bytes_per_token": round(kv_bytes_per_token(geometry, type_k, type_v)),
    }


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------

def evaluate(geometry: dict, settings: dict, prediction: dict,
             gpus: list[dict] | None = None, host: dict | None = None) -> dict:
    """Structured issues for a configuration, worst first.

    Severity is `error` only where the configuration cannot work as written -
    it will fail to allocate, or a request will be rejected. Everything the
    operator might have meant is a `warn`.
    """
    issues: list[dict] = []
    gpus = gpus or []
    host = host or {}

    # VRAM. Compared against the upper bound so "fits" means fits.
    devices = prediction["vram"]["per_device"]
    for index, device in enumerate(devices):
        gpu = gpus[index] if index < len(gpus) else None
        if not gpu or not gpu.get("mem_total"):
            continue
        capacity = gpu["mem_total"]
        if device["upper_mib"] > capacity:
            issues.append({
                "level": "error", "code": "vram_overcommit",
                "text": f"GPU {gpu.get('index', index)} needs up to {device['upper_mib']:,} MiB "
                        f"but has {capacity:,} MiB. Reduce context, quantise the KV cache further, "
                        f"or move layers to another device.",
            })
        elif device["upper_mib"] > capacity * 0.95:
            issues.append({
                "level": "warn", "code": "vram_tight",
                "text": f"GPU {gpu.get('index', index)} is predicted to sit within 5% of its "
                        f"{capacity:,} MiB capacity. A restart may fail to allocate.",
            })

    # Host prompt cache. This is the eviction storm, stated before it happens.
    host_budget = prediction["host"]
    shortfall = host_budget["cache_ram_shortfall_mib"]
    if host_budget["checkpoints_per_slot"] and shortfall > 0:
        issues.append({
            "level": "warn", "code": "cache_ram_shortfall",
            "text": f"{host_budget['checkpoints_per_slot']} checkpoints x {prediction['slots']} slots "
                    f"need up to {host_budget['checkpoint_total_mib']:,} MiB but --cache-ram is "
                    f"{host_budget['cache_ram_mib']:,} MiB — {shortfall:,} MiB short. Expect the cache "
                    f"to evict entries on most requests, which reprocesses prompts it exists to keep.",
        })

    if geometry["is_hybrid"] and host_budget["checkpoint_fixed_mib"] >= 1:
        issues.append({
            "level": "info", "code": "hybrid_checkpoints",
            "text": f"Hybrid attention model: every context checkpoint costs "
                    f"{host_budget['checkpoint_fixed_mib']} MiB of recurrent state before a single "
                    f"token is stored. Checkpoint count is the dominant prompt-cache lever here.",
        })

    available = host.get("mem_available_mib")
    if available and host_budget["checkpoint_total_mib"] > available:
        issues.append({
            "level": "warn", "code": "host_ram_overcommit",
            "text": f"The prompt cache could want {host_budget['checkpoint_total_mib']:,} MiB against "
                    f"{available:,} MiB of available host RAM, which pushes the box into swap.",
        })

    # Dead and contradictory flags.
    if str(settings.get("swa_full") or "off").lower() == "on" and not geometry["supports_swa"]:
        issues.append({
            "level": "warn", "code": "swa_full_unsupported",
            "text": "Full SWA KV cache is enabled, but this model has no sliding-window attention. "
                    "llama-server logs 'swa_full is not supported by this model' and ignores it.",
        })

    fit_ctx = str(settings.get("fit_ctx") or "").strip()
    if str(settings.get("fit") or "on").lower() == "off" and fit_ctx and fit_ctx != "0":
        issues.append({
            "level": "warn", "code": "fit_ctx_without_fit",
            "text": f"Minimum Fit Context is {fit_ctx} but auto-fit is off, so --fit-ctx does nothing. "
                    "Clear one of the two.",
        })

    cache_reuse = str(settings.get("cache_reuse") or "0").strip()
    if cache_reuse not in ("", "0") and settings.get("mmproj_path"):
        issues.append({
            "level": "warn", "code": "cache_reuse_with_multimodal",
            "text": f"Cache Reuse Chunk is {cache_reuse}, but this backend loads a multimodal "
                    "projector and llama-server disables --cache-reuse for multimodal models "
                    "('cache_reuse is not supported by multimodal'). Partial prefix reuse is off; "
                    "whole-prefix slot reuse still applies.",
        })

    # Context accounting, the mistake the per-slot division invites.
    train_ctx = geometry.get("train_context_length") or 0
    if train_ctx and prediction["total_context"] > train_ctx:
        issues.append({
            "level": "warn", "code": "context_above_trained",
            "text": f"Total context {prediction['total_context']:,} exceeds the model's trained "
                    f"{train_ctx:,}. Quality past that point is not guaranteed.",
        })
    if prediction["slots"] > 1:
        issues.append({
            "level": "info", "code": "per_slot_context",
            "text": f"--ctx-size {prediction['total_context']:,} across {prediction['slots']} slots "
                    f"gives {prediction['per_slot_context']:,} tokens per slot. Requests above that "
                    f"are rejected, whatever the total says.",
        })

    order = {"error": 0, "warn": 1, "info": 2}
    issues.sort(key=lambda issue: order.get(issue["level"], 3))
    return {
        "ok": not any(issue["level"] == "error" for issue in issues),
        "issues": issues,
    }


# --------------------------------------------------------------------------
# env plumbing
# --------------------------------------------------------------------------

# Each backend the model can price, and the env prefix its settings live under.
BACKEND_PREFIXES = {
    "chat-primary": "CHAT_PRIMARY",
    "chat-secondary": "CHAT2",
    "embed": "EMBED",
    "embed2": "EMBED2",
    "rerank": "RERANK",
    "task": "TASK",
    "ocr": "OCR",
}

# Settings key -> env suffix. Absent suffixes fall back to the defaults in
# `predict`, which matches how the launchers treat an unset value.
_SETTING_SUFFIXES = {
    "ctx_size": "CTX_SIZE",
    "parallel": "N_PARALLEL",
    "ubatch": "UBATCH_SIZE",
    "batch": "BATCH_SIZE",
    "cache_type_k": "CACHE_TYPE_K",
    "cache_type_v": "CACHE_TYPE_V",
    "ctx_checkpoints": "CTX_CHECKPOINTS",
    "cache_ram": "CACHE_RAM",
    "tensor_split": "TENSOR_SPLIT",
    "swa_full": "SWA_FULL",
    "fit": "FIT",
    "fit_ctx": "FIT_CTX",
    "cache_reuse": "CACHE_REUSE",
    "spec_method": "SPEC_METHOD",
    "spec_draft_type_k": "SPEC_DRAFT_TYPE_K",
    "spec_draft_type_v": "SPEC_DRAFT_TYPE_V",
}


# Settings a caller may override to price a change before saving it. Anything
# outside this set is ignored, so a stray query parameter cannot reshape the
# model the prediction is built from.
OVERRIDABLE_SETTINGS = frozenset(_SETTING_SUFFIXES) | {
    "devices", "model_path", "mmproj_path",
}


def settings_from_env(env: dict, backend: str = "chat-primary") -> dict:
    """Read one backend's launcher settings out of the env file.

    `read_env` has already backfilled the `CHAT_PRIMARY_*` keys from their
    legacy `CHAT_*` twins, so this reads the new names only.
    """
    prefix = BACKEND_PREFIXES.get(backend)
    if prefix is None:
        raise ValueError(f"unknown backend {backend!r}")

    settings = {"backend": backend, "prefix": prefix}
    for name, suffix in _SETTING_SUFFIXES.items():
        value = env.get(f"{prefix}_{suffix}")
        if value not in (None, ""):
            settings[name] = value

    settings["model_path"] = env.get(f"{prefix}_MODEL_PATH") or ""
    settings["mmproj_path"] = env.get(f"{prefix}_MMPROJ_PATH") or ""
    visible = env.get(f"{prefix}_GPU_VISIBLE_DEVICES") or ""
    settings["devices"] = len([part for part in visible.split(",") if part.strip()]) or 1
    return settings


def budget_for(env: dict, backend: str = "chat-primary",
               gpus: list[dict] | None = None, host: dict | None = None,
               overrides: dict | None = None) -> dict:
    """Full budget for one backend: geometry, prediction and verdict.

    Degrades to an error payload rather than raising, so a missing or
    unreadable model never takes down the caller's response.
    """
    settings = settings_from_env(env, backend)
    settings.update(overrides or {})

    model_path = settings.get("model_path") or ""
    result = {
        "backend": backend,
        "model_path": model_path,
        "settings": {k: v for k, v in settings.items() if k not in {"prefix"}},
        "geometry": None,
        "prediction": None,
        "verdict": None,
        "error": None,
    }
    if not model_path or not Path(model_path).is_file():
        result["error"] = f"model not found: {model_path or '(unset)'}"
        return result

    try:
        geometry = model_geometry(read_gguf_metadata(model_path))
    except (GGUFError, OSError) as exc:
        result["error"] = f"could not read model metadata: {exc}"
        return result

    projector = settings.get("mmproj_path") or ""
    if projector and Path(projector).is_file():
        settings["projector_mib"] = Path(projector).stat().st_size / MIB

    prediction = predict(geometry, settings)
    result["geometry"] = geometry
    result["prediction"] = prediction
    result["verdict"] = evaluate(geometry, settings, prediction, gpus, host)
    return result


# Terms of an existing configuration that change the footprint without being
# things a recommendation should choose. Pricing candidate contexts without
# them recommends a configuration that then fails its own pre-flight: this box
# loads an 885 MiB projector and an MTP draft head, together worth several
# gigabytes that a bare `predict` call never sees.
_RECOMMEND_CARRIED_SETTINGS = (
    "ubatch", "batch", "weights_mib", "projector_mib", "tensor_split",
    "spec_method", "spec_draft_type_k", "spec_draft_type_v",
)

# Candidate total contexts, largest first. Not powers of two throughout: the
# useful sizes are the ones that divide evenly across two slots.
_CONTEXT_LADDER = (524288, 393216, 262144, 196608, 131072, 98304, 65536, 32768, 16384, 8192)


def recommend(geometry: dict, gpus: list[dict], host: dict, slots: int = 2,
              cache_type_k: str = "q8_0", cache_type_v: str = "q8_0",
              base_settings: dict | None = None) -> dict:
    """Derive a configuration that fits this host, instead of a constant.

    `cache-aware-scheduling.js` used to hardcode a preset that was measured to
    thrash this box. This is the replacement: pick the largest context whose
    predicted footprint fits detected VRAM, then size checkpoints and
    prompt-cache RAM from what the model actually costs per checkpoint.

    `base_settings` carries the parts of the live configuration that a
    recommendation has no business choosing but that still move the number —
    the projector, the draft head, the micro-batch, the tensor split. Priced
    without them, the recommendation is for a backend nobody is launching.

    The result is a configuration `evaluate` has nothing to say about. That is
    the point of deriving it: a recommendation that trips the pre-flight on the
    way in is not a recommendation.
    """
    capacity = sum(gpu.get("mem_total") or 0 for gpu in gpus)
    devices = max(1, len(gpus))
    usable = capacity * 0.92
    slots = max(1, slots)

    carried = {key: value for key, value in (base_settings or {}).items()
               if key in _RECOMMEND_CARRIED_SETTINGS and value not in (None, "")}

    # Never recommend past what the model was trained for. `evaluate` warns
    # about it, and a recommendation should not be the thing that earns the
    # warning.
    ceiling = geometry.get("train_context_length") or 0

    best_ctx = 0
    for ctx in _CONTEXT_LADDER:
        if ceiling and ctx > ceiling:
            continue
        prediction = predict(geometry, dict(carried, **{
            "ctx_size": ctx, "parallel": slots, "devices": devices,
            "cache_type_k": cache_type_k, "cache_type_v": cache_type_v,
        }))
        if prediction["vram"]["upper_mib"] <= usable:
            best_ctx = ctx
            break

    available = host.get("mem_available_mib") or 0
    # Leave the host most of its memory: the prompt cache is a nice-to-have and
    # swapping it costs more than missing it.
    cache_budget = max(2048, int(available * 0.25))

    fixed_b, per_token_b = checkpoint_bytes(geometry, best_ctx, cache_type_k, cache_type_v)
    per_slot = best_ctx // slots
    each_mib = (fixed_b + per_token_b * per_slot) / MIB
    checkpoints = 8
    if each_mib > 0:
        checkpoints = int(cache_budget / (each_mib * slots))
        checkpoints = max(2, min(32, checkpoints))
    # Size the budget to what the checkpoints actually claim rather than to a
    # quarter of RAM the cache will never reach for.
    cache_budget = min(cache_budget, max(2048, int(each_mib * checkpoints * slots) + 256))

    return {
        "ctx_size": best_ctx,
        "per_slot_context": per_slot,
        "parallel": slots,
        "ctx_checkpoints": checkpoints,
        "cache_ram": cache_budget,
        "cache_type_k": cache_type_k,
        "cache_type_v": cache_type_v,
        # llama-server disables --cache-reuse outright for multimodal backends,
        # so recommending a chunk size there recommends a dead flag.
        "cache_reuse": 0 if carried.get("projector_mib") else 256,
        "swa_full": "on" if geometry["supports_swa"] else "off",
        # Auto-fit exists to shrink the context at launch, which is the one
        # thing a cache-aware slot layout cannot tolerate. Off means --fit-ctx
        # has nothing to act on, so it is cleared with it.
        "fit": "off",
        "fit_ctx": "",
        "cache_idle_slots": "on",
        "checkpoint_each_mib": round(each_mib, 1),
        "fits_vram_mib": round(usable),
        "train_context_length": ceiling,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _read_env_file(path: Path) -> dict:
    """Minimal env reader, so the launchers can call this without Flask."""
    env: dict = {}
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, _, value = stripped.partition("=")
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                    value = value[1:-1]
                env[key.strip()] = value
    except OSError:
        pass
    # Mirror the manager's legacy backfill so the CLI and the UI agree.
    for key in [k for k in env if k.startswith("CHAT_") and not k.startswith(("CHAT_PRIMARY_", "CHAT_SECONDARY_"))]:
        env.setdefault("CHAT_PRIMARY_" + key[len("CHAT_"):], env[key])
    return env


def _nvidia_gpus() -> list[dict]:
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"index": int(parts[0]), "mem_used": int(parts[1]), "mem_total": int(parts[2])})
    return gpus


def _host_memory() -> dict:
    info: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, raw = line.partition(":")
                parts = raw.strip().split()
                if parts and parts[0].isdigit():
                    info[key] = int(parts[0])
    except OSError:
        return {}
    return {
        "mem_total_mib": round(info.get("MemTotal", 0) / 1024),
        "mem_available_mib": round(info.get("MemAvailable", 0) / 1024),
    }


def _format_report(result: dict) -> str:
    if result.get("error"):
        return f"{result['backend']}: {result['error']}"

    geometry, prediction = result["geometry"], result["prediction"]
    vram, host = prediction["vram"], prediction["host"]
    lines = [
        f"{result['backend']}: {geometry['name'] or geometry['architecture']} "
        f"({geometry['architecture']}, {geometry['file_type']})",
        f"  layers            {geometry['layers']} = {geometry['full_attention_layers']} full-attention "
        f"+ {geometry['recurrent_layers']} recurrent",
        f"  context           {prediction['total_context']:,} total / "
        f"{prediction['per_slot_context']:,} per slot x {prediction['slots']} slots",
        f"  KV per token      {prediction['kv_bytes_per_token']:,} bytes",
        "",
        "  VRAM (MiB)",
        f"    weights         {vram['weights_mib']:>8,}",
        f"    projector       {vram['projector_mib']:>8,}",
        f"    KV cache        {vram['kv_mib']:>8,}",
        f"    recurrent       {vram['recurrent_mib']:>8,}",
        f"    draft           {vram['draft_mib']:>8,}",
        f"    compute+ctx     {vram['estimated_mib']:>8,}  (estimated, +/-{int(COMPUTE_UNCERTAINTY*100)}%)",
        f"    total           {vram['total_mib']:>8,}  (upper bound {vram['upper_mib']:,})",
    ]
    for device in vram["per_device"]:
        lines.append(f"    device {device['device']}        {device['total_mib']:>8,}  "
                     f"(upper {device['upper_mib']:,})")
    lines += [
        "",
        "  Host prompt cache (MiB)",
        f"    per checkpoint  {host['checkpoint_each_mib']:>8}  "
        f"({host['checkpoint_fixed_mib']} fixed + {host['checkpoint_per_token_mib']}/token)",
        f"    required        {host['checkpoint_total_mib']:>8,}  "
        f"({host['checkpoints_per_slot']} checkpoints x {prediction['slots']} slots)",
        f"    --cache-ram     {host['cache_ram_mib']:>8,}",
    ]
    verdict = result.get("verdict") or {}
    if verdict.get("issues"):
        lines.append("")
        for issue in verdict["issues"]:
            lines.append(f"  [{issue['level']}] {issue['text']}")
    return "\n".join(lines)


def _observed_vram_mib(model_path: str) -> int | None:
    """VRAM held by the processes actually serving this model.

    Summing whole-GPU usage would fold in every other backend sharing the
    device, which is exactly the comparison that misleads. Match on the model
    path in each compute process's command line instead.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    total = 0
    matched = False
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        try:
            cmdline = Path(f"/proc/{parts[0]}/cmdline").read_bytes().decode("utf-8", errors="replace")
        except OSError:
            continue
        if model_path and model_path in cmdline:
            matched = True
            total += int(parts[1])
    return total if matched else None


def _validate(result: dict) -> str:
    """Predicted vs observed, so the estimate can be checked rather than trusted."""
    if result.get("error"):
        return result["error"]
    vram = result["prediction"]["vram"]
    observed = _observed_vram_mib(result.get("model_path") or "")
    if observed is None:
        return "no running process was found serving this model; start the backend to validate."

    predicted, exact = vram["total_mib"], vram["exact_mib"]
    lines = [
        "predicted vs observed (processes serving this model only)",
        f"  exact terms       {exact:>8,} MiB  weights + projector + KV + recurrent + draft",
        f"  estimated terms   {vram['estimated_mib']:>8,} MiB  compute buffers + CUDA contexts",
        f"  predicted total   {predicted:>8,} MiB  (upper bound {vram['upper_mib']:,})",
        f"  observed in use   {observed:>8,} MiB",
        f"  difference        {observed - predicted:>+8,} MiB "
        f"({100 * (observed - predicted) / max(1, predicted):+.1f}%)",
        "",
        f"  residual over the exact terms is {observed - exact:,} MiB against an estimate of "
        f"{vram['estimated_mib']:,} MiB.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict the memory footprint of a backend configuration.")
    default_env = Path(__file__).resolve().parent.parent / "config" / "llm-stack.env"
    parser.add_argument("--env", type=Path, default=default_env, help="path to llm-stack.env")
    parser.add_argument("--backend", default="chat-primary", choices=sorted(BACKEND_PREFIXES),
                        help="which backend to price")
    parser.add_argument("--model", help="price this model file instead of the backend's configured one")
    parser.add_argument("--mmproj", help="multimodal projector accompanying --model")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override one setting, repeatable. A launcher passes the values it "
                             "actually resolved, so the report describes the real launch rather "
                             "than a second reading of the env file.")
    parser.add_argument("--json", action="store_true", help="emit the full payload as JSON")
    parser.add_argument("--field", metavar="PATH",
                        help="print one dotted value and exit, e.g. geometry.supports_swa")
    parser.add_argument("--validate", action="store_true", help="compare the prediction against nvidia-smi")
    parser.add_argument("--quiet", action="store_true", help="print only errors and warnings")
    args = parser.parse_args(argv)

    env = _read_env_file(args.env)
    gpus, host = _nvidia_gpus(), _host_memory()
    overrides: dict = {}
    if args.model:
        overrides["model_path"] = args.model
    if args.mmproj:
        overrides["mmproj_path"] = args.mmproj
    for item in args.set:
        key, _, value = item.partition("=")
        key = key.strip()
        if key in OVERRIDABLE_SETTINGS and value != "":
            overrides[key] = value
        elif key not in OVERRIDABLE_SETTINGS:
            print(f"ignoring unknown setting {key!r}", file=sys.stderr)

    result = budget_for(env, args.backend, gpus, host, overrides=overrides)

    if args.field:
        # Shell callers want one value, not a document. Booleans print as
        # true/false so `[[ "$(... --field ...)" == "true" ]]` reads naturally.
        value = result
        for part in args.field.split("."):
            if not isinstance(value, dict) or part not in value:
                return 2
            value = value[part]
        print("true" if value is True else "false" if value is False else value)
        return 0

    if args.json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif args.quiet:
        for issue in (result.get("verdict") or {}).get("issues", []):
            if issue["level"] in {"error", "warn"}:
                print(f"[{issue['level']}] {issue['text']}")
        if result.get("error"):
            print(result["error"])
    else:
        print(_format_report(result))
        if args.validate:
            print()
            print(_validate(result))

    if result.get("error"):
        return 2
    return 0 if (result.get("verdict") or {}).get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
