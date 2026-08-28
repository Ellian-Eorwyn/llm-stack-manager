#!/usr/bin/env python3
"""The slot registry.

Transcribed from the launcher scripts rather than reimagined: every fallback
chain and default here is the one that shipped, and `tests/test_launchers.py`
holds the resulting command lines against a golden file captured from the
original scripts.

Order matters. The flags come out in the order below because that is the order
the launchers emitted them, and keeping it means the golden file can be a
straight comparison rather than a set comparison -- which would not have caught
a flag moving from before a positional to after it.
"""

from __future__ import annotations

from .spec import Flag, Slot, Toggle

# The flags every llama.cpp slot takes, in emission order. Suffixes are
# relative to the slot's prefix; a leading "!" marks an absolute key.
COMMON_FLAGS = (
    Flag("--ctx-size",        ("CTX_SIZE",)),
    Flag("--n-gpu-layers",    ("N_GPU_LAYERS", "!CHAT_N_GPU_LAYERS"), "-1"),
    # --split-mode / --tensor-split / --main-gpu are inserted here by the
    # engine: they are vetted against the platform and the model, not simply
    # read (one Metal device cannot be split, and CUDA dropped `row`).
    Flag("--batch-size",      ("BATCH_SIZE",)),
    Flag("--ubatch-size",     ("UBATCH_SIZE", "!CHAT_UBATCH_SIZE"), "512"),
    Flag("--parallel",        ("N_PARALLEL",), "1"),
    Flag("--threads",         ("THREADS",), "-1"),
    Flag("--threads-batch",   ("THREADS_BATCH",), "-1"),
    Flag("--cache-type-k",    ("CACHE_TYPE_K",), "q8_0"),
    Flag("--cache-type-v",    ("CACHE_TYPE_V",), "q8_0"),
    Flag("--flash-attn",      ("FLASH_ATTN",), "on"),
    Flag("--temp",            ("TEMP",), "1.0"),
    Flag("--top-p",           ("TOP_P",), "0.95"),
    Flag("--top-k",           ("TOP_K",), "20"),
    Flag("--min-p",           ("MIN_P",), "0.00"),
    Flag("--reasoning-format", ("REASONING_FORMAT",), "none"),
    Flag("--fit",             ("FIT",), "on"),
)

# Emitted after the literals, in this order.
COMMON_TOGGLES = (
    Toggle("--log-prefix", "LOG_PREFIX", when="true", default="true"),
    Toggle("--metrics",    "METRICS",    when="on",   default="on"),
    Toggle("--no-mmap",    "NO_MMAP",    when="true", default="false"),
    Toggle("--mlock",      "MLOCK",      when="true", default="false"),
    Toggle("--jinja",      "JINJA",      when="on",   default="off"),
)

SLOTS = {
    "embed": Slot(
        name="embed",
        prefix="EMBED",
        model_keys=("!EMBEDDING_MODEL_PATH",),
        alias_default="embed",
        port_default="8005",
        # The embedding slot has always defaulted its KV cache to f16 rather
        # than the q8_0 the chat backends use.
        defaults={"CACHE_TYPE_K": "f16", "CACHE_TYPE_V": "f16"},
        literals=("--embedding", "--pooling", "mean"),
    ),
    "rerank": Slot(
        name="rerank",
        prefix="RERANK",
        model_keys=("!RERANKER_MODEL_PATH",),
        # Not "rerank": the router's INI section name overwrites the child's
        # --alias, and callers send "rank". See docs/model-router.md.
        alias_default="rank",
        port_default="8006",
        defaults={"CACHE_TYPE_K": "f16", "CACHE_TYPE_V": "f16"},
        literals=("--reranking",),
    ),
    "ocr": Slot(
        name="ocr",
        prefix="OCR",
        model_keys=("!OCR_MODEL_PATH",),
        alias_default="ocr",
        port_default="8009",
        host_keys=("OCR_HOST", "!LISTEN_HOST"),
        # A layout model wants determinism, not sampling.
        defaults={"CACHE_TYPE_K": "f16", "CACHE_TYPE_V": "f16",
                  "TEMP": "0.1", "TOP_K": "1", "FIT": "off",
                  "CTX_SIZE": "8192", "BATCH_SIZE": "2048"},
        # OCR never passed one; llama-server's own default applies.
        omit=frozenset({"--reasoning-format"}),
        extra_toggles=(
            Toggle("--kv-offload",     "KV_OFFLOAD",     when="on", default="on",
                   otherwise="--no-kv-offload"),
            Toggle("--op-offload",     "OP_OFFLOAD",     when="on", default="on",
                   otherwise="--no-op-offload"),
            Toggle("--mmproj-offload", "MMPROJ_OFFLOAD", when="on", default="on",
                   otherwise="--no-mmproj-offload"),
        ),
        custom_args_key="OCR_CUSTOM_ARGS_JSON",
    ),
}
