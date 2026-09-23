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

from . import options
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
    # The prompt cache. Only the slots that hold a conversation take these; the
    # auxiliary slots omit them.
    Flag("--cache-ram",       ("CACHE_RAM",), "8192"),
    Flag("--ctx-checkpoints", ("CTX_CHECKPOINTS",), "8"),
    Flag("--flash-attn",      ("FLASH_ATTN",), "on"),
    Flag("--temp",            ("TEMP",), "1.0"),
    Flag("--top-p",           ("TOP_P",), "0.95"),
    Flag("--top-k",           ("TOP_K",), "20"),
    Flag("--min-p",           ("MIN_P",), "0.00"),
    # Task only, and between --min-p and --reasoning-format where it emitted
    # them.
    Flag("--presence-penalty", ("PRESENCE_PENALTY",), "0.00"),
    Flag("--repeat-penalty",   ("REPEAT_PENALTY",), "1.00"),
    Flag("--reasoning-format", ("REASONING_FORMAT",), "none"),
    Flag("--fit",             ("FIT",), "on"),
)

# Emitted after the literals, before the slot's own tail.
COMMON_TOGGLES = (
    Toggle("--log-prefix", "LOG_PREFIX", when="true", default="true"),
    Toggle("--metrics",    "METRICS",    when="on",   default="on"),
    Toggle("--no-mmap",    "NO_MMAP",    when="true", default="false"),
    Toggle("--mlock",      "MLOCK",      when="true", default="false"),
)

#: What the memory-fit report carries for a slot holding a conversation, in
#: the order the launchers passed it. An empty suffix is one the launcher
#: resolves rather than reads.
_LARGE_PREFLIGHT = (
    ("ctx_size", "CTX_SIZE"), ("parallel", "N_PARALLEL"), ("ubatch", "UBATCH_SIZE"),
    ("cache_type_k", "CACHE_TYPE_K"), ("cache_type_v", "CACHE_TYPE_V"),
    ("ctx_checkpoints", "CTX_CHECKPOINTS"), ("cache_ram", "CACHE_RAM"),
    ("tensor_split", ""), ("devices", ""),
    ("swa_full", "SWA_FULL"), ("fit", "FIT"), ("fit_ctx", "FIT_CTX"),
    ("spec_method", "SPEC_METHOD"),
)

#: An auxiliary slot has no prompt cache, no auto-fit and no draft model, so it
#: reports the five settings it does have.
_AUX_PREFLIGHT = (
    ("ctx_size", "CTX_SIZE"), ("parallel", "N_PARALLEL"),
    ("cache_type_k", "CACHE_TYPE_K"), ("cache_type_v", "CACHE_TYPE_V"),
    ("tensor_split", ""),
)

#: The prompt-cache and sampling flags an auxiliary slot never passed.
_AUX_OMIT = frozenset({"--cache-ram", "--ctx-checkpoints",
                       "--presence-penalty", "--repeat-penalty"})

#: The three offload flags llama.cpp states both ways round, in the order the
#: launchers emitted them.
_OFFLOAD = (
    Toggle("--kv-offload",     "KV_OFFLOAD",     when="on", default="on",
           otherwise="--no-kv-offload"),
    Toggle("--op-offload",     "OP_OFFLOAD",     when="on", default="on",
           otherwise="--no-op-offload"),
    Toggle("--mmproj-offload", "MMPROJ_OFFLOAD", when="on", default="on",
           otherwise="--no-mmproj-offload"),
)

#: What a large-model slot emits after the common toggles. Both chat slots and
#: the task slot share it apart from where `--mmproj` falls and which shape of
#: template kwargs they ask for, so the two differences are arguments.
def _large_model_tail(kwargs: options.TemplateKwargs, template: options.TemplateFile,
                      mmproj_early: bool = False) -> tuple:
    mmproj = (options.MMProj(),)
    return (
        options.Device(),
        *_OFFLOAD,
        options.SwaFull(),
        *(mmproj if mmproj_early else ()),
        Flag("--fit-target", ("FIT_TARGET",), empty_is_set=True),
        options.FitCtx(),
        Toggle("--cache-idle-slots", "CACHE_IDLE_SLOTS", when="on", default="on",
               otherwise="--no-cache-idle-slots"),
        options.CacheReuse(),
        options.Jinja(),
        kwargs,
        template,
        *(() if mmproj_early else mmproj),
        options.Speculative(),
        # Last, so that a slot with no adapters configured emits exactly the
        # command line it emitted before adapters existed.
        options.Lora(),
    )


SLOTS = {
    # In the order the services panel renders them, which is also the order
    # `telemetry.BACKEND_TARGETS` and `app.SERVICES` used to state separately.
    "llm-a": Slot(
        name="llm-a",
        component="llm-a",
        group="chat",
        label="LLM A",
        desc="Primary model backend",
        config_section="LLM A",
        ports_display="8010 internal",
        probe_host_keys=("!CHAT_BACKEND_HOST",),
        prefix="LLM_A",
        # Three renames deep now. The slot was the bare `CHAT_*` names, then
        # `CHAT_DENSE_*`, then `CHAT_PRIMARY_*`, and is `LLM_A_*`. Each is read
        # behind the one before it, so a config written at any point still
        # starts this backend -- which is the whole reason the rename is safe to
        # do while the fleet spans versions.
        legacy_prefixes=("CHAT_PRIMARY", "CHAT"),
        model_keys=("!LLM_A_MODEL_PATH", "!CHAT_PRIMARY_MODEL_PATH",
                    "!CHAT_DENSE_MODEL_PATH", "!CHAT_MODEL_PATH"),
        alias_keys=("!LLM_A_MODEL_NAME", "!CHAT_PRIMARY_MODEL_NAME",
                    "!CHAT_DENSE_MODEL_NAME"),
        alias_default="chat-dense",
        mmproj_keys=("!LLM_A_MMPROJ_PATH", "!CHAT_PRIMARY_MMPROJ_PATH",
                     "!CHAT_DENSE_MMPROJ_PATH", "!CHAT_MMPROJ_PATH"),
        key_chains={"CTX_SIZE": ("!LLM_A_CTX_SIZE", "!CHAT_PRIMARY_CTX_SIZE",
                                 "!CHAT_DENSE_CTX_SIZE", "!CHAT_CTX_SIZE")},
        # Not LLM_A_PORT: both chat slots bind the shared backend port,
        # and every consumer in the stack -- the proxies, telemetry, health --
        # talks to that name.
        port_keys=("!CHAT_BACKEND_PORT",),
        port_default="8010",
        host_keys=("!CHAT_BACKEND_HOST",),
        defaults={"CTX_SIZE": "32768", "BATCH_SIZE": "2048", "FLASH_ATTN": "auto",
                  "REASONING_FORMAT": "deepseek"},
        omit=frozenset({"--presence-penalty", "--repeat-penalty"}),
        tail=_large_model_tail(options.TemplateKwargs(style="preserve"),
                               options.TemplateFile(keys=("TEMPLATE_ID",))),
        custom_args_keys=("CUSTOM_ARGS_JSON",),
        # `budget.py` knows the slot by what it holds, not by its unit name.
        budget_name="llm-a",
        preflight_fields=_LARGE_PREFLIGHT,
    ),
    "llm-b": Slot(
        name="llm-b",
        component="llm-b",
        group="chat",
        label="LLM B",
        desc="Secondary model backend",
        config_section="LLM B",
        ports_display="8020 internal",
        probe_host_keys=("!CHAT2_BACKEND_HOST",),
        prefix="LLM_B",
        # `CHAT2_*` was the canonical spelling until this rename. The port and
        # host keys keep their old names on purpose: they are the proxy's half
        # of the wiring and section 2.1 freezes the ports.
        legacy_prefixes=("CHAT2",),
        model_keys=("!LLM_B_MODEL_PATH", "!CHAT2_MODEL_PATH"),
        alias_default="chat-moe",
        mmproj_keys=("!LLM_B_MMPROJ_PATH", "!CHAT2_MMPROJ_PATH"),
        port_keys=("!CHAT2_BACKEND_PORT",),
        port_default="8020",
        host_keys=("!CHAT2_BACKEND_HOST",),
        defaults={"CTX_SIZE": "32768", "BATCH_SIZE": "2048", "FLASH_ATTN": "auto",
                  "REASONING_FORMAT": "deepseek"},
        omit=frozenset({"--presence-penalty", "--repeat-penalty"}),
        tail=_large_model_tail(options.TemplateKwargs(style="preserve"),
                               options.TemplateFile(keys=("TEMPLATE_ID",))),
        custom_args_keys=("CUSTOM_ARGS_JSON",),
        budget_name="llm-b",
        preflight_fields=_LARGE_PREFLIGHT,
    ),
    "embed": Slot(
        name="embed",
        component="embedding",
        label="Embedding",
        desc="Embedding model",
        config_section="Embedding",
        ports_display="8005",
        probe_host_keys=("!EMBED_BACKEND_HOST",),
        prefix="EMBED",
        model_keys=("!EMBEDDING_MODEL_PATH",),
        alias_default="embed",
        port_default="8005",
        # No mmproj: an embedding model has no projector, and the setup wizard
        # has recorded that as an empty key since it was written.
        mmproj_keys=(),
        # The embedding slot has always defaulted its KV cache to f16 rather
        # than the q8_0 the chat backends use.
        defaults={"CACHE_TYPE_K": "f16", "CACHE_TYPE_V": "f16"},
        omit=_AUX_OMIT,
        literals=("--embedding", "--pooling", "mean"),
        preflight_fields=_AUX_PREFLIGHT,
        tail=(Toggle("--jinja", "JINJA", when="on", default="off"), options.MMProj()),
    ),
    "rerank": Slot(
        name="rerank",
        component="reranker",
        label="Reranker",
        desc="Reranker model",
        config_section="Reranker",
        ports_display="8006",
        probe_host_keys=("!RERANK_BACKEND_HOST",),
        prefix="RERANK",
        model_keys=("!RERANKER_MODEL_PATH",),
        # Not "rerank": the router's INI section name overwrites the child's
        # --alias, and callers send "rank". See docs/model-router.md.
        alias_default="rank",
        port_default="8006",
        mmproj_keys=(),
        defaults={"CACHE_TYPE_K": "f16", "CACHE_TYPE_V": "f16"},
        omit=_AUX_OMIT,
        literals=("--reranking",),
        preflight_fields=_AUX_PREFLIGHT,
        tail=(Toggle("--jinja", "JINJA", when="on", default="off"), options.MMProj()),
    ),
    "task": Slot(
        name="task",
        component="task",
        label="Task Model",
        desc="Small fast task model",
        config_section="Task Model",
        ports_display="8007",
        probe_host_keys=("!TASK_BACKEND_HOST",),
        prefix="TASK",
        model_keys=("!TASK_MODEL_PATH",),
        alias_default="task",
        port_default="8007",
        defaults={"CTX_SIZE": "32000", "BATCH_SIZE": "2048", "FLASH_ATTN": "auto"},
        # The task slot emits --mmproj early, before --fit-target, where the
        # chat slots emit it last. Declared rather than normalised: the golden
        # file is only worth having if it is compared against unchanged.
        tail=_large_model_tail(options.TemplateKwargs(
                                   style="enable", thinking_keys=("THINKING",)),
                               options.TemplateFile(keys=("CHAT_TEMPLATE_ID",)),
                               mmproj_early=True),
        custom_args_keys=("CUSTOM_ARGS_JSON",),
        preflight_fields=_LARGE_PREFLIGHT,
    ),
    "ocr": Slot(
        name="ocr",
        component="ocr",
        label="OCR Model",
        desc="GLM-OCR llama.cpp model backend",
        config_section="OCR",
        ports_display="8009",
        probe_host_keys=("!OCR_BACKEND_HOST",),
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
        omit=_AUX_OMIT | frozenset({"--reasoning-format"}),
        tail=(Toggle("--jinja", "JINJA", when="on", default="off"),
              *_OFFLOAD,
              options.MMProj()),
        preflight_fields=_AUX_PREFLIGHT,
        custom_args_keys=("CUSTOM_ARGS_JSON",),
    ),
}
