# shellcheck shell=bash
# =============================================================================
# backend-preflight.sh
# Shared helpers for the llama.cpp launcher scripts.
#
# Two problems these solve. First, llama-server accepts flags it cannot act on:
# --swa-full on a model without sliding-window attention logs a warning and is
# ignored, and --fit-ctx does nothing when --fit is off. Both looked like
# working configuration for months. Rather than passing them and hoping someone
# reads the journal, the launcher checks and says so.
#
# Second, nothing recorded what a configuration was expected to cost until it
# either fit or crashed. preflight_report writes the predicted footprint into
# the journal immediately before exec, next to what llama-server then actually
# allocates.
#
# Every function degrades to permissive behaviour when web/budget.py cannot
# run: a launcher must never fail to start a backend because a helper could
# not form an opinion about it.
# =============================================================================

BUDGET_PY="${BUDGET_PY:-${STACK_DIR}/web/budget.py}"

# Which platform's rules to apply. `uname -s` unless overridden.
#
# The override exists so the Linux placement logic below stays testable on a
# Mac and the Metal logic on Linux -- the same reason `platforms.set_active`
# exists on the Python side. Without it, half of this file would only ever be
# exercised on half of the runners, which is exactly how the macOS support that
# preceded this decayed into four silent failures.
_bp_platform() {
    echo "${LLM_STACK_PLATFORM:-$(uname -s)}"
}


# Keep a Metal backend's buffers wired while it sits idle.
#
# llama.cpp holds its Metal buffers in a residency set and keeps asking for
# residency only for GGML_METAL_RESIDENCY_KEEP_ALIVE_S after the last
# computation -- three minutes by default. After that the buffers become
# ordinary pageable memory, and on a box running several models macOS
# compresses and swaps them: measured on the Studio, 21.5 GB of the idle 27B
# model's 37 GB was compressed or swapped, and the next request paid to bring
# it back. A server whose job is to answer at any moment wants the opposite.
#
# 100 days rather than "forever": llama.cpp counts the window in 5 ms ticks in
# an int, and 8,640,000 s is comfortably inside it. The heartbeat that keeps the
# set resident costs ~0.1% CPU. An explicit GGML_METAL_RESIDENCY_KEEP_ALIVE_S
# wins; METAL_KEEP_MODELS_RESIDENT=off restores llama.cpp's default.
metal_keep_resident() {
    [[ "$(_bp_platform)" == "Darwin" ]] || return 0
    [[ "${METAL_KEEP_MODELS_RESIDENT:-on}" == "on" ]] || return 0
    [[ -n "${GGML_METAL_RESIDENCY_KEEP_ALIVE_S:-}" ]] && return 0
    export GGML_METAL_RESIDENCY_KEEP_ALIVE_S=8640000
}


# Ask the budget model one question about a model file. Prints nothing and
# returns non-zero when the answer is unavailable.
budget_field() {
    local model_path="$1" field="$2"
    [[ -f "${BUDGET_PY}" && -f "${model_path}" ]] || return 1
    python3 "${BUDGET_PY}" --env "${STACK_DIR}/config/llm-stack.env" \
        --model "${model_path}" --field "${field}" 2>/dev/null
}

# True when --swa-full would do something for this model. Unknown counts as
# supported, so an unreadable model keeps the operator's setting.
model_supports_swa() {
    local answer
    answer="$(budget_field "$1" geometry.supports_swa)" || return 0
    [[ "${answer}" == "true" ]]
}

# Append --swa-full when the model has sliding-window attention, and explain
# the omission when it does not.
add_swa_full_opt() {
    local prefix="$1" setting="$2" model_path="$3"
    [[ "${setting}" == "on" ]] || return 0
    if model_supports_swa "${model_path}"; then
        OPTS+=(--swa-full)
    else
        echo "${prefix} Ignoring Full SWA KV Cache: this model has no sliding-window attention, so --swa-full has no effect."
    fi
}

# Append --fit-ctx only when auto-fit is on to act on it.
add_fit_ctx_opt() {
    local prefix="$1" fit="$2" fit_ctx="$3"
    [[ -n "${fit_ctx}" && "${fit_ctx}" != "0" ]] || return 0
    if [[ "${fit}" == "off" ]]; then
        echo "${prefix} Ignoring Minimum Fit Context ${fit_ctx}: auto-fit is off, so --fit-ctx has no effect."
    else
        OPTS+=(--fit-ctx "${fit_ctx}")
    fi
}

# Build the GPU split flags this llama.cpp build can actually act on.
#
# Two of the four modes llama-server advertises cannot run on a CUDA build of
# the pinned revision, and both fail *after* the launcher has exec'd, so the
# only symptom is a systemd restart loop:
#
#   row     Upstream 74976e1ae ("CUDA: remove -sm row") deleted the CUDA split
#           buffer implementation. make_gpu_buft_list (src/llama-model.cpp)
#           throws "device CUDA0 does not support split buffers" on the first
#           device, so this dies even with one GPU. Only SYCL still implements
#           it. Nothing about the config can make it work here.
#
#   tensor  Tensor parallelism is unimplemented for hybrid/recurrent
#           architectures. llm_arch_supports_sm_tensor (src/llama-arch.cpp)
#           blacklists every hybrid arch *except* qwen35, qwen35moe and
#           qwen3next, which were added to llm_arch_is_hybrid and missed in the
#           blacklist. Those three therefore pass the arch gate and then abort
#           in the meta backend during the warmup decode:
#               ggml.c: GGML_ASSERT(obj_new) failed
#               ggml_new_object: not enough space in the context's memory pool
#           Every Qwen3.5/3.8 GGUF in models/ is qwen35, so the whole family is
#           affected. Refusing here is what keeps the backend up.
#
# In tensor mode llama.cpp folds all visible GPUs into a single "Meta device",
# which changes what the other placement flags mean: --tensor-split describes a
# ratio between devices that no longer exist separately, and --main-gpu selects
# among them. Both are dropped rather than passed and ignored. --fit is inert
# too (common/fit.cpp refuses SPLIT_MODE_TENSOR and downgrades to a warning),
# and flash attention is mandatory (src/llama-context.cpp errors without it).
#
# Populates SPLIT_OPTS, plus SPLIT_MODE_EFFECTIVE / TENSOR_SPLIT_EFFECTIVE /
# MAIN_GPU_EFFECTIVE for the launcher's own startup banner, so the banner
# reports the placement llama-server is actually given rather than the one that
# was asked for. Call as:
#
#   resolve_split_opts "[llm-a]" "${MODE}" "${MODEL}" \
#       "${TENSOR_SPLIT}" "${MAIN_GPU}" "${FLASH_ATTN}"
#
# Pass an empty main_gpu for the backends that never emitted --main-gpu
# (embed, rerank); the helper will not introduce one.
resolve_split_opts() {
    local prefix="$1" mode="$2" model_path="$3"
    local tensor_split="$4" main_gpu="$5" flash_attn="${6:-auto}"

    SPLIT_OPTS=()
    SPLIT_MODE_EFFECTIVE=""
    TENSOR_SPLIT_EFFECTIVE=""
    MAIN_GPU_EFFECTIVE=""
    [[ -n "${mode}" ]] || mode="layer"

    # Apple silicon has one device and one pool of memory, so every mode that
    # exists to divide a model between cards is inapplicable rather than
    # unsupported. `none` is the honest description of what Metal does, and it
    # is what the launcher uses: `layer` on a single device is `none` with extra
    # steps, and passing --tensor-split or --main-gpu asks llama-server to
    # choose between GPUs there is only one of.
    if [[ "$(_bp_platform)" == "Darwin" ]]; then
        if [[ "${mode}" != "none" ]]; then
            echo "${prefix} Split Mode '${mode}' does not apply on Metal: one device, one unified memory pool. Using 'none'."
        fi
        SPLIT_OPTS+=(--split-mode none)
        SPLIT_MODE_EFFECTIVE="none"
        [[ -n "${tensor_split}" ]] && \
            echo "${prefix} Ignoring Tensor Split '${tensor_split}': there is one Metal device to split across."
        [[ -n "${main_gpu}" ]] && [[ "${main_gpu}" != "0" ]] && \
            echo "${prefix} Ignoring Main GPU Index ${main_gpu}: there is one Metal device."
        return 0
    fi

    case "${mode}" in
        row)
            echo "${prefix} Ignoring Split Mode 'row': this CUDA build has no split-buffer support, so llama-server would fail to load the model. Using 'layer' instead."
            mode="layer"
            ;;
        tensor)
            _split_mode_vet_tensor "${prefix}" "${model_path}" "${flash_attn}"
            mode="${_SPLIT_MODE_RESOLVED}"
            ;;
        none|layer)
            ;;
        *)
            echo "${prefix} Ignoring Split Mode '${mode}': expected none, layer, or tensor. Using 'layer' instead."
            mode="layer"
            ;;
    esac

    SPLIT_OPTS+=(--split-mode "${mode}")
    SPLIT_MODE_EFFECTIVE="${mode}"

    if [[ "${mode}" == "tensor" ]]; then
        # Deliberately no --tensor-split / --main-gpu: see the header comment.
        [[ -n "${tensor_split}" ]] && \
            echo "${prefix} Ignoring Tensor Split '${tensor_split}': split-mode=tensor merges the visible GPUs into one device, so there is no ratio between them to set."
        [[ -n "${main_gpu}" ]] && \
            echo "${prefix} Ignoring Main GPU Index ${main_gpu}: split-mode=tensor merges the visible GPUs into one device, so there is no main GPU to choose."
        return 0
    fi

    if [[ -n "${main_gpu}" ]]; then
        SPLIT_OPTS+=(--main-gpu "${main_gpu}")
        MAIN_GPU_EFFECTIVE="${main_gpu}"
    fi
    # An empty ratio must be omitted, not passed as "": llama.cpp reads
    # `--tensor-split ""` as an explicit empty split and refuses it.
    if [[ -n "${tensor_split}" ]]; then
        SPLIT_OPTS+=(--tensor-split "${tensor_split}")
        TENSOR_SPLIT_EFFECTIVE="${tensor_split}"
    fi
    return 0
}

# Expand a tensor-split ratio of `auto` into an even share per visible device.
#
# "1" for one device, "1,1" for two. llama.cpp does not understand the literal
# string `auto`, so it has to be expanded before it is passed -- and against the
# *visible* devices rather than every device on the host, because a slot pinned
# to one card of two wants "1", not "1,1".
auto_tensor_split() {
    local ratio="$1" devices="${2//[[:space:]]/}"
    if [[ -n "${ratio}" && "${ratio}" != "auto" ]]; then
        printf '%s' "${ratio}"
        return 0
    fi
    if [[ -z "${ratio}" ]]; then
        printf ''
        return 0
    fi
    local count=0 part split="1" i
    IFS=',' read -ra parts <<< "${devices}"
    for part in ${parts[@]+"${parts[@]}"}; do
        [[ -n "${part}" ]] && count=$((count + 1))
    done
    if ((count < 1)); then count=1; fi
    for ((i = 1; i < count; i++)); do split+=",1"; done
    printf '%s' "${split}"
}

# Add --device, if the configured device exists on this machine's backend.
#
# The device name is backend-specific: a CUDA build enumerates CUDA0, CUDA1, a
# Metal build enumerates Metal0. The shipped default is CUDA0, so an Apple
# silicon host that never edited it would pass `--device CUDA0` to a Metal
# build, which fails at load with a device-not-found rather than falling back --
# the same shape of failure as the split modes above, arriving after exec and
# looking like a crash loop.
#
# Dropped rather than translated: on a single-device Metal machine there is
# nothing for --device to choose between, and silently rewriting an operator's
# CUDA0 into Metal0 would hide a config that is wrong for the host.
add_device_opt() {
    local prefix="$1" device="$2"
    [[ -n "${device}" ]] || return 0
    if [[ "$(_bp_platform)" == "Darwin" ]]; then
        # llama.cpp names it MTL0, not Metal0 -- `--list-devices` prints
        # "MTL0: Apple M1 Pro". Matching on "Metal" refuses the correct name.
        if [[ "${device}" != MTL* ]]; then
            echo "${prefix} Ignoring Device '${device}': this is a Metal build, which enumerates MTL0. Letting llama-server choose."
            return 0
        fi
    fi
    OPTS+=(--device "${device}")
    return 0
}

# Decide whether split-mode=tensor can run against this model, setting
# _SPLIT_MODE_RESOLVED to the mode to actually use. Falls back to 'layer' with a
# reason on every refusal.
#
# Note this degrades to 'layer' rather than to the operator's setting when the
# model cannot be read. The rule elsewhere in this file is that a helper must
# never stop a backend starting, and falling back still starts it — in the mode
# known to work. Passing an unvetted 'tensor' through is what produces the
# core-dump loop, so permissiveness here would defeat the check entirely.
_split_mode_vet_tensor() {
    local prefix="$1" model_path="$2" flash_attn="$3"

    if [[ "${flash_attn}" == "off" ]]; then
        echo "${prefix} Ignoring Split Mode 'tensor': it requires flash attention, which is turned off for this backend. Using 'layer' instead."
        _SPLIT_MODE_RESOLVED=layer
        return 0
    fi

    local is_hybrid arch
    if ! is_hybrid="$(budget_field "${model_path}" geometry.is_hybrid)" || [[ -z "${is_hybrid}" ]]; then
        echo "${prefix} Ignoring Split Mode 'tensor': could not read the architecture of $(basename "${model_path}") to confirm tensor parallelism supports it. Using 'layer' instead."
        _SPLIT_MODE_RESOLVED=layer
        return 0
    fi

    if [[ "${is_hybrid}" == "true" ]]; then
        arch="$(budget_field "${model_path}" geometry.architecture)" || arch="unknown"
        echo "${prefix} Ignoring Split Mode 'tensor': llama.cpp has not implemented tensor parallelism for hybrid attention models, and this one is ${arch:-unknown}. Using 'layer' instead."
        _SPLIT_MODE_RESOLVED=layer
        return 0
    fi

    echo "${prefix} Split Mode 'tensor' is experimental: auto-fit does not apply to it, so context and layer counts are used exactly as configured."
    _SPLIT_MODE_RESOLVED=tensor
}

# Append the --chat-template-kwargs the backend should start with.
#
# These are defaults the template sees when a request carries no
# chat_template_kwargs of its own; the proxy overrides them per endpoint. Both
# settings are only meaningful under a template that reads them — Qwen 3.8 reads
# all three of enable_thinking, preserve_thinking and reasoning_effort, older
# Qwen templates read only the first two, and a template that reads none is
# unaffected either way.
#
# reasoning_effort is validated here rather than passed through: the Qwen 3.8
# template raises on an unrecognized level, which would fail every request
# against this backend rather than degrade.
add_chat_template_kwargs_opt() {
    local prefix="$1" preserve="$2" effort="$3"
    local -a pairs=()
    [[ "${preserve}" == "on" ]] && pairs+=('"preserve_thinking": true')
    if [[ -n "${effort}" ]]; then
        case "${effort}" in
            xhigh|medium|low)
                pairs+=("\"reasoning_effort\": \"${effort}\"")
                ;;
            *)
                echo "${prefix} Ignoring Reasoning Effort '${effort}': expected xhigh, medium, or low."
                ;;
        esac
    fi
    [[ ${#pairs[@]} -gt 0 ]] || return 0
    local joined
    printf -v joined '%s, ' "${pairs[@]}"
    OPTS+=(--chat-template-kwargs "{${joined%, }}")
}

# Record the predicted memory footprint and any configuration warnings, so the
# journal carries the prediction alongside llama-server's own allocation log.
#
# The values come from the launcher rather than from a second reading of the
# env file, so the report describes the process about to start. Re-deriving
# settings independently is precisely how --fit-ctx stayed live after it had
# been cleared. Call as:
#
#   preflight_report "[llm-a]" llm-a \
#       "${MODEL}" "${MMPROJ}" ctx_size="${CTX}" parallel="${SLOTS}" ...
preflight_report() {
    local prefix="$1" backend="$2" model_path="$3" mmproj_path="$4"
    shift 4
    [[ -f "${BUDGET_PY}" && -f "${model_path}" ]] || return 0

    local args=(--env "${STACK_DIR}/config/llm-stack.env" --backend "${backend}" --model "${model_path}")
    [[ -n "${mmproj_path}" && -f "${mmproj_path}" ]] && args+=(--mmproj "${mmproj_path}")
    local setting
    for setting in "$@"; do
        [[ "${setting}" == *=?* ]] && args+=(--set "${setting}")
    done

    local report
    report="$(python3 "${BUDGET_PY}" "${args[@]}" 2>/dev/null)" || true
    [[ -n "${report:-}" ]] || return 0
    while IFS= read -r line; do
        echo "${prefix} ${line}"
    done <<< "${report}"
}
