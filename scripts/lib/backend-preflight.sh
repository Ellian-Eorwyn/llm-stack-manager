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
#   preflight_report "[chat-backend-dense]" chat-primary \
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
