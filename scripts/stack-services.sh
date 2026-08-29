#!/usr/bin/env bash
# The shell's one list of what this stack runs.
#
# Four scripts used to carry their own copy of this -- restore-active-stack.sh,
# activate-selected-stack.sh, llm-stack-manager and this file -- and they had
# drifted: two still named `think` and `nothink` as current services a year
# after they were retired, and none of them agreed on whether the transcription
# sidecar counted.
#
# `web/backends/slots.py` is the registry on the Python side.
# `STACK_MODEL_BACKENDS` below must name exactly the slots in it, and
# `tests/test_slot_registry.py` fails if it does not -- the same arrangement
# `update.sh` and `deploy.BACKEND_SENSITIVE_PATHS` already have.

# One unit per servable position. The same six, in the same order, as
# `backends.SLOTS`.
STACK_MODEL_BACKENDS=(
  chat-backend-dense
  chat-backend2
  embed
  rerank
  task
  ocr
)

# In front of a slot, not one of them.
STACK_PROXY_SERVICES=(
  chat-proxy
  chat-proxy2
)

STACK_SUPPORT_SERVICES=(
  llm-manager
  "${STACK_PROXY_SERVICES[@]}"
  glmocr-sdk
  playwright-server
)

# Units this stack no longer installs, kept by name because a host that predates
# their retirement still has them and they must be stopped rather than left
# running against a launcher that no longer exists.
STACK_RETIRED_SERVICES=(
  think
  nothink
  embed2
  chat-backend
  chat-backend-moe
)

STACK_CORE_SERVICES=(
  llm-manager
  "${STACK_MODEL_BACKENDS[@]}"
  "${STACK_PROXY_SERVICES[@]}"
  llama-router
  glmocr-sdk
  playwright-server
  honcho-api
  honcho-deriver
)

STACK_LEGACY_CORE_SERVICES=(
  qwen-chat-backend
  qwen-chat-backend-27b
  qwen-chat-backend-35b
  qwen-chat-proxy
  qwen-embedding
  qwen-reranker
  qwen-task
  qwen-think
  qwen-nothink
)

STACK_OPTIONAL_SERVICES=(
  graphiti
  transcript-backend
  tts-gateway
  tts-backend-kokoro
  tts-backend-chatterbox
  tts-backend-vibevoice
)

STACK_SERVICES=(
  "${STACK_CORE_SERVICES[@]}"
  "${STACK_LEGACY_CORE_SERVICES[@]}"
  "${STACK_OPTIONAL_SERVICES[@]}"
)

# Everything a cutover may need to stop before starting the selected set.
STACK_STOPPABLE_SERVICES=(
  "${STACK_RETIRED_SERVICES[@]}"
  "${STACK_MODEL_BACKENDS[@]}"
  "${STACK_PROXY_SERVICES[@]}"
  llama-router
  glmocr-sdk
  transcript-backend
  honcho-api
  honcho-deriver
  "${STACK_LEGACY_CORE_SERVICES[@]}"
)

# Every unit activate-selected-stack.sh disables before starting the selection.
STACK_SELECTABLE_UNITS=(
  "${STACK_MODEL_BACKENDS[@]}"
  "${STACK_PROXY_SERVICES[@]}"
  glmocr-sdk
  llama-router
  transcript-backend
  playwright-server
  honcho-api
  honcho-deriver
)

# What `update.sh` restarts after a code update. Everything the manager owns
# that can come back in seconds; the model router is excluded because its
# children hold weights it would have to reload.
STACK_UPDATE_RESTART_SERVICES=(
  llm-manager
  "${STACK_MODEL_BACKENDS[@]}"
  "${STACK_PROXY_SERVICES[@]}"
  glmocr-sdk
  transcript-backend
  playwright-server
  honcho-api
  honcho-deriver
)

# True when $1 is one of the remaining arguments. bash 3.2 has no associative
# arrays, and macOS ships bash 3.2.
stack_contains() {
    local needle="$1"; shift
    local item
    for item in "$@"; do
        [[ "${item}" == "${needle}" ]] && return 0
    done
    return 1
}
