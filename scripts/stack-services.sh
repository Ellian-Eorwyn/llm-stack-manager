#!/usr/bin/env bash
# Shared service lists used by backup/cutover/rollback helpers.

STACK_CORE_SERVICES=(
  llm-manager
  chat-backend
  chat-backend-dense
  chat-backend-moe
  chat-backend-bee
  chat-proxy
  embed
  rerank
  task
  ocr
  glmocr-sdk
  honcho-api
  honcho-deriver
  think
  nothink
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
