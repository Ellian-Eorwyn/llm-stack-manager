#!/usr/bin/env bash
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/config/llm-stack.env"
SELECTED=",${LLM_STACK_SELECTED_COMPONENTS:-primary,embedding,task,ocr,glmocr-sdk,searxng,playwright},"

has_component() { [[ "${SELECTED}" == *",$1,"* ]]; }

declare -a ALL_UNITS=(chat-backend-dense chat-proxy chat-backend2 chat-proxy2 embed embed2 task ocr glmocr-sdk rerank playwright-server honcho-api honcho-deriver)
declare -a START_UNITS=()
has_component primary && START_UNITS+=(chat-backend-dense chat-proxy)
has_component secondary && START_UNITS+=(chat-backend2 chat-proxy2)
has_component embedding && START_UNITS+=(embed)
has_component embedding2 && START_UNITS+=(embed2)
has_component reranker && START_UNITS+=(rerank)
has_component task && START_UNITS+=(task)
has_component ocr && START_UNITS+=(ocr)
has_component glmocr-sdk && START_UNITS+=(glmocr-sdk)
has_component playwright && START_UNITS+=(playwright-server)
has_component honcho && START_UNITS+=(honcho-api honcho-deriver)

for unit in "${ALL_UNITS[@]}"; do
  systemctl disable --now "${unit}" 2>/dev/null || true
done
systemctl daemon-reload
for unit in "${START_UNITS[@]}"; do
  systemctl enable "${unit}"
  systemctl restart "${unit}"
done

if ! has_component searxng; then
  [[ -L /etc/uwsgi/apps-enabled/searxng.ini ]] && unlink /etc/uwsgi/apps-enabled/searxng.ini
  [[ -L /etc/nginx/default.d/searxng.conf ]] && unlink /etc/nginx/default.d/searxng.conf
  systemctl restart uwsgi 2>/dev/null || true
  nginx -t >/dev/null 2>&1 && systemctl reload nginx || true
fi

if command -v ufw >/dev/null 2>&1 && ufw status | grep -q '^Status: active'; then
  LAN_IFACE="$(ip route show default | awk 'NR==1 {print $5}')"
  LAN_CIDR="$(ip -o -f inet addr show dev "${LAN_IFACE}" scope global | awk 'NR==1 {print $4}')"
  if [[ -n "${LAN_CIDR}" ]]; then
    PORTS=(8077)
    has_component primary && PORTS+=(8003 8004 8008)
    has_component secondary && PORTS+=(8103 8104 8108)
    has_component embedding && PORTS+=(8005)
    has_component embedding2 && PORTS+=(8011)
    has_component reranker && PORTS+=(8006)
    has_component task && PORTS+=(8007)
    has_component ocr && PORTS+=(8009)
    has_component glmocr-sdk && PORTS+=(5002)
    (has_component searxng || has_component playwright) && PORTS+=(80)
    for port in "${PORTS[@]}"; do
      ufw allow from "${LAN_CIDR}" to any port "${port}" proto tcp >/dev/null
    done
  fi
else
  echo "WARNING: UFW is not active. This unauthenticated manager must remain on a trusted LAN." >&2
fi

failed=0
for unit in "${START_UNITS[@]}"; do
  for _ in $(seq 1 30); do
    systemctl is-active --quiet "${unit}" && break
    sleep 2
  done
  systemctl is-active --quiet "${unit}" || { echo "Service failed to become active: ${unit}" >&2; failed=1; }
done
exit "${failed}"
