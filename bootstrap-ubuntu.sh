#!/usr/bin/env bash
# One-command Ubuntu bootstrap: prerequisites + manager only. The UI installs the stack.
set -euo pipefail

if [[ "${1:-}" == "--dry-run" ]]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  bash -n "${ROOT_DIR}/install.sh" "${ROOT_DIR}/scripts/"*.sh
  python3 -m py_compile "${ROOT_DIR}/scripts/setup_engine.py" "${ROOT_DIR}/web/app.py"
  echo "Bootstrap dry-run checks passed; no system changes made."
  exit 0
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo, for example: curl -fsSL <bootstrap-url> | sudo bash" >&2
  exit 1
fi

if [[ ! -f /etc/os-release ]] || ! grep -q '^ID=ubuntu' /etc/os-release || ! grep -q '^VERSION_ID="\?24\.04"\?' /etc/os-release; then
  echo "Supported target is Ubuntu 24.04." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "Install a working NVIDIA driver first; nvidia-smi must succeed." >&2
  exit 1
fi
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl git sudo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
if [[ -n "${LLM_STACK_SOURCE_DIR:-}" ]]; then
  STACK_DIR="${LLM_STACK_SOURCE_DIR}"
elif [[ -f "${SCRIPT_DIR}/web/app.py" ]]; then
  STACK_DIR="${SCRIPT_DIR}"
else
  STACK_DIR="${LLM_STACK_INSTALL_DIR:-/opt/llm-stack-manager}"
  if ! id llmstack >/dev/null 2>&1; then
    useradd --system --create-home --home-dir /var/lib/llm-stack-manager --shell /bin/bash llmstack
  fi
  if [[ ! -d "${STACK_DIR}/.git" ]]; then
    mkdir -p "$(dirname "${STACK_DIR}")"
    git clone https://github.com/Ellian-Eorwyn/llm-stack-manager.git "${STACK_DIR}"
  fi
  chown -R llmstack:llmstack "${STACK_DIR}"
fi

bash "${STACK_DIR}/scripts/install-system-dependencies.sh" --bootstrap
bash "${STACK_DIR}/install.sh" --manager-only

LAN_IFACE="$(ip route show default | awk 'NR==1 {print $5}')"
LAN_IP="$(ip -o -4 addr show dev "${LAN_IFACE}" scope global | awk '{split($4,a,"/"); print a[1]; exit}')"
if ufw status | grep -q '^Status: active'; then
  LAN_CIDR="$(ip -o -f inet addr show dev "${LAN_IFACE}" scope global | awk 'NR==1 {print $4}')"
  [[ -n "${LAN_CIDR}" ]] && ufw allow from "${LAN_CIDR}" to any port 8077 proto tcp >/dev/null
else
  echo "WARNING: UFW is not active; the manager is reachable by every device that can route to this host." >&2
fi
echo
echo "LLM Stack Manager is ready."
echo "Open: http://${LAN_IP:-127.0.0.1}:8077"
echo "This interface is unauthenticated. Keep it on a trusted LAN and never port-forward it."
