#!/usr/bin/env bash
# Ubuntu 24.04 package/toolkit installer. Does not install or replace NVIDIA drivers.
set -euo pipefail

MODE="${1:---full}"
if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo: sudo bash $0 ${MODE}" >&2
  exit 1
fi

source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" || "$(uname -m)" != "x86_64" ]]; then
  echo "Supported target is Ubuntu 24.04 x86_64." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "A working NVIDIA driver is required before bootstrap. Fix nvidia-smi and retry." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl wget gnupg git build-essential cmake ninja-build pkg-config \
  python3 python3-dev python3-pip python3-venv nginx jq iproute2 ufw sudo

install_node22() {
  if command -v node >/dev/null 2>&1 && [[ "$(node -p 'process.versions.node.split(`.`)[0]')" == "22" ]]; then
    return
  fi
  mkdir -p /etc/apt/keyrings
  curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
    | gpg --dearmor --yes -o /etc/apt/keyrings/nodesource.gpg
  echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
    > /etc/apt/sources.list.d/nodesource.list
  apt-get update
  apt-get install -y nodejs
  [[ "$(node -p 'process.versions.node.split(`.`)[0]')" == "22" ]] || { echo "Node.js 22 installation failed" >&2; exit 1; }
}
install_node22

if [[ "${MODE}" == "--bootstrap" ]]; then
  exit 0
fi

DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
case "${DRIVER_CUDA}" in
  13.3|13.4|13.5|13.6|13.7|13.8|13.9) TOOLKIT="13-3" ;;
  13.*) TOOLKIT="13-0" ;;
  12.8|12.9) TOOLKIT="12-8" ;;
  *) echo "NVIDIA driver reports CUDA compatibility ${DRIVER_CUDA:-unknown}; need 12.8 or newer." >&2; exit 1 ;;
esac

NVCC_RELEASE="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1 || true)"
DESIRED_RELEASE="${TOOLKIT/-/.}"
if [[ "${NVCC_RELEASE}" != "${DESIRED_RELEASE}" ]]; then
  KEYRING=/tmp/cuda-keyring_1.1-1_all.deb
  curl -fL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o "${KEYRING}"
  dpkg -i "${KEYRING}"
  rm -f "${KEYRING}"
  apt-get update
  apt-get install -y "cuda-toolkit-${TOOLKIT}"
fi

NVCC_PATH="$(find "/usr/local/cuda-${DESIRED_RELEASE}" /usr/local/cuda -maxdepth 2 -type f -name nvcc -print -quit 2>/dev/null || true)"
[[ -n "${NVCC_PATH}" ]] || NVCC_PATH="$(command -v nvcc || true)"
[[ -x "${NVCC_PATH}" ]] || { echo "CUDA toolkit installed but nvcc was not found." >&2; exit 1; }
echo "CUDA compiler: ${NVCC_PATH}"
