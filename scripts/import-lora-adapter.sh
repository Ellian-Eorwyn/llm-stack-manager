#!/usr/bin/env bash
# =============================================================================
# import-lora-adapter.sh
# Convert a PEFT LoRA adapter directory into a GGUF the stack can attach to a
# backend, and drop it in models/loras/.
#
# The adapter is converted, not merged. Merging a 27B would produce a ~54 GB
# fp16 checkpoint to requantise, and needs more RAM than this class of machine
# has; the converted adapter is tens of megabytes and applies on top of a base
# GGUF that is already loaded. It also means several fine-tunes of one base can
# be resident at once and switched between without a reload.
#
# The converter needs the base model's config.json -- not its weights -- to know
# the tensor layout it is writing against. Point --base at the Hugging Face
# snapshot the adapter was trained on.
#
# Usage:
#   scripts/import-lora-adapter.sh --base <hf-snapshot-dir> <adapter-dir> [name]
# =============================================================================
set -euo pipefail
STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_DIR=""
PYTHON="${LORA_IMPORT_PYTHON:-python3}"
OUTTYPE="${LORA_IMPORT_OUTTYPE:-f16}"

usage() {
    sed -n '3,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)    BASE_DIR="$2"; shift 2 ;;
        --python)  PYTHON="$2";   shift 2 ;;
        --outtype) OUTTYPE="$2";  shift 2 ;;
        -h|--help) usage 0 ;;
        --) shift; break ;;
        -*) echo "unknown option: $1" >&2; usage 2 ;;
        *)  break ;;
    esac
done

ADAPTER_DIR="${1:-}"
[[ -n "${ADAPTER_DIR}" ]] || usage 2
NAME="${2:-$(basename "${ADAPTER_DIR}")}"

if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
    echo "not a PEFT adapter directory (no adapter_config.json): ${ADAPTER_DIR}" >&2
    exit 1
fi

CONVERTER="${STACK_DIR}/deps/llama.cpp/convert_lora_to_gguf.py"
if [[ ! -f "${CONVERTER}" ]]; then
    echo "converter not found: ${CONVERTER}" >&2
    echo "deps/llama.cpp is the tree the stack builds llama-server from; the" >&2
    echo "adapter must be converted with the same one that will load it." >&2
    exit 1
fi

BASE_ARGS=()
SANITISED=""
cleanup() { [[ -n "${SANITISED}" ]] && rm -rf "${SANITISED}"; }
trap cleanup EXIT

if [[ -n "${BASE_DIR}" ]]; then
    if [[ ! -f "${BASE_DIR}/config.json" ]]; then
        echo "no config.json in --base ${BASE_DIR}" >&2
        exit 1
    fi
    # A 4-bit training checkpoint declares `quantization_config`, and the
    # converter refuses any quant method it cannot dequantise -- on the hparams
    # alone, before it looks at a single tensor. That guard is about base
    # weights, and this conversion never reads them: the tensors being written
    # are the adapter's own fp32 deltas. So hand it a config with that block
    # removed, symlinking everything else so the tokenizer is still found.
    if "${PYTHON}" -c "
import json, sys
sys.exit(0 if json.load(open('${BASE_DIR}/config.json')).get('quantization_config') else 1)
"; then
        SANITISED="$(mktemp -d)"
        for f in "${BASE_DIR}"/*; do
            [[ "$(basename "${f}")" == "config.json" ]] && continue
            ln -s "${f}" "${SANITISED}/"
        done
        "${PYTHON}" -c "
import json
c = json.load(open('${BASE_DIR}/config.json'))
c.pop('quantization_config', None)
json.dump(c, open('${SANITISED}/config.json', 'w'), indent=2)
"
        echo "base declares a quantised checkpoint; converting against a config"
        echo "with quantization_config removed (adapter tensors are not quantised)"
        BASE_DIR="${SANITISED}"
    fi
    BASE_ARGS=(--base "${BASE_DIR}")
fi
# With no --base the converter reads base_model_name_or_path from the adapter
# config and fetches that model's config from Hugging Face, which needs network.

OUT_DIR="${STACK_DIR}/models/loras"
OUT="${OUT_DIR}/${NAME}.gguf"
mkdir -p "${OUT_DIR}"

echo "converting ${ADAPTER_DIR}"
echo "        -> ${OUT}"
"${PYTHON}" "${CONVERTER}" "${BASE_ARGS[@]}" \
    --outtype "${OUTTYPE}" --outfile "${OUT}" "${ADAPTER_DIR}"

# A converted file that is not recognisably an adapter would be accepted by the
# config UI and then refused by llama-server after exec, so check it here.
"${PYTHON}" -c "
import sys
sys.path.insert(0, '${STACK_DIR}/web')
from budget import read_gguf_metadata
m = read_gguf_metadata('${OUT}')
kind, adapter = m.get('general.type'), m.get('adapter.type')
print(f'  type={kind!r} adapter={adapter!r} arch={m.get(\"general.architecture\")!r} '
      f'alpha={m.get(\"adapter.lora.alpha\")} tensors={m[\"__tensor_count__\"]}')
if kind != 'adapter' or adapter != 'lora':
    sys.exit('  refusing: converted file is not a LoRA adapter')
"

echo "done. Add '${NAME}.gguf' to a backend's LoRA Adapters field in the config UI."
