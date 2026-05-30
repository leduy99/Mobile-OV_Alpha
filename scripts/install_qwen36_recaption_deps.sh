#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="${ENV_PATH:-${CONDA_ENV_PATH:-/proj/cvl/users/x_fahkh2/envs/mobileov}}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
DOWNLOAD_MODEL=0
export MODEL_ID

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_ROOT="${CACHE_ROOT:-/proj/cvl/users/x_fahkh2/caches}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT}"
export TMPDIR="${TMPDIR:-$CACHE_ROOT}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$TRITON_CACHE_DIR"

for arg in "$@"; do
  case "$arg" in
    --download-model)
      DOWNLOAD_MODEL=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$ENV_PATH/bin/python" ]]; then
    echo "Python env not found: $ENV_PATH/bin/python" >&2
    exit 1
fi
export PATH="$ENV_PATH/bin:$PATH"
export PYTHONNOUSERSITE=1
PYTHON_BIN="${PYTHON_BIN:-$ENV_PATH/bin/python}"
HF_BIN="${HF_BIN:-$ENV_PATH/bin/hf}"
if [[ ! -x "$HF_BIN" ]]; then
    HF_BIN="hf"
fi

echo "Qwen recaption cache:"
echo "  ENV_PATH=$ENV_PATH"
echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"
echo "  TORCH_HOME=$TORCH_HOME"
echo "  PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "  TMPDIR=$TMPDIR"
echo "  TRITON_CACHE_DIR=$TRITON_CACHE_DIR"

"$PYTHON_BIN" -m pip install -U \
  "transformers>=4.57.0" \
  "accelerate>=1.4.0" \
  "huggingface_hub>=0.30.0" \
  "safetensors>=0.4.5" \
  qwen-vl-utils

"$PYTHON_BIN" - <<'PY'
import os
from transformers import AutoConfig
model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3.6-35B-A3B")
try:
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
except Exception as exc:
    raise SystemExit(
        "Qwen3.6 config load failed after dependency install. "
        "Try: python -m pip install -U git+https://github.com/huggingface/transformers.git\n"
        f"Original error: {exc}"
    )
print("Qwen recaption environment OK:", type(cfg), getattr(cfg, "model_type", None))
PY

if [[ "$DOWNLOAD_MODEL" == "1" ]]; then
  "$HF_BIN" download "$MODEL_ID" --repo-type model
fi
