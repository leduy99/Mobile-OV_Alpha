#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-mobileov}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
DOWNLOAD_MODEL=0
export MODEL_ID

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/download_data/checkpoints/huggingface}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

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

source /share_0/conda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1

echo "Qwen recaption cache:"
echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"

python -m pip install -U \
  "transformers>=4.57.0" \
  "accelerate>=1.4.0" \
  "huggingface_hub>=0.30.0" \
  "safetensors>=0.4.5" \
  qwen-vl-utils

python - <<'PY'
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
  hf download "$MODEL_ID" --repo-type model
fi
