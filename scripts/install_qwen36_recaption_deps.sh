#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-mobileov}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
DOWNLOAD_MODEL=0
export MODEL_ID

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
