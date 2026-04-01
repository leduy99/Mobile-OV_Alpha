#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-mobileov}"
PY_BIN="${2:-}"

if [[ -z "${PY_BIN}" ]]; then
  PY_BIN="$(conda run -n "${ENV_NAME}" which python)"
fi

echo "[validate] using python: ${PY_BIN}"

PYTHONPATH="${ROOT_DIR}" PYTHONNOUSERSITE=1 "${PY_BIN}" - <<'PY'
import torch, torchvision, torchaudio, diffusers, transformers, peft
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("torchaudio", torchaudio.__version__)
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("cuda_available", torch.cuda.is_available())
PY

PYTHONPATH="${ROOT_DIR}" PYTHONNOUSERSITE=1 "${PY_BIN}" "${ROOT_DIR}/tools/train_stage1_teacher_free.py" --help >/dev/null
PYTHONPATH="${ROOT_DIR}" PYTHONNOUSERSITE=1 "${PY_BIN}" "${ROOT_DIR}/tools/inference/test_q1_student_video.py" --help >/dev/null
PYTHONPATH="${ROOT_DIR}" PYTHONNOUSERSITE=1 "${PY_BIN}" "${ROOT_DIR}/tools/data_prepare/encode_laion_coyo_images_sana_ar.py" --help >/dev/null

echo "[validate] trainer, inference, and data-prep help checks passed"
