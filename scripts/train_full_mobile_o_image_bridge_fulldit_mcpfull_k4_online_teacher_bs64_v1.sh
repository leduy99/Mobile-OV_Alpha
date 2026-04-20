#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-mobileov}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
mkdir -p output/logs
CONFIG_PATH="configs/stage1_teacher_free_full_mobile_o_image_bridge_fulldit_mcpfull_k4_online_teacher_bs64_v1_8gpu_20260420.yaml"
RUN_ROOT="output/stage1_bridge_fulldit_full_mobile_o_smolvlm2_500m_mcpfull_k4_online_teacher_bs64_v1_20260420_8gpu"
LOG_PATH="output/logs/train_full_mobile_o_image_bridge_fulldit_mcpfull_k4_online_teacher_bs64_v1_8gpu.log"
MAX_GPUS="${MAX_GPUS:-8}"
EXTRA_ARGS=()

if [[ -n "${RESUME_FROM:-}" ]]; then
  EXTRA_ARGS+=(--resume-from "$RESUME_FROM")
  if [[ -n "${RESUME_OUTPUT_DIR:-}" ]]; then
    EXTRA_ARGS+=(--output-dir "$RESUME_OUTPUT_DIR")
  fi
else
  LATEST_RUN=""
  if [[ -d "$RUN_ROOT" ]]; then
    while IFS= read -r run_dir; do
      LATEST_RUN="$run_dir"
    done < <(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
  fi
  if [[ -n "$LATEST_RUN" ]]; then
    if [[ -f "$LATEST_RUN/checkpoint_latest.pt" ]]; then
      EXTRA_ARGS+=(--resume-from "$LATEST_RUN/checkpoint_latest.pt" --output-dir "$LATEST_RUN")
    else
      LATEST_STEP_CKPT="$(find "$LATEST_RUN" -maxdepth 1 -type f -name 'checkpoint_step*.pt' | sort | tail -n 1 || true)"
      if [[ -n "$LATEST_STEP_CKPT" ]]; then
        EXTRA_ARGS+=(--resume-from "$LATEST_STEP_CKPT" --output-dir "$LATEST_RUN")
      fi
    fi
  fi
fi

python tools/train_stage1_teacher_free.py \
  --config "$CONFIG_PATH" \
  --max-gpus "$MAX_GPUS" \
  "${EXTRA_ARGS[@]}" \
  "$@" \
  2>&1 | tee "$LOG_PATH"
