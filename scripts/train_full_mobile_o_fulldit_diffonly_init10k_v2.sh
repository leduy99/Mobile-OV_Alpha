#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAX_GPUS="${MAX_GPUS:-8}"

CFG="${CFG:-configs/stage1_full_mobile_o_fulldit_diffonly_init10k_v2_bs64_8gpu.yaml}"
INIT_CKPT="${INIT_CKPT:-output/stage1_bridge_only_full_mobile_o_smolvlm2_500m_lexical_gated_k2_online_teacher_bs64_v2_20260417_8gpu/20260417_093000/checkpoint_step10000.pt}"
LOG_PATH="${LOG_PATH:-output/logs/train_full_mobile_o_fulldit_diffonly_init10k_v2.log}"

if [[ ! -f "$INIT_CKPT" ]]; then
  echo "Init checkpoint not found: $INIT_CKPT" >&2
  exit 1
fi

mkdir -p output/logs

if [[ "$MAX_GPUS" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="$MAX_GPUS" \
    tools/train_stage1_teacher_free.py \
    --config "$CFG" \
    --init-from "$INIT_CKPT" \
    --max-gpus "$MAX_GPUS" \
    2>&1 | tee "$LOG_PATH"
else
  python tools/train_stage1_teacher_free.py \
    --config "$CFG" \
    --init-from "$INIT_CKPT" \
    --max-gpus "$MAX_GPUS" \
    2>&1 | tee "$LOG_PATH"
fi
