#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-mobileov}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p output/logs
python tools/train_stage1_teacher_free.py \
  --config configs/stage1_teacher_free_full_mobile_o_image_bridge_only_lexical_gated_k2_online_teacher_bs4_v2_1gpu_20260417.yaml \
  --max-gpus 1 \
  2>&1 | tee output/logs/train_full_mobile_o_image_bridge_only_lexical_gated_k2_online_teacher_bs4_v2.log
