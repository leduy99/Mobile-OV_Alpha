#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-mobileov}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
mkdir -p output/logs
python tools/train_stage1_teacher_free.py \
  --config configs/stage1_teacher_free_laion_coyo_clean10k_image_bridge_only_lexical_gated_k2_online_teacher_bs16_v2_1gpu_20260328.yaml \
  --max-gpus 1 \
  2>&1 | tee output/logs/train_laion_coyo_clean10k_image_bridge_only_lexical_gated_k2_online_teacher_bs16_v2_gpu6_20260328.log
