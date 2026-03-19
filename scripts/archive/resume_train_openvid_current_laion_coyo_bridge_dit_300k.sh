#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-mobileov}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
CONFIG="configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_dit_online_teacher_300k_resume_2gpu_20260316.yaml"
RESUME_CKPT="output/stage1_bridge_dit_openvid_current_laion_coyo_online_teacher_30k_20260315_2gpu/20260315_201712/checkpoint_final.pt"

source /share_0/conda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

python -m torch.distributed.run --nproc_per_node=2 \
  tools/train_stage1_teacher_free.py \
  --config "$CONFIG" \
  --max-gpus 2 \
  --resume-from "$RESUME_CKPT"
