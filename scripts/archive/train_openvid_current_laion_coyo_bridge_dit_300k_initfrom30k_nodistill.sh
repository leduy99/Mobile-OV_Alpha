#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-mobileov}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export MASTER_PORT="${MASTER_PORT:-29731}"
python -m torch.distributed.run --nproc_per_node=2 \
  tools/train_stage1_teacher_free.py \
  --config configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_dit_300k_initfrom30k_nodistill_2gpu_20260316.yaml \
  --max-gpus 2 \
  --init-from output/stage1_bridge_dit_openvid_current_laion_coyo_online_teacher_30k_20260315_2gpu/20260315_201712/checkpoint_final.pt
