#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "/share_0/conda/etc/profile.d/conda.sh" ]]; then
  source "/share_0/conda/etc/profile.d/conda.sh"
else
  echo "Could not find conda.sh" >&2
  exit 1
fi

conda activate "${CONDA_ENV:-mobileov}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export MASTER_PORT="${MASTER_PORT:-29825}"

CONFIG="${CONFIG_STAGE25:-configs/stage25_teacher_free_joint_openvid_current_laion_coyo_hybrid_semvis_funcdistill_5v1i_2gpu_20260320.yaml}"
INIT_CKPT="${INIT_CKPT:-output/stage1_prompt_distill_openvid_current_laion_coyo_gemma_teacher_5v1i_20260318_2gpu/20260318_144252/checkpoint_step50000.pt}"

python -m torch.distributed.run --nproc_per_node=2 \
  tools/train_stage1_teacher_free.py \
  --config "$CONFIG" \
  --max-gpus 2 \
  --init-from "$INIT_CKPT"
