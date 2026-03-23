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
export MASTER_PORT="${MASTER_PORT:-29832}"
export SMOLVLM2_MODEL_ID="${SMOLVLM2_MODEL_ID:-HuggingFaceTB/SmolVLM2-2.2B-Instruct}"

CONFIG="${CONFIG_STAGE1_SEMVIS_2P2B:-configs/stage1_teacher_free_joint_openvid_current_laion_coyo_semvis_frombase_smolvlm2_2p2b_5v1i_2gpu_20260321.yaml}"

python -m torch.distributed.run --nproc_per_node=2 \
  tools/train_stage1_teacher_free.py \
  --config "$CONFIG" \
  --max-gpus 2
