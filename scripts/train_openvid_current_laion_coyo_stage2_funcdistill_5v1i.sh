#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "Could not find conda initialization script. Set CONDA_SH or put conda on PATH." >&2
  exit 1
fi

conda activate "${CONDA_ENV:-mobileov}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

CONFIG_STAGE2="${CONFIG_STAGE2:-configs/stage2_teacher_free_joint_openvid_current_laion_coyo_bridge_only_gemma_funcdistill_5v1i_2gpu_20260319.yaml}"
INIT_FROM="${INIT_FROM:-output/stage1_prompt_distill_openvid_current_laion_coyo_gemma_teacher_5v1i_20260318_2gpu/20260318_144252/checkpoint_step50000.pt}"
GPUS="${GPUS:-1,2}"
MASTER_PORT="${MASTER_PORT:-29752}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

[[ -f "$INIT_FROM" ]] || { echo "Missing init checkpoint: $INIT_FROM" >&2; exit 1; }

NPROC="$(python - <<PY
print(len([x for x in '${GPUS}'.split(',') if x.strip()]))
PY
)"

echo "[stage2-v2] config=$CONFIG_STAGE2 init=$INIT_FROM gpus=$GPUS"
CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$MASTER_PORT" \
python -m torch.distributed.run --nproc_per_node="$NPROC" \
  tools/train_stage1_teacher_free.py \
  --config "$CONFIG_STAGE2" \
  --max-gpus "$NPROC" \
  --init-from "$INIT_FROM" \
  $EXTRA_ARGS
