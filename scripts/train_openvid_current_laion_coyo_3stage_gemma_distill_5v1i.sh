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

CONFIG_STAGE1="${CONFIG_STAGE1:-configs/stage1_prompt_teacher_distill_openvid_current_laion_coyo_5v1i_2gpu_20260318.yaml}"
CONFIG_STAGE2="${CONFIG_STAGE2:-configs/stage2_teacher_free_joint_openvid_current_laion_coyo_bridge_only_gemma_distill_5v1i_2gpu_20260318.yaml}"
CONFIG_STAGE3="${CONFIG_STAGE3:-configs/stage3_teacher_free_joint_openvid_current_laion_coyo_bridge_dit_gemma_distill_5v1i_2gpu_20260318.yaml}"
GPUS="${GPUS:-1,2}"
STAGE1_MASTER_PORT="${STAGE1_MASTER_PORT:-29741}"
STAGE2_MASTER_PORT="${STAGE2_MASTER_PORT:-29742}"
STAGE3_MASTER_PORT="${STAGE3_MASTER_PORT:-29743}"
EXTRA_STAGE1_ARGS="${EXTRA_STAGE1_ARGS:-}"
EXTRA_STAGE2_ARGS="${EXTRA_STAGE2_ARGS:-}"
EXTRA_STAGE3_ARGS="${EXTRA_STAGE3_ARGS:-}"

NPROC="$(python - <<PY
print(len([x for x in '${GPUS}'.split(',') if x.strip()]))
PY
)"

config_output_dir() {
  python - <<PY
import yaml
with open('$1', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg['run']['output_dir'])
PY
}

latest_run_dir() {
  find "$1" -mindepth 1 -maxdepth 1 -type d | sort | tail -n1
}

STAGE1_OUTPUT="$(config_output_dir "$CONFIG_STAGE1")"
STAGE2_OUTPUT="$(config_output_dir "$CONFIG_STAGE2")"
STAGE3_OUTPUT="$(config_output_dir "$CONFIG_STAGE3")"

echo "[stage1] prompt-only distill config=$CONFIG_STAGE1 gpus=$GPUS output=$STAGE1_OUTPUT"
CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$STAGE1_MASTER_PORT" \
python -m torch.distributed.run --nproc_per_node="$NPROC" \
  tools/train_q1_sana_bridge.py \
  --config "$CONFIG_STAGE1" \
  --max-gpus "$NPROC" \
  $EXTRA_STAGE1_ARGS

STAGE1_RUN_DIR="$(latest_run_dir "$STAGE1_OUTPUT")"
STAGE1_CKPT="$STAGE1_RUN_DIR/checkpoint_final.pt"
[[ -f "$STAGE1_CKPT" ]] || { echo "Missing stage1 checkpoint: $STAGE1_CKPT" >&2; exit 1; }

echo "[stage2] bridge reinjection init=$STAGE1_CKPT config=$CONFIG_STAGE2"
CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$STAGE2_MASTER_PORT" \
python -m torch.distributed.run --nproc_per_node="$NPROC" \
  tools/train_stage1_teacher_free.py \
  --config "$CONFIG_STAGE2" \
  --max-gpus "$NPROC" \
  --init-from "$STAGE1_CKPT" \
  $EXTRA_STAGE2_ARGS

STAGE2_RUN_DIR="$(latest_run_dir "$STAGE2_OUTPUT")"
STAGE2_CKPT="$STAGE2_RUN_DIR/checkpoint_final.pt"
[[ -f "$STAGE2_CKPT" ]] || { echo "Missing stage2 checkpoint: $STAGE2_CKPT" >&2; exit 1; }

echo "[stage3] bridge+DiT init=$STAGE2_CKPT config=$CONFIG_STAGE3"
CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$STAGE3_MASTER_PORT" \
python -m torch.distributed.run --nproc_per_node="$NPROC" \
  tools/train_stage1_teacher_free.py \
  --config "$CONFIG_STAGE3" \
  --max-gpus "$NPROC" \
  --init-from "$STAGE2_CKPT" \
  $EXTRA_STAGE3_ARGS

STAGE3_RUN_DIR="$(latest_run_dir "$STAGE3_OUTPUT")"
echo "Done. Final stage3 run dir: $STAGE3_RUN_DIR"
