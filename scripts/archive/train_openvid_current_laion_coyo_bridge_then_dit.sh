#!/usr/bin/env bash
set -euo pipefail

CONFIG_PHASE1="${CONFIG_PHASE1:-configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_only_20k_online_teacher_2gpu_20260315.yaml}"
CONFIG_PHASE2="${CONFIG_PHASE2:-configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_dit_online_teacher_30k_2gpu_20260315.yaml}"
GPUS="${GPUS:-1,2}"
CONDA_ENV="${CONDA_ENV:-mobileov}"
PHASE1_MASTER_PORT="${PHASE1_MASTER_PORT:-29621}"
PHASE2_MASTER_PORT="${PHASE2_MASTER_PORT:-29622}"
EXTRA_PHASE1_ARGS="${EXTRA_PHASE1_ARGS:-}"
EXTRA_PHASE2_ARGS="${EXTRA_PHASE2_ARGS:-}"

ROOT="/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2"
cd "$ROOT"
source /share_0/conda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1 PYTHONPATH=. NCCL_P2P_DISABLE=1

NPROC="$(python - <<PY
print(len([x for x in '${GPUS}'.split(',') if x.strip()]))
PY
)"

phase_output_dir() {
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

PHASE1_OUTPUT="$(phase_output_dir "$CONFIG_PHASE1")"
PHASE2_OUTPUT="$(phase_output_dir "$CONFIG_PHASE2")"

echo "[phase1] config=$CONFIG_PHASE1 gpus=$GPUS output=$PHASE1_OUTPUT"
CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$PHASE1_MASTER_PORT" \
python -m torch.distributed.run --nproc_per_node="$NPROC" \
  tools/train_stage1_teacher_free.py \
  --config "$CONFIG_PHASE1" \
  --max-gpus "$NPROC" \
  $EXTRA_PHASE1_ARGS

PHASE1_RUN_DIR="$(latest_run_dir "$PHASE1_OUTPUT")"
if [[ -z "$PHASE1_RUN_DIR" ]]; then
  echo "phase1 finished but no run dir found under $PHASE1_OUTPUT" >&2
  exit 1
fi
PHASE1_CKPT="$PHASE1_RUN_DIR/checkpoint_final.pt"
if [[ ! -f "$PHASE1_CKPT" ]]; then
  echo "phase1 checkpoint not found: $PHASE1_CKPT" >&2
  exit 1
fi

echo "[phase2] config=$CONFIG_PHASE2 resume=$PHASE1_CKPT gpus=$GPUS output=$PHASE2_OUTPUT"
CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$PHASE2_MASTER_PORT" \
python -m torch.distributed.run --nproc_per_node="$NPROC" \
  tools/train_stage1_teacher_free.py \
  --config "$CONFIG_PHASE2" \
  --max-gpus "$NPROC" \
  --resume-from "$PHASE1_CKPT" \
  --resume-skip-optimizer-state \
  $EXTRA_PHASE2_ARGS
