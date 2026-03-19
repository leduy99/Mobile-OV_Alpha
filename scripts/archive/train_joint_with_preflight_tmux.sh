#!/usr/bin/env bash
set -euo pipefail

# Run preflight checks first; only launch training if checks pass.
#
# Example:
#   bash scripts/train_joint_with_preflight_tmux.sh \
#     --config configs/stage1_teacher_free_joint_msrvtt_laion_coyo_ivjoint_3gpu.yaml \
#     --gpus 1,2,3 \
#     --master-port 29613 \
#     --session train_joint_clean_preflight

CONFIG="configs/stage1_teacher_free_joint_msrvtt_laion_coyo_ivjoint_3gpu.yaml"
GPUS="1,2,3"
MASTER_PORT="29613"
SESSION=""
EXTRA_ARGS=""
CONDA_ENV="mobileov"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${SESSION}" ]]; then
  SESSION="train_joint_preflight_${TS}"
fi

NPROC="$(python - <<PY
gpus='${GPUS}'.strip()
print(len([x for x in gpus.split(',') if x.strip()]))
PY
)"

LOG_DIR="output/logs"
mkdir -p "${LOG_DIR}"
PREFLIGHT_LOG="${LOG_DIR}/preflight_${SESSION}.log"
TRAIN_LOG="${LOG_DIR}/train_${SESSION}.log"

CMD="
cd /share_4/users/duy/project/unified_video/Omni-Video-smolvlm2 && \
source /share_0/conda/etc/profile.d/conda.sh && \
conda activate ${CONDA_ENV} && \
export PYTHONUNBUFFERED=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPUS} NCCL_P2P_DISABLE=1 MASTER_PORT=${MASTER_PORT} && \
python tools/data_prepare/preflight_joint_pipeline_check.py --config ${CONFIG} 2>&1 \
  | tee ${PREFLIGHT_LOG} && \
python -m torch.distributed.run --nproc_per_node=${NPROC} tools/train_stage1_teacher_free.py --config ${CONFIG} --max-gpus ${NPROC} ${EXTRA_ARGS} 2>&1 \
  | tee ${TRAIN_LOG}
"

tmux new-session -d -s "${SESSION}" "${CMD}"
echo "session=${SESSION}"
echo "preflight_log=${PREFLIGHT_LOG}"
echo "train_log=${TRAIN_LOG}"
