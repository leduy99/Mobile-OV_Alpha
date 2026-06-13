#!/usr/bin/env bash
#SBATCH --job-name=mobile-ov-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm-train-%j.out
#SBATCH --error=logs/slurm-train-%j.out
#SBATCH --account=berzelius-2025-436
##SBATCH --gres=gpu:8
#SBATCH --gres=gpu:A100-SXM4-80GB:8

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load buildenv-gcccuda/12.1.1-gcc12.3.0 || true
fi

positive_int() {
  local value="${1:-}"
  [[ "$value" =~ ^[0-9]+$ ]] && [[ "$value" -gt 0 ]]
}

parse_gpu_count() {
  local value="${1:-}"
  if [[ "$value" =~ ^([0-9]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  elif [[ "$value" =~ :([0-9]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  else
    return 1
  fi
}

count_visible_devices() {
  local value="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -z "$value" ]]; then
    return 1
  fi
  awk -F',' '{print NF}' <<<"$value"
}

DEFAULT_REMOTE_ROOT="/proj/cvl/users/x_fahkh2"
ENV_PATH="${ENV_PATH:-${CONDA_ENV_PATH:-}}"
if [[ -z "$ENV_PATH" && -x "$DEFAULT_REMOTE_ROOT/envs/mobileov/bin/python" ]]; then
  ENV_PATH="$DEFAULT_REMOTE_ROOT/envs/mobileov"
fi
if [[ -n "$ENV_PATH" ]]; then
  if [[ ! -x "$ENV_PATH/bin/python" ]]; then
    echo "Python env not found: $ENV_PATH/bin/python" >&2
    exit 1
  fi
  export PATH="$ENV_PATH/bin:$PATH"
fi

if [[ -d "$DEFAULT_REMOTE_ROOT" ]]; then
  export HF_HOME="${HF_HOME:-$DEFAULT_REMOTE_ROOT/caches}"
  export TORCH_HOME="${TORCH_HOME:-$DEFAULT_REMOTE_ROOT/caches}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DEFAULT_REMOTE_ROOT/caches}"
  export TMPDIR="${TMPDIR:-$DEFAULT_REMOTE_ROOT/caches}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$DEFAULT_REMOTE_ROOT/caches}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
fi

if [[ -n "$ENV_PATH" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$ENV_PATH/bin/python}"
  TORCHRUN_BIN="${TORCHRUN_BIN:-$ENV_PATH/bin/torchrun}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
  TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
fi

NNODES="${NNODES:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}}"
if ! positive_int "$NNODES"; then
  NNODES=1
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-}"
if [[ -z "$GPUS_PER_NODE" ]]; then
  if positive_int "${MAX_GPUS:-}"; then
    GPUS_PER_NODE="$MAX_GPUS"
  elif parsed_gpus="$(parse_gpu_count "${SLURM_GPUS_ON_NODE:-}" 2>/dev/null)"; then
    GPUS_PER_NODE="$parsed_gpus"
  elif visible_gpus="$(count_visible_devices 2>/dev/null)"; then
    GPUS_PER_NODE="$visible_gpus"
  else
    GPUS_PER_NODE=8
  fi
fi
if ! positive_int "$GPUS_PER_NODE"; then
  echo "Invalid GPUS_PER_NODE=$GPUS_PER_NODE" >&2
  exit 1
fi

if positive_int "${SLURM_JOB_NUM_NODES:-}" && [[ "$NNODES" -gt "$SLURM_JOB_NUM_NODES" ]]; then
  echo "Requested NNODES=$NNODES but SLURM allocated only SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES." >&2
  echo "Submit with: sbatch --nodes=$NNODES ..." >&2
  exit 1
fi
if allocated_gpus_per_node="$(parse_gpu_count "${SLURM_GPUS_ON_NODE:-}" 2>/dev/null)"; then
  if [[ "$GPUS_PER_NODE" -gt "$allocated_gpus_per_node" ]]; then
    echo "Requested GPUS_PER_NODE=$GPUS_PER_NODE but SLURM allocated only SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE." >&2
    echo "Submit with matching --gres, for example: sbatch --gres=gpu:A100-SXM4-80GB:$GPUS_PER_NODE ..." >&2
    exit 1
  fi
fi

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
TRAIN_MAX_GPUS="${TRAIN_MAX_GPUS:-$WORLD_SIZE}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${SLURM_JOB_ID:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPUS_PER_NODE - 1)))"
  export CUDA_VISIBLE_DEVICES
fi
export PYTHONNOUSERSITE=1

CFG="${CFG:-configs/stage1_joint_openvid_fullmobile_o_fulldit_diffonly_initlatest_v2_bs64_8gpu.yaml}"
DEFAULT_OPENVID_CSV="download_data/data/openvid/manifests/openvid_all.csv"
DEFAULT_OPENVID_RECAPTION_CSV="download_data/data/openvid/manifests/openvid_all_recaptions.csv"
if [[ -z "${OPENVID_CSV:-}" ]]; then
  if [[ -f "$DEFAULT_OPENVID_RECAPTION_CSV" ]]; then
    OPENVID_CSV="$DEFAULT_OPENVID_RECAPTION_CSV"
  else
    OPENVID_CSV="$DEFAULT_OPENVID_CSV"
  fi
fi
OPENVID_ENC="${OPENVID_ENC:-download_data/data/openvid/encoded/wan_vae_openvid_all}"
IMAGE_CSV="${IMAGE_CSV:-data/full_mobile-o/manifests/journeydb_short_caption_train_ready.csv}"
JOINT_PREFIX="${JOINT_PREFIX:-data/mix/manifests/joint_openvid_fullmobile_5v1i}"
INIT_FROM_LATEST="${INIT_FROM_LATEST:-output/stage1_bridge_fulldit_full_mobile_o_smolvlm2_500m_lexical_gated_k2_diffonly_init10k_bs64_v2_20260420_8gpu/20260425_135135/checkpoint_latest.pt}"
RESUME_FROM="${RESUME_FROM:-}"
REBUILD_MANIFEST="${REBUILD_MANIFEST:-0}"
PREP_ONLY="${PREP_ONLY:-0}"
LOG_PATH="${LOG_PATH:-output/logs/train_joint_openvid_fullmobile_o_fulldit_diffonly_initlatest_v2.log}"
GPU_HEARTBEAT="${GPU_HEARTBEAT:-1}"
GPU_HEARTBEAT_INTERVAL="${GPU_HEARTBEAT_INTERVAL:-15}"
GPU_HEARTBEAT_TENSOR_MB="${GPU_HEARTBEAT_TENSOR_MB:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

RAW_JOINT_CSV="${JOINT_PREFIX}.csv"
RAW_VIDEO_CSV="${JOINT_PREFIX}_video.csv"
RAW_IMAGE_CSV="${JOINT_PREFIX}_image.csv"
CLEAN_JOINT_CSV="${JOINT_PREFIX}_clean.csv"
CLEAN_VIDEO_CSV="${JOINT_PREFIX}_clean_video.csv"
CLEAN_IMAGE_CSV="${JOINT_PREFIX}_clean_image.csv"

mkdir -p output/logs logs "${HF_HOME:-output/logs}" "${HF_HUB_CACHE:-output/logs}" \
  "${TORCH_HOME:-output/logs}" "${PIP_CACHE_DIR:-output/logs}" "${TMPDIR:-output/logs}" \
  "${TRITON_CACHE_DIR:-output/logs}"

SETUP_HEARTBEAT_PID=""
SETUP_HEARTBEAT_STOP_FILE=""
stop_setup_heartbeat() {
  if [[ -n "${SETUP_HEARTBEAT_PID:-}" ]] && kill -0 "$SETUP_HEARTBEAT_PID" 2>/dev/null; then
    if [[ -n "${SETUP_HEARTBEAT_STOP_FILE:-}" ]]; then
      touch "$SETUP_HEARTBEAT_STOP_FILE" 2>/dev/null || true
    fi
    for _ in $(seq 1 30); do
      if ! kill -0 "$SETUP_HEARTBEAT_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$SETUP_HEARTBEAT_PID" 2>/dev/null; then
      kill "$SETUP_HEARTBEAT_PID" 2>/dev/null || true
    fi
    wait "$SETUP_HEARTBEAT_PID" 2>/dev/null || true
  fi
  SETUP_HEARTBEAT_PID=""
  SETUP_HEARTBEAT_STOP_FILE=""
}
trap stop_setup_heartbeat EXIT

start_setup_heartbeat() {
  if [[ "$GPU_HEARTBEAT" != "1" || -z "${SLURM_JOB_ID:-}" ]]; then
    return 0
  fi
  SETUP_HEARTBEAT_STOP_FILE="${TMPDIR:-/tmp}/mobileov_train_setup_heartbeat_stop_${SLURM_JOB_ID}.flag"
  rm -f "$SETUP_HEARTBEAT_STOP_FILE"
  echo "Starting setup heartbeat: nodes=$NNODES gpus_per_node=$GPUS_PER_NODE"
  srun --overlap \
    --nodes="$NNODES" \
    --ntasks="$NNODES" \
    --ntasks-per-node=1 \
    --gpus-per-task="$GPUS_PER_NODE" \
    bash -lc '
set -euo pipefail
cd "'"$PWD"'"
if [[ -n "'"${ENV_PATH:-}"'" ]]; then
  export PATH="'"${ENV_PATH:-}"'/bin:$PATH"
fi
export PYTHONNOUSERSITE=1
export PYTHONPATH="./:${PYTHONPATH:-}"
"'"$PYTHON_BIN"'" tools/data_prepare/gpu_heartbeat.py \
  --label "train-setup-'"${SLURM_JOB_ID}"'-node-${SLURMD_NODENAME:-unknown}" \
  --all-devices \
  --interval "'"$GPU_HEARTBEAT_INTERVAL"'" \
  --tensor-mb "'"$GPU_HEARTBEAT_TENSOR_MB"'" \
  --stop-file "'"$SETUP_HEARTBEAT_STOP_FILE"'"
' &
  SETUP_HEARTBEAT_PID="$!"
  sleep 3
  if ! kill -0 "$SETUP_HEARTBEAT_PID" 2>/dev/null; then
    wait "$SETUP_HEARTBEAT_PID" 2>/dev/null || true
    echo "Setup GPU heartbeat failed to start; aborting to avoid idle-GPU job cancellation." >&2
    exit 1
  fi
}

echo "Training launcher resources:"
echo "  nnodes=$NNODES"
echo "  gpus_per_node=$GPUS_PER_NODE"
echo "  world_size=$WORLD_SIZE"
echo "  train_max_gpus=$TRAIN_MAX_GPUS"
echo "  slurm_job_id=${SLURM_JOB_ID:-unset}"
echo "  slurm_job_nodelist=${SLURM_JOB_NODELIST:-unset}"
echo "  cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "  env_path=${ENV_PATH:-unset}"
echo "  python_bin=$PYTHON_BIN"
echo "  torchrun_bin=$TORCHRUN_BIN"
echo "  gpu_heartbeat=$GPU_HEARTBEAT"

start_setup_heartbeat

need_manifest_rebuild=0
for path in "$RAW_JOINT_CSV" "$RAW_VIDEO_CSV" "$RAW_IMAGE_CSV" "$CLEAN_JOINT_CSV" "$CLEAN_VIDEO_CSV" "$CLEAN_IMAGE_CSV"; do
  if [[ ! -f "$path" ]]; then
    need_manifest_rebuild=1
    break
  fi
done
if [[ "$REBUILD_MANIFEST" == "1" ]]; then
  need_manifest_rebuild=1
fi

if [[ "$need_manifest_rebuild" == "1" ]]; then
  echo "Preparing joint manifests under: $JOINT_PREFIX"
  PYTHONPATH=. "$PYTHON_BIN" tools/data_prepare/build_joint_manifest_openvid_fullmobile.py \
    --image-manifest "$IMAGE_CSV" \
    --openvid-manifest "$OPENVID_CSV" \
    --openvid-preprocessed-dir "$OPENVID_ENC" \
    --output-prefix "$JOINT_PREFIX"

  PYTHONPATH=. "$PYTHON_BIN" tools/data_prepare/sanitize_joint_manifest.py \
    --input-csv "$RAW_JOINT_CSV" \
    --output-csv "$CLEAN_JOINT_CSV" \
    --image-datasets "" \
    --min-image-bytes 0 \
    --dedup-image-path \
    --drop-missing-preprocessed \
    --drop-empty-caption

  CLEAN_JOINT_CSV="$CLEAN_JOINT_CSV" \
  CLEAN_VIDEO_CSV="$CLEAN_VIDEO_CSV" \
  CLEAN_IMAGE_CSV="$CLEAN_IMAGE_CSV" \
  "$PYTHON_BIN" - <<'PY'
import os
import pandas as pd

joint = pd.read_csv(os.environ["CLEAN_JOINT_CSV"])
joint["modality"] = joint["modality"].fillna("").astype(str).str.strip().str.lower()
video_df = joint[joint["modality"].eq("video")].copy()
image_df = joint[joint["modality"].eq("image")].copy()
video_df.to_csv(os.environ["CLEAN_VIDEO_CSV"], index=False)
image_df.to_csv(os.environ["CLEAN_IMAGE_CSV"], index=False)
print(f"wrote {os.environ['CLEAN_VIDEO_CSV']} rows={len(video_df)}")
print(f"wrote {os.environ['CLEAN_IMAGE_CSV']} rows={len(image_df)}")
PY
else
  echo "Reusing existing joint manifests under: $JOINT_PREFIX"
fi

if [[ "$PREP_ONLY" == "1" ]]; then
  echo "PREP_ONLY=1, stopping after manifest preparation."
  exit 0
fi

OUTPUT_ROOT="$(
  CFG_PATH="$CFG" "$PYTHON_BIN" - <<'PY'
import os
import yaml

with open(os.environ["CFG_PATH"], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["run"]["output_dir"])
PY
)"

if [[ -n "$RESUME_FROM" ]]; then
  CKPT_MODE="resume"
  CKPT_PATH="$RESUME_FROM"
else
  AUTO_RESUME=""
  if [[ -d "$OUTPUT_ROOT" ]]; then
    AUTO_RESUME="$(
      find "$OUTPUT_ROOT" -type f \( -name 'checkpoint_latest.pt' -o -name 'checkpoint_step*.pt' \) -printf '%T@ %p\n' 2>/dev/null \
        | sort -n \
        | tail -n 1 \
        | cut -d' ' -f2-
    )"
  fi
  if [[ -n "$AUTO_RESUME" ]]; then
    CKPT_MODE="resume"
    CKPT_PATH="$AUTO_RESUME"
  else
    CKPT_MODE="init"
    CKPT_PATH="$INIT_FROM_LATEST"
  fi
fi

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint not found for $CKPT_MODE: $CKPT_PATH" >&2
  exit 1
fi

echo "Training config: $CFG"
echo "Output root: $OUTPUT_ROOT"
echo "Checkpoint mode: $CKPT_MODE"
echo "Checkpoint path: $CKPT_PATH"

if [[ "$CKPT_MODE" == "resume" ]]; then
  CKPT_ARGS=(--resume-from "$CKPT_PATH")
else
  CKPT_ARGS=(--init-from "$CKPT_PATH")
fi

stop_setup_heartbeat

export CFG CKPT_MODE CKPT_PATH TRAIN_MAX_GPUS PYTHONNOUSERSITE=1
export PYTHONPATH="./:${PYTHONPATH:-}"

if [[ "$WORLD_SIZE" -gt 1 && "$NNODES" -gt 1 ]]; then
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Multi-node launch requires SLURM. Use sbatch/salloc with NNODES>1." >&2
    exit 1
  fi
  MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
  RDZV_ID="${RDZV_ID:-${SLURM_JOB_ID}}"
  export MASTER_ADDR MASTER_PORT RDZV_ID TORCHRUN_BIN ENV_PATH
  echo "Launching multi-node training:"
  echo "  master_addr=$MASTER_ADDR"
  echo "  master_port=$MASTER_PORT"
  echo "  rdzv_id=$RDZV_ID"
  srun \
    --nodes="$NNODES" \
    --ntasks="$NNODES" \
    --ntasks-per-node=1 \
    --gpus-per-task="$GPUS_PER_NODE" \
    bash -lc '
set -euo pipefail
cd "'"$PWD"'"
if [[ -n "${ENV_PATH:-}" ]]; then
  export PATH="$ENV_PATH/bin:$PATH"
fi
export PYTHONNOUSERSITE=1
export PYTHONPATH="./:${PYTHONPATH:-}"
CKPT_ARGS=()
if [[ "$CKPT_MODE" == "resume" ]]; then
  CKPT_ARGS=(--resume-from "$CKPT_PATH")
else
  CKPT_ARGS=(--init-from "$CKPT_PATH")
fi
"$TORCHRUN_BIN" \
  --nnodes "'"$NNODES"'" \
  --nproc_per_node "'"$GPUS_PER_NODE"'" \
  --node_rank "${SLURM_NODEID:-0}" \
  --rdzv_backend c10d \
  --rdzv_endpoint "'"$MASTER_ADDR:$MASTER_PORT"'" \
  --rdzv_id "'"$RDZV_ID"'" \
  tools/train_stage1_teacher_free.py \
  --config "$CFG" \
  "${CKPT_ARGS[@]}" \
  --max-gpus "$TRAIN_MAX_GPUS"
' 2>&1 | tee "$LOG_PATH"
elif [[ "$WORLD_SIZE" -gt 1 ]]; then
  echo "Launching single-node distributed training: gpus_per_node=$GPUS_PER_NODE"
  "$TORCHRUN_BIN" --standalone --nproc_per_node="$GPUS_PER_NODE" \
    tools/train_stage1_teacher_free.py \
    --config "$CFG" \
    "${CKPT_ARGS[@]}" \
    --max-gpus "$TRAIN_MAX_GPUS" \
    2>&1 | tee "$LOG_PATH"
else
  echo "Launching single-GPU training"
  "$PYTHON_BIN" tools/train_stage1_teacher_free.py \
    --config "$CFG" \
    "${CKPT_ARGS[@]}" \
    --max-gpus "$TRAIN_MAX_GPUS" \
    2>&1 | tee "$LOG_PATH"
fi
