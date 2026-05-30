#!/usr/bin/env bash
#SBATCH --job-name=mobile-ov
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurmm1.0-%j.out
#SBATCH --error=logs/slurmm1.0-%j.out
#SBATCH --account=berzelius-2025-436
##SBATCH --gres=gpu:8
#SBATCH --gres=gpu:A100-SXM4-80GB:8

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load buildenv-gcccuda/12.1.1-gcc12.3.0
fi

ENV_PATH="${ENV_PATH:-${CONDA_ENV_PATH:-/proj/cvl/users/x_fahkh2/envs/mobileov}}"
if [[ ! -x "$ENV_PATH/bin/python" ]]; then
  echo "Python env not found: $ENV_PATH/bin/python" >&2
  exit 1
fi
export PATH="$ENV_PATH/bin:$PATH"

export PYTHONNOUSERSITE=1
export PYTHONPATH="./:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

PYTHON_BIN="${PYTHON_BIN:-$ENV_PATH/bin/python}"
HF_BIN="${HF_BIN:-$ENV_PATH/bin/hf}"
if [[ ! -x "$HF_BIN" ]]; then
  HF_BIN="hf"
fi
export ENV_PATH PYTHON_BIN HF_BIN

CACHE_ROOT="${CACHE_ROOT:-$HF_HOME}"

INPUT_CSV="${INPUT_CSV:-download_data/data/openvid/manifests/openvid_all.csv}"
OUT_DIR="${OUT_DIR:-download_data/data/openvid/recaption/qwen3p6_35b_a3b}"
OUTPUT_CSV="${OUTPUT_CSV:-download_data/data/openvid/manifests/openvid_all_recaptions.csv}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
CAPTIONER="${CAPTIONER:-qwen}"
DTYPE="${DTYPE:-bf16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.9}"
SAVE_EVERY="${SAVE_EVERY:-1}"
RETRY_FAILED="${RETRY_FAILED:-0}"
OVERWRITE_PARTS="${OVERWRITE_PARTS:-0}"
GLOBAL_RESUME="${GLOBAL_RESUME:-1}"
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-1}"
PRE_DOWNLOAD_MODEL="${PRE_DOWNLOAD_MODEL:-1}"
FAIL_ON_ERROR="${FAIL_ON_ERROR:-1}"
GPU_HEARTBEAT="${GPU_HEARTBEAT:-1}"
GPU_HEARTBEAT_INTERVAL="${GPU_HEARTBEAT_INTERVAL:-15}"
GPU_HEARTBEAT_TENSOR_MB="${GPU_HEARTBEAT_TENSOR_MB:-4}"
MODEL_PRE_DOWNLOADED=0
TASKS_PER_NODE="${TASKS_PER_NODE:-}"

positive_int() {
  local value="${1:-}"
  [[ "$value" =~ ^[0-9]+$ ]] && [[ "$value" -gt 0 ]]
}

if [[ -z "$TASKS_PER_NODE" ]]; then
  if [[ "${SLURM_TASKS_PER_NODE:-}" =~ ^([0-9]+) ]]; then
    TASKS_PER_NODE="${BASH_REMATCH[1]}"
  else
    TASKS_PER_NODE=8
  fi
fi

if [[ -z "${NUM_SHARDS:-}" ]]; then
  if positive_int "${SLURM_NTASKS:-}"; then
    NUM_SHARDS="$SLURM_NTASKS"
  elif positive_int "${SLURM_GPUS:-}"; then
    NUM_SHARDS="$SLURM_GPUS"
  elif positive_int "${SLURM_JOB_NUM_NODES:-}" && positive_int "${SLURM_GPUS_ON_NODE:-}"; then
    NUM_SHARDS=$((SLURM_JOB_NUM_NODES * SLURM_GPUS_ON_NODE))
  else
    NUM_SHARDS=8
  fi
fi

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

mkdir -p "$OUT_DIR" output/logs logs "$(dirname "$OUTPUT_CSV")" \
  "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$TRITON_CACHE_DIR"

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "Input CSV not found: $INPUT_CSV" >&2
  exit 1
fi

echo "input_csv=$INPUT_CSV"
echo "out_dir=$OUT_DIR"
echo "output_csv=$OUTPUT_CSV"
echo "model_id=$MODEL_ID"
echo "num_shards=$NUM_SHARDS"
echo "slurm_nodes=${SLURM_JOB_NUM_NODES:-unset}"
echo "slurm_ntasks=${SLURM_NTASKS:-unset}"
echo "tasks_per_node=$TASKS_PER_NODE"
echo "auto_install_deps=$AUTO_INSTALL_DEPS"
echo "pre_download_model=$PRE_DOWNLOAD_MODEL"
echo "fail_on_error=$FAIL_ON_ERROR"
echo "global_resume=$GLOBAL_RESUME"
echo "retry_failed=$RETRY_FAILED"
echo "overwrite_parts=$OVERWRITE_PARTS"
echo "save_every=$SAVE_EVERY"
echo "python_bin=$PYTHON_BIN"
echo "env_path=$ENV_PATH"
echo "gpu_heartbeat=$GPU_HEARTBEAT"
echo "hf_home=$HF_HOME"
echo "hf_hub_cache=$HF_HUB_CACHE"
echo "torch_home=$TORCH_HOME"
echo "pip_cache_dir=$PIP_CACHE_DIR"
echo "tmpdir=$TMPDIR"
echo "triton_cache_dir=$TRITON_CACHE_DIR"

if [[ "$GPU_HEARTBEAT" == "1" && -n "${SLURM_JOB_ID:-}" ]]; then
  SETUP_HEARTBEAT_STOP_FILE="${TMPDIR%/}/mobileov_setup_heartbeat_stop_${SLURM_JOB_ID}.flag"
  rm -f "$SETUP_HEARTBEAT_STOP_FILE"
  echo "Starting setup heartbeat: tasks=$NUM_SHARDS tasks_per_node=$TASKS_PER_NODE gpus_per_task=1 gpu_bind=single:1"
  srun --overlap \
    --ntasks="$NUM_SHARDS" \
    --ntasks-per-node="$TASKS_PER_NODE" \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    bash -lc '
set -euo pipefail
export PATH="'"$ENV_PATH"'/bin:$PATH"
cd "'"$PWD"'"
export PYTHONNOUSERSITE=1
export PYTHONPATH="./:${PYTHONPATH:-}"
"'"$PYTHON_BIN"'" tools/data_prepare/gpu_heartbeat.py \
  --label "recaption-setup-'"${SLURM_JOB_ID}"'-node-${SLURMD_NODENAME}-rank-${SLURM_PROCID}" \
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
fi

if [[ "$AUTO_INSTALL_DEPS" == "1" ]]; then
  INSTALL_ARGS=()
  if [[ "$PRE_DOWNLOAD_MODEL" == "1" ]]; then
    INSTALL_ARGS+=(--download-model)
    MODEL_PRE_DOWNLOADED=1
  fi
  ENV_PATH="$ENV_PATH" MODEL_ID="$MODEL_ID" CACHE_ROOT="$CACHE_ROOT" HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" \
    TORCH_HOME="$TORCH_HOME" PIP_CACHE_DIR="$PIP_CACHE_DIR" TMPDIR="$TMPDIR" TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    bash scripts/install_qwen36_recaption_deps.sh "${INSTALL_ARGS[@]}"
fi

if [[ "$CAPTIONER" == "qwen" ]]; then
  "$PYTHON_BIN" - <<PY
from transformers import AutoConfig
model_id = "$MODEL_ID"
cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
print("Qwen config OK:", type(cfg), getattr(cfg, "model_type", None))
PY
fi

if [[ "$CAPTIONER" == "qwen" && "$PRE_DOWNLOAD_MODEL" == "1" && "$MODEL_PRE_DOWNLOADED" != "1" ]]; then
  "$HF_BIN" download "$MODEL_ID" --repo-type model
fi

stop_setup_heartbeat

srun \
  --ntasks="$NUM_SHARDS" \
  --ntasks-per-node="$TASKS_PER_NODE" \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  bash -lc '
set -euo pipefail
export PATH="'"$ENV_PATH"'/bin:$PATH"
cd "'"$PWD"'"
export PYTHONNOUSERSITE=1
export PYTHONPATH="./:${PYTHONPATH:-}"
WORKER_HEARTBEAT_PID=""
cleanup_worker_heartbeat() {
  if [[ -n "${WORKER_HEARTBEAT_PID:-}" ]] && kill -0 "$WORKER_HEARTBEAT_PID" 2>/dev/null; then
    kill "$WORKER_HEARTBEAT_PID" 2>/dev/null || true
    wait "$WORKER_HEARTBEAT_PID" 2>/dev/null || true
  fi
}
trap cleanup_worker_heartbeat EXIT
LIMIT_ARGS=()
if [[ -n "${LIMIT_PER_SHARD:-}" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT_PER_SHARD")
fi
FAIL_ARGS=()
if [[ "'"$FAIL_ON_ERROR"'" == "1" ]]; then
  FAIL_ARGS+=(--fail-on-error)
fi
RESUME_ARGS=(--save-every "'"$SAVE_EVERY"'")
if [[ "'"$RETRY_FAILED"'" == "1" ]]; then
  RESUME_ARGS+=(--retry-failed)
fi
if [[ "'"$OVERWRITE_PARTS"'" == "1" ]]; then
  RESUME_ARGS+=(--overwrite)
fi
if [[ "'"$GLOBAL_RESUME"'" != "1" ]]; then
  RESUME_ARGS+=(--no-global-resume)
fi
if [[ "'"$GPU_HEARTBEAT"'" == "1" ]]; then
  "'"$PYTHON_BIN"'" tools/data_prepare/gpu_heartbeat.py \
    --label "recaption-worker-${SLURM_PROCID}" \
    --interval "'"$GPU_HEARTBEAT_INTERVAL"'" \
    --tensor-mb "'"$GPU_HEARTBEAT_TENSOR_MB"'" &
  WORKER_HEARTBEAT_PID="$!"
  sleep 3
fi
"'"$PYTHON_BIN"'" tools/data_prepare/recaption_openvid_text.py \
  --input-csv "'"$INPUT_CSV"'" \
  --output-dir "'"$OUT_DIR"'" \
  --captioner "'"$CAPTIONER"'" \
  --model-id "'"$MODEL_ID"'" \
  --num-shards "'"$NUM_SHARDS"'" \
  --shard-id "$SLURM_PROCID" \
  --device cuda:0 \
  --dtype "'"$DTYPE"'" \
  --max-new-tokens "'"$MAX_NEW_TOKENS"'" \
  --temperature "'"$TEMPERATURE"'" \
  --top-p "'"$TOP_P"'" \
  "${LIMIT_ARGS[@]}" \
  "${FAIL_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
'

"$PYTHON_BIN" tools/data_prepare/merge_recaption_parts.py \
  --input-csv "$INPUT_CSV" \
  --parts-dir "$OUT_DIR" \
  --output-csv "$OUTPUT_CSV"

echo "Merged recaption manifest: $OUTPUT_CSV"
