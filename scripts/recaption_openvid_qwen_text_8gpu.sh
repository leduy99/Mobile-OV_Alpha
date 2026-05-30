#!/usr/bin/env bash
#SBATCH --job-name=openvid_qwen_recap
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --output=output/logs/%x_%j.out

set -euo pipefail

source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-mobileov}"

export PYTHONNOUSERSITE=1
export PYTHONPATH=.
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
HF_BIN="${HF_BIN:-hf}"

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

positive_int() {
  local value="${1:-}"
  [[ "$value" =~ ^[0-9]+$ ]] && [[ "$value" -gt 0 ]]
}

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
stop_setup_heartbeat() {
  if [[ -n "${SETUP_HEARTBEAT_PID:-}" ]] && kill -0 "$SETUP_HEARTBEAT_PID" 2>/dev/null; then
    kill "$SETUP_HEARTBEAT_PID" 2>/dev/null || true
    wait "$SETUP_HEARTBEAT_PID" 2>/dev/null || true
  fi
  SETUP_HEARTBEAT_PID=""
}
trap stop_setup_heartbeat EXIT

mkdir -p "$OUT_DIR" output/logs "$(dirname "$OUTPUT_CSV")"

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
echo "auto_install_deps=$AUTO_INSTALL_DEPS"
echo "pre_download_model=$PRE_DOWNLOAD_MODEL"
echo "fail_on_error=$FAIL_ON_ERROR"
echo "global_resume=$GLOBAL_RESUME"
echo "retry_failed=$RETRY_FAILED"
echo "overwrite_parts=$OVERWRITE_PARTS"
echo "save_every=$SAVE_EVERY"
echo "python_bin=$PYTHON_BIN"
echo "gpu_heartbeat=$GPU_HEARTBEAT"

if [[ "$GPU_HEARTBEAT" == "1" && -n "${SLURM_JOB_ID:-}" ]]; then
  srun --ntasks="$NUM_SHARDS" --gpus-per-task=1 bash -lc '
set -euo pipefail
source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-mobileov}"
cd "'"$PWD"'"
export PYTHONNOUSERSITE=1
export PYTHONPATH=.
"'"$PYTHON_BIN"'" tools/data_prepare/gpu_heartbeat.py \
  --label "recaption-setup-'"${SLURM_JOB_ID}"'-${SLURM_PROCID}" \
  --interval "'"$GPU_HEARTBEAT_INTERVAL"'" \
  --tensor-mb "'"$GPU_HEARTBEAT_TENSOR_MB"'"
' &
  SETUP_HEARTBEAT_PID="$!"
  sleep 3
fi

if [[ "$AUTO_INSTALL_DEPS" == "1" ]]; then
  INSTALL_ARGS=()
  if [[ "$PRE_DOWNLOAD_MODEL" == "1" ]]; then
    INSTALL_ARGS+=(--download-model)
    MODEL_PRE_DOWNLOADED=1
  fi
  CONDA_ENV="${CONDA_ENV:-mobileov}" MODEL_ID="$MODEL_ID" \
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

srun --ntasks="$NUM_SHARDS" --gpus-per-task=1 bash -lc '
set -euo pipefail
source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-mobileov}"
cd "'"$PWD"'"
export PYTHONNOUSERSITE=1
export PYTHONPATH=.
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
