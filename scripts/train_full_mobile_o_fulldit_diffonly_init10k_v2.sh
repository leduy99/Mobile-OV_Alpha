#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAX_GPUS="${MAX_GPUS:-8}"

CFG="${CFG:-configs/stage1_full_mobile_o_fulldit_diffonly_init10k_v2_bs64_8gpu.yaml}"
INIT_FROM_LATEST="${INIT_FROM_LATEST:-omni_ckpts/hf_mobile_ov/stage1_bridge_fulldit_full_mobile_o_smolvlm2_500m_lexical_gated_k2_diffonly_init10k_bs64_v2_20260420_8gpu_5k.pt}"
RESUME_FROM="${RESUME_FROM:-}"
LOG_PATH="${LOG_PATH:-output/logs/train_full_mobile_o_fulldit_diffonly_initfromlatest_v2.log}"

mkdir -p output/logs

OUTPUT_ROOT="$(
  CFG_PATH="$CFG" python - <<'PY'
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

if [[ "$MAX_GPUS" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="$MAX_GPUS" \
    tools/train_stage1_teacher_free.py \
    --config "$CFG" \
    "${CKPT_ARGS[@]}" \
    --max-gpus "$MAX_GPUS" \
    2>&1 | tee "$LOG_PATH"
else
  python tools/train_stage1_teacher_free.py \
    --config "$CFG" \
    "${CKPT_ARGS[@]}" \
    --max-gpus "$MAX_GPUS" \
    2>&1 | tee "$LOG_PATH"
fi
