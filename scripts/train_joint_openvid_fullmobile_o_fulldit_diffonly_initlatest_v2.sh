#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAX_GPUS="${MAX_GPUS:-8}"

CFG="${CFG:-configs/stage1_joint_openvid_fullmobile_o_fulldit_diffonly_initlatest_v2_bs64_8gpu.yaml}"
OPENVID_CSV="${OPENVID_CSV:-data/openvid_1m/manifests/openvid_manifest_0_112.csv}"
OPENVID_ENC="${OPENVID_ENC:-data/openvid_1m/encoded/wan_vae_fp16_stream}"
IMAGE_CSV="${IMAGE_CSV:-data/full_mobile-o/manifests/journeydb_short_caption_train_ready.csv}"
JOINT_PREFIX="${JOINT_PREFIX:-data/mix/manifests/joint_openvid_fullmobile_5v1i}"
INIT_FROM_LATEST="${INIT_FROM_LATEST:-output/stage1_bridge_fulldit_full_mobile_o_smolvlm2_500m_lexical_gated_k2_diffonly_init10k_bs64_v2_20260420_8gpu/20260425_135135/checkpoint_latest.pt}"
RESUME_FROM="${RESUME_FROM:-}"
REBUILD_MANIFEST="${REBUILD_MANIFEST:-0}"
PREP_ONLY="${PREP_ONLY:-0}"
LOG_PATH="${LOG_PATH:-output/logs/train_joint_openvid_fullmobile_o_fulldit_diffonly_initlatest_v2.log}"

RAW_JOINT_CSV="${JOINT_PREFIX}.csv"
RAW_VIDEO_CSV="${JOINT_PREFIX}_video.csv"
RAW_IMAGE_CSV="${JOINT_PREFIX}_image.csv"
CLEAN_JOINT_CSV="${JOINT_PREFIX}_clean.csv"
CLEAN_VIDEO_CSV="${JOINT_PREFIX}_clean_video.csv"
CLEAN_IMAGE_CSV="${JOINT_PREFIX}_clean_image.csv"

mkdir -p output/logs

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
  PYTHONPATH=. python tools/data_prepare/build_joint_manifest_openvid_fullmobile.py \
    --image-manifest "$IMAGE_CSV" \
    --openvid-manifest "$OPENVID_CSV" \
    --openvid-preprocessed-dir "$OPENVID_ENC" \
    --output-prefix "$JOINT_PREFIX"

  PYTHONPATH=. python tools/data_prepare/sanitize_joint_manifest.py \
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
  python - <<'PY'
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
