#!/usr/bin/env bash
set -euo pipefail

# Example pipeline for parts [1,2,4,5]
# Run from download_data/ directory.

PARTS='[1,2,4,5]'
MANIFEST='openvid_p1_p2_p4_p5.csv'
CKPT_DIR='checkpoints/wan/wanxiang1_3b'

python -m openvid_dataops download --parts "$PARTS" --extract
python -m openvid_dataops build-manifest --parts "$PARTS" --output-name "$MANIFEST"

bash scripts/run_encode_4gpu.sh \
  --manifest-csv "data/openvid/manifests/${MANIFEST}" \
  --ckpt-dir "$CKPT_DIR" \
  --task t2v-1.3B \
  --frame-num 81 \
  --sampling-rate 1 \
  --target-size 480,832 \
  --output-subdir wan_vae_p1_p2_p4_p5
