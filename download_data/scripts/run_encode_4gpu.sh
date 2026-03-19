#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <python -m openvid_dataops encode args>"
  echo "Example:"
  echo "  $0 --manifest-csv data/openvid/manifests/openvid.csv --ckpt-dir checkpoints/wan/wanxiang1_3b"
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  -m openvid_dataops encode "$@"
