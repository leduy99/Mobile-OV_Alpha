#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT_DIR/scripts/train_full_mobile_o_image_bridge_fulldit_mcpfull_k4_online_teacher_bs64_v1.sh" "$@"
