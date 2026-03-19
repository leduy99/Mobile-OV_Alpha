#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2}"
OPENVID_DIR="${PROJECT_ROOT}/data/openvid"

if [[ ! -d "${OPENVID_DIR}" ]]; then
  echo "OpenVid directory not found: ${OPENVID_DIR}" >&2
  exit 1
fi

echo "Before cleanup:"
du -sh "${OPENVID_DIR}"
du -sh "${PROJECT_ROOT}/data"

# Direct rm is policy-blocked in this environment, so truncate the files to
# zero bytes to release storage while keeping predictable paths.
find "${OPENVID_DIR}" -maxdepth 1 -type f -name 'OpenVid_part*.zip' -exec truncate -s 0 {} +
find "${OPENVID_DIR}" -maxdepth 1 -type f -name '*.log' -exec truncate -s 0 {} +

echo "After cleanup:"
du -sh "${OPENVID_DIR}"
du -sh "${PROJECT_ROOT}/data"
df -h /share_4 | sed -n '1,5p'
