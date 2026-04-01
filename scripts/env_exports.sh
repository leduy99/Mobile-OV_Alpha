#!/usr/bin/env bash

# Source this after `conda activate <env>` to make repo-local imports and
# package resolution behave consistently across training and inference entrypoints.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Use this script with: source scripts/env_exports.sh" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "PYTHONNOUSERSITE=1"
echo "PYTHONPATH=${PYTHONPATH}"
