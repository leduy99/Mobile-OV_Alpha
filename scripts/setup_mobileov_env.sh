#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-mobileov}"

if [[ -f /share_0/conda/etc/profile.d/conda.sh ]]; then
  # Canonical conda location on the current server.
  # shellcheck disable=SC1091
  source /share_0/conda/etc/profile.d/conda.sh
else
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] updating existing env: ${ENV_NAME}"
  conda env update -n "${ENV_NAME}" -f "${ROOT_DIR}/environment.yml" --prune
else
  echo "[setup] creating new env: ${ENV_NAME}"
  conda env create -n "${ENV_NAME}" -f "${ROOT_DIR}/environment.yml"
fi

conda activate "${ENV_NAME}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "[setup] installing editable OpenVid DataOps package"
pip install -e "${ROOT_DIR}/download_data"

echo "[setup] running smoke checks"
"${ROOT_DIR}/scripts/validate_mobileov_env.sh" "${ENV_NAME}" "$(which python)"

cat <<EOF

[setup] environment is ready: ${ENV_NAME}

Next shell:
  source /share_0/conda/etc/profile.d/conda.sh
  conda activate ${ENV_NAME}
  source ${ROOT_DIR}/scripts/env_exports.sh

EOF
