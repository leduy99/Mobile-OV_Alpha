#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2}"
DATA_DOWNLOAD_ROOT="${DATA_DOWNLOAD_ROOT:-/share_4/users/duy/project/unified_video/data_download}"
PYTHON_BIN="${PYTHON_BIN:-python}"

HDVILA_TARGET_ROWS="${HDVILA_TARGET_ROWS:-300000}"
COYO_TARGET_ROWS="${COYO_TARGET_ROWS:-700000}"
LAION_TARGET_ROWS="${LAION_TARGET_ROWS:-350000}"

COYO_SHARDS="${COYO_SHARDS:-16}"
LAION_SHARDS="${LAION_SHARDS:-1}"

HDVILA_WORKERS="${HDVILA_WORKERS:-4}"
IMAGE_WORKERS="${IMAGE_WORKERS:-16}"

HDVILA_MANIFEST_NAME="${HDVILA_MANIFEST_NAME:-hdvila_manifest_300k_20260308.csv}"
LAION_COYO_MANIFEST_NAME="${LAION_COYO_MANIFEST_NAME:-laion_coyo_manifest_700k_350k_20260308.csv}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
HDVILA_SHARD_DIR="${PROJECT_ROOT}/data/hd_vila/manifests/materialized_${RUN_STAMP}"
IMAGE_SHARD_DIR="${PROJECT_ROOT}/data/laion_coyo/manifests/materialized_${RUN_STAMP}"

mkdir -p "${HDVILA_SHARD_DIR}" "${IMAGE_SHARD_DIR}"

if [[ -z "${YTDLP_COOKIES_FILE:-}" && -z "${YTDLP_COOKIES_FROM_BROWSER:-}" ]]; then
  echo "HD-VILA download requires YTDLP_COOKIES_FILE or YTDLP_COOKIES_FROM_BROWSER." >&2
  exit 1
fi

pushd "${DATA_DOWNLOAD_ROOT}" >/dev/null

if [[ ! -f "${PROJECT_ROOT}/data/hd_vila/manifests/${HDVILA_MANIFEST_NAME}" ]]; then
  "${PYTHON_BIN}" -m openvid_dataops download-hdvila \
    --root "${PROJECT_ROOT}" \
    --max-rows "${HDVILA_TARGET_ROWS}" \
    --output-name "${HDVILA_MANIFEST_NAME}"
fi

if [[ ! -f "${PROJECT_ROOT}/data/laion_coyo/manifests/${LAION_COYO_MANIFEST_NAME}" ]]; then
  "${PYTHON_BIN}" -m openvid_dataops download-laion-coyo \
    --root "${PROJECT_ROOT}" \
    --coyo-rows "${COYO_TARGET_ROWS}" \
    --laion-rows "${LAION_TARGET_ROWS}" \
    --coyo-shards "${COYO_SHARDS}" \
    --laion-shards "${LAION_SHARDS}" \
    --min-coyo-clip-sim 0.22 \
    --max-coyo-nsfw 0.20 \
    --max-coyo-watermark 0.50 \
    --min-coyo-res 256 \
    --min-laion-sim 0.24 \
    --max-laion-nsfw 0.20 \
    --min-laion-res 256 \
    --min-caption-words 4 \
    --output-name "${LAION_COYO_MANIFEST_NAME}"
fi

popd >/dev/null

PIDS=()
for ((worker=0; worker<HDVILA_WORKERS; worker++)); do
  "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/data_prepare/materialize_unified_manifest.py" \
    --manifest-csv "${PROJECT_ROOT}/data/hd_vila/manifests/${HDVILA_MANIFEST_NAME}" \
    --dataset-root "${PROJECT_ROOT}/data/hd_vila" \
    --output-manifest "${HDVILA_SHARD_DIR}/hdvila_worker_${worker}.csv" \
    --modality video \
    --worker-id "${worker}" \
    --num-workers "${HDVILA_WORKERS}" \
    --cleanup-full-video \
    --log-every 50 &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/data_prepare/merge_unified_manifest_shards.py" \
  --input-glob "${HDVILA_SHARD_DIR}/hdvila_worker_*.csv" \
  --output-csv "${PROJECT_ROOT}/data/hd_vila/manifests/hdvila_materialized_${RUN_STAMP}.csv"

PIDS=()
for ((worker=0; worker<IMAGE_WORKERS; worker++)); do
  "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/data_prepare/materialize_unified_manifest.py" \
    --manifest-csv "${PROJECT_ROOT}/data/laion_coyo/manifests/${LAION_COYO_MANIFEST_NAME}" \
    --dataset-root "${PROJECT_ROOT}/data/laion_coyo" \
    --output-manifest "${IMAGE_SHARD_DIR}/laion_coyo_worker_${worker}.csv" \
    --modality image \
    --worker-id "${worker}" \
    --num-workers "${IMAGE_WORKERS}" \
    --log-every 500 &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/data_prepare/merge_unified_manifest_shards.py" \
  --input-glob "${IMAGE_SHARD_DIR}/laion_coyo_worker_*.csv" \
  --output-csv "${PROJECT_ROOT}/data/laion_coyo/manifests/laion_coyo_materialized_${RUN_STAMP}.csv"

echo "HD-VILA materialized manifest:"
echo "  ${PROJECT_ROOT}/data/hd_vila/manifests/hdvila_materialized_${RUN_STAMP}.csv"
echo "LAION+COYO materialized manifest:"
echo "  ${PROJECT_ROOT}/data/laion_coyo/manifests/laion_coyo_materialized_${RUN_STAMP}.csv"
