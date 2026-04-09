#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

INPUT_ROOT=""
OUTPUT_ROOT=""
DATASET_NAME=""
FILENAMES="all"
BOOTSTRAP_JOBS=8
BOOTSTRAP_LOG_EVERY=1000
ENCODE_LOG_EVERY=200
NPROC_PER_NODE=1
VAE_CKPT="omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
MAX_SAMPLES=""
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/prepare_local_wds_image_dataset.sh \
    --input-root /path/to/local_wds_dir \
    --output-root data/local_wds_run \
    --dataset-name local_wds_run \
    [--filenames all] \
    [--bootstrap-jobs 8] \
    [--bootstrap-log-every 1000] \
    [--encode-log-every 200] \
    [--nproc-per-node 1] \
    [--vae-ckpt omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth] \
    [--max-samples 100] \
    [--overwrite]

This one-command pipeline:
1. reads local .tar shards containing paired image/text files,
2. materializes local raw images + a source manifest CSV,
3. encodes WAN VAE latent pickles,
4. writes a final train-ready manifest CSV.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root)
      INPUT_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --dataset-name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --filenames)
      FILENAMES="$2"
      shift 2
      ;;
    --bootstrap-jobs)
      BOOTSTRAP_JOBS="$2"
      shift 2
      ;;
    --bootstrap-log-every)
      BOOTSTRAP_LOG_EVERY="$2"
      shift 2
      ;;
    --encode-log-every)
      ENCODE_LOG_EVERY="$2"
      shift 2
      ;;
    --nproc-per-node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --vae-ckpt)
      VAE_CKPT="$2"
      shift 2
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --torchrun-bin)
      TORCHRUN_BIN="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${INPUT_ROOT}" || -z "${OUTPUT_ROOT}" || -z "${DATASET_NAME}" ]]; then
  echo "--input-root, --output-root, and --dataset-name are required." >&2
  usage >&2
  exit 1
fi

SOURCE_MANIFEST="${OUTPUT_ROOT}/manifests/${DATASET_NAME}_source.csv"
ENCODED_DIR="${OUTPUT_ROOT}/encoded/wan_vae_sana_ar"
TRAIN_READY_CSV="${OUTPUT_ROOT}/manifests/${DATASET_NAME}_train_ready.csv"

bootstrap_cmd=(
  "${PYTHON_BIN}"
  "tools/data_prepare/bootstrap_local_wds_source_manifest.py"
  "--input-root" "${INPUT_ROOT}"
  "--filenames" "${FILENAMES}"
  "--output-root" "${OUTPUT_ROOT}"
  "--dataset-name" "${DATASET_NAME}"
  "--jobs" "${BOOTSTRAP_JOBS}"
  "--log-every" "${BOOTSTRAP_LOG_EVERY}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  bootstrap_cmd+=("--max-samples" "${MAX_SAMPLES}")
fi
if [[ "${OVERWRITE}" -eq 1 ]]; then
  bootstrap_cmd+=("--overwrite")
fi

echo "==> Step 1/3: bootstrap local WebDataset shards"
printf '    input-root: %s\n' "${INPUT_ROOT}"
printf '    output-root: %s\n' "${OUTPUT_ROOT}"
printf '    dataset-name: %s\n' "${DATASET_NAME}"
printf '    filenames: %s\n' "${FILENAMES}"
printf '    bootstrap-jobs: %s\n' "${BOOTSTRAP_JOBS}"
"${bootstrap_cmd[@]}"

encode_cmd=(
  "--manifest-csv" "${SOURCE_MANIFEST}"
  "--output-dir" "${ENCODED_DIR}"
  "--vae-ckpt" "${VAE_CKPT}"
  "--log-every" "${ENCODE_LOG_EVERY}"
)
if [[ -n "${MAX_SAMPLES}" ]]; then
  encode_cmd+=("--max-samples" "${MAX_SAMPLES}")
fi

echo "==> Step 2/3: encode WAN VAE latents"
printf '    source-manifest: %s\n' "${SOURCE_MANIFEST}"
printf '    encoded-dir: %s\n' "${ENCODED_DIR}"
printf '    nproc-per-node: %s\n' "${NPROC_PER_NODE}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  "${TORCHRUN_BIN}" \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    tools/data_prepare/encode_laion_coyo_images_sana_ar.py \
    "${encode_cmd[@]}"
else
  "${PYTHON_BIN}" \
    tools/data_prepare/encode_laion_coyo_images_sana_ar.py \
    "${encode_cmd[@]}"
fi

echo "==> Step 3/3: build train-ready manifest"
printf '    train-ready-csv: %s\n' "${TRAIN_READY_CSV}"
"${PYTHON_BIN}" \
  tools/data_prepare/build_laion_coyo_encoded_manifest.py \
  --source-manifest "${SOURCE_MANIFEST}" \
  --encoded-dir "${ENCODED_DIR}" \
  --output-csv "${TRAIN_READY_CSV}" \
  --datasets "${DATASET_NAME}" \
  --modality image

echo "==> Done"
printf '    source-manifest: %s\n' "${SOURCE_MANIFEST}"
printf '    encoded-dir: %s\n' "${ENCODED_DIR}"
printf '    train-ready-csv: %s\n' "${TRAIN_READY_CSV}"
