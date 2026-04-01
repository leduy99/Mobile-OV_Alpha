# LAION / COYO Full-Mix Data Preparation: Step-by-Step Guide

This guide explains how to rebuild the current `OpenVid + LAION / COYO` data
mix used by this repository.

Use this guide if you want to go from:

- a LAION / COYO source manifest CSV
- to downloaded raw media files
- to WAN VAE encoded image latents
- to a final mixed train manifest that can be consumed by the current training
  configs

## Read This First

### What "full mix" means in the current repo

In the current training scope, the mixed dataset means:

- OpenVid video rows
- LAION image rows
- COYO image rows

The current repo does **not** use generic COYO video rows in the canonical
OpenVid + LAION / COYO training path.

### What this repo can and cannot do

This repo can:

- materialize raw media from a unified LAION / COYO source manifest
- repair LAION image filename collisions
- encode LAION / COYO image rows into WAN VAE latent pickles
- build the final OpenVid + LAION / COYO train manifest
- validate the final manifest before training

This repo does **not** currently provide a polished crawler that discovers and
filters LAION / COYO rows from scratch.

That means you must start from one of these two inputs:

- the exact source manifest from an existing workspace
- your own curated source manifest

If you want to reproduce the **exact current mix**, copy the exact manifest CSV
from the existing server.

## Before You Start

Use the canonical environment:

```bash
conda activate mobileov
source scripts/env_exports.sh
pip install -e download_data
```

Recommended system tools:

```bash
ffmpeg
wget
yt-dlp
```

If needed:

```bash
huggingface-cli login
```

## Files and Scripts Used in This Guide

Main scripts:

- `tools/data_prepare/materialize_unified_manifest.py`
- `tools/data_prepare/merge_unified_manifest_shards.py`
- `tools/data_prepare/recover_laion_images_unique.py`
- `tools/data_prepare/encode_laion_coyo_images_sana_ar.py`
- `tools/data_prepare/build_laion_coyo_encoded_manifest.py`
- `tools/data_prepare/build_joint_manifest_openvid_current_laion_coyo.py`
- `tools/data_prepare/sanitize_joint_manifest.py`
- `tools/data_prepare/preflight_joint_pipeline_check.py`

OpenVid dependency:

- `download_data/README.md`

## Directory Layout Used Below

The commands below assume this layout:

```text
data/
  openvid_1m/
    manifests/
    encoded/
  laion_coyo/
    manifests/
    raw/
    encoded/
  mix/
    manifests/
```

## Step 1: Prepare OpenVid Latents First

This full-mix pipeline assumes that OpenVid video latents already exist.

### Command

```bash
python download_data/scripts/download_wan_vae_2_1.py \
  --output-dir download_data/checkpoints/wan/wanxiang1_3b

python -m openvid_dataops download \
  --parts all \
  --extract

python -m openvid_dataops build-manifest \
  --parts all \
  --output-name openvid_all.csv

python -m openvid_dataops encode \
  --manifest-csv download_data/data/openvid/manifests/openvid_all.csv \
  --ckpt-dir download_data/checkpoints/wan/wanxiang1_3b \
  --task t2v-1.3B \
  --frame-num 81 \
  --sampling-rate 1 \
  --target-size 480,832 \
  --output-subdir wan_vae_openvid_all
```

### What this step does

This creates the OpenVid side of the final mixed dataset.

### Arguments that matter most

- `--parts`: which OpenVid parts to download
- `--manifest-csv`: which OpenVid manifest to encode
- `--ckpt-dir`: WAN checkpoint directory
- `--frame-num`: number of frames per video sample
- `--sampling-rate`: temporal stride when sampling frames
- `--target-size`: spatial resolution in `H,W`
- `--output-subdir`: where the OpenVid latent pickles will be stored

### Expected output

You should end up with:

- an OpenVid manifest CSV
- an encoded directory containing `sample_XXXXXXXX.pkl` files

## Step 2: Obtain a LAION / COYO Source Manifest

This is the one input that the repo does not generate for you.

### What the source manifest should contain

At minimum:

- `sample_idx`
- `dataset`
- `modality`
- `caption`
- `source_url`

Useful optional columns:

- `source_id`
- `image_path`
- `media_path`
- `video_path`
- `extra_json`

For the current training path, the LAION / COYO rows should be `image` rows.

### Best option if you want exact reproducibility

Copy the exact source manifest from the existing server, for example a file such
as:

```text
data/laion_coyo/manifests/laion_coyo_selected_media_existing_58k.csv
```

### Example environment variable used below

```bash
export LAION_COYO_SOURCE_MANIFEST=data/laion_coyo/manifests/laion_coyo_source_selected.csv
```

## Step 3: Download Raw LAION / COYO Media From the Source Manifest

### Single-worker command

```bash
PYTHONPATH=. python tools/data_prepare/materialize_unified_manifest.py \
  --manifest-csv "$LAION_COYO_SOURCE_MANIFEST" \
  --dataset-root data/laion_coyo/raw \
  --output-manifest data/laion_coyo/manifests/laion_coyo_materialized.csv \
  --modality all \
  --worker-id 0 \
  --num-workers 1
```

### What this step does

This reads `source_url` from each row, downloads the referenced media, and
writes local paths back into the output manifest.

### Arguments that matter most

- `--manifest-csv`: input source manifest
- `--dataset-root`: root directory where raw files will be stored
- `--output-manifest`: output CSV with updated local paths
- `--modality`: which rows to materialize
  - `image` for image rows only
  - `video` for video rows only
  - `all` for both
- `--worker-id`: current worker id for sharded downloads
- `--num-workers`: total number of workers
- `--url-timeout-s`: per-request timeout
- `--max-retries`: retry count for failed downloads
- `--cleanup-full-video`: delete full downloaded videos after clipped outputs are made
- `--cookies-file`: browser/exported cookies file for protected video URLs
- `--cookies-from-browser`: ask `yt-dlp` to pull cookies from a local browser profile
- `--log-every`: progress log frequency

### When to change `--worker-id` and `--num-workers`

Only change these when you want to split the download job across multiple
processes or machines.

### Optional: multi-worker materialization

Worker 0:

```bash
PYTHONPATH=. python tools/data_prepare/materialize_unified_manifest.py \
  --manifest-csv "$LAION_COYO_SOURCE_MANIFEST" \
  --dataset-root data/laion_coyo/raw \
  --output-manifest data/laion_coyo/manifests/laion_coyo_materialized.worker00.csv \
  --modality all \
  --worker-id 0 \
  --num-workers 4
```

Repeat the same command for worker ids `1`, `2`, and `3`, then merge the shard
CSVs:

```bash
PYTHONPATH=. python tools/data_prepare/merge_unified_manifest_shards.py \
  --input-glob 'data/laion_coyo/manifests/laion_coyo_materialized.worker*.csv' \
  --output-csv data/laion_coyo/manifests/laion_coyo_materialized.csv
```

## Step 4: Repair LAION Image Filename Collisions

This step is recommended if your manifest contains LAION image rows.

### Why this exists

Different LAION rows can produce the same raw filename, which can lead to:

- image overwrites
- image-caption mismatch
- bad training examples

### Command

```bash
PYTHONPATH=. python tools/data_prepare/recover_laion_images_unique.py \
  --input-manifest data/laion_coyo/manifests/laion_coyo_materialized.csv \
  --output-manifest data/laion_coyo/manifests/laion_coyo_recovered_unique.csv \
  --output-image-dir data/laion_coyo/raw/media/images_laion_recovered \
  --failures-csv data/laion_coyo/manifests/laion_recovered_unique_failures.csv \
  --summary-json data/laion_coyo/manifests/laion_recovered_unique_summary.json \
  --workers 32 \
  --timeout 20 \
  --retries 2
```

### What this step does

It rewrites LAION image rows so each image is stored under a unique,
`sample_idx`-based file name.

### Arguments that matter most

- `--input-manifest`: materialized manifest from Step 3
- `--output-manifest`: fixed manifest to use in later steps
- `--output-image-dir`: directory for repaired LAION images
- `--workers`: download concurrency
- `--timeout`: network timeout in seconds
- `--retries`: retry count
- `--min-side`: reject images below this minimum side length
- `--min-bytes`: reject tiny/broken files below this size
- `--max-samples`: useful for smoke tests
- `--overwrite`: force redownload even if files already exist

### If your manifest is COYO-only

If there are no LAION image rows, you can skip this step and keep using the
materialized manifest from Step 3.

## Step 5: Encode LAION / COYO Image Rows Into WAN VAE Latents

### Single-GPU command

```bash
PYTHONPATH=. python tools/data_prepare/encode_laion_coyo_images_sana_ar.py \
  --manifest-csv data/laion_coyo/manifests/laion_coyo_recovered_unique.csv \
  --output-dir data/laion_coyo/encoded/wan_vae_openvid_mix_sana_ar \
  --vae-ckpt omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth
```

If you skipped Step 4, replace the manifest path with:

```text
data/laion_coyo/manifests/laion_coyo_materialized.csv
```

### Multi-GPU command

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
  tools/data_prepare/encode_laion_coyo_images_sana_ar.py \
  --manifest-csv data/laion_coyo/manifests/laion_coyo_recovered_unique.csv \
  --output-dir data/laion_coyo/encoded/wan_vae_openvid_mix_sana_ar \
  --vae-ckpt omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth
```

### What this step does

This command:

1. reads image rows from the manifest
2. picks the nearest SANA 480 bucket for each image
3. center-crops and resizes the image
4. encodes it with WAN VAE
5. writes `sample_XXXXXXXX.pkl` files

### Arguments that matter most

- `--manifest-csv`: input manifest containing image rows
- `--output-dir`: where encoded latent pickles will be written
- `--vae-ckpt`: path to the WAN VAE checkpoint
- `--max-samples`: useful for smoke tests
- `--log-every`: progress logging interval
- `--seed`: random seed

### Expected output

```text
data/laion_coyo/encoded/wan_vae_openvid_mix_sana_ar/sample_00000000.pkl
...
data/laion_coyo/encoded/wan_vae_openvid_mix_sana_ar/summary_rank00.json
```

## Step 6: Build a Train-Ready LAION / COYO Encoded Manifest

### Command

```bash
PYTHONPATH=. python tools/data_prepare/build_laion_coyo_encoded_manifest.py \
  --source-manifest data/laion_coyo/manifests/laion_coyo_recovered_unique.csv \
  --encoded-dir data/laion_coyo/encoded/wan_vae_openvid_mix_sana_ar \
  --output-csv data/laion_coyo/manifests/laion_coyo_encoded_for_openvid_mix.csv \
  --datasets laion,coyo_700m \
  --modality image
```

If you skipped Step 4, use the materialized manifest as `--source-manifest`
instead.

### What this step does

This converts:

- the source manifest
- plus the encoded output directory

into a normalized train-ready manifest with explicit `preprocessed_path` values.

### Arguments that matter most

- `--source-manifest`: LAION / COYO manifest that still contains metadata such
  as `caption`, `dataset`, and raw media paths
- `--encoded-dir`: directory containing `sample_XXXXXXXX.pkl` files
- `--output-csv`: normalized train-ready output CSV
- `--datasets`: comma-separated dataset names to keep
- `--modality`: keep only image rows, only video rows, or all rows

### Expected output

```text
data/laion_coyo/manifests/laion_coyo_encoded_for_openvid_mix.csv
data/laion_coyo/manifests/laion_coyo_encoded_for_openvid_mix.summary.json
```

## Step 7: Build the Final OpenVid + LAION / COYO Joint Manifest

### Command

```bash
PYTHONPATH=. python tools/data_prepare/build_joint_manifest_openvid_current_laion_coyo.py \
  --existing-mix-manifest data/laion_coyo/manifests/laion_coyo_encoded_for_openvid_mix.csv \
  --openvid-manifest data/openvid_1m/manifests/openvid_manifest_0_112.csv \
  --openvid-preprocessed-dir data/openvid_1m/encoded/wan_vae_fp16_stream \
  --output-prefix data/mix/manifests/joint_openvid_current_laion_coyo_rebuilt
```

### What this step does

This merges:

- OpenVid video rows
- LAION / COYO image rows

into the final manifest format expected by the current training configs.

### Arguments that matter most

- `--existing-mix-manifest`: the LAION / COYO encoded manifest from Step 6
- `--openvid-manifest`: OpenVid manifest with `sample_idx` values
- `--openvid-preprocessed-dir`: directory containing OpenVid latent pickles
- `--output-prefix`: prefix used for the combined, video-only, image-only, and
  summary outputs

### Expected output

```text
data/mix/manifests/joint_openvid_current_laion_coyo_rebuilt.csv
data/mix/manifests/joint_openvid_current_laion_coyo_rebuilt_video.csv
data/mix/manifests/joint_openvid_current_laion_coyo_rebuilt_image.csv
data/mix/manifests/joint_openvid_current_laion_coyo_rebuilt.summary.json
```

### Important note

This builder keeps LAION / COYO **image rows only** in the current training
setup. If you want a different policy, create a different builder instead of
quietly reusing this one.

## Step 8: Sanitize the Joint Manifest

### Command

```bash
PYTHONPATH=. python tools/data_prepare/sanitize_joint_manifest.py \
  --input-csv data/mix/manifests/joint_openvid_current_laion_coyo_rebuilt.csv \
  --output-csv data/mix/manifests/joint_openvid_current_laion_coyo_rebuilt_clean.csv \
  --image-datasets coyo_700m,laion \
  --min-image-bytes 2000 \
  --dedup-image-path \
  --drop-missing-preprocessed \
  --drop-empty-caption
```

### What this step does

This removes rows that are likely to be bad training samples.

### Arguments that matter most

- `--input-csv`: raw combined manifest
- `--output-csv`: cleaned output manifest
- `--image-datasets`: which image datasets to keep
- `--min-image-bytes`: reject very small image files
- `--dedup-image-path`: remove duplicate image-path rows
- `--drop-missing-preprocessed`: remove rows whose latent pickle is missing
- `--drop-empty-caption`: remove rows with empty captions
- `--reindex-sample-idx`: rewrite sample ids to contiguous values after filtering

## Step 9: Run Preflight Checks Before Training

### Command

```bash
PYTHONPATH=. python tools/data_prepare/preflight_joint_pipeline_check.py \
  --config configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_only_smolvlm2_2p2b_online_teacher_20k_2gpu_20260321.yaml \
  --csv-path data/mix/manifests/joint_openvid_current_laion_coyo_rebuilt_clean.csv \
  --report-json data/mix/manifests/joint_openvid_current_laion_coyo_rebuilt_clean.preflight.json
```

### What this step does

This validates the manifest and a sample of latent pickles before you launch a
long training run.

### Arguments that matter most

- `--config`: training config to validate against
- `--csv-path`: manifest to test; use this when your rebuilt file name differs
  from the one inside the config
- `--report-json`: where to save the preflight report
- `--sample-video`: number of video rows to sample for latent checks
- `--sample-image`: number of image rows to sample for latent checks
- `--max-loader-batches`: dataloader dry-run batch count
- `--tiny-image-bytes`: threshold used to flag suspiciously small images

### What to do with the result

Treat a non-zero exit code as a blocker. Fix the manifest before training.

## Step 10: Point Training To the Rebuilt Manifest

You have two clean options.

### Option A: Reuse the existing config names

Write the rebuilt files to the canonical paths expected by the current config.

### Option B: Copy a config and edit the manifest paths

Update these fields in the copied config:

- `data.openvid.csv_path`
- `data.openvid.csv_path_video`
- `data.openvid.csv_path_image`

Use this option if you want to keep the rebuilt files under new names.

## Step 11: Train

### Example bridge-only run

```bash
PYTHONPATH=. torchrun --nproc_per_node=2 \
  tools/train_stage1_teacher_free.py \
  --config configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_only_smolvlm2_2p2b_online_teacher_20k_2gpu_20260321.yaml \
  --max-gpus 2
```

### Example full 3-stage run

```bash
GPUS=0,1 bash scripts/train_openvid_current_laion_coyo_3stage_gemma_distill_5v1i.sh
```

## Smallest Useful Smoke-Test Plan

If you want to test the pipeline before committing to the full rebuild, do this:

1. use a tiny source manifest
2. materialize only that subset
3. run `encode_laion_coyo_images_sana_ar.py --max-samples 16`
4. build the encoded manifest
5. build the final joint manifest
6. run preflight

If that succeeds, scale up.

## The Most Important Truth About This Pipeline

The hardest part is not the encoder. The hardest part is the **source manifest**.

If you want exact reproducibility, preserve and share:

- the canonical LAION / COYO source manifest
- the encoded-manifest summary JSON
- the final joint manifest summary JSON
- the preflight report JSON

That is the minimum metadata bundle that makes this workflow reproducible on a
new server.

## Help Commands

```bash
PYTHONPATH=. python tools/data_prepare/materialize_unified_manifest.py --help
PYTHONPATH=. python tools/data_prepare/merge_unified_manifest_shards.py --help
PYTHONPATH=. python tools/data_prepare/recover_laion_images_unique.py --help
PYTHONPATH=. python tools/data_prepare/encode_laion_coyo_images_sana_ar.py --help
PYTHONPATH=. python tools/data_prepare/build_laion_coyo_encoded_manifest.py --help
PYTHONPATH=. python tools/data_prepare/build_joint_manifest_openvid_current_laion_coyo.py --help
PYTHONPATH=. python tools/data_prepare/sanitize_joint_manifest.py --help
PYTHONPATH=. python tools/data_prepare/preflight_joint_pipeline_check.py --help
```
