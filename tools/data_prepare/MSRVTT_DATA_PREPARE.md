# MSR-VTT Data Preparation: Step-by-Step Guide

This guide explains how to rebuild the MSR-VTT preprocessing outputs used by
this repository.

Use this guide if you want to go from:

- raw MSR-VTT files on Hugging Face
- to an OpenVid-style CSV
- to WAN VAE encoded `*_features.pkl` files

## What This Pipeline Produces

After you finish this guide, you will have:

- raw downloaded MSR-VTT files
- extracted videos and metadata
- an OpenVid-style CSV manifest
- WAN VAE encoded features under `data/msrvtt/preprocessed/`

## Before You Start

Use the canonical repository environment:

```bash
conda activate mobileov
```

If needed:

```bash
huggingface-cli login
```

## Folder Layout

Typical outputs live under:

```text
data/msrvtt/
  raw/
  videos/
  metadata/
  preprocessed/
  OpenVid_extracted_subset_unique.csv
```

## Step 1: Download Raw MSR-VTT Assets

### Command

```bash
PYTHONPATH=. python tools/data_prepare/msrvtt_data_prepare.py download \
  --root-dir data/msrvtt \
  --repo-id AlexZigma/msr-vtt \
  --extract
```

### What this does

This command downloads the raw MSR-VTT assets from the Hugging Face dataset and
extracts the zip files.

### Important arguments

- `--root-dir`: root directory for all MSR-VTT outputs
- `--repo-id`: Hugging Face dataset id
- `--extract`: extract zip files after download
- `--hf-token`: optional token override if you do not want to rely on
  `huggingface-cli login`
- `--video-zip`: override the video zip file name if the upstream dataset changes
- `--meta-zip`: override the metadata zip file name
- `--train-parquet`: override the train parquet file name
- `--val-parquet`: override the validation parquet file name

### Expected output

```text
data/msrvtt/raw/
data/msrvtt/videos/TestVideo/
data/msrvtt/metadata/
```

## Step 2: Build an OpenVid-Style CSV

### Command

```bash
PYTHONPATH=. python tools/data_prepare/msrvtt_data_prepare.py build-csv \
  --root-dir data/msrvtt \
  --manifest-name OpenVid_extracted_subset_unique.csv \
  --caption-policy longest \
  --captions-per-video 1
```

### What this does

This command reads the downloaded metadata and creates a simple CSV with one row
per video and a single caption per video.

### Important arguments

- `--root-dir`: the same root used in Step 1
- `--manifest-name`: output CSV file name
- `--caption-policy`: how to choose one caption when multiple captions exist
  - `first`: take the first caption
  - `longest`: take the longest caption
  - `random`: sample one caption randomly
- `--captions-per-video`: how many captions to keep per video
- `--max-videos`: useful for smoke tests or partial runs
- `--seed`: random seed for deterministic caption selection when using random

### Expected output

```text
data/msrvtt/OpenVid_extracted_subset_unique.csv
```

## Step 3: Encode Videos Into WAN VAE Latents

### Single-GPU command

```bash
PYTHONPATH=. python tools/data_prepare/msrvtt_data_prepare.py encode \
  --root-dir data/msrvtt \
  --manifest-name OpenVid_extracted_subset_unique.csv \
  --ckpt-dir omni_ckpts/wan/wanxiang1_3b \
  --task t2v-1.3B \
  --frame-num 81 \
  --sampling-rate 1 \
  --target-size 480,832
```

### Multi-GPU command

```bash
torchrun --standalone --nproc_per_node=4 \
  tools/data_prepare/msrvtt_data_prepare.py encode \
  --root-dir data/msrvtt \
  --manifest-name OpenVid_extracted_subset_unique.csv \
  --ckpt-dir omni_ckpts/wan/wanxiang1_3b \
  --task t2v-1.3B \
  --frame-num 81 \
  --sampling-rate 1 \
  --target-size 480,832
```

### What this does

This command reads each MSR-VTT video, samples frames, runs WAN VAE encoding,
and writes one `*_features.pkl` file per sample.

### Important arguments

- `--root-dir`: the same root used in previous steps
- `--manifest-name`: which CSV to encode
- `--ckpt-dir`: WAN checkpoint directory
- `--task`: WAN task preset
- `--frame-num`: number of frames per sample
- `--sampling-rate`: temporal stride when sampling video frames
- `--skip-num`: number of frames to skip before sampling starts
- `--target-size`: output size in `H,W`
- `--max-samples`: useful for smoke tests or partial runs
- `--log-every`: progress logging interval
- `--allow-pad-short`: whether to pad short videos by repeating the last frame

### Expected output

```text
data/msrvtt/preprocessed/<video_id>_features.pkl
data/msrvtt/preprocessed/encode_summary_rank00.json
data/msrvtt/preprocessed/encode_failed_rank00.csv
```

Each pickle contains fields such as:

- `latent_feature`
- `prompt`
- `video_path`
- `frame_num`

## Step 4: Run the Whole Pipeline in One Command

### Command

```bash
PYTHONPATH=. python tools/data_prepare/msrvtt_data_prepare.py all \
  --root-dir data/msrvtt \
  --repo-id AlexZigma/msr-vtt \
  --manifest-name OpenVid_extracted_subset_unique.csv \
  --ckpt-dir omni_ckpts/wan/wanxiang1_3b \
  --frame-num 81 \
  --sampling-rate 1 \
  --target-size 480,832
```

### What this does

This runs:

1. `download`
2. `build-csv`
3. `encode`

in one pipeline.

Use this only after you are confident the individual steps work on your server.

## Quick Smoke-Test Strategy

Before processing the full dataset, try a tiny run:

```bash
PYTHONPATH=. python tools/data_prepare/msrvtt_data_prepare.py encode \
  --root-dir data/msrvtt \
  --manifest-name OpenVid_extracted_subset_unique.csv \
  --ckpt-dir omni_ckpts/wan/wanxiang1_3b \
  --task t2v-1.3B \
  --frame-num 81 \
  --sampling-rate 1 \
  --target-size 480,832 \
  --max-samples 16
```

## Which Arguments Usually Matter Most?

If you are just trying to get the pipeline working, focus on these flags first:

- `--root-dir`
- `--repo-id`
- `--manifest-name`
- `--ckpt-dir`
- `--frame-num`
- `--sampling-rate`
- `--target-size`
- `--max-samples`

Most users do not need to change the rest on day one.

## Help Commands

```bash
PYTHONPATH=. python tools/data_prepare/msrvtt_data_prepare.py --help
PYTHONPATH=. python tools/data_prepare/msrvtt_data_prepare.py download --help
PYTHONPATH=. python tools/data_prepare/msrvtt_data_prepare.py build-csv --help
PYTHONPATH=. python tools/data_prepare/msrvtt_data_prepare.py encode --help
```
