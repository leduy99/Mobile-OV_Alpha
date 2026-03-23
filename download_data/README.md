# OpenVid DataOps: Step-by-Step Guide

This guide explains how to rebuild OpenVid latents with the bundled DataOps
package inside this repository.

Use this guide if you want to go from:

- raw OpenVid parts on Hugging Face
- to an OpenVid manifest CSV
- to WAN VAE encoded `sample_XXXXXXXX.pkl` files

## What This Pipeline Produces

After you finish this guide, you will have:

- raw OpenVid zip files
- extracted video files
- a manifest CSV
- WAN VAE latents ready for training

The main outputs live under:

```text
download_data/data/openvid/
  raw/
  manifests/
  encoded/
  state/
  logs/
```

## Before You Start

Use the canonical repository environment:

```bash
conda activate mobileov
pip install -e download_data
```

If you need Hugging Face access:

```bash
huggingface-cli login
```

## Step 1: Download the WAN VAE Checkpoint

### Command

```bash
python download_data/scripts/download_wan_vae_2_1.py \
  --output-dir download_data/checkpoints/wan/wanxiang1_3b
```

### What this does

This downloads the WAN 2.1 VAE weights used to encode videos into latent
pickles.

### Important argument

- `--output-dir`: where the WAN checkpoint directory will be stored

### Expected output

You should get a checkpoint directory under:

```text
download_data/checkpoints/wan/wanxiang1_3b/
```

## Step 2: Download Raw OpenVid Parts

### Command

```bash
python -m openvid_dataops download \
  --parts all \
  --extract
```

### What this does

This command:

1. looks up the available OpenVid parts on Hugging Face
2. downloads the selected zip files
3. optionally extracts them
4. caches the discovered part index for resume-friendly reruns

### Important arguments

- `--parts`: which parts to download
  - `all` means every discovered part
  - `"[1,2,4,5]"` means only those user-facing part ids
- `--extract`: extract the downloaded zip files after download
- `--keep-zip`: keep zip files after extraction
- `--no-csv`: skip downloading the main OpenVid CSV metadata file
- `--part-index-base`: how to interpret the part ids you pass
  - `0` means user part `N` maps directly to remote `OpenVid_partN.zip`
  - `1` means your numbering is one-based
- `--root`: override the mini-repo root if you do not want to use
  `download_data/` as the working root

### Resume behavior

- existing zip files are skipped
- extracted folders are reused
- the HF part index is cached under:

```text
download_data/data/openvid/state/hf_openvid_index_cache.json
```

### Expected output

Typical raw outputs are:

```text
download_data/data/openvid/raw/OpenVid-1M.csv
download_data/data/openvid/raw/zips/
download_data/data/openvid/raw/parts/
```

## Step 3: Build a Manifest CSV

### Command

```bash
python -m openvid_dataops build-manifest \
  --parts all \
  --output-name openvid_all.csv
```

### What this does

This command scans the extracted OpenVid parts and joins them with the OpenVid
CSV metadata to produce a manifest that the encoder can consume.

### Important arguments

- `--parts`: optional part filter
  - use `all` for every available part
  - or use a subset such as `"[1,2,4,5]"`
- `--output-name`: output file name under `download_data/data/openvid/manifests/`
- `--part-index-base`: same meaning as in the download step; only change this
  if your part numbering convention is different
- `--root`: override the working root

### Expected output

```text
download_data/data/openvid/manifests/openvid_all.csv
download_data/data/openvid/manifests/openvid_all.summary.json
```

The summary JSON is useful when you want to quickly confirm row counts and
selected parts.

## Step 4: Encode Videos Into WAN VAE Latents

### Single-GPU command

```bash
python -m openvid_dataops encode \
  --manifest-csv download_data/data/openvid/manifests/openvid_all.csv \
  --ckpt-dir download_data/checkpoints/wan/wanxiang1_3b \
  --task t2v-1.3B \
  --frame-num 81 \
  --sampling-rate 1 \
  --target-size 480,832 \
  --output-subdir wan_vae_openvid_all
```

### Multi-GPU command

```bash
bash download_data/scripts/run_encode_4gpu.sh \
  --manifest-csv download_data/data/openvid/manifests/openvid_all.csv \
  --ckpt-dir download_data/checkpoints/wan/wanxiang1_3b \
  --task t2v-1.3B \
  --frame-num 81 \
  --sampling-rate 1 \
  --target-size 480,832 \
  --output-subdir wan_vae_openvid_all
```

### What this does

This command reads each manifest row, loads the corresponding video, samples
frames, runs WAN VAE encoding, and writes one pickle per sample.

### Important arguments

- `--manifest-csv`: which manifest CSV to encode
- `--ckpt-dir`: WAN checkpoint directory downloaded in Step 1
- `--task`: WAN task preset
  - keep `t2v-1.3B` unless you explicitly need another WAN task
- `--frame-num`: number of frames per encoded sample
  - `81` is the canonical choice in this repo
- `--sampling-rate`: temporal stride when sampling frames from the raw video
  - `1` means dense sampling
  - larger values skip more frames
- `--skip-num`: number of initial frames to skip before sampling starts
- `--target-size`: output size in `H,W`
  - current canonical value is `480,832`
- `--max-samples`: useful for smoke tests or partial runs
- `--log-every`: how often to report progress
- `--output-subdir`: subfolder name under `download_data/data/openvid/encoded/`

### Expected output

```text
download_data/data/openvid/encoded/wan_vae_openvid_all/sample_00000000.pkl
download_data/data/openvid/encoded/wan_vae_openvid_all/sample_00000001.pkl
...
download_data/data/openvid/encoded/wan_vae_openvid_all/summary_rank00.json
download_data/data/openvid/encoded/wan_vae_openvid_all/failed_rank00.csv
```

If you rerun the encoder, existing `sample_*.pkl` files are skipped.

## Step 5: Point Training To the Encoded OpenVid Data

Once encoding is done, you have two common choices:

### Option A: use the encoded directory directly in your own manifest-building flow

This is the usual choice when OpenVid is only one part of a larger training mix.

### Option B: rename the manifest and encoded directory to match an existing config

This is convenient when you want to reuse a config without editing paths.

## Quick Smoke-Test Strategy

Before launching a very large run, it is worth doing a tiny encode pass:

```bash
python -m openvid_dataops encode \
  --manifest-csv download_data/data/openvid/manifests/openvid_all.csv \
  --ckpt-dir download_data/checkpoints/wan/wanxiang1_3b \
  --task t2v-1.3B \
  --frame-num 81 \
  --sampling-rate 1 \
  --target-size 480,832 \
  --output-subdir wan_vae_smoke \
  --max-samples 16
```

If this succeeds, scale up to the full encode job.

## Command Reference

### `openvid_dataops download`

Use this when you want raw OpenVid parts on disk.

Most important flags:

- `--parts`
- `--extract`
- `--keep-zip`
- `--part-index-base`
- `--no-csv`

### `openvid_dataops build-manifest`

Use this after raw download and extraction.

Most important flags:

- `--parts`
- `--output-name`
- `--part-index-base`

### `openvid_dataops encode`

Use this after the manifest exists and the WAN checkpoint is available.

Most important flags:

- `--manifest-csv`
- `--ckpt-dir`
- `--task`
- `--frame-num`
- `--sampling-rate`
- `--skip-num`
- `--target-size`
- `--max-samples`
- `--output-subdir`

## Troubleshooting

### The downloader cannot find a zip file

The downloader can fall back to chunked downloads for parts that are stored in a
split form on Hugging Face.

### Encoding is too slow

Use the multi-GPU wrapper:

```bash
bash download_data/scripts/run_encode_4gpu.sh ...
```

### I only want a subset first

Use:

- a smaller `--parts` selection
- or `--max-samples` during encoding

## Help Commands

```bash
python -m openvid_dataops --help
python -m openvid_dataops download --help
python -m openvid_dataops build-manifest --help
python -m openvid_dataops encode --help
```
