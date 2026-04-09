# Mobile-O-Pre-Train Data Guide

This guide explains how to use the current repo to:

1. download `Amshaker/Mobile-O-Pre-Train`
2. convert it into the repo's internal image manifest format
3. encode WAN VAE latents
4. build a train-ready manifest for the existing image-training pipeline

This path is now supported by:

- [bootstrap_mobile_o_pretrain_source_manifest.py](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/data_prepare/bootstrap_mobile_o_pretrain_source_manifest.py)
- [encode_laion_coyo_images_sana_ar.py](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/data_prepare/encode_laion_coyo_images_sana_ar.py)
- [build_laion_coyo_encoded_manifest.py](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/data_prepare/build_laion_coyo_encoded_manifest.py)

## 1. What Mobile-O-Pre-Train looks like

`Amshaker/Mobile-O-Pre-Train` is stored as WebDataset tar shards on Hugging Face.

Inside each shard, samples look like:

```text
000000000.jpg
000000000.txt
000000002.jpg
000000002.txt
...
```

The bootstrap script streams those tar shards, pairs each image with its text,
saves local images, and writes a source manifest CSV compatible with the
existing image-encoding and training code in this repo.

## 2. Environment

Use the verified env flow:

```bash
cd /share_4/users/duy/project/unified_video/Omni-Video-smolvlm2
source /share_0/conda/etc/profile.d/conda.sh
conda activate /share_4/users/duy/.conda/envs/mobileov_onepass_20260401
source scripts/env_exports.sh
export PYTHONNOUSERSITE=1
export PYTHONPATH=.
```

`Mobile-O-Pre-Train` is public, so no Hugging Face login is required for the
default flow.

## 3. Smoke test: download a few samples

```bash
python tools/data_prepare/bootstrap_mobile_o_pretrain_source_manifest.py \
  --filenames 00000.tar \
  --output-root /tmp/mobile_o_pretrain_smoke \
  --max-samples 8 \
  --log-every 2
```

Expected outputs:

```text
/tmp/mobile_o_pretrain_smoke/raw/images/00000/000000000.jpg
/tmp/mobile_o_pretrain_smoke/raw/images/00000/000000002.jpg
/tmp/mobile_o_pretrain_smoke/manifests/mobile_o_pretrain_source.csv
```

The script now prints progress to the terminal while it runs:

- when shard resolution starts
- when each shard starts
- periodic in-shard progress every `--log-every` paired samples
- when each shard finishes
- when the final manifest is written

## 4. Full dataset download and conversion

```bash
python tools/data_prepare/bootstrap_mobile_o_pretrain_source_manifest.py \
  --filenames all \
  --output-root data/mobile_o_pretrain_full \
  --jobs 8 \
  --log-every 1000
```

Notes:

- `--jobs` parallelizes across tar shards.
- Start with `--jobs 4` or `--jobs 8`.
- Increase only if network and disk are stable.

## 5. Encode WAN VAE latents

```bash
python tools/data_prepare/encode_laion_coyo_images_sana_ar.py \
  --manifest-csv data/mobile_o_pretrain_full/manifests/mobile_o_pretrain_source.csv \
  --output-dir data/mobile_o_pretrain_full/encoded/wan_vae_sana_ar \
  --vae-ckpt omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth
```

This writes files such as:

```text
data/mobile_o_pretrain_full/encoded/wan_vae_sana_ar/sample_00000000.pkl
data/mobile_o_pretrain_full/encoded/wan_vae_sana_ar/sample_00000001.pkl
```

## 6. Build a train-ready encoded manifest

```bash
python tools/data_prepare/build_laion_coyo_encoded_manifest.py \
  --source-manifest data/mobile_o_pretrain_full/manifests/mobile_o_pretrain_source.csv \
  --encoded-dir data/mobile_o_pretrain_full/encoded/wan_vae_sana_ar \
  --output-csv data/mobile_o_pretrain_full/manifests/mobile_o_pretrain_train_ready.csv \
  --datasets mobile_o_pretrain \
  --modality image
```

The resulting CSV can be pointed at the existing image-only training configs in
this repo the same way other image datasets are used.
