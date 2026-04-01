# Mobile-O-SFT Training Guide

This guide explains how to use the current repo to:

1. download `Amshaker/Mobile-O-SFT`
2. convert it into the repo's internal image manifest format
3. encode WAN VAE latents
4. build a train-ready manifest
5. train the current `ver 2` image-generation recipe

This path is now supported by:

- [bootstrap_mobile_o_sft_source_manifest.py](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/data_prepare/bootstrap_mobile_o_sft_source_manifest.py)
- [encode_laion_coyo_images_sana_ar.py](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/data_prepare/encode_laion_coyo_images_sana_ar.py)
- [build_laion_coyo_encoded_manifest.py](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/data_prepare/build_laion_coyo_encoded_manifest.py)
- [train_stage1_teacher_free.py](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/train_stage1_teacher_free.py)

## 1. What Mobile-O-SFT looks like

`Amshaker/Mobile-O-SFT` is stored as WebDataset tar shards on Hugging Face.

Inside each shard, samples look like:

```text
303.jpg
303.txt
45.jpg
45.txt
...
```

The new bootstrap script streams those tar shards, pairs each `jpg` with its `txt`, saves local images, and writes a source manifest CSV that the existing repo pipeline already understands.

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

`Mobile-O-SFT` is public, so no Hugging Face login is required for the default flow.

## 3. Smoke test: download a few samples

This command downloads a few samples from the smallest shard and writes:

- local raw images
- a source manifest CSV

```bash
python tools/data_prepare/bootstrap_mobile_o_sft_source_manifest.py \
  --filenames object_2.tar \
  --output-root /tmp/mobile_o_sft_smoke \
  --max-samples 8
```

Expected outputs:

```text
/tmp/mobile_o_sft_smoke/raw/images/object_2/sample_00000000.jpg
/tmp/mobile_o_sft_smoke/raw/images/object_2/sample_00000001.jpg
/tmp/mobile_o_sft_smoke/manifests/mobile_o_sft_source.csv
```

The source manifest schema is:

```csv
sample_idx,dataset,modality,caption,image_path,media_path,video_path,source_id,source_url,width,height,extra_json
```

This is the exact format expected by the image encoder path in this repo.

## 4. Full dataset download and conversion

To process the full dataset:

```bash
python tools/data_prepare/bootstrap_mobile_o_sft_source_manifest.py \
  --filenames all \
  --output-root data/mobile_o_sft_full \
  --jobs 8
```

Notes:

- `--jobs` parallelizes across tar shards.
- Start with `--jobs 4` or `--jobs 8`.
- Increase only if network and disk are stable.

If you want a subset of specific shards:

```bash
python tools/data_prepare/bootstrap_mobile_o_sft_source_manifest.py \
  --filenames object_1.tar,object_2.tar,text_1.tar \
  --output-root data/mobile_o_sft_subset \
  --jobs 3
```

## 5. Do I need to resize images manually?

No.

Do not resize raw images by hand.

Use:

- [encode_laion_coyo_images_sana_ar.py](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/data_prepare/encode_laion_coyo_images_sana_ar.py)

It already does:

- read original image
- choose closest SANA 480 AR bucket
- center-crop to target aspect ratio
- resize to the bucket size
- encode with WAN VAE

## 6. Encode WAN VAE latents

Single GPU:

```bash
python tools/data_prepare/encode_laion_coyo_images_sana_ar.py \
  --manifest-csv data/mobile_o_sft_full/manifests/mobile_o_sft_source.csv \
  --output-dir data/mobile_o_sft_full/encoded/wan_vae_sana_ar \
  --vae-ckpt omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth
```

### Parallel encode

This script already supports distributed encode through `RANK/WORLD_SIZE`.

For multi-GPU encode:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  tools/data_prepare/encode_laion_coyo_images_sana_ar.py \
  --manifest-csv data/mobile_o_sft_full/manifests/mobile_o_sft_source.csv \
  --output-dir data/mobile_o_sft_full/encoded/wan_vae_sana_ar \
  --vae-ckpt omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth
```

For 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  tools/data_prepare/encode_laion_coyo_images_sana_ar.py \
  --manifest-csv data/mobile_o_sft_full/manifests/mobile_o_sft_source.csv \
  --output-dir data/mobile_o_sft_full/encoded/wan_vae_sana_ar \
  --vae-ckpt omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth
```

This was verified on a small real sample set with `2` GPUs.

Expected outputs:

```text
data/mobile_o_sft_full/encoded/wan_vae_sana_ar/sample_00000000.pkl
data/mobile_o_sft_full/encoded/wan_vae_sana_ar/sample_00000001.pkl
...
```

For image rows, the latent contract is:

```text
latent_feature.shape = [16, 1, H, W]
frame_num = 1
```

## 7. Build the train-ready manifest

```bash
python tools/data_prepare/build_laion_coyo_encoded_manifest.py \
  --source-manifest data/mobile_o_sft_full/manifests/mobile_o_sft_source.csv \
  --encoded-dir data/mobile_o_sft_full/encoded/wan_vae_sana_ar \
  --output-csv data/mobile_o_sft_full/manifests/mobile_o_sft_train_ready.csv \
  --datasets mobile_o_sft \
  --modality image
```

This creates the normalized training CSV:

```csv
video,caption,preprocessed_path,video_path,dataset,modality,sample_idx
```

That CSV is accepted directly by the current training code.

## 8. Train the current ver2 image recipe

Start from the known-good config:

```bash
cp \
  configs/stage1_teacher_free_laion_coyo_clean100_image_bridge_fulldit_lexical_gated_k2_nodistill_bs4_1gpu_20260330.yaml \
  configs/mobile_o_sft_ver2_bs4_1gpu.yaml
```

Patch it:

```bash
python - <<'PY'
from pathlib import Path
import yaml

cfg_path = Path("configs/mobile_o_sft_ver2_bs4_1gpu.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

manifest = "data/mobile_o_sft_full/manifests/mobile_o_sft_train_ready.csv"

cfg["run"]["output_dir"] = "output/mobile_o_sft_ver2_bs4_1gpu"
cfg["data"]["openvid"]["csv_path"] = manifest
cfg["data"]["openvid"]["csv_path_video"] = manifest
cfg["data"]["openvid"]["csv_path_image"] = manifest
cfg["data"]["openvid"]["max_samples"] = None
cfg["data"]["batching"]["batch_size"] = 4
cfg["data"]["batching"]["batch_size_image"] = 4
cfg["train"]["total_steps"] = 50000
cfg["run"]["save_every_steps"] = 5000

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(cfg_path)
PY
```

Train:

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/train_stage1_teacher_free.py \
  --config configs/mobile_o_sft_ver2_bs4_1gpu.yaml \
  --max-gpus 1
```

## 9. Quick smoke train

If you only want to verify the pipeline, run a tiny dataset first:

```bash
python tools/data_prepare/bootstrap_mobile_o_sft_source_manifest.py \
  --filenames object_2.tar \
  --output-root /tmp/mobile_o_sft_smoke \
  --max-samples 4
```

```bash
python tools/data_prepare/encode_laion_coyo_images_sana_ar.py \
  --manifest-csv /tmp/mobile_o_sft_smoke/manifests/mobile_o_sft_source.csv \
  --output-dir /tmp/mobile_o_sft_smoke/encoded/wan_vae_sana_ar \
  --vae-ckpt omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth \
  --max-samples 4 \
  --log-every 1
```

```bash
python tools/data_prepare/build_laion_coyo_encoded_manifest.py \
  --source-manifest /tmp/mobile_o_sft_smoke/manifests/mobile_o_sft_source.csv \
  --encoded-dir /tmp/mobile_o_sft_smoke/encoded/wan_vae_sana_ar \
  --output-csv /tmp/mobile_o_sft_smoke/manifests/mobile_o_sft_train_ready.csv \
  --datasets mobile_o_sft \
  --modality image
```

Then point a tiny copied config at `/tmp/mobile_o_sft_smoke/manifests/mobile_o_sft_train_ready.csv` and train for `2-10` steps.

## 10. What is already supported vs not supported

Supported now:

- download `Mobile-O-SFT`
- process a few samples or the full dataset
- parallel shard ingestion with `--jobs`
- parallel VAE encode with `torchrun`
- training with the current repo recipe

Not added here:

- direct training from tar shards without local materialization
- automatic parallel training launcher generation
- caption filtering or dataset-quality filtering beyond what `Mobile-O-SFT` already provides

## 11. Practical recommendation

For first full runs:

1. bootstrap with `--jobs 4` or `--jobs 8`
2. encode with `torchrun --nproc_per_node=2` or `4`
3. train on a small subset first
4. only then lift `max_samples` / run full data
