# Image Generation Ver2 Training Guide

This guide explains how to train the current `ver 2` image-generation recipe in this repo using a custom image-only dataset.

In this repo, `ver 2` means:

- student text path: `SmolVLM2 -> bridge`
- bridge projector: `mcp_lexical_gated`
- `mcp_num_fuse_layers: 2`
- `bridge + full DiT`
- `flow-matching / diffusion loss only`
- `no distill`

Real reference files in this repo:

- config: `/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/configs/stage1_teacher_free_laion_coyo_clean100_image_bridge_fulldit_lexical_gated_k2_nodistill_bs4_1gpu_20260330.yaml`
- launcher: `/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/scripts/train_laion_coyo_clean100_image_bridge_fulldit_lexical_gated_k2_nodistill_bs4_gpu2_20260330.sh`
- trainer: `/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/train_stage1_teacher_free.py`
- dataset loader: `/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/nets/omni/datasets/openvid_dataset.py`

## 1. Environment

Run everything from repo root and use the same env as the training runs in this project:

```bash
cd /share_4/users/duy/project/unified_video/Omni-Video-smolvlm2
source /share_0/conda/etc/profile.d/conda.sh
conda activate mobileov
source scripts/env_exports.sh
```

## 2. What this recipe trains

This recipe trains:

- the `ver 2` bridge
- the full SANA DiT

This recipe does not use teacher distillation losses.

## 3. What the trainer expects at train time

The trainer does not read raw JPG files directly during training.

It expects a CSV manifest that points to pre-encoded WAN VAE latent `.pkl` files.

For image rows, the important contract is:

- `frame_num = 1`
- `latent_feature.shape = [16, 1, H, W]`

`H, W` depend on the aspect-ratio bucket chosen during preprocessing.

## 4. Raw image size and resize policy

Do not resize images manually.

Use the repo image encoder:

- `/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/data_prepare/encode_laion_coyo_images_sana_ar.py`

It already does the right thing:

- reads the original image
- picks the closest SANA 480 aspect-ratio bucket
- center-crops to the bucket aspect ratio
- resizes to the bucket size
- encodes with WAN VAE

So the correct policy is:

- keep original raw images as they are
- let the repo encoder handle crop + resize

## 5. Recommended folder layout

```text
my_data/
  images/
    img_000001.jpg
    img_000002.jpg
    ...
  encoded/
    wan_vae_sana_ar/
      sample_00000000.pkl
      sample_00000001.pkl
      ...
  manifests/
    image_source.csv
    image_train_ready.csv
```

## 6. Raw source manifest format

Create `my_data/manifests/image_source.csv` like this:

```csv
sample_idx,dataset,modality,caption,image_path
0,my_images,image,"a red bicycle leaning against a wall",/abs/path/to/my_data/images/img_000001.jpg
1,my_images,image,"a bowl of ramen on a wooden table",/abs/path/to/my_data/images/img_000002.jpg
```

Practical rules:

- `sample_idx` must be unique
- `modality` must be `image`
- `caption` should be non-empty
- `image_path` must point to a real file

## 7. Encode raw images into WAN VAE latents

```bash
PYTHONPATH=. python tools/data_prepare/encode_laion_coyo_images_sana_ar.py \
  --manifest-csv my_data/manifests/image_source.csv \
  --output-dir my_data/encoded/wan_vae_sana_ar \
  --vae-ckpt omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth
```

This writes files such as:

```text
my_data/encoded/wan_vae_sana_ar/sample_00000000.pkl
my_data/encoded/wan_vae_sana_ar/sample_00000001.pkl
```

## 8. Build the train-ready CSV

Use the repo manifest builder:

- `/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/data_prepare/build_laion_coyo_encoded_manifest.py`

```bash
PYTHONPATH=. python tools/data_prepare/build_laion_coyo_encoded_manifest.py \
  --source-manifest my_data/manifests/image_source.csv \
  --encoded-dir my_data/encoded/wan_vae_sana_ar \
  --output-csv my_data/manifests/image_train_ready.csv \
  --datasets my_images \
  --modality image
```

The resulting CSV follows the normalized schema that the current trainer handles well:

```csv
video,caption,preprocessed_path,video_path,dataset,modality,sample_idx
img_000001,"a red bicycle leaning against a wall",/abs/path/to/sample_00000000.pkl,/abs/path/to/img_000001.jpg,my_images,image,0
```

## 9. Latent pickle contract

A safe image pickle should contain at least:

- `latent_feature`
- `prompt`
- `frame_num`

Recommended extra fields:

- `sample_idx`
- `dataset`
- `modality`
- `img_hw`
- `aspect_ratio`
- `target_size`

For image rows, the key requirements are:

```text
latent_feature: [16, 1, H, W]
frame_num: 1
```

## 10. Copy a known-good config

The simplest path is to copy the tested config and only change dataset paths and run metadata:

```bash
cp \
  configs/stage1_teacher_free_laion_coyo_clean100_image_bridge_fulldit_lexical_gated_k2_nodistill_bs4_1gpu_20260330.yaml \
  configs/my_image_ver2_bs4_1gpu.yaml
```

Then patch the copied config in one shot:

```bash
python - <<'PY'
from pathlib import Path
import yaml

cfg_path = Path("configs/my_image_ver2_bs4_1gpu.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

cfg["run"]["output_dir"] = "output/my_image_ver2_run"
cfg["data"]["openvid"]["csv_path"] = "my_data/manifests/image_train_ready.csv"
cfg["data"]["openvid"]["csv_path_video"] = "my_data/manifests/image_train_ready.csv"
cfg["data"]["openvid"]["csv_path_image"] = "my_data/manifests/image_train_ready.csv"
cfg["data"]["openvid"]["max_samples"] = 1000
cfg["train"]["total_steps"] = 50000
cfg["run"]["save_every_steps"] = 5000

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"patched {cfg_path}")
PY
```

Edit these fields in `configs/my_image_ver2_bs4_1gpu.yaml`:

- `run.output_dir`
- `data.openvid.csv_path`
- `data.openvid.csv_path_video`
- `data.openvid.csv_path_image`
- `data.openvid.max_samples`
- `run.save_every_steps`
- `train.total_steps`
- optionally `data.batching.batch_size`
- optionally `train.lr.bridge`
- optionally `train.lr.dit`

For image-only training, keep these values aligned with the image preprocessing contract:

- `data.openvid.expected_latent_t: 1`
- `data.openvid.expected_frame_num: 1`
- `data.joint.enabled: false`

## 11. One command to train

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python tools/train_stage1_teacher_free.py \
  --config configs/my_image_ver2_bs4_1gpu.yaml \
  --max-gpus 1
```

## 12. Launcher template

If you prefer a launcher script:

```bash
cat > scripts/train_my_image_ver2.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source /share_0/conda/etc/profile.d/conda.sh
conda activate mobileov
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
python tools/train_stage1_teacher_free.py \
  --config configs/my_image_ver2_bs4_1gpu.yaml \
  --max-gpus 1 \
  2>&1 | tee output/logs/train_my_image_ver2.log
SH
chmod +x scripts/train_my_image_ver2.sh
bash scripts/train_my_image_ver2.sh
```

## 13. Preflight checklist

Before launching, make sure all of these are true:

- `my_data/manifests/image_train_ready.csv` exists
- every row has a real `preprocessed_path`
- every pickle has `latent_feature`
- every pickle has `frame_num = 1`
- the config points to the correct CSV
- `expected_latent_t = 1`
- `expected_frame_num = 1`
- `joint.enabled = false`

## 14. Minimal end-to-end example

If your dataset is already in `my_data/images`, the minimum runnable flow is:

```bash
cd /share_4/users/duy/project/unified_video/Omni-Video-smolvlm2
source /share_0/conda/etc/profile.d/conda.sh
conda activate mobileov
export PYTHONPATH=$PWD:${PYTHONPATH:-}

PYTHONPATH=. python tools/data_prepare/encode_laion_coyo_images_sana_ar.py \
  --manifest-csv my_data/manifests/image_source.csv \
  --output-dir my_data/encoded/wan_vae_sana_ar \
  --vae-ckpt omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth

PYTHONPATH=. python tools/data_prepare/build_laion_coyo_encoded_manifest.py \
  --source-manifest my_data/manifests/image_source.csv \
  --encoded-dir my_data/encoded/wan_vae_sana_ar \
  --output-csv my_data/manifests/image_train_ready.csv \
  --datasets my_images \
  --modality image

cp \
  configs/stage1_teacher_free_laion_coyo_clean100_image_bridge_fulldit_lexical_gated_k2_nodistill_bs4_1gpu_20260330.yaml \
  configs/my_image_ver2_bs4_1gpu.yaml

python - <<'PY'
from pathlib import Path
import yaml

cfg_path = Path("configs/my_image_ver2_bs4_1gpu.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

cfg["run"]["output_dir"] = "output/my_image_ver2_run"
cfg["data"]["openvid"]["csv_path"] = "my_data/manifests/image_train_ready.csv"
cfg["data"]["openvid"]["csv_path_video"] = "my_data/manifests/image_train_ready.csv"
cfg["data"]["openvid"]["csv_path_image"] = "my_data/manifests/image_train_ready.csv"
cfg["data"]["openvid"]["max_samples"] = 1000
cfg["train"]["total_steps"] = 50000
cfg["run"]["save_every_steps"] = 5000

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"patched {cfg_path}")
PY

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python tools/train_stage1_teacher_free.py \
  --config configs/my_image_ver2_bs4_1gpu.yaml \
  --max-gpus 1
```
