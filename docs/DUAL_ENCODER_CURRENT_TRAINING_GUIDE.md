# Current Dual-Encoder Training Guide

This guide explains how to run the current dual-encoder recipe in this repo using your own video + image data.

This guide is for the current design family where:

- native SANA text encoder is the main text branch
- `SmolVLM2 -> ver2 bridge` is the auxiliary text branch
- fusion happens inside SANA cross-attention
- training uses joint video + image data

Real reference files in this repo:

- config: `/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/configs/stage1_teacher_free_joint_openvid_current_laion_coyo_dualtext_ver2_auxonly_gate005_bs1_1gpu_20260331.yaml`
- launcher: `/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/scripts/train_joint_openvid_current_laion_coyo_dualtext_ver2_auxonly_gate005_bs1_gpu2_20260331.sh`
- trainer: `/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/train_stage1_teacher_free.py`
- dataset loader: `/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/nets/omni/datasets/openvid_dataset.py`

## 1. Environment

Run everything from repo root and use the training env:

```bash
cd /share_4/users/duy/project/unified_video/Omni-Video-smolvlm2
source /share_0/conda/etc/profile.d/conda.sh
conda activate mobileov
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

## 2. What this recipe trains

The current dual-encoder recipe trains:

- the Smol `ver 2` bridge
- the auxiliary text-injection path inside DiT:
  - `image_embedder`
  - `image_kv_linear`
  - `image_k_norm`
  - `image_gate`

It does not train the whole native SANA text path.

The loss is diffusion / flow-matching only.

## 3. What the trainer expects

The trainer uses two dataloaders when joint mode is enabled:

- one video dataloader
- one image dataloader

The current tested scheduling is:

- `interval = 5`
- meaning `5 video steps : 1 image step`

Both image rows and video rows are still represented by OpenVid-style CSV rows that point to preprocessed latent pickles.

## 4. The manifest files you should prepare

Prepare these files:

1. `video_train_ready.csv`
2. `image_train_ready.csv`
3. optionally `joint_all.csv`

The config uses:

- `data.openvid.csv_path_video`
- `data.openvid.csv_path_image`
- `data.openvid.csv_path`

If `csv_path_video` and `csv_path_image` are both set correctly, joint training works cleanly.

## 5. Recommended folder layout

```text
my_joint_data/
  videos/
    clip_000001.mp4
    clip_000002.mp4
    ...
  images/
    img_000001.jpg
    img_000002.jpg
    ...
  encoded/
    videos/
      clip_000001_features.pkl
      clip_000002_features.pkl
      ...
    images/
      sample_00000000.pkl
      sample_00000001.pkl
      ...
  manifests/
    video_source.csv
    image_source.csv
    video_train_ready.csv
    image_train_ready.csv
    joint_all.csv
```

## 6. Image-side contract

The image side is the same as the image-only `ver 2` recipe.

Raw images can be any reasonable size.
Do not resize them manually.

Use the repo image encoder:

- `/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/data_prepare/encode_laion_coyo_images_sana_ar.py`

The resulting image pickle should behave like:

```text
latent_feature: [16, 1, H, W]
frame_num: 1
modality: image
```

A normalized image manifest row should look like:

```csv
video,caption,preprocessed_path,video_path,dataset,modality,sample_idx
img_000001,"a red bicycle leaning against a wall",/abs/path/to/sample_00000000.pkl,/abs/path/to/img_000001.jpg,my_images,image,0
```

## 7. Video-side contract

For the current SANA-video path, the safest canonical setup is:

- raw frame count: `81`
- target size: `480,832`
- latent temporal length after VAE: typically `21`

So a canonical video pickle usually behaves like:

```text
latent_feature: [16, 21, 60, 104]
frame_num: 81
modality: video
```

Raw video resolution can vary.
You do not need to resize videos manually if you preprocess them with the repo video encoder using `81` frames and target size `480,832`.

## 8. Prepare the image side

Create `my_joint_data/manifests/image_source.csv`:

```csv
sample_idx,dataset,modality,caption,image_path
0,my_images,image,"a red bicycle leaning against a wall",/abs/path/to/my_joint_data/images/img_000001.jpg
1,my_images,image,"a bowl of ramen on a wooden table",/abs/path/to/my_joint_data/images/img_000002.jpg
```

Encode images:

```bash
PYTHONPATH=. python tools/data_prepare/encode_laion_coyo_images_sana_ar.py \
  --manifest-csv my_joint_data/manifests/image_source.csv \
  --output-dir my_joint_data/encoded/images \
  --vae-ckpt omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth
```

Build the normalized image manifest:

```bash
PYTHONPATH=. python tools/data_prepare/build_laion_coyo_encoded_manifest.py \
  --source-manifest my_joint_data/manifests/image_source.csv \
  --encoded-dir my_joint_data/encoded/images \
  --output-csv my_joint_data/manifests/image_train_ready.csv \
  --datasets my_images \
  --modality image
```

## 9. Prepare the video side

Create `my_joint_data/manifests/video_source.csv` like this:

```csv
video,caption,video_path,dataset,modality,sample_idx
clip_000001.mp4,"a golden retriever running through shallow water at sunset",/abs/path/to/my_joint_data/videos/clip_000001.mp4,my_videos,video,0
clip_000002.mp4,"a cyclist riding through neon city streets at night",/abs/path/to/my_joint_data/videos/clip_000002.mp4,my_videos,video,1
```

### Recommended video encoder command

Use the repo-local script that lets you choose the output directory directly:

```bash
PYTHONPATH=. python tools/data_prepare/extract_openvid_features.py \
  --csv_path my_joint_data/manifests/video_source.csv \
  --video_dir my_joint_data/videos \
  --output_dir my_joint_data/encoded/videos \
  --ckpt_dir omni_ckpts/wan \
  --frame_num 81 \
  --target_size 480,832 \
  --sampling_rate 1 \
  --skip_num 0
```

### Alternative encoder command

If you prefer the repo-local `openvid_dataops` module, it also works, but note that `--output-subdir` is written under `download_data/data/openvid/encoded/`:

```bash
PYTHONPATH=download_data python -m openvid_dataops encode \
  --manifest-csv my_joint_data/manifests/video_source.csv \
  --ckpt-dir checkpoints/wan/wanxiang1_3b \
  --task t2v-1.3B \
  --frame-num 81 \
  --sampling-rate 1 \
  --target-size 480,832 \
  --output-subdir my_joint_videos
```

With this alternative path, encoded pickles land under:

```text
download_data/data/openvid/encoded/my_joint_videos/
```

## 10. Build the normalized video manifest

Use this inline command to convert `video_source.csv` plus encoded pickles into the normalized CSV the trainer expects:

```bash
python - <<'PY'
from pathlib import Path
import os
import pandas as pd

src = Path('my_joint_data/manifests/video_source.csv')
out = Path('my_joint_data/manifests/video_train_ready.csv')
enc = Path('my_joint_data/encoded/videos')

df = pd.read_csv(src)
rows = []
for i, row in df.iterrows():
    video = str(row['video']).strip()
    caption = str(row['caption']).strip()
    sample_idx = int(row['sample_idx']) if 'sample_idx' in row and pd.notna(row['sample_idx']) else int(i)
    stem = os.path.splitext(os.path.basename(video))[0]
    pkl = enc / f'{stem}_features.pkl'
    if not pkl.exists():
        raise FileNotFoundError(pkl)
    rows.append({
        'video': video,
        'caption': caption,
        'preprocessed_path': str(pkl.resolve()),
        'video_path': str(row.get('video_path', '') or ''),
        'dataset': str(row.get('dataset', 'my_videos')),
        'modality': 'video',
        'sample_idx': sample_idx,
    })

pd.DataFrame(rows).to_csv(out, index=False)
print(f'wrote {out} with {len(rows)} rows')
PY
```

## 11. Optional combined CSV

The current trainer mainly uses `csv_path_video` and `csv_path_image`, but a combined CSV is still useful for bookkeeping:

```bash
python - <<'PY'
import pandas as pd
v = pd.read_csv('my_joint_data/manifests/video_train_ready.csv')
i = pd.read_csv('my_joint_data/manifests/image_train_ready.csv')
out = 'my_joint_data/manifests/joint_all.csv'
pd.concat([v, i], ignore_index=True).to_csv(out, index=False)
print(f'wrote {out} with {len(v) + len(i)} rows')
PY
```

## 12. Copy a known-good config

```bash
cp \
  configs/stage1_teacher_free_joint_openvid_current_laion_coyo_dualtext_ver2_auxonly_gate005_bs1_1gpu_20260331.yaml \
  configs/my_dual_encoder_joint_bs1_1gpu.yaml
```

Then patch the copied config in one shot:

```bash
python - <<'PY'
from pathlib import Path
import yaml

cfg_path = Path("configs/my_dual_encoder_joint_bs1_1gpu.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

cfg["run"]["output_dir"] = "output/my_dual_encoder_joint_run"
cfg["data"]["openvid"]["csv_path"] = "my_joint_data/manifests/video_train_ready.csv"
cfg["data"]["openvid"]["csv_path_video"] = "my_joint_data/manifests/video_train_ready.csv"
cfg["data"]["openvid"]["csv_path_image"] = "my_joint_data/manifests/image_train_ready.csv"
cfg["data"]["openvid"]["video_dir"] = "."
cfg["train"]["total_steps"] = 30000
cfg["run"]["save_every_steps"] = 5000
cfg["data"]["batching"]["batch_size"] = 1
cfg["data"]["joint"]["enabled"] = True
cfg["data"]["joint"]["interval"] = 5

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"patched {cfg_path}")
PY
```

Edit these fields in `configs/my_dual_encoder_joint_bs1_1gpu.yaml`:

- `run.output_dir`
- `data.openvid.csv_path`
- `data.openvid.csv_path_video`
- `data.openvid.csv_path_image`
- `data.openvid.video_dir`
- optionally `data.openvid.max_samples`
- optionally `run.save_every_steps`
- optionally `train.total_steps`
- optionally `model.dual_text.image_gate_init`
- optionally `data.joint.interval`
- optionally `data.batching.batch_size`

For the current design, the key joint values are:

- `data.joint.enabled: true`
- `data.joint.interval: 5`
- `data.joint.video_modality: video`
- `data.joint.image_modality: image`

## 13. One command to train

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python tools/train_stage1_teacher_free.py \
  --config configs/my_dual_encoder_joint_bs1_1gpu.yaml \
  --max-gpus 1
```

## 14. Launcher template

```bash
cat > scripts/train_my_dual_encoder_joint.sh <<'SH'
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
  --config configs/my_dual_encoder_joint_bs1_1gpu.yaml \
  --max-gpus 1 \
  2>&1 | tee output/logs/train_my_dual_encoder_joint.log
SH
chmod +x scripts/train_my_dual_encoder_joint.sh
bash scripts/train_my_dual_encoder_joint.sh
```

## 15. Preflight checklist

Video side:

- every row in `video_train_ready.csv` has `modality=video`
- every row points to a real `.pkl`
- every video pickle has `frame_num=81`
- every video pickle has `latent_feature` with temporal length matching 81-frame preprocessing

Image side:

- every row in `image_train_ready.csv` has `modality=image`
- every row points to a real `.pkl`
- every image pickle has `frame_num=1`
- every image pickle has `latent_feature = [16, 1, H, W]`

Config side:

- `csv_path_video` points to the video CSV
- `csv_path_image` points to the image CSV
- `joint.enabled = true`
- `joint.interval > 0`
- `batch_size = 1` if you want to mirror the current tested run exactly

## 16. Minimal end-to-end example

```bash
cd /share_4/users/duy/project/unified_video/Omni-Video-smolvlm2
source /share_0/conda/etc/profile.d/conda.sh
conda activate mobileov
export PYTHONPATH=$PWD:${PYTHONPATH:-}

PYTHONPATH=. python tools/data_prepare/encode_laion_coyo_images_sana_ar.py \
  --manifest-csv my_joint_data/manifests/image_source.csv \
  --output-dir my_joint_data/encoded/images \
  --vae-ckpt omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth

PYTHONPATH=. python tools/data_prepare/build_laion_coyo_encoded_manifest.py \
  --source-manifest my_joint_data/manifests/image_source.csv \
  --encoded-dir my_joint_data/encoded/images \
  --output-csv my_joint_data/manifests/image_train_ready.csv \
  --datasets my_images \
  --modality image

PYTHONPATH=. python tools/data_prepare/extract_openvid_features.py \
  --csv_path my_joint_data/manifests/video_source.csv \
  --video_dir my_joint_data/videos \
  --output_dir my_joint_data/encoded/videos \
  --ckpt_dir omni_ckpts/wan \
  --frame_num 81 \
  --target_size 480,832 \
  --sampling_rate 1 \
  --skip_num 0

python - <<'PY'
from pathlib import Path
import os
import pandas as pd
src = Path('my_joint_data/manifests/video_source.csv')
out = Path('my_joint_data/manifests/video_train_ready.csv')
enc = Path('my_joint_data/encoded/videos')
df = pd.read_csv(src)
rows = []
for i, row in df.iterrows():
    video = str(row['video']).strip()
    stem = os.path.splitext(os.path.basename(video))[0]
    pkl = enc / f'{stem}_features.pkl'
    rows.append({
        'video': video,
        'caption': str(row['caption']).strip(),
        'preprocessed_path': str(pkl.resolve()),
        'video_path': str(row.get('video_path', '') or ''),
        'dataset': str(row.get('dataset', 'my_videos')),
        'modality': 'video',
        'sample_idx': int(row.get('sample_idx', i)),
    })
pd.DataFrame(rows).to_csv(out, index=False)
print(f'wrote {out} with {len(rows)} rows')
PY

cp \
  configs/stage1_teacher_free_joint_openvid_current_laion_coyo_dualtext_ver2_auxonly_gate005_bs1_1gpu_20260331.yaml \
  configs/my_dual_encoder_joint_bs1_1gpu.yaml

python - <<'PY'
from pathlib import Path
import yaml

cfg_path = Path("configs/my_dual_encoder_joint_bs1_1gpu.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

cfg["run"]["output_dir"] = "output/my_dual_encoder_joint_run"
cfg["data"]["openvid"]["csv_path"] = "my_joint_data/manifests/video_train_ready.csv"
cfg["data"]["openvid"]["csv_path_video"] = "my_joint_data/manifests/video_train_ready.csv"
cfg["data"]["openvid"]["csv_path_image"] = "my_joint_data/manifests/image_train_ready.csv"
cfg["data"]["openvid"]["video_dir"] = "."
cfg["train"]["total_steps"] = 30000
cfg["run"]["save_every_steps"] = 5000
cfg["data"]["batching"]["batch_size"] = 1
cfg["data"]["joint"]["enabled"] = True
cfg["data"]["joint"]["interval"] = 5

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"patched {cfg_path}")
PY

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python tools/train_stage1_teacher_free.py \
  --config configs/my_dual_encoder_joint_bs1_1gpu.yaml \
  --max-gpus 1
```
