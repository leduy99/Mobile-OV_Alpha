# Image-Gen Ver2 Training Guide

This note explains how to prepare a custom **image-only** dataset so it can be
used directly by the current `ver 2` training path in this repo.

In this guide, `ver 2` means the current image-generation bridge recipe that
uses:

- `SmolVLM2 -> SANA bridge`
- projector type `mcp_lexical_gated`
- `mcp_num_fuse_layers: 2`
- `full DiT` training
- `no-distill`

This is the same family as:

- [stage1_teacher_free_laion_coyo_clean100_image_bridge_fulldit_lexical_gated_k2_nodistill_bs4_1gpu_20260330.yaml](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/configs/stage1_teacher_free_laion_coyo_clean100_image_bridge_fulldit_lexical_gated_k2_nodistill_bs4_1gpu_20260330.yaml)

## 1. What the Current Trainer Actually Expects

For the current image-generation path, the trainer does **not** want raw JPGs
at train time.

It expects each row to already have a WAN VAE latent pickle, and the training
loop reads:

- `latent_feature`
- `prompt`
- `frame_num`
- optionally `img_hw`
- optionally `aspect_ratio`

Relevant code:

- [openvid_dataset.py](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/nets/omni/datasets/openvid_dataset.py)
- [train_stage1_teacher_free.py](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/train_stage1_teacher_free.py)

Important consequence:

- if you want to use the current training path as-is, you should **pre-encode**
  your images into WAN VAE latent `.pkl` files first
- do **not** rely on raw-mode image loading for this path

## 2. Recommended Folder Layout

Use a dataset root like this:

```text
data/my_imagegen_dataset/
  raw/
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
    source.csv
    train_ready.csv
```

Recommended meaning:

- `raw/images/`: your original images
- `encoded/wan_vae_sana_ar/`: WAN VAE latent pickles
- `manifests/source.csv`: your initial image manifest
- `manifests/train_ready.csv`: final manifest used by training

## 3. Best Practice: Two Manifest Stages

The cleanest path is:

1. create a **source manifest**
2. encode images into `sample_XXXXXXXX.pkl`
3. build a **train-ready manifest** with explicit `preprocessed_path`

This matches the repo's current LAION / COYO pipeline.

## 4. Source Manifest Format

The source manifest is what the encoder reads.

Minimum columns I recommend:

- `sample_idx`
- `dataset`
- `modality`
- `caption`
- one of:
  - `image_path`
  - `media_path`
  - `video_path`

Recommended schema:

```csv
sample_idx,dataset,modality,caption,image_path
0,my_dataset,image,"a red bicycle leaning against a wall",data/my_imagegen_dataset/raw/images/img_000001.jpg
1,my_dataset,image,"a bowl of ramen on a wooden table",data/my_imagegen_dataset/raw/images/img_000002.jpg
```

Notes:

- `sample_idx` should be unique and stable
- `modality` should be `image`
- `dataset` can be any short name, for example `my_dataset`
- `caption` must be non-empty
- `image_path` should point to a real image file

## 5. Encode Images Into WAN VAE Latents

Use the existing repo encoder:

- [encode_laion_coyo_images_sana_ar.py](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/data_prepare/encode_laion_coyo_images_sana_ar.py)

Example:

```bash
PYTHONPATH=. python tools/data_prepare/encode_laion_coyo_images_sana_ar.py \
  --manifest-csv data/my_imagegen_dataset/manifests/source.csv \
  --output-dir data/my_imagegen_dataset/encoded/wan_vae_sana_ar \
  --vae-ckpt omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth
```

What this script does:

1. reads `image` rows from your manifest
2. picks the nearest SANA 480 aspect-ratio bucket
3. center-crops and resizes the image
4. encodes it with WAN VAE
5. writes:

```text
data/my_imagegen_dataset/encoded/wan_vae_sana_ar/sample_00000000.pkl
data/my_imagegen_dataset/encoded/wan_vae_sana_ar/sample_00000001.pkl
...
```

## 6. Train-Ready Manifest Format

After encoding, the trainer works best with a normalized manifest like:

```csv
video,caption,preprocessed_path,video_path,dataset,modality,sample_idx
img_000001,"a red bicycle leaning against a wall",/abs/path/to/sample_00000000.pkl,/abs/path/to/img_000001.jpg,my_dataset,image,0
img_000002,"a bowl of ramen on a wooden table",/abs/path/to/sample_00000001.pkl,/abs/path/to/img_000002.jpg,my_dataset,image,1
```

This is the exact schema produced by:

- [build_laion_coyo_encoded_manifest.py](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/data_prepare/build_laion_coyo_encoded_manifest.py)

Recommended command:

```bash
PYTHONPATH=. python tools/data_prepare/build_laion_coyo_encoded_manifest.py \
  --source-manifest data/my_imagegen_dataset/manifests/source.csv \
  --encoded-dir data/my_imagegen_dataset/encoded/wan_vae_sana_ar \
  --output-csv data/my_imagegen_dataset/manifests/train_ready.csv \
  --datasets my_dataset \
  --modality image
```

## 7. Minimal CSV Contract the Current Loader Supports

The dataset loader supports two useful direct-preprocessed schemas:

### Option A: recommended

- `caption`
- `preprocessed_path`

### Option B: also supported

- `caption`
- `sample_idx`
- and set `data.openvid.preprocessed_dir` in config

But for practical training, I recommend the full normalized schema:

- `video`
- `caption`
- `preprocessed_path`
- `video_path`
- `dataset`
- `modality`
- `sample_idx`

Why:

- easier debugging
- easier filtering
- easier provenance tracking
- matches current mixed-manifest tooling

## 8. Latent Pickle Contract

The latent `.pkl` file is the most important part.

From a real working image latent in this repo, the pickle looks like:

- `latent_feature`: `torch.Tensor` with shape `(16, 1, H, W)`
- `prompt`: `str`
- `frame_num`: `int`, must be `1` for image rows
- `sample_idx`: `int`
- `modality`: `image`
- `img_hw`: tensor of shape `(2,)`
- `aspect_ratio`: scalar tensor

Example real keys seen in a working file:

- `sample_idx`
- `dataset`
- `modality`
- `video`
- `video_path`
- `image_path`
- `media_path`
- `prompt`
- `frame_num`
- `target_size`
- `aspect_ratio`
- `img_hw`
- `closest_ratio`
- `latent_feature`

### Minimum safe fields for the current image config

If you want the current `ver 2` config to work without edits, your pickle should
contain at least:

- `latent_feature`
- `prompt`
- `frame_num`

And I strongly recommend also including:

- `sample_idx`
- `modality`
- `img_hw`
- `aspect_ratio`

### Exact shape expectations

For image-only training:

- `latent_feature` should be shaped:

```text
[C, T, H, W] = [16, 1, H, W]
```

- `T` must be `1`
- `frame_num` must be `1`

This matches the current config:

- `data.openvid.expected_latent_t: 1`
- `data.openvid.expected_frame_num: 1`

## 9. Spatial Size and Aspect Ratio

For image rows, `H` and `W` in latent space do **not** have to be identical for
every sample, but they should come from a **single consistent encoding
pipeline**.

The repo's recommended encoder uses SANA 480 aspect-ratio buckets.

That is the safest choice because it:

- matches the current SANA setup
- writes `img_hw` and `aspect_ratio`
- keeps training/inference parity cleaner

I do **not** recommend mixing arbitrary latent shapes produced by unrelated
encoders.

## 10. What to Put in the Config

For a custom image-only dataset, the easiest approach is:

1. copy the current `ver 2` config
2. change only the manifest paths, output dir, and maybe batch size / steps

Base config to copy:

- [stage1_teacher_free_laion_coyo_clean100_image_bridge_fulldit_lexical_gated_k2_nodistill_bs4_1gpu_20260330.yaml](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/configs/stage1_teacher_free_laion_coyo_clean100_image_bridge_fulldit_lexical_gated_k2_nodistill_bs4_1gpu_20260330.yaml)

The dataset section should look like:

```yaml
data:
  openvid:
    csv_path: data/my_imagegen_dataset/manifests/train_ready.csv
    csv_path_video: data/my_imagegen_dataset/manifests/train_ready.csv
    csv_path_image: data/my_imagegen_dataset/manifests/train_ready.csv
    video_dir: .
    preprocessed_dir: null
    use_preprocessed: true
    max_samples: null
    expected_latent_t: 1
    expected_frame_num: 1
  joint:
    enabled: false
```

Why `preprocessed_dir: null` works here:

- because `train_ready.csv` already contains `preprocessed_path`

## 11. If You Already Have Your Own `.pkl` Files

If your dataset is already encoded and you do not want to rerun the encoder,
that is fine.

Just make sure:

1. each pickle contains a compatible `latent_feature`
2. each pickle has `frame_num = 1`
3. your CSV points to it through `preprocessed_path`

In that case, the minimal train-ready manifest can be:

```csv
caption,preprocessed_path,modality,sample_idx
"a red bicycle leaning against a wall",/abs/path/to/sample_00000000.pkl,image,0
"a bowl of ramen on a wooden table",/abs/path/to/sample_00000001.pkl,image,1
```

But again, I still recommend the fuller schema.

## 12. Quick Sanity Check Before Training

Before launching a long run, check one row manually:

```bash
python - <<'PY'
import pandas as pd, pickle
from pathlib import Path

manifest = Path("data/my_imagegen_dataset/manifests/train_ready.csv")
df = pd.read_csv(manifest)
row = df.iloc[0]
print(df.columns.tolist())
print(row.to_dict())

with open(row["preprocessed_path"], "rb") as f:
    item = pickle.load(f)

print(type(item))
print(item.keys())
print(item["latent_feature"].shape)
print(item["frame_num"])
print(item.get("prompt", ""))
PY
```

For image-gen `ver 2`, the important outputs should look like:

- `latent_feature.shape -> (16, 1, H, W)`
- `frame_num -> 1`
- `prompt` present or at least `caption` present in the CSV

## 13. Minimal End-to-End Example

If you already have a folder of images and captions, the shortest safe path is:

```bash
# 1) Build source.csv with:
# sample_idx,dataset,modality,caption,image_path

# 2) Encode images
PYTHONPATH=. python tools/data_prepare/encode_laion_coyo_images_sana_ar.py \
  --manifest-csv data/my_imagegen_dataset/manifests/source.csv \
  --output-dir data/my_imagegen_dataset/encoded/wan_vae_sana_ar \
  --vae-ckpt omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth

# 3) Build train-ready manifest
PYTHONPATH=. python tools/data_prepare/build_laion_coyo_encoded_manifest.py \
  --source-manifest data/my_imagegen_dataset/manifests/source.csv \
  --encoded-dir data/my_imagegen_dataset/encoded/wan_vae_sana_ar \
  --output-csv data/my_imagegen_dataset/manifests/train_ready.csv \
  --datasets my_dataset \
  --modality image

# 4) Copy the current ver2 config and replace csv_path/output_dir
```

## 14. Practical Recommendation

If the goal is to train the current `ver 2` recipe on a new image dataset with
minimum risk, do this:

1. build a `source.csv`
2. run the existing SANA-AR image encoder
3. build a normalized `train_ready.csv`
4. copy the current `ver 2` config and only edit paths + training budget

That path is the closest to what is already working in this repo.
