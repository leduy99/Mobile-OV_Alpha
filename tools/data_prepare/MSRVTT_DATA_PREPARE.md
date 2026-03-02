# MSR-VTT Data Prepare (Download + WAN VAE Encode)

Script:
- `tools/data_prepare/msrvtt_data_prepare.py`

## Auth (HuggingFace)

Do not commit raw tokens into git.

Recommended:

```bash
hf auth login --token <YOUR_HF_TOKEN>
# or
export HF_TOKEN=<YOUR_HF_TOKEN>
```

Current local token note (masked): `hf_NRZK...TTvGt`

## 1) Download + extract raw MSR-VTT files

```bash
python tools/data_prepare/msrvtt_data_prepare.py download \
  --root-dir data/msrvtt \
  --repo-id AlexZigma/msr-vtt \
  --extract
```

This downloads:
- `test_videos.zip`
- `test_videodatainfo.json.zip`
- `train.parquet`
- `val.parquet`

into `data/msrvtt/raw/`, then extracts to:
- videos: `data/msrvtt/videos/TestVideo/*.mp4`
- metadata: `data/msrvtt/metadata/*.json` (or fallback from `raw/`)

## 2) Build OpenVid-style CSV (`video,caption`)

```bash
python tools/data_prepare/msrvtt_data_prepare.py build-csv \
  --root-dir data/msrvtt \
  --manifest-name OpenVid_extracted_subset_unique.csv \
  --caption-policy longest \
  --captions-per-video 1
```

Output:
- `data/msrvtt/OpenVid_extracted_subset_unique.csv`

## 3) Encode WAN VAE (single GPU)

```bash
python tools/data_prepare/msrvtt_data_prepare.py encode \
  --root-dir data/msrvtt \
  --manifest-name OpenVid_extracted_subset_unique.csv \
  --ckpt-dir omni_ckpts/wan/wanxiang1_3b \
  --task t2v-1.3B \
  --frame-num 81 \
  --sampling-rate 1 \
  --target-size 480,832
```

Output:
- `data/msrvtt/preprocessed/*_features.pkl`

Each pickle contains:
- `latent_feature`
- `prompt`
- `video_path`
- `frame_num`

## 4) Encode WAN VAE (multi-GPU)

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

## 5) One-shot full pipeline

```bash
python tools/data_prepare/msrvtt_data_prepare.py all \
  --root-dir data/msrvtt \
  --repo-id AlexZigma/msr-vtt \
  --manifest-name OpenVid_extracted_subset_unique.csv \
  --ckpt-dir omni_ckpts/wan/wanxiang1_3b \
  --frame-num 81 \
  --sampling-rate 1 \
  --target-size 480,832
```
