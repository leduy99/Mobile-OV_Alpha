# MSR-VTT Data Preparation

Canonical script:
- `tools/data_prepare/msrvtt_data_prepare.py`

## Environment

Use the canonical repo environment:

```bash
conda activate mobileov
```

## Hugging Face Auth

Authenticate if required by your local setup:

```bash
huggingface-cli login
# or
export HF_TOKEN=<YOUR_HF_TOKEN>
```

## 1. Download and Extract Raw MSR-VTT Files

```bash
PYTHONPATH=. python tools/data_prepare/msrvtt_data_prepare.py download \
  --root-dir data/msrvtt \
  --repo-id AlexZigma/msr-vtt \
  --extract
```

This downloads:
- `test_videos.zip`
- `test_videodatainfo.json.zip`
- `train.parquet`
- `val.parquet`

Outputs:
- videos under `data/msrvtt/videos/TestVideo/`
- metadata under `data/msrvtt/metadata/`

## 2. Build an OpenVid-style CSV

```bash
PYTHONPATH=. python tools/data_prepare/msrvtt_data_prepare.py build-csv \
  --root-dir data/msrvtt \
  --manifest-name OpenVid_extracted_subset_unique.csv \
  --caption-policy longest \
  --captions-per-video 1
```

Output:
- `data/msrvtt/OpenVid_extracted_subset_unique.csv`

## 3. Encode WAN VAE Latents (Single GPU)

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

Output:
- `data/msrvtt/preprocessed/*_features.pkl`

Each pickle contains:
- `latent_feature`
- `prompt`
- `video_path`
- `frame_num`

## 4. Encode WAN VAE Latents (Multi-GPU)

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

## 5. One-shot Pipeline

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
