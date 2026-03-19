# OpenVid Guide

This guide covers the OpenVid workflows that are still active in this repository.

## Recommended Path: OpenVid DataOps

The bundled `download_data/` package is the cleanest OpenVid workflow here.

### Install once in the canonical env

```bash
conda activate mobileov
pip install -e download_data
```

### Download WAN VAE checkpoint

```bash
python download_data/scripts/download_wan_vae_2_1.py \
  --output-dir download_data/checkpoints/wan/wanxiang1_3b
```

### Download OpenVid parts

```bash
python -m openvid_dataops download --parts "[1,2,4,5]" --extract
```

### Build manifest

```bash
python -m openvid_dataops build-manifest \
  --parts "[1,2,4,5]" \
  --output-name openvid_p1_p2_p4_p5.csv
```

### Encode WAN latents

```bash
python -m openvid_dataops encode \
  --manifest-csv download_data/data/openvid/manifests/openvid_p1_p2_p4_p5.csv \
  --ckpt-dir download_data/checkpoints/wan/wanxiang1_3b \
  --task t2v-1.3B \
  --frame-num 81 \
  --sampling-rate 1 \
  --target-size 480,832 \
  --output-subdir wan_vae_p1_p2_p4_p5
```

## Lightweight Alternative

If you only need a quick CSV / part downloader, use:

```bash
PYTHONPATH=. python tools/data_prepare/download_openvid.py \
  --output_dir data/openvid \
  --num_parts 2
```

Notes:
- `--csv_only` downloads only the main CSV.
- `--num_parts` is optional; omit it to target all parts visible from the HF index.

## Using OpenVid in Training

The current stage-2 / stage-3 training configs expect manifest CSVs and encoded
latents to already exist.

The canonical 3-stage training flow is:
1. prepare OpenVid data,
2. prepare LAION / COYO image rows,
3. build the mixed manifests used by the kept configs,
4. run the 3-stage launcher.

## Useful Files

- `download_data/README.md`
- `tools/data_prepare/download_openvid.py`
- `tools/data_prepare/build_joint_manifest_openvid_current_laion_coyo.py`
- `configs/stage1_prompt_teacher_distill_openvid_current_laion_coyo_5v1i_2gpu_20260318.yaml`
- `scripts/train_openvid_current_laion_coyo_3stage_gemma_distill_5v1i.sh`
