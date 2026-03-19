# MobileOV + SANA Training Repo

This repository is the cleaned training and inference workspace we use for the
`SmolVLM2 -> bridge/projector -> SANA Video DiT` pipeline.

It is focused on a narrow, practical scope:
- text-to-video training and inference,
- OpenVid data preparation,
- LAION / COYO image materialization and encoding,
- MSR-VTT download + WAN-VAE encoding,
- the 3-stage Gemma-distilled training pipeline.

It is **not** intended to be a polished release of every Omni-Video research
path. Legacy understanding pipelines, HD-VILA flows, and old experiment notes
are intentionally de-emphasized or archived.

## Supported Scope

### Training
- Stage 1: prompt-only teacher distillation
- Stage 2: bridge reinjection with frozen DiT
- Stage 3: bridge + full DiT training
- student adaptation modes:
  - top-N SmolVLM2 text layers,
  - final norm on/off,
  - LoRA on/off,
  - bridge-only vs bridge+DiT

### Inference
- canonical backend: `fixed`
- student checkpoint loading with:
  - SmolVLM2 trainable text state,
  - bridge/projector state,
  - optional DiT trainable state,
  - optional LoRA state

### Data workflows in scope
- OpenVid
- LAION / COYO
- MSR-VTT

## Canonical Environment

The canonical environment name is `mobileov`.

```bash
conda env create -f environment.yml
conda activate mobileov
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate mobileov
```

For OpenVid DataOps convenience commands, install the local package once:

```bash
pip install -e download_data
```

## Model Asset Bootstrap

### Auto-bootstrap is available
These entrypoints will auto-download / auto-convert model assets when missing:
- `tools/train_stage1_teacher_free.py`
- `tools/train_q1_sana_bridge.py`
- `tools/inference/sana_video_inference_fixed.py`
- `tools/inference/test_q1_student_video.py`

This covers:
- SANA video checkpoint assets
- converted SmolVLM2 checkpoint

### Still manual
WAN VAE assets used by OpenVid / MSR-VTT latent encoding are still explicit
setup steps. See the dataset guides below.

## Quick Start

### 1. Train the 3-stage pipeline

```bash
GPUS=0,1 bash scripts/train_openvid_current_laion_coyo_3stage_gemma_distill_5v1i.sh
```

### 2. Run inference from a student checkpoint

```bash
PYTHONPATH=. python tools/inference/test_q1_student_video.py \
  --bridge-ckpt /path/to/checkpoint_stepXXXX.pt \
  --prompt "a golden retriever running along a beach at sunset" \
  --output-dir output/inference_example \
  --sana-backend fixed \
  --steps 24 \
  --cfg-scale 3.0
```

## Dataset Guides

### OpenVid
Canonical options:
- `download_data/` for OpenVid DataOps
- `tools/data_prepare/download_openvid.py` for lightweight CSV / part download

See:
- `OPENVID_TRAINING_GUIDE.md`
- `download_data/README.md`

### MSR-VTT
Use:
- `tools/data_prepare/msrvtt_data_prepare.py`

See:
- `tools/data_prepare/MSRVTT_DATA_PREPARE.md`

### LAION / COYO
Current repo support is manifest-based rather than one-shot raw crawling.
Canonical tools are:
- `tools/data_prepare/materialize_unified_manifest.py`
- `tools/data_prepare/recover_laion_images_unique.py`
- `tools/data_prepare/encode_laion_coyo_images_sana_ar.py`

## Main Entry Points

### Training
- `tools/train_q1_sana_bridge.py`
- `tools/train_stage1_teacher_free.py`
- `scripts/train_openvid_current_laion_coyo_3stage_gemma_distill_5v1i.sh`

### Inference
- `tools/inference/test_q1_student_video.py`
- `tools/inference/sana_video_inference_fixed.py`

### Data
- `tools/data_prepare/download_openvid.py`
- `tools/data_prepare/msrvtt_data_prepare.py`
- `tools/data_prepare/materialize_unified_manifest.py`
- `tools/data_prepare/recover_laion_images_unique.py`
- `tools/data_prepare/encode_laion_coyo_images_sana_ar.py`
- `download_data/`

## What To Ignore

These paths are not the current recommended public surface:
- legacy inference backend,
- archived configs and scripts,
- old paper/demo instructions in archived notes,
- HD-VILA-specific workflows.

## Repository Notes

- Top-level configs were intentionally trimmed to the most recent, representative
  training settings.
- Older experiments live under archive folders.
- Runtime outputs, checkpoints, and datasets remain git-ignored.

If you are onboarding a teammate, start with `SETUP_AND_TRAINING.md`.
