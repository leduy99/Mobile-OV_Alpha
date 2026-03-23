# Setup and Training

This guide is the canonical setup document for the cleaned training / inference
scope of this repository.

If you want one file that covers the full onboarding path from repo setup to
dataset preparation to training to inference, start with:
- `docs/END_TO_END_GUIDE.md`

## Scope

The supported workflows in this guide are:
- 3-stage SmolVLM2 + bridge + SANA training
- checkpoint-based video inference with the fixed backend
- OpenVid data preparation
- LAION / COYO manifest materialization and image encoding
- MSR-VTT download and WAN-VAE encoding

The following are intentionally out of scope here:
- HD-VILA pipelines
- legacy paper/demo instructions
- deprecated inference backends

## 1. Prerequisites

Recommended system tools:
- `conda`
- `ffmpeg`
- `wget`
- `git`

Optional but useful:
- `yt-dlp` for generic video materialization paths in unified manifests

You will also need Hugging Face access for:
- SANA checkpoint assets
- SmolVLM2 conversion source model
- OpenVid dataset files
- MSR-VTT dataset files

Authenticate once if needed:

```bash
huggingface-cli login
```

## 2. Create the Canonical Environment

```bash
conda env create -f environment.yml
conda activate mobileov
```

If the env already exists:

```bash
conda env update -f environment.yml --prune
conda activate mobileov
```

For OpenVid DataOps commands, install the local package once:

```bash
pip install -e download_data
```

## 3. Model Asset Bootstrap

### Auto-bootstrap
The following entrypoints will automatically download or convert model assets if
they are missing locally:
- `tools/train_q1_sana_bridge.py`
- `tools/train_stage1_teacher_free.py`
- `tools/inference/sana_video_inference_fixed.py`
- `tools/inference/test_q1_student_video.py`

This auto-bootstrap covers:
- SANA video checkpoint files
- SmolVLM2 converted checkpoint

### Manual assets
WAN VAE checkpoints used for OpenVid / MSR-VTT latent encoding are still manual.
For OpenVid DataOps, use:

```bash
python download_data/scripts/download_wan_vae_2_1.py \
  --output-dir download_data/checkpoints/wan/wanxiang1_3b
```

## 4. Canonical Training Commands

### Full 3-stage pipeline

```bash
GPUS=0,1 bash scripts/train_openvid_current_laion_coyo_3stage_gemma_distill_5v1i.sh
```

This launcher runs:
1. Stage 1 prompt-only teacher distillation
2. Stage 2 bridge reinjection with frozen DiT
3. Stage 3 bridge + full DiT training

### Manual Stage 1

```bash
PYTHONPATH=. torchrun --nproc_per_node=2 \
  tools/train_q1_sana_bridge.py \
  --config configs/stage1_prompt_teacher_distill_openvid_current_laion_coyo_5v1i_2gpu_20260318.yaml \
  --max-gpus 2
```

### Manual Stage 2

```bash
PYTHONPATH=. torchrun --nproc_per_node=2 \
  tools/train_stage1_teacher_free.py \
  --config configs/stage2_teacher_free_joint_openvid_current_laion_coyo_bridge_only_gemma_distill_5v1i_2gpu_20260318.yaml \
  --init-from /path/to/stage1/checkpoint_final.pt \
  --max-gpus 2
```

### Manual Stage 3

```bash
PYTHONPATH=. torchrun --nproc_per_node=2 \
  tools/train_stage1_teacher_free.py \
  --config configs/stage3_teacher_free_joint_openvid_current_laion_coyo_bridge_dit_gemma_distill_5v1i_2gpu_20260318.yaml \
  --init-from /path/to/stage2/checkpoint_final.pt \
  --max-gpus 2
```

## 5. Canonical Inference Command

```bash
PYTHONPATH=. python tools/inference/test_q1_student_video.py \
  --bridge-ckpt /path/to/checkpoint_stepXXXX.pt \
  --prompt "a golden retriever running along a beach at sunset" \
  --output-dir output/inference_example \
  --sana-backend fixed \
  --steps 24 \
  --cfg-scale 3.0
```

Notes:
- `fixed` is the canonical backend.
- If the SANA checkpoint directory is missing, the fixed inference backend can
  bootstrap it automatically.
- The wrapper also auto-downloads the checkpoint directory when missing.

## 6. Dataset Workflows

### OpenVid
For a robust OpenVid workflow, use the bundled DataOps package:
- `download_data/README.md` (step-by-step guide)

### LAION / COYO
The current workflow is manifest-based.

Typical steps are:
1. materialize raw media from a unified manifest
2. recover unique LAION image files if needed
3. encode image rows into WAN VAE latents

Canonical scripts:
- `tools/data_prepare/materialize_unified_manifest.py`
- `tools/data_prepare/recover_laion_images_unique.py`
- `tools/data_prepare/encode_laion_coyo_images_sana_ar.py`
- `tools/data_prepare/build_laion_coyo_encoded_manifest.py`

See:
- `docs/LAION_COYO_DATA_PREPARE.md` (step-by-step guide)

### MSR-VTT
Canonical script:
- `tools/data_prepare/msrvtt_data_prepare.py`

See:
- `tools/data_prepare/MSRVTT_DATA_PREPARE.md` (step-by-step guide)

## 7. Recommended Sanity Checks

From the repo root:

```bash
PYTHONPATH=. python tools/train_q1_sana_bridge.py --help
PYTHONPATH=. python tools/train_stage1_teacher_free.py --help
PYTHONPATH=. python tools/inference/test_q1_student_video.py --help
PYTHONPATH=. python tools/data_prepare/download_openvid.py --help
PYTHONPATH=. python tools/data_prepare/msrvtt_data_prepare.py --help
```

## 8. Known Non-Goals

This repository is not currently trying to provide:
- a polished public release for every Omni-Video capability,
- a single command that downloads every dataset automatically,
- a stable HD-VILA path.

For the current training + inference scope, the main remaining responsibility on
new machines is still dataset preparation.
