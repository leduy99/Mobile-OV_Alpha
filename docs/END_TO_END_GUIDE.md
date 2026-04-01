# End-to-End Guide: Setup, Data, Training, and Inference

This guide is the single-file onboarding path for this repository.

It explains how to go from:

- cloning the repo
- creating the conda environment
- preparing datasets
- launching training
- running inference

The goal is to make a fresh server usable without having to jump between many
notes.

## 1. Scope

This guide covers the workflows that are currently supported and actively used
in this repository:

- canonical conda environment setup
- OpenVid download and WAN VAE encoding
- MSR-VTT download and WAN VAE encoding
- LAION / COYO full-mix preparation from a source manifest
- 3-stage SmolVLM2 + bridge + SANA training
- checkpoint-based video inference with the fixed backend

This guide does not try to cover:

- HD-VILA
- archived legacy paths
- deprecated inference backends

## 2. Clone the Repository

### Command

```bash
git clone <YOUR_REPO_URL>
cd Omni-Video-smolvlm2
```

### What this does

This gives you the training code, data-preparation scripts, configs, and
launchers used by the current pipeline.

## 3. Create the Conda Environment

Recommended path:

```bash
bash scripts/setup_mobileov_env.sh mobileov
```

Detailed environment guide:
- [ENV_SETUP_GUIDE.md](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/docs/ENV_SETUP_GUIDE.md)

### Command

```bash
conda env create -f environment.yml
conda activate mobileov
export PYTHONNOUSERSITE=1
source scripts/env_exports.sh
pip install -e download_data
```

### What this does

- creates the canonical conda environment
- activates it
- installs the bundled `openvid_dataops` package in editable mode

### If the environment already exists

```bash
conda env update -f environment.yml --prune
conda activate mobileov
export PYTHONNOUSERSITE=1
source scripts/env_exports.sh
pip install -e download_data
```

All commands below assume those two exports are active. They avoid user-site package
conflicts and make repo-local imports such as `nets.*` resolve correctly.

## 4. Install System Tools

Recommended system packages:

```bash
ffmpeg
wget
git
```

Optional but useful:

```bash
yt-dlp
```

`yt-dlp` is useful for generic video materialization paths in unified manifests.

## 5. Authenticate With Hugging Face

### Command

```bash
huggingface-cli login
```

### What this does

This is used for:

- SANA checkpoint assets
- SmolVLM2 source model conversion
- OpenVid downloads
- MSR-VTT downloads

## 6. Understand Which Model Assets Are Automatic and Which Are Manual

### Auto-bootstrap

These entrypoints can download or convert model assets automatically when
missing:

- `tools/train_q1_sana_bridge.py`
- `tools/train_stage1_teacher_free.py`
- `tools/inference/sana_video_inference_fixed.py`
- `tools/inference/test_q1_student_video.py`

This covers:

- SANA video checkpoint assets
- converted SmolVLM2 checkpoints

### Still manual

WAN VAE assets used for OpenVid / MSR-VTT / LAION / COYO latent encoding are
still manual.

### Command

```bash
python download_data/scripts/download_wan_vae_2_1.py \
  --output-dir download_data/checkpoints/wan/wanxiang1_3b
```

If you also want to use the SANA-packaged VAE path for LAION / COYO image
encoding, make sure the SANA assets have been bootstrapped once by one of the
training or inference entrypoints.

## 7. Run Basic Sanity Checks

### Command

```bash
PYTHONPATH=. python tools/train_q1_sana_bridge.py --help
PYTHONPATH=. python tools/train_stage1_teacher_free.py --help
PYTHONPATH=. python tools/inference/test_q1_student_video.py --help
PYTHONPATH=. python tools/data_prepare/msrvtt_data_prepare.py --help
python -m openvid_dataops --help
```

### What this does

This confirms that the environment can import the main training, inference, and
data-preparation entrypoints.

## 8. Choose Your Dataset Workflow

You do not always need every dataset.

Use this quick rule:

- If you only want OpenVid: follow Section 9
- If you want MSR-VTT too: follow Section 10
- If you want the OpenVid + LAION / COYO full mix: follow Section 11

## 9. OpenVid: Download and Encode

The detailed step-by-step guide lives here:

- [download_data/README.md](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/download_data/README.md)

### Minimal command flow

```bash
# 1) Download the WAN VAE checkpoint used for OpenVid latent encoding
python download_data/scripts/download_wan_vae_2_1.py \
  --output-dir download_data/checkpoints/wan/wanxiang1_3b

# 2) Download raw OpenVid parts and extract them
python -m openvid_dataops download \
  --parts all \
  --extract

# 3) Build the manifest CSV that lists the extracted samples
python -m openvid_dataops build-manifest \
  --parts all \
  --output-name openvid_all.csv

# 4) Encode videos into WAN VAE latent pickles
python -m openvid_dataops encode \
  --manifest-csv download_data/data/openvid/manifests/openvid_all.csv \
  --ckpt-dir download_data/checkpoints/wan/wanxiang1_3b \
  --task t2v-1.3B \
  --frame-num 81 \
  --sampling-rate 1 \
  --target-size 480,832 \
  --output-subdir wan_vae_openvid_all
```

### Final outputs you should expect

- raw OpenVid files under `download_data/data/openvid/raw/`
- manifest CSV under `download_data/data/openvid/manifests/`
- latent pickles under `download_data/data/openvid/encoded/wan_vae_openvid_all/`

### Important arguments

- `--parts`: which OpenVid parts to download; use `all` for the full dataset
- `--extract`: extract zip files immediately after download
- `--output-name`: name of the generated OpenVid manifest CSV
- `--manifest-csv`: which OpenVid manifest to encode
- `--ckpt-dir`: WAN VAE checkpoint directory
- `--task`: WAN task preset; keep `t2v-1.3B` unless you know you need another one
- `--frame-num`: number of frames per sample; current canonical value is `81`
- `--sampling-rate`: frame stride; `1` means dense sampling
- `--target-size`: encoded spatial size in `H,W`; current canonical value is `480,832`
- `--output-subdir`: folder name under `download_data/data/openvid/encoded/`

## 10. MSR-VTT: Download and Encode

The detailed step-by-step guide lives here:

- [MSRVTT_DATA_PREPARE.md](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/data_prepare/MSRVTT_DATA_PREPARE.md)

### Minimal command flow

```bash
# 1) Download the raw MSR-VTT dataset files and extract zip archives
PYTHONPATH=. python tools/data_prepare/msrvtt_data_prepare.py download \
  --root-dir data/msrvtt \
  --repo-id AlexZigma/msr-vtt \
  --extract

# 2) Build a simple OpenVid-style CSV with one caption per video
PYTHONPATH=. python tools/data_prepare/msrvtt_data_prepare.py build-csv \
  --root-dir data/msrvtt \
  --manifest-name OpenVid_extracted_subset_unique.csv \
  --caption-policy longest \
  --captions-per-video 1

# 3) Encode MSR-VTT videos into WAN VAE latent pickles
PYTHONPATH=. python tools/data_prepare/msrvtt_data_prepare.py encode \
  --root-dir data/msrvtt \
  --manifest-name OpenVid_extracted_subset_unique.csv \
  --ckpt-dir omni_ckpts/wan/wanxiang1_3b \
  --task t2v-1.3B \
  --frame-num 81 \
  --sampling-rate 1 \
  --target-size 480,832
```

### Final outputs you should expect

- raw files under `data/msrvtt/raw/`
- extracted videos under `data/msrvtt/videos/`
- metadata under `data/msrvtt/metadata/`
- CSV manifest `data/msrvtt/OpenVid_extracted_subset_unique.csv`
- latent pickles under `data/msrvtt/preprocessed/`

### Important arguments

- `--root-dir`: root directory for all MSR-VTT outputs
- `--repo-id`: Hugging Face dataset id to download from
- `--manifest-name`: output CSV name used by later encode steps
- `--caption-policy`: how to choose one caption when multiple captions exist
- `--captions-per-video`: number of captions to keep per video
- `--ckpt-dir`: WAN VAE checkpoint directory
- `--task`: WAN task preset; keep `t2v-1.3B` unless you explicitly need a different one
- `--frame-num`: number of frames per sample; canonical value is `81`
- `--sampling-rate`: temporal stride used during frame sampling
- `--target-size`: encoded spatial size in `H,W`
- `--max-samples`: useful for smoke tests before running the full encode

## 11. LAION / COYO Full Mix: From Source Manifest to Train-Ready Mix

The detailed step-by-step guide lives here:

- [LAION_COYO_DATA_PREPARE.md](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/docs/LAION_COYO_DATA_PREPARE.md)

### Important limitation

The repo does not currently build a curated LAION / COYO source manifest from
zero. You must start from:

- the exact source manifest copied from an existing workspace
- or your own curated source manifest

### Minimal command flow

```bash
# 0) Point to the curated LAION / COYO source manifest you want to rebuild
export LAION_COYO_SOURCE_MANIFEST=data/laion_coyo/manifests/laion_coyo_source_selected.csv

# 1) Download raw media referenced by the source manifest
PYTHONPATH=. python tools/data_prepare/materialize_unified_manifest.py \
  --manifest-csv "$LAION_COYO_SOURCE_MANIFEST" \
  --dataset-root data/laion_coyo/raw \
  --output-manifest data/laion_coyo/manifests/laion_coyo_materialized.csv \
  --modality all \
  --worker-id 0 \
  --num-workers 1

# 2) Repair LAION image filename collisions so captions cannot attach to the wrong file
PYTHONPATH=. python tools/data_prepare/recover_laion_images_unique.py \
  --input-manifest data/laion_coyo/manifests/laion_coyo_materialized.csv \
  --output-manifest data/laion_coyo/manifests/laion_coyo_recovered_unique.csv \
  --output-image-dir data/laion_coyo/raw/media/images_laion_recovered \
  --failures-csv data/laion_coyo/manifests/laion_recovered_unique_failures.csv \
  --summary-json data/laion_coyo/manifests/laion_recovered_unique_summary.json \
  --workers 32 \
  --timeout 20 \
  --retries 2

# 3) Encode LAION / COYO image rows into WAN VAE latent pickles
PYTHONPATH=. python tools/data_prepare/encode_laion_coyo_images_sana_ar.py \
  --manifest-csv data/laion_coyo/manifests/laion_coyo_recovered_unique.csv \
  --output-dir data/laion_coyo/encoded/wan_vae_openvid_mix_sana_ar \
  --vae-ckpt omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth

# 4) Convert the encoded directory into a normalized LAION / COYO train manifest
PYTHONPATH=. python tools/data_prepare/build_laion_coyo_encoded_manifest.py \
  --source-manifest data/laion_coyo/manifests/laion_coyo_recovered_unique.csv \
  --encoded-dir data/laion_coyo/encoded/wan_vae_openvid_mix_sana_ar \
  --output-csv data/laion_coyo/manifests/laion_coyo_encoded_for_openvid_mix.csv \
  --datasets laion,coyo_700m \
  --modality image

# 5) Merge OpenVid video rows with LAION / COYO image rows
PYTHONPATH=. python tools/data_prepare/build_joint_manifest_openvid_current_laion_coyo.py \
  --existing-mix-manifest data/laion_coyo/manifests/laion_coyo_encoded_for_openvid_mix.csv \
  --openvid-manifest data/openvid_1m/manifests/openvid_manifest_0_112.csv \
  --openvid-preprocessed-dir data/openvid_1m/encoded/wan_vae_fp16_stream \
  --output-prefix data/mix/manifests/joint_openvid_current_laion_coyo_rebuilt

# 6) Remove broken or low-quality rows from the combined manifest
PYTHONPATH=. python tools/data_prepare/sanitize_joint_manifest.py \
  --input-csv data/mix/manifests/joint_openvid_current_laion_coyo_rebuilt.csv \
  --output-csv data/mix/manifests/joint_openvid_current_laion_coyo_rebuilt_clean.csv \
  --image-datasets coyo_700m,laion \
  --min-image-bytes 2000 \
  --dedup-image-path \
  --drop-missing-preprocessed \
  --drop-empty-caption

# 7) Validate the final manifest before training
PYTHONPATH=. python tools/data_prepare/preflight_joint_pipeline_check.py \
  --config configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_only_smolvlm2_2p2b_online_teacher_20k_2gpu_20260321.yaml \
  --csv-path data/mix/manifests/joint_openvid_current_laion_coyo_rebuilt_clean.csv \
  --report-json data/mix/manifests/joint_openvid_current_laion_coyo_rebuilt_clean.preflight.json
```

### Final outputs you should expect

- raw media under `data/laion_coyo/raw/`
- repaired LAION images under `data/laion_coyo/raw/media/images_laion_recovered/`
- latent pickles under `data/laion_coyo/encoded/`
- normalized LAION / COYO encoded manifest under `data/laion_coyo/manifests/`
- final mixed train manifests under `data/mix/manifests/`

### Important arguments

- `--manifest-csv`: input source manifest for download/materialization
- `--dataset-root`: root directory where raw LAION / COYO media will be stored
- `--output-manifest`: output CSV produced by each transformation step
- `--modality`: keep `image`, `video`, or `all` rows at the materialization stage
- `--worker-id` / `--num-workers`: sharding controls for multi-worker downloads
- `--workers`: thread concurrency for LAION recovery
- `--timeout` / `--retries`: network robustness controls during recovery
- `--output-dir`: where encoded latent pickles are written
- `--vae-ckpt`: WAN VAE checkpoint used for image encoding
- `--datasets`: dataset filter when building the normalized LAION / COYO encoded manifest
- `--image-datasets`: which image datasets to keep during manifest sanitization
- `--drop-missing-preprocessed`: remove rows whose latent pickle does not exist
- `--drop-empty-caption`: remove rows with empty captions
- `--config`: training config used by the preflight checker
- `--csv-path`: final manifest path to validate before training

## 12. Training

Once the dataset paths are ready, you can train.

### Easiest path: full 3-stage launcher

```bash
# Run Stage 1 -> Stage 2 -> Stage 3 in sequence
GPUS=0,1 bash scripts/train_openvid_current_laion_coyo_3stage_gemma_distill_5v1i.sh
```

### What this does

This launcher runs:

1. Stage 1 prompt-only teacher distillation
2. Stage 2 bridge reinjection with frozen DiT
3. Stage 3 bridge + full DiT training

### Manual Stage 1

```bash
# Stage 1: prompt-only teacher distillation
PYTHONPATH=. torchrun --nproc_per_node=2 \
  tools/train_q1_sana_bridge.py \
  --config configs/stage1_prompt_teacher_distill_openvid_current_laion_coyo_5v1i_2gpu_20260318.yaml \
  --max-gpus 2
```

### Manual Stage 2

```bash
# Stage 2: bridge reinjection with frozen DiT, initialized from Stage 1
PYTHONPATH=. torchrun --nproc_per_node=2 \
  tools/train_stage1_teacher_free.py \
  --config configs/stage2_teacher_free_joint_openvid_current_laion_coyo_bridge_only_gemma_distill_5v1i_2gpu_20260318.yaml \
  --init-from /path/to/stage1/checkpoint_final.pt \
  --max-gpus 2
```

### Manual Stage 3

```bash
# Stage 3: bridge + DiT training, initialized from Stage 2
PYTHONPATH=. torchrun --nproc_per_node=2 \
  tools/train_stage1_teacher_free.py \
  --config configs/stage3_teacher_free_joint_openvid_current_laion_coyo_bridge_dit_gemma_distill_5v1i_2gpu_20260318.yaml \
  --init-from /path/to/stage2/checkpoint_final.pt \
  --max-gpus 2
```

### Important arguments

- `--config`: which training recipe to run
- `--init-from`: checkpoint used to initialize the next stage
- `--max-gpus`: maximum number of GPUs the script may use
- `--nproc_per_node`: number of local distributed workers launched by `torchrun`

### Most important training note

If you rebuilt your manifests under new file names, either:

- rename them to match the paths already used by the config
- or copy the config and update:
  - `data.openvid.csv_path`
  - `data.openvid.csv_path_video`
  - `data.openvid.csv_path_image`

## 13. Inference

### Canonical command

```bash
# Load a trained student checkpoint and generate one video
PYTHONPATH=. python tools/inference/test_q1_student_video.py \
  --bridge-ckpt /path/to/checkpoint_stepXXXX.pt \
  --prompt "a golden retriever running along a beach at sunset" \
  --output-dir output/inference_example \
  --sana-backend fixed \
  --steps 24 \
  --cfg-scale 3.0
```

### What this does

This loads:

- the trainable student checkpoint
- the bridge/projector state
- optional DiT trainable state if present

and generates a video with the fixed backend.

### Important arguments

- `--bridge-ckpt`: student checkpoint to load
- `--prompt`: text prompt
- `--output-dir`: where generated video files will be written
- `--sana-backend`: use `fixed`
- `--steps`: diffusion sampling steps
- `--cfg-scale`: classifier-free guidance scale

### Notes

- if the SANA checkpoint directory is missing, the fixed backend can bootstrap
  it automatically
- the wrapper can also auto-bootstrap the checkpoint directory when needed

## 14. Recommended End-to-End Smoke Test

If you are onboarding a new machine, do not start with a huge run.

Use this order:

1. create the env
2. run the `--help` commands
3. download WAN VAE
4. do a tiny OpenVid encode with `--max-samples 16`
5. if using MSR-VTT, do a tiny encode with `--max-samples 16`
6. if using LAION / COYO, run:
   - a tiny source manifest
   - a tiny encode
   - preflight
7. only then launch a real training run

## 15. The Three Most Important Things to Preserve for Reproducibility

If you want another server to reproduce your exact training data, preserve:

1. the source manifests
2. the cleaned final joint manifests
3. the preflight report JSONs

For LAION / COYO, preserving the exact source manifest is especially important.

## 16. Where To Go Next

If you want more detail on a specific dataset path, use the dedicated guide:

- OpenVid: [download_data/README.md](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/download_data/README.md)
- MSR-VTT: [MSRVTT_DATA_PREPARE.md](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/data_prepare/MSRVTT_DATA_PREPARE.md)
- LAION / COYO: [LAION_COYO_DATA_PREPARE.md](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/docs/LAION_COYO_DATA_PREPARE.md)
