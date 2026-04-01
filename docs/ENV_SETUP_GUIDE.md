# Environment Setup Guide

This is the recommended setup path for this repo as of April 1, 2026.

It has been re-tested from a fresh environment and is intended to work in one pass.

## Recommended one-shot setup

Run from repo root:

```bash
cd /share_4/users/duy/project/unified_video/Omni-Video-smolvlm2
bash scripts/setup_mobileov_env.sh mobileov_fresh_test
```

What this does:

- creates or updates the conda env from [environment.yml](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/environment.yml)
- installs `download_data` in editable mode
- validates:
  - `tools/train_stage1_teacher_free.py --help`
  - `tools/inference/test_q1_student_video.py --help`
  - `tools/data_prepare/encode_laion_coyo_images_sana_ar.py --help`

## Daily activation

In a new shell:

```bash
cd /share_4/users/duy/project/unified_video/Omni-Video-smolvlm2
source /share_0/conda/etc/profile.d/conda.sh
conda activate mobileov_fresh_test
source scripts/env_exports.sh
```

The `env_exports.sh` step is important because it sets:

- `PYTHONNOUSERSITE=1`
- `PYTHONPATH=$PWD:${PYTHONPATH:-}`

This avoids two common problems:

- user-site packages in `~/.local` shadowing the env
- repo-local imports like `nets.*` failing when scripts are launched from repo root

## Manual setup

If you prefer to run the steps yourself:

```bash
cd /share_4/users/duy/project/unified_video/Omni-Video-smolvlm2
source /share_0/conda/etc/profile.d/conda.sh
conda env create -n mobileov_fresh_test -f environment.yml
conda activate mobileov_fresh_test
source scripts/env_exports.sh
pip install -e download_data
```

Then validate:

```bash
PYTHONPATH=. PYTHONNOUSERSITE=1 python tools/train_stage1_teacher_free.py --help
PYTHONPATH=. PYTHONNOUSERSITE=1 python tools/inference/test_q1_student_video.py --help
PYTHONPATH=. PYTHONNOUSERSITE=1 python tools/data_prepare/encode_laion_coyo_images_sana_ar.py --help
```

## What changed in the environment spec

The canonical spec now includes the runtime dependencies that were missing in fresh env creation:

- `peft`
- `termcolor`
- `timm`
- `omegaconf`
- `qwen-vl-utils`

It also removes two problematic packages from the canonical env:

- `apex`
- `flash-attn`

These are treated as optional accelerators rather than hard requirements.

The spec also now includes the PyTorch CUDA wheel index:

```text
--extra-index-url https://download.pytorch.org/whl/cu118
```

## Smoke-check script

You can rerun validation anytime:

```bash
bash scripts/validate_mobileov_env.sh mobileov_fresh_test
```

## Notes

- The canonical env name in the repo remains `mobileov`, but the setup script accepts any env name.
- `apex` is optional. If it is absent, SANA will fall back to vanilla RMSNorm.
- Warnings from `timm` deprecations or `torch.cuda.amp.autocast` do not block training.
