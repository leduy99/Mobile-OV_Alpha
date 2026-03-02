# Data Mix Strategy (LAION + COYO + HD-VILA) - 2026-02-22

## Current storage snapshot
- Filesystem: `/share_4` = `7.0T total`, `~2.0T free` after cleanup.
- Old checkpoints cleaned: removed 73 old `checkpoint*.pt`, freed ~154.85 GB, kept 5 latest checkpoints.

## Downloaded artifacts in this workspace
- HD-VILA full metadata file:
  - `data/hd_vila/raw/metadata/hdvila-100M.jsonl` (~21 GB)
- HD-VILA manifest:
  - `data/hd_vila/manifests/hdvila_manifest_500k.csv` (500,000 rows)
- LAION+COYO manifests:
  - `data/laion_coyo/manifests/laion_coyo_manifest_1p2m_nomedia_v3.csv`
  - Rows: 1,200,000 = 800,000 LAION + 400,000 COYO
- Small media sanity sample:
  - `data/laion_coyo/manifests/laion_media30_v2.csv` (30 rows with local image files)

## LAION source note
- Default LAION source in `data_download` was switched to:
  - `candido-ai/laion400m-pt` (public parquet, non-gated in this environment)
- Many official LAION mirrors were gated (403) in this environment.

## Recommended joint mix (image+video)
Given current available pools:
- Images (LAION+COYO): very large
- Videos (HD-VILA): smaller and noisier captions

Use staged training:

1. Warmup (semantic stabilization):
- Image 70%, Video 30%

2. Core (motion/temporal quality):
- Image 45%, Video 55%

3. Refine (video fidelity):
- Image 40%, Video 60%

## Prebuilt phase manifests
Generated at:
- `data/mix/manifests/joint_phase0_warmup_300k.csv`
  - 210,000 image + 90,000 video
- `data/mix/manifests/joint_phase1_core_600k.csv`
  - 270,000 image + 330,000 video
- `data/mix/manifests/joint_phase2_refine_300k.csv`
  - 120,000 image + 180,000 video

All files keep unified schema from `data_download/openvid_dataops/schema.py`.

## Practical next step for training throughput
1. Materialize media for selected subsets (especially video clips and image URLs needed by WAN VAE encode).
2. Run WAN VAE encode on each phase manifest.
3. Start training with phase curriculum above (and keep same train/infer preprocessing).
