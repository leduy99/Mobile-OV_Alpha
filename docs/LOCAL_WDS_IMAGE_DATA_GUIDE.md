# Local WebDataset Image Data Guide

This guide explains how to process a local image/text WebDataset tar dataset into
the repo's final image-training format with one command.

It covers datasets stored like this:

```text
JourneyDB_215.tar
ShortCaption_000.tar
...
```

and inside each tar shard:

```text
02150000.jpg
02150000.txt
02150001.jpg
02150001.txt
...
```

The new one-command entrypoint is:

- [prepare_local_wds_image_dataset.sh](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/scripts/prepare_local_wds_image_dataset.sh)

Under the hood it uses:

- [bootstrap_local_wds_source_manifest.py](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/data_prepare/bootstrap_local_wds_source_manifest.py)
- [encode_laion_coyo_images_sana_ar.py](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/data_prepare/encode_laion_coyo_images_sana_ar.py)
- [build_laion_coyo_encoded_manifest.py](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/tools/data_prepare/build_laion_coyo_encoded_manifest.py)

## 1. What the one-command pipeline does

Running the script once will:

1. read local `.tar` shards from disk
2. pair image and text members by shared key
3. materialize local images under `raw/images`
4. write a source manifest CSV
5. encode WAN VAE latent pickles
6. write a final train-ready manifest CSV

The final train-ready CSV is what the existing image-training configs should use.

## 2. Environment

Use the verified env flow:

```bash
cd /share_4/users/duy/project/unified_video/Omni-Video-smolvlm2
source /share_0/conda/etc/profile.d/conda.sh
conda activate /share_4/users/duy/.conda/envs/mobileov_onepass_20260401
source scripts/env_exports.sh
export PYTHONNOUSERSITE=1
export PYTHONPATH=.
```

## 3. One-command usage

Example for JourneyDB:

```bash
bash scripts/prepare_local_wds_image_dataset.sh \
  --input-root /proj/cvl/users/x_fahkh2/BLIP3o/dataset/BLIP3o-Pretrain-JourneyDB \
  --output-root data/blip3o_pretrain_journeydb \
  --dataset-name blip3o_pretrain_journeydb \
  --bootstrap-jobs 8 \
  --nproc-per-node 1
```

Example for Short Caption:

```bash
bash scripts/prepare_local_wds_image_dataset.sh \
  --input-root /proj/cvl/users/x_fahkh2/BLIP3o/dataset/BLIP3o-Pretrain-Short-Caption \
  --output-root data/blip3o_pretrain_short_caption \
  --dataset-name blip3o_pretrain_short_caption \
  --bootstrap-jobs 8 \
  --nproc-per-node 1
```

If you want to process only a subset of tar shards:

```bash
bash scripts/prepare_local_wds_image_dataset.sh \
  --input-root /path/to/local_wds \
  --output-root data/local_wds_subset \
  --dataset-name local_wds_subset \
  --filenames JourneyDB_215.tar,JourneyDB_216.tar \
  --bootstrap-jobs 2 \
  --nproc-per-node 1
```

## 4. Parallelism

There are two independent knobs:

- `--bootstrap-jobs`: number of tar shards to read in parallel while building the source manifest
- `--nproc-per-node`: number of encoding workers for WAN VAE latent generation

Typical starting points:

- small smoke test: `--bootstrap-jobs 2 --nproc-per-node 1`
- medium run on one machine: `--bootstrap-jobs 8 --nproc-per-node 1`
- multi-GPU encode: `--bootstrap-jobs 8 --nproc-per-node 2`

For multi-GPU encode, set `CUDA_VISIBLE_DEVICES` yourself if needed.

## 5. Terminal logging

You will now see progress in the terminal while the pipeline runs:

- shard discovery logs
- shard start and finish logs
- periodic bootstrap progress logs inside each shard
- encoder progress logs per rank
- manifest write summaries

## 6. Output layout

Example output structure:

```text
data/blip3o_pretrain_journeydb/
  raw/images/JourneyDB_215/02150000.jpg
  manifests/blip3o_pretrain_journeydb_source.csv
  manifests/blip3o_pretrain_journeydb_source.summary.json
  encoded/wan_vae_sana_ar/sample_00000000.pkl
  encoded/wan_vae_sana_ar/summary_rank00.json
  manifests/blip3o_pretrain_journeydb_train_ready.csv
  manifests/blip3o_pretrain_journeydb_train_ready.summary.json
```

## 7. Resume behavior

- bootstrap will reuse existing materialized image files unless `--overwrite` is set
- encoding will skip existing `sample_XXXXXXXX.pkl` files
- rebuilding the train-ready manifest is always safe

This makes reruns suitable for partial resumes.
