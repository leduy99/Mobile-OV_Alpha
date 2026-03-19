# OpenVid DataOps

This is the bundled OpenVid toolkit kept inside the main repository.

It covers:
- downloading selected OpenVid-1M parts,
- extracting videos,
- building a manifest CSV,
- encoding videos into WAN VAE latents.

## Recommended Environment

Use the repository's canonical environment:

```bash
conda activate mobileov
pip install -e download_data
```

If you prefer a dedicated env for OpenVid-only work, that is still possible, but
it is no longer the primary recommendation.

## Folder Structure

```text
download_data/
  openvid_dataops/
  scripts/
    download_wan_vae_2_1.py
    run_encode_4gpu.sh
    run_pipeline_example.sh
  checkpoints/
    wan/wanxiang1_3b/
  data/openvid/
    raw/{OpenVid-1M.csv,zips/,parts/}
    manifests/
    encoded/
    state/
    logs/
```

## 1. Download WAN VAE 2.1 Checkpoint

```bash
python download_data/scripts/download_wan_vae_2_1.py \
  --output-dir download_data/checkpoints/wan/wanxiang1_3b
```

If the model repo is gated, login first:

```bash
huggingface-cli login
```

## 2. Download and Extract OpenVid Parts

```bash
# Example: parts 1,2,4,5
python -m openvid_dataops download --parts "[1,2,4,5]" --extract

# Or discover all available parts from HF dynamically
python -m openvid_dataops download --parts all --extract
```

Notes:
- default `--part-index-base 0` maps user part `N` to remote `OpenVid_partN.zip`
- if the zip is missing on HF, the downloader falls back to split chunks and
  merges them automatically

## 3. Build a Manifest

```bash
python -m openvid_dataops build-manifest \
  --parts "[1,2,4,5]" \
  --output-name openvid_p1_p2_p4_p5.csv
```

Outputs:
- `download_data/data/openvid/manifests/openvid_p1_p2_p4_p5.csv`
- `download_data/data/openvid/manifests/openvid_p1_p2_p4_p5.summary.json`

## 4. Encode WAN VAE Latents

Single GPU:

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

Multi-GPU:

```bash
bash download_data/scripts/run_encode_4gpu.sh \
  --manifest-csv download_data/data/openvid/manifests/openvid_p1_p2_p4_p5.csv \
  --ckpt-dir download_data/checkpoints/wan/wanxiang1_3b \
  --task t2v-1.3B \
  --frame-num 81 \
  --sampling-rate 1 \
  --target-size 480,832 \
  --output-subdir wan_vae_p1_p2_p4_p5
```

## 5. Resume Behavior

- re-running `download` skips existing zip files
- re-running `encode` skips existing `sample_*.pkl`
- the downloader caches the Hugging Face OpenVid index in
  `download_data/data/openvid/state/hf_openvid_index_cache.json`

## 6. Outputs

- latents: `download_data/data/openvid/encoded/<output-subdir>/sample_XXXXXXXX.pkl`
- per-rank summaries: `summary_rankXX.json`
- failures: `failed_rankXX.csv`

## 7. Help

```bash
python -m openvid_dataops --help
python -m openvid_dataops download --help
python -m openvid_dataops build-manifest --help
python -m openvid_dataops encode --help
```
