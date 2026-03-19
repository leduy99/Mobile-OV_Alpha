# 5TB Data Strategy for SANA-Based Mobile-OV Training

This note documents the current data plan for `Omni-Video-smolvlm2` after the OpenVid archive cleanup on March 8, 2026.

It is intended to answer four practical questions:

1. Why we should not attempt to mirror the full upstream corpora.
2. How much data we actually want to materialize within a 5TB budget.
3. Why the dataset ratios are video-heavy.
4. Which training constraints matter so other people can reproduce the same assumptions.

## 1. Scope: 5TB curated subset, not full-source replication

The 5TB target must be interpreted as a curated local training pool, not a full copy of the source datasets.

Using the storage footprint measured from the current encoded data on this machine, full-source scale would be far beyond 5TB:

- HD-VILA 100M clips: roughly 839TB to 1258TB
- COYO-700M: roughly 281TB
- LAION-400M: roughly 160TB

Those estimates are based on the local encoded footprint already present in this workspace:

- COYO encoded sample size: about 0.40MB per sample
- LAION encoded sample size: about 0.40MB per sample
- Video encoded sample size: about 8.4MB to 12.6MB per sample

Conclusion: the correct target is a filtered subset that preserves the video prior and semantic diversity, not a raw mirror.

## 2. Post-cleanup storage state

The OpenVid archive cleanup zeroed the old `OpenVid_part*.zip` files and root download logs inside `data/openvid`.

Observed state after cleanup:

- `data/openvid`: about 150GB
- `data/` total: about 147GB
- `/share_4` free space: about 4.9TB

This means the 5TB data budget is now effectively available for the new curated pool.

## 3. Why the mix should be video-heavy

The guiding principle comes from Omni-Video 2: large-scale captioned videos should remain the backbone of training, while image-text data is used to improve semantic alignment and controllability rather than replace the video prior.

For our setup, that maps naturally to:

- HD-VILA = captioned video backbone
- COYO = higher-value image-text supplement
- LAION = additional broad-coverage image-text supplement

This is especially important because the current conditioning path is still fragile:

- The bridge is `SmolVLM2 -> MCP -> SANA cross-attn space`
- The current MCP path is simpler than the legacy adapter+resampler path
- If image data dominates, the model can drift toward prompt-insensitive or near-unconditional behavior

Because of that, the data plan should protect the text-to-video prior first, then add image-text diversity second.

## 4. Recommended dataset targets inside the 5TB budget

Recommended materialized targets:

- HD-VILA: 300,000 clips
- COYO: 700,000 images
- LAION: 350,000 images

Why these numbers:

- 300k HD-VILA clips keep the pool decisively video-centric.
- 700k COYO gives enough image-text diversity without overwhelming the video lane.
- 350k LAION adds breadth but stays below COYO, because COYO is usually the more useful image-text source for this training regime.

This target remains safely inside the 5TB budget based on current measured storage per sample.

## 5. Recommended training-time sampling ratios

Do not sample according to raw file counts. Use weighted sampling.

Recommended schedule:

- Warmup phase: 50% HD-VILA, 35% COYO, 15% LAION
- Main phase: 60% HD-VILA, 28% COYO, 12% LAION
- Late/refinement phase: 70% HD-VILA, 21% COYO, 9% LAION

The main-phase ratio is the default recommendation if only one static mixture is used.

Rationale:

- Warmup should keep enough image-text pressure to stabilize semantics.
- Main training should prioritize video to preserve motion and temporal prior.
- Late training should become even more video-heavy if prompt collapse is under control.

## 6. Current training assumptions that affect the data plan

The current run assumptions matter because the data mix was selected for this exact regime.

Relevant settings from the active SANA-based joint config:

- Base model: SANA Video 2B 480p
- Prompt conditioning target width: 2304
- Prompt sequence length: 300
- Strict SANA parity text path: enabled
- Fail-fast mask check: enabled
- Flow shift: 3.0
- Optimizer: AdamW
- Bridge LR: 5e-5
- DiT LR: 1e-5
- External CFG dropout: 0.0
- Teacher distillation: enabled
- Semantic anti-collapse probe/loss: enabled

The current bridge path uses:

- `projector.type = mcp_full`
- last-4-layer fusion
- lightweight sequence refinement
- direct projection into the SANA conditioning space

That bridge is usable, but it is still a likely bottleneck. The data plan therefore favors video-heavy sampling so the model does not lose the text-to-video prior while the conditioning path is still being improved.

## 7. Download workflow

Two scripts were added to this repo:

- `scripts/cleanup_openvid_storage.sh`
- `scripts/download_5tb_data_parallel.sh`

### Cleanup

```bash
bash scripts/cleanup_openvid_storage.sh
```

This environment blocks direct `rm` in some cases, so the cleanup script truncates the old OpenVid archives and logs to zero bytes instead.

### Parallel download

```bash
export YTDLP_COOKIES_FILE=/path/to/cookies.txt
bash scripts/download_5tb_data_parallel.sh
```

Default targets:

- HD-VILA rows: 300000
- COYO rows: 700000
- LAION rows: 350000
- HD-VILA workers: 4
- image workers: 16

The script will:

1. Create metadata-only manifests if they do not exist.
2. Launch multiple workers for HD-VILA materialization.
3. Launch multiple workers for LAION+COYO image downloads.
4. Merge worker shard manifests into final materialized manifests.

## 8. Important operational notes

HD-VILA is the main bottleneck.

- It requires valid YouTube cookies.
- Without `YTDLP_COOKIES_FILE` or `YTDLP_COOKIES_FROM_BROWSER`, HD-VILA download will fail.
- Workers are partitioned by video id so clips from the same source reuse the same full-video download inside one worker.
- Full videos are removed after clipping when it is safe to do so.

LAION and COYO are easier to scale.

- They are downloaded from unified metadata manifests.
- Image workers are partitioned by sample id.
- Materialized shard manifests are merged after all workers finish.

## 9. Why this plan is preferable to a naive larger image pool

A naive plan would be to maximize LAION/COYO count because images are cheap.

That is the wrong objective for this project.

The current problem is not just visual quality; it is maintaining prompt sensitivity while preserving a strong text-to-video prior. Omni-Video 2 strongly suggests that captioned video remains the backbone. For this reason, the local plan intentionally spends most of the 5TB budget on HD-VILA rather than on the cheapest possible image corpus.

## 10. Recommended next steps

1. Keep the cleaned OpenVid state.
2. Materialize HD-VILA first and confirm YouTube cookie stability.
3. Fill COYO second.
4. Fill LAION third.
5. Build weighted manifests with the main-phase ratio: `HD-VILA 60 / COYO 28 / LAION 12`.

## References

- Omni-Video 2 paper: https://arxiv.org/abs/2602.08820
- Omni-Video project page: https://sais-fuxi.github.io/Omni-Video/
