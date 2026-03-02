# Next Run Plan: SANA Parity Checklist (2026-02-22)

This note captures the exact gaps we observed between current Mobile-OV stage-3 training and upstream SANA video training.

Goal for next run:
- Implement these parity items first.
- Run a clean ablation with minimal extra losses.
- Verify prompt-following before adding extra objectives back.

## 1) Current Snapshot (what happened in the last run)

- Training run: `output/logs/stage3_msrvtt_bridge_fulldit_diffusion_3gpu_20260221_122105.log`
- It improved visual quality vs older runs, but prompt-following is still weak.
- Important mismatch found:
  - Current run logged `chunk_index=None` in SANA objective startup line.
  - This means we did not actually run true chunk-causal schedule in that run.

Reference:
- `output/logs/stage3_msrvtt_bridge_fulldit_diffusion_3gpu_20260221_122105.log:283`

## 2) Key Gaps vs Upstream SANA (must-fix)

## Gap A: Not using full chunk-causal recipe
- Upstream chunk config uses:
  - `attn_type: chunkcausal`
  - `ffn_type: ChunkGLUMBConvTemp`
  - `chunk_index: [0, 11]`
- Our last run used non-chunk config and effectively had `chunk_index=None`.

References:
- `../Sana/configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp_chunk.yaml:51`
- `../Sana/configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp_chunk.yaml:53`
- `../Sana/configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp_chunk.yaml:65`
- `configs/stage1_teacher_free_msrvtt_train_stage3_crossattn_deepspeed_3gpu.yaml:91`

## Gap B: Missing IncrementalTimesteps in chunk timestep processing
- Upstream chunk trainer creates `IncrementalTimesteps(...)` and passes `time_sampler` into `process_timesteps(...)`.
- Our trainer calls `process_timesteps(...)` but does not pass `time_sampler`.

References:
- `../Sana/train_video_scripts/train_video_ivjoint_chunk.py:308`
- `../Sana/train_video_scripts/train_video_ivjoint_chunk.py:579`
- `tools/train_stage1_teacher_free.py:1723`

## Gap C: Missing ivjoint image+video curriculum
- Upstream ivjoint alternates video and image batches using `joint_training_interval` and two dataloaders.
- Our current stage-3 run is single-lane video only.

References:
- `../Sana/train_video_scripts/train_video_ivjoint.py:321`
- `../Sana/train_video_scripts/train_video_ivjoint.py:906`
- `../Sana/train_video_scripts/train_video_ivjoint.py:922`

## Gap D: Missing i2v loss-mask/noise path
- Upstream chunk script supports `do_i2v` with:
  - `loss_mask` (first frame excluded)
  - `noise_multiplier`
  - `training_losses(..., loss_mask=...)`
- Our script currently does not include this path.

References:
- `../Sana/train_video_scripts/train_video_ivjoint_chunk.py:560`
- `../Sana/train_video_scripts/train_video_ivjoint_chunk.py:563`
- `../Sana/train_video_scripts/train_video_ivjoint_chunk.py:588`
- `../Sana/train_video_scripts/train_video_ivjoint_chunk.py:611`

## Gap E: LR schedule mismatch
- Upstream uses scheduler from config (`build_lr_scheduler`, commonly constant + warmup in SANA configs).
- Our script currently hard-codes warmup + cosine lambda.

References:
- `../Sana/train_video_scripts/train_video_ivjoint.py:1215`
- `tools/train_stage1_teacher_free.py:1135`

## Gap F: EMA mismatch
- Upstream DDP path uses EMA updates during training.
- Our script currently has no EMA branch.

References:
- `../Sana/train_video_scripts/train_video_ivjoint.py:490`

## Gap G: Non-SANA auxiliary losses active during parity run
- For strict parity diagnosis, objective should be primarily SANA `training_losses(...).mean()`.
- Our config had semantic auxiliary enabled in some runs, which can distort early behavior.

References:
- `tools/train_stage1_teacher_free.py:1764`
- `tools/train_stage1_teacher_free.py:1866`
- `configs/stage1_teacher_free_msrvtt_train_stage3_crossattn_deepspeed_3gpu.yaml:155`

## 3) Implementation Plan For Next Run

Phase 1 (strict parity baseline):
1. Use chunk config semantics in training (`chunkcausal` + chunk index active).
2. Add `IncrementalTimesteps` and pass `time_sampler` to `process_timesteps`.
3. Disable all extra losses:
   - semantic_probe = off
   - distill = off
   - norm/gate losses = off
4. Keep only diffusion/flow objective from `training_losses`.

Phase 2 (ivjoint parity):
1. Add image dataloader lane.
2. Add `joint_training_interval` alternation (video/image).
3. Implement i2v loss-mask/noise path (`do_i2v`, `loss_mask`, `noise_multiplier`).

Phase 3 (stability and quality):
1. Align scheduler behavior with upstream config mode.
2. Add optional EMA branch for non-FSDP mode.
3. Re-enable extra losses one-by-one with ablation.

## 4) Acceptance Checks (must pass)

1. Startup log confirms:
- chunk attention mode active
- chunk index non-null
- process_timesteps + time_sampler active

2. Training log confirms:
- no auxiliary loss terms in Phase 1 baseline
- only diffusion/flow objective drives updates

3. Inference checks (fixed prompts, fixed seed):
- checkpoint 600 / 1000 / 2000 show progressive prompt-following gains
- no early collapse trend reappears

## 5) Notes

- Current LAION+COYO selected download runs independently and should not block this plan.
- Keep this file as the source-of-truth checklist before the next large run.
