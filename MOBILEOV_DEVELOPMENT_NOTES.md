# MobileOV Development Notes

Tài liệu tổng hợp về quá trình phát triển MobileOV, các thay đổi, fixes, và verification results.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Key Changes Made](#key-changes-made)
3. [Flow Fixes](#flow-fixes)
4. [Verification Results](#verification-results)
5. [Performance Analysis](#performance-analysis)
6. [Stage-3 SANA Tracking (2026-02-25)](#stage-3-sana-tracking-2026-02-25)
7. [Current 3-Stage Status (2026-03-19)](#current-3-stage-status-2026-03-19)
8. [Stage-1 Sem+Vis Restart (2026-03-21)](#stage-1-semvis-restart-2026-03-21)
9. [Smol Visual Recipe Repro (2026-03-25)](#smol-visual-recipe-repro-2026-03-25)
10. [Experiment Report (2026-03-25)](#experiment-report-2026-03-25)
11. [Overfit And Recaption Diagnostics (2026-03-25)](#overfit-and-recaption-diagnostics-2026-03-25)
12. [Full Mobile-O Image Recipe Status (2026-04-20)](#full-mobile-o-image-recipe-status-2026-04-20)

---

## Full Mobile-O Image Recipe Status (2026-04-20)

We now have one `full_mobile_o` image-only line that is important to preserve as
a known-working recipe family.

### What is now considered "confirmed working"

The `bridge-only + online teacher distill` recipe for merged Mobile-O image data
is no longer just a speculative setup. It ran stably enough to produce a usable
`step10000` checkpoint and that checkpoint passed prompt-based image inference
sanity checks.

This does **not** mean the recipe is final or already optimal. It means the
training path itself is valid and worth keeping as a baseline instead of
discarding it as another dead branch.

### Known-working bridge-only teacher recipe

- Config:
  `configs/stage1_teacher_free_full_mobile_o_image_bridge_only_lexical_gated_k2_online_teacher_bs4_v2_1gpu_20260417.yaml`
- Launcher:
  `scripts/train_full_mobile_o_image_bridge_only_lexical_gated_k2_online_teacher_bs4_v2.sh`
- Data:
  merged image-only manifest
  `data/full_mobile-o/manifests/journeydb_short_caption_train_ready.csv`
  built from `JourneyDB + Short-Caption`
- Student family:
  `SmolVLM2-500M`
- Projector:
  `mcp_lexical_gated`, `K=2`
- DiT setting:
  `train_modules: []` (bridge-only, no DiT finetuning)
- Effective train batch:
  `batch_size=64`, `batch_size_image=64`, `grad_accum_steps=1`
- Hardware used for the real long run:
  `8 x H200` on Berzelius

Loss stack used by this known-working bridge-only run:
- diffusion loss enabled with `diff.weight=1.0`
- online teacher distill enabled
- `freeze_sana_conditioner: true`
- `online_fallback_on_missing: true`
- token losses enabled:
  `token_mse_weight=1.0`, `token_cos_weight=0.5`, `pooled_cos_weight=0.2`
- teacher-side auxiliary alignment enabled:
  `contrastive_weight=0.1`, `hidden0_geom_weight=0.05`
- functional loss enabled:
  `pred_mse_weight=0.05`, `pred_cos_weight=0.02`
- disabled regularizers:
  `semantic_probe`, `norm`, and `gate`

Most important practical takeaway:
- this recipe produced the bridge-only checkpoint
  `output/stage1_bridge_only_full_mobile_o_smolvlm2_500m_lexical_gated_k2_online_teacher_bs64_v2_20260417_8gpu/20260417_093000/checkpoint_step10000.pt`
- that `step10000` checkpoint was later used successfully for image inference
  sanity checks and for initializing the next full-DiT run family

### Follow-up full-DiT recipe that now branches from the 10k bridge checkpoint

We also now have a clean continuation recipe that starts from the known-working
bridge-only `10k` checkpoint and then trains both bridge + full DiT.

- Config:
  `configs/stage1_full_mobile_o_fulldit_diffonly_init10k_v2_bs64_8gpu.yaml`
- Launcher:
  `scripts/train_full_mobile_o_fulldit_diffonly_init10k_v2.sh`
- Init checkpoint:
  `output/stage1_bridge_only_full_mobile_o_smolvlm2_500m_lexical_gated_k2_online_teacher_bs64_v2_20260417_8gpu/20260417_093000/checkpoint_step10000.pt`
- Student family:
  `SmolVLM2-500M`
- Projector:
  `mcp_lexical_gated`, `K=2`
- DiT setting:
  `train_modules: [all]`
- Sharding:
  `model.dit.fsdp: true`
- Effective train batch:
  `batch_size=64`, `batch_size_image=64`, `grad_accum_steps=1`
- Intended hardware:
  `8 x H200`
- Current default duration:
  `total_steps=500000`

Loss stack used by this follow-up run:
- diffusion loss only
- `distill.enabled: false`
- `functional.enabled: false`
- `semantic_probe`, `norm`, and `gate` all disabled

### Small operational note

In logs, this image-only merged-data run may still print `mode=video`. In this
recipe that label is only a loader-path naming artifact; the actual manifest is
image-only and the dataset contract is `expected_latent_t=1`,
`expected_frame_num=1`.

---

## Current 3-Stage Status (2026-03-19)

The repo now has a dedicated note for the current `SmolVLM2 -> bridge -> SANA`
3-stage pipeline and the Stage 2 redesign:

- Detailed note: `docs/STAGE2_FUNCTIONAL_RETRAIN_NOTE_20260319.md`

Why this note exists:
- Stage 1 looked directionally successful and reduced semantic collapse.
- The first Stage 2 showed that token-level teacher matching alone was not
  enough for a frozen DiT.
- On 2026-03-19, Stage 2 was revised to add functional DiT-response
  distillation and retrained from the best Stage 1 checkpoint.

Use the dedicated note above as the current reference for:
- the original 3-stage plan and why we adopted it,
- what Stage 1 and the first Stage 2 taught us,
- why we changed Stage 2 instead of jumping to Stage 3,
- what the new Stage 2 objective is trying to match,
- what happened by the 5k and 10k checkpoints of Stage 2 v2,
- why better loss did not yet translate into clearly better inference,
- and why the current plan is to hold Stage 2 v2 until 20k before deciding on the next redesign.

---

## Stage-1 Sem+Vis Restart (2026-03-21)

After checking the hybrid continuation line at both `5k` and `10k`, we decided
to stop that run and restart from pretrained base instead of continuing to patch
later stages.

Reference note:
- `docs/STAGE1_SEMVIS_RESTART_NOTE_20260321.md`

Canonical files for the new restart:
- `configs/stage1_teacher_free_joint_openvid_current_laion_coyo_semvis_frombase_5v1i_2gpu_20260321.yaml`
- `scripts/train_openvid_current_laion_coyo_stage1_semvis_frombase_5v1i.sh`

Prepared follow-up if the new 500M restart still fails:
- `configs/stage1_teacher_free_joint_openvid_current_laion_coyo_semvis_frombase_smolvlm2_2p2b_5v1i_2gpu_20260321.yaml`
- `scripts/train_openvid_current_laion_coyo_stage1_semvis_frombase_2p2b_5v1i.sh`

Why this restart exists:
- the hybrid continuation remained numerically stable but did not improve
  qualitative inference in a meaningful way,
- the `5k -> 10k` comparison stayed in the same failure family,
- this suggested the earlier semantic-only Stage 1 had already biased the
  student manifold in a way that later patching could not cleanly repair.

The new recipe changes the starting assumption:
- restart from pretrained base,
- keep semantic teacher anchoring from step 0,
- allow a small amount of DiT text-facing adaptation from step 0,
- reduce proxy-loss dominance so diffusion quality matters more,
- and only move to `SmolVLM2-2.2B` if this cleaner 500M restart still fails.

Smoke status:
- the new `500M` restart recipe passed a real `1-step` two-GPU smoke run
  including forward, backward, and checkpoint save.

---

## Smol Visual Recipe Repro (2026-03-25)

We now have a dedicated note for the `SmolVLM2-500M + bridge + full DiT`
recipe that looks much healthier than the recent Qwen experiments and is worth
preserving as a candidate "known-good family".

Detailed note:
- `docs/SMOL_VISUAL_RECIPE_NOTE_20260325.md`

Why this note exists:
- the old `2026-03-16 initfrom30k_nodistill` line could not be replayed
  directly because its upstream init checkpoint was no longer on disk,
- to recover that lineage faithfully, we relaunched the upstream
  `2026-03-15 online_teacher_30k` family first,
- this run restored the old training shape:
  - `SmolVLM2-500M`
  - bridge trainable
  - full DiT trainable
  - online teacher distill
  - semantic anti-collapse enabled
  - `1V:1I`
  - `grad_accum_steps=1`
- by `4k`, the line still was not "solved", but it looked materially more
  stable than the Qwen lines in both loss geometry and qualitative behavior.

Use the dedicated note above as the reference for:
- the exact config/launcher/log paths for the reproduction run,
- why this recipe is currently the strongest candidate to keep around,
- which metrics looked healthy enough to justify waiting longer,
- what the early `4k` inference did and did not prove,
- and how to branch from this family later if we want to recover the old
  `initfrom30k_nodistill` behavior again.

---

## Experiment Report (2026-03-25)

We now also have a longer, presentation-friendly experiment report that
summarizes the recent Smol and Qwen runs, quotes representative log lines, and
records the main failure modes and next-step hypotheses.

Detailed report:
- `docs/EXPERIMENT_REPORT_20260325.md`

Use this report when we need:
- a shareable summary for other collaborators,
- a single place that explains the architecture + dataset + loss setup,
- a chronological view of what we tried and what failed,
- or a structured explanation of why the current diagnosis is
  \"semantic stability without reliable prompt grounding\".

---

## Colleague Handoff (2026-03-25)

We now have a new single-note handoff written specifically for sending to a
teammate without asking them to open multiple notes.

Handoff note:
- `docs/COLLEAGUE_HANDOFF_20260325.md`

This version is intentionally self-contained. It includes:
- the full architecture summary,
- dataset and schedule choices,
- all major experiment families from `2026-03-15` to `2026-03-25`,
- representative quoted log lines from the real runs,
- the ladder-test findings,
- the current diagnosis,
- and the concrete next-step recommendations.

Use this note when we want:
- one document to send directly to a collaborator,
- one place to explain both the experiment history and the current state,
- or one self-contained summary that does not rely on cross-referencing the
  older notes.

---

## Overfit And Recaption Diagnostics (2026-03-25)

We now have a dedicated note for the focused diagnostic work done on
`2026-03-25` around two questions:

- can the current `SmolVLM2 -> bridge -> full DiT` stack actually overfit a
  tiny clean video set when we simplify the problem enough, and
- would regenerating OpenVid captions with `SmolVLM2-500M-Video-Instruct` make
  the prompt space materially less clustered before training?

Detailed note:
- `docs/OVERFIT_AND_RECAPTION_NOTE_20260325.md`

This note is meant to be self-contained and records:
- the exact `clean16` overfit setup,
- the earlier `init-from-10k` overfit run with online distill + semantic probe,
- the scratch `diffusion-only` overfit run,
- representative log lines from both runs,
- prompt-level inference outcomes at `step1000`, `step2000`, and final,
- the raw Smol prompt-similarity measurement on `clean16`,
- the OpenVid part-0 recaption experiment and its exact query prompt,
- concrete good/bad recaption examples,
- and the current diagnosis that the main bottleneck is more likely
  `objective / functional prompt influence` than a simple data-loading bug.

Use this note when we need a tight record of what we learned from the
`overfit` and `recaption` tracks specifically, without re-reading the broader
experiment history.

---

## Stage-3 SANA Tracking (2026-02-25)

### Current run judged "usable but not final"
- Training log: `output/logs/stage3_sana_nochunk_nochi_anticollapse_nccl_full_0224_200214.log`
- Active config: `configs/stage1_teacher_free_msrvtt_sana_nochunk_nochi_anticollapse_2gpu.yaml`
- Runtime setup:
  - 2 GPUs (`CUDA_VISIBLE_DEVICES=1,2`)
  - NCCL manual sync (`manual_sync_backend: nccl`, `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`)
  - `batch_size=1`, `grad_accum_steps=1`
  - `train_modules: [all]` for DiT
  - `chunk_index: null` (no chunk-causal split in this run)
  - `use_chi_prompt: false`
  - `cfg_dropout_prob: 0.0` (no CFG dropout)
  - `semantic_probe.enabled: true`, weight=0.2 (anti-collapse monitor/loss)

### Why this run is considered better than recent failed ones
- Training remained stable beyond 10k steps (no hard divergence).
- Bridge probe trend improved compared to earlier collapsed runs:
  - `mcp_offdiag` around `~0.50` early and drifting to `~0.48` by 10k+ steps.
  - `smol_offdiag` stayed around `~0.757` (reference baseline).
- Visual outputs are still imperfect for prompt following, but the generated videos look less degenerate and more aesthetically coherent than the immediately previous retries.

### Concrete inference checkpoints tested
- `step1000`, `step6800`, `step9000`, `step10200` were manually inferred and compared.
- Example output folders:
  - `output/infer_stage3_sana_nochunk_nochi_anticollapse_step1000_gpu7_20260225_004159`
  - `output/infer_stage3_sana_nochunk_nochi_anticollapse_step9000_gpu7_20260225_004318`
  - `output/infer_stage3_sana_nochunk_nochi_anticollapse_step10200_gpu7_dual_20260225_011728`

### Failed experiments to remember (important)
These are explicitly marked as failed/regressed and should not be reused as-is:

1. `stage3_msrvtt_sana_nochunk_nochi_2gpu_nccl_p2poff_tmux_20260224_051630.log`
   - Outcome: unstable semantic quality, weak prompt following.
2. `stage3_msrvtt_sana_nochunk_nochi_2gpu_nccl_p2poff_tmux_rerun_20260224_104652.log`
   - Outcome: repeated collapse-like behavior, visually similar outputs across prompts.
3. `stage3_msrvtt_sana_chunk_like_2gpu_nccl_p2poff_tmux_resume_step2000_20260224_010248.log`
   - Outcome: chunk-like resume did not recover quality; outputs remained weak.
4. `stage3_msrvtt_t2v_sana_parity_3gpu_resume_step3000_20260223_013700.log`
   - Outcome: regression vs previous stronger run; quality trended worse during continuation.
5. Uniform/incremental chunk strategy toggles without other safeguards
   - Outcome: no consistent visual improvement; several runs stopped due to poor qualitative trend.

### Decision snapshot
- Keep current run family as temporary "best available" baseline for further debugging.
- Do not interpret anti-collapse metrics alone as success criteria; final decision must include side-by-side prompt-level inference quality.
- Preserve failed run names/configs in this note to prevent repeating the same settings blindly.

---

## Architecture Overview

### MobileOV vs OmniVideo

**OmniVideo:**
- AR VisionHead (pretrained) → DM_Adapter (pretrained) → WAN
- Context: ~103 tokens (64 adapter + 77 T5)

**MobileOV:**
- SmolVLM2-500M → SmolVLM2VisionHead (trainable) → DM_Adapter (pretrained) → WAN
- Context: 141 tokens (64 adapter + 77 T5) - **FIXED** (was 260 tokens)

### Key Differences
1. **Understanding Module**: SmolVLM2-500M (smaller, trainable) vs AR VisionHead (pretrained)
2. **Context Length**: Now matches OmniVideo (64 tokens) after truncation fix
3. **Training**: VisionHead + Adapter trainable, WAN + SmolVLM2 frozen

---

## Key Changes Made

### 1. Adapter Output Processing (Match OmniVideo)
- **Before**: Used all 256 tokens from adapter output
- **After**: Truncate to 64 tokens to match OmniVideo
- **Files**: `nets/omni/modules/mobile_ov_model.py`
  - Main path: Extract → Truncate to 64 → Concat
  - CFG path: Unconditioned → Truncate to 64 → Concat
  - Clue-only path: Truncate to 64 → Concat

### 2. Context Concatenation Logic
- **Flow**: Extract adapter_item → Normalize to 2D → Concat with T5 → Truncate if needed
- **Match**: Identical to OmniVideo processing logic
- **Result**: Context length 64 + 77 = 141 tokens (vs OmniVideo ~103-141)

### 3. Config Update
- **Before**: `adapter.query_length: 256`
- **After**: `adapter.query_length: 64`
- **Impact**: Ready for retrain with 64 tokens

---

## Flow Fixes

### Problem Identified
- MobileOV context was 260 tokens (256 adapter + 77 T5, truncated to 260)
- OmniVideo context was ~103 tokens (64 adapter + 77 T5)
- **11.5x slower** WAN forward pass due to attention O(n²) complexity

### Solution Implemented
1. **Truncation Logic**: Added truncation to 64 tokens in all adapter output paths
2. **Config Update**: Changed `adapter.query_length` from 256 → 64
3. **Verification**: Tested forward pass, confirmed 64 tokens → 141 total context

### Expected Results
- **Context length**: 141 tokens (64 + 77) - matches OmniVideo
- **Speedup**: ~2-3x faster WAN forward pass
- **Attention ops**: ~3.5x reduction (from ~69k → ~20k)

---

## Verification Results

### Shape Verification
✅ **All shapes match from adapter onwards:**
- Adapter output (raw): `[1, 256, 4096]` (from checkpoint)
- Adapter item (truncated): `[64, 4096]` ✅
- Mixed context: `[141, 4096]` (64 + 77) ✅
- WAN input: `[141, 4096]` ✅

### Flow Verification
✅ **Processing logic matches OmniVideo:**
1. Extract adapter_item: Take `[0]` if `dim==3`
2. Truncate to 64 tokens if needed
3. Normalize to 2D `[L, C]`
4. Concatenate: `torch.cat([adapter_item, context_item], dim=0)`
5. Truncate if `max_context_len` exceeded
6. Pass to WAN as `List[[L, C]]`

### Performance Verification
- **Before fix**: WAN forward ~911ms/step (260 tokens)
- **After fix**: WAN forward ~202ms/step (71 tokens for short prompt)
- **Speedup**: ~4.5x faster!

---

## Performance Analysis

### Current Flow (After Optimizations)

1. **Initialization** (1 time): ~97s
2. **Pre-compute** (1 time): ~10s
   - T5 encoding: ~9.1s
   - SmolVLM2 encoding: ~0.94s
   - VisionHead: ~3.6ms
   - Adapter: ~3.0ms
3. **Denoising Loop** (50 steps):
   - Use pre-computed adapter output ✅
   - Context: 64 + 77 = 141 tokens ✅
   - WAN forward: ~200-300ms/step (expected)

### Bottleneck Analysis

**Root Cause**: Context length was too long (260 tokens vs ~103 in OmniVideo)

**Solution**: Truncate adapter output to 64 tokens
- Context: 260 → 141 tokens
- Attention ops: ~69k → ~20k (3.5x reduction)
- Expected speedup: ~2-3x

---

## Training Notes

### Current Training Setup
- **Config**: `configs/mobile_ov_openvid_overfit.yaml`
- **Adapter query_length**: 64 (updated)
- **use_precomputed_features**: false (must be false to train VisionHead)
- **disable_t5_context**: false (enable T5 to match OmniVideo)
- **Training**: VisionHead + Adapter (WAN + SmolVLM2 frozen)

### Next Steps
1. ✅ Code fixed: Truncation logic added
2. ✅ Config updated: `query_length: 64`
3. ⏳ **Retrain**: Train with new config
4. ⏳ **Test**: Verify context length = 141 tokens
5. ⏳ **Profile**: Measure speedup (expected 2-3x)

---

## Q1 Overfit Notes (Temporary)

We are running a **small overfit Q1 distillation** (1 prompt → 20 prompts) to prove feasibility for Mobile-OV-SANA.
This is **temporary** and **not the long-term plan**. Long-term will use larger-scale Q1/Q2 with effect distillation.
Current overfit recipe for proof-of-concept:
- Guidance scale = 1.0, fixed seed, single prompt first
- Masked embed loss + optional effect distillation (SANA transformer MSE)

---

## Files Modified

### Core Model
- `nets/omni/modules/mobile_ov_model.py`: Truncation logic added

### Config
- `configs/mobile_ov_openvid_overfit.yaml`: `query_length: 256 → 64`

### Inference
- `tools/inference/inference_trained_extract_frames.py`: Pre-compute optimization

---

*Last updated: 2025-01-17*

---

## Stage1 Teacher-Free Postmortem (2026-02-08)

### Symptoms we repeatedly observed
- Inference from different prompts looked very similar (semantic collapse).
- Training loss decreased, but prompt-following quality did not improve.
- Logs sometimes looked "silent" for a few minutes because only sync-step logs were enabled (not micro-step logs).

### Root causes confirmed by measurements
- **Not a tokenizer bug**: probe prompts produce different token ids/lengths.
- **Student embedding collapse during training**:
  - Step 0 probe offdiag cosine: ~0.8608
  - Step 50 probe offdiag cosine: ~0.9964
- **DiT became less condition-sensitive**:
  - `||pred(promptA)-pred(promptB)||` (same `z_t,t`) dropped strongly from step 0 to step 50.
- Training full DiT (`train_modules: [all]`) gave a shortcut: reduce diffusion loss while ignoring conditioning.

### Infra/training bugs fixed earlier
- Fixed multi-GPU gradient sync path:
  - Ensured DDP/FSDP/manual-sync modes are not mixed incorrectly.
  - Added guardrails for invalid multi-GPU settings.
- Fixed LoRA grad-path issue:
  - SmolVLM2 internal no-grad behavior in eval mode was bypassing LoRA updates.
  - Kept correct train/eval handling so LoRA grads flow.
- Fixed LoRA target scope:
  - Added include/exclude filtering to avoid training unused vision-side LoRA modules.
- Fixed misleading LoRA grad logs:
  - Collected grad stats before `optimizer.zero_grad()` instead of after.

### Lessons learned (actionable)
- Diffusion loss alone is underconstrained for teacher-free bridge alignment.
- Do **not** start from full DiT finetune while conditioning bridge is unstable.
- For batch size 1, batch offdiag metric is not informative; always use fixed-prompt probes.
- Keep a mandatory preflight:
  - Token hash/length sanity.
  - Probe embedding cosine (step 0 and early steps).
  - Condition sensitivity delta on same noisy latent/timestep.

### Anti-collapse stage1 smoke (student-only, 4 GPU)
- Config: `configs/stage1_teacher_free_openvid_train_stage1_studentonly_debug_4gpu.yaml`
- Core setup:
  - DiT frozen (`train_modules: []`, trainable DiT params = 0)
  - LoRA disabled
  - Gate clamp off (`gate_min_value=0`)
  - Gate loss off, norm loss off
  - bridge LR = `5e-5`, weight_decay = `0`
- Run log: `output/logs/stage1_teacher_free_stage1_studentonly_debug_4gpu_20260209_020729.log`
- Results:
  - `probe_semantic` at step 50: `0.931553` (improved vs previous ~`0.996407`)
  - `probe_semantic` at step 100: `0.973179` (regressed vs step 50 but still better than ~`0.996407`)
  - Condition sensitivity test (same `z_t,t`, prompt A vs B):
    - step 0: `l2=246.33`, `mean_abs=0.13398`
    - step 100: `l2=491.51`, `mean_abs=0.25671`
  - Interpretation: model did not collapse as severely as previous full-DiT run; however stability is not solved yet and still needs explicit anti-collapse objective/CFG training.

### New instrumentation added (2026-02-09)
- File: `tools/train_stage1_teacher_free.py`
- Added `cfg_dropout_prob`:
  - Randomly replaces training prompt with unconditioned prompt (`""`) during training micro-steps.
- Added `cfg_delta_every_steps` monitor:
  - On sync-step, computes `pred_cond` vs `pred_uncond` for same noisy latent/timestep and logs:
  - `cfg_delta pred_cond_vs_uncond l2=... mean_abs=...`
- Step log now includes realized drop ratio:
  - `cfg_drop=...`

### CFG/Delta smoke verification
- Config: `configs/stage1_teacher_free_openvid_train_stage1_studentonly_debug_4gpu.yaml`
  - `cfg_dropout_prob: 0.15`
  - `cfg_delta_every_steps: 10`
  - `cfg_delta_uncond_prompt: ""`
- Log: `output/logs/stage1_teacher_free_cfgdrop_delta_smoke_4gpu_20260209_031414.log`
- Verified lines:
  - `CFG settings: dropout_prob=0.150 ...`
  - `Step 10 | ... cfg_drop=...`
  - `Step 10 | cfg_delta pred_cond_vs_uncond ...`
  - `Step 20 | cfg_delta pred_cond_vs_uncond ...`

## OmniVideo2-Inspired Staged Recipe (2026-02-10)

### Why this update
- Keep DiT from learning the shortcut "ignore condition" too early.
- First stabilize bridge semantics, then open limited DiT modules.

### New configs added
- Stage-2 (adapter alignment, DiT frozen):
  - `configs/stage1_teacher_free_openvid_train_stage2_adapter_align_4gpu.yaml`
  - Core settings:
    - `model.dit.train_modules: []`
    - `student_ddp: true`
    - `distill.enabled: true`
    - `semantic_probe.enabled: true`
    - `cfg_dropout_prob: 0.10`
    - `norm/gate loss: off`
- Stage-3 (light DiT adaptation):
  - `configs/stage1_teacher_free_openvid_train_stage3_crossattn_4gpu.yaml`
  - Core settings:
    - `model.dit.train_modules: [cross_attn]`
    - `model.dit.ddp: true`
    - keep CFG-drop + light distill + semantic probe

### Intended usage order
1. Run stage-2 first until semantic probe is stable (offdiag not drifting to ~0.99).
2. Resume/initialize stage-3 only after stage-2 quality is acceptable.

## Failure Note (2026-02-24)

- Setting marked as failed:
  - Config: `configs/stage1_teacher_free_msrvtt_sana_nochunk_nochi_2gpu.yaml`
  - Runtime: 2-GPU NCCL (`CUDA_VISIBLE_DEVICES=1,2`, `NCCL_P2P_DISABLE=1`)
  - Log file (removed): `output/logs/stage3_msrvtt_sana_nochunk_nochi_2gpu_nccl_p2poff_tmux_20260224_051630.log`
  - Checkpoint dir (removed): `output/stage1_teacher_free_msrvtt_sana_nochunk_nochi/20260224_051638`
- Observed behavior:
  - Visual quality remained poor despite long training.
  - Prompt-following stayed weak.
  - Semantic probe trend deteriorated (MCP embeddings became increasingly collapsed).
- Action taken:
  - Stopped the run.
  - Deleted the run artifacts above.
  - Re-run started from scratch with current code for fresh validation.

## Conditioning Collapse Triage Note (2026-02-26)

- Temporary diagnostics were added to `tools/train_stage1_teacher_free.py` to debug prompt-conditioning collapse:
  - `cond_shuffle_dloss`: diffusion loss delta when condition embeddings are shuffled across batch.
  - `cond_uncond_dloss`: diffusion loss delta when replacing condition with unconditioned embedding.
  - `cond_grad`: gradient norm on conditioning embedding (`student_embeds_for_dit`).
- Config switches:
  - `run.conditioning_diag_every_steps`
  - `run.conditioning_diag_shuffle`
  - `run.conditioning_diag_uncond`
  - `run.conditioning_diag_grad`
- Cleanup intention:
  - These diagnostics are explicitly temporary and should be removed after collapse root-cause is fixed.
