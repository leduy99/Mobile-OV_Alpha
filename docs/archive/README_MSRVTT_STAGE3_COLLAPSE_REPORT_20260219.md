# MSRVTT Stage-3 Collapse Report (2026-02-19)

This document summarizes the current Mobile-OV training behavior on MSRVTT so an external expert can quickly review the exact setup, symptoms, and likely failure modes.

## 1) Experiment Scope

- Pipeline: `SmolVLM2 (LoRA) -> MCP projector -> SANA-Video DiT (cross-attn LoRA)`
- Task: text-to-video training on MSRVTT preprocessed latents
- Main concern: semantic collapse (different prompts produce increasingly similar conditioning behavior and outputs)

## 2) Runs Covered

- Base run log:
  - `output/logs/stage3_msrvtt_deepspeed_3gpu_safe_20260218_233142.log`
- Resume run log:
  - `output/logs/stage3_msrvtt_deepspeed_3gpu_safe_resume_20260219_023044.log`
- Resume source checkpoint:
  - `output/stage1_teacher_free_msrvtt_real/20260218_233150/checkpoint_step1000.pt`
- Resume output dir (new checkpoints):
  - `output/stage1_teacher_free_msrvtt_real/20260219_023052`

## 3) Exact Training Config

Config file:
- `configs/stage1_teacher_free_msrvtt_train_stage3_crossattn_deepspeed_3gpu.yaml`

Key settings:
- Distributed:
  - `world_size=3` (resume launched on GPUs `1,2,3`)
  - DiT uses DeepSpeed ZeRO-1
  - Student gradient sync uses manual all-reduce with backend `gloo`
- Precision:
  - `bf16`
- Batch:
  - per-GPU micro-batch `1`
  - grad accumulation `8`
  - global batch (`3 * 1 * 8`) = `24`
- Data:
  - CSV: `data/msrvtt/OpenVid_extracted_subset_unique.csv`
  - preprocessed latents dir: `data/msrvtt/preprocessed`
  - valid samples in log: `2990`
- Objective:
  - SANA scheduler params:
    - `train_sampling_steps=1000`
    - `noise_schedule=linear_flow`
    - `predict_flow_v=True`
    - `flow_shift=3.0`
    - `weighting_scheme=logit_normal`
    - `use_sana_process_timesteps=True`
  - `latent_window.frames=13` (from latent T=21)
- LR:
  - bridge: `5e-5`
  - DiT: `1e-5`
- Loss toggles:
  - `loss.diff.weight=1.0` (active)
  - `loss.distill.enabled=false`
  - `loss.semantic_probe.enabled=false` (only debug probe logging is active)
  - `loss.norm.enabled=false`
  - `loss.gate.enabled=false`
- CFG in training:
  - `cfg_dropout_prob=0.10`
  - `cfg_delta_every_steps=20`

## 4) What Is Trainable (Current Run)

From logs:
- Trainable bridge params: `4.74M`
- Trainable DiT params: `2.51M`

From checkpoint state composition:
- `student_state.smolvlm2_lora`: `1,638,400` params
- `student_state.projector` (MCP): `3,106,567` params
- `student_state.adapter/resampler`: not used (`0` tensors in this run)
- `dit_trainable_state` (cross-attn LoRA): `2,508,800` params

So student-side trainable path is effectively:
- SmolVLM2 LoRA + MCP projector

## 5) Behavior Observed in Logs

### 5.1 Semantic Probe Drift (critical)

`probe_semantic` trend (selected points):

| Step | mcp_offdiag_mean | smol_offdiag_mean |
|---|---:|---:|
| 100 | 0.896436 | 0.757832 |
| 500 | 0.896236 | 0.757567 |
| 1000 | 0.901724 | 0.764932 |
| 1100 | 0.902994 | 0.766997 |
| 1200 | 0.903061 | 0.768252 |
| 1400 | 0.906199 | 0.771826 |
| 1500 | 0.908422 | 0.774852 |
| 1600 | 0.911630 | 0.779807 |

Interpretation:
- Both MCP output similarity and raw Smol embedding similarity increase over time.
- This indicates collapse is not only at projector output; upstream Smol branch is also drifting toward less prompt-separable representations.

### 5.2 CFG delta trend (also concerning)

`cfg_delta_l2` decreases in later resume steps (example):
- Step 1020: `149.31`
- Step 1400: `90.50`
- Step 1600: `70.21`

Interpretation:
- Conditional vs unconditional prediction gap is shrinking, often a sign that conditioning is becoming less informative.

### 5.3 Diffusion loss does not explode

- Loss remains numerically stable (`~0.09` to `~0.26` in most later steps).
- Therefore collapse is happening while diffusion objective still optimizes, suggesting objective under-constraint for semantic separability.

## 6) Important Implementation Notes

### 6.1 Resume behavior

On resume, log reports:
- `Sharded resume mode: skipping optimizer/scheduler state load...`

Meaning:
- Model weights and step counter restored.
- Optimizer/scheduler internal state is not restored in this mode.
- This can alter training dynamics after resume.

### 6.2 Student sync mode

Current config uses:
- `run.student_ddp=false`
- manual student gradient all-reduce on `gloo`

Log warns this is slower with DeepSpeed DiT. It may not be the root cause of collapse, but it is a nonstandard path vs pure NCCL/DDP and can make debugging harder.

### 6.3 Optimizer grouping

In current code path (DeepSpeed for DiT):
- Student-side trainables (Smol LoRA + MCP projector) are optimized together by one bridge optimizer group at LR `5e-5`.
- No separate LR policy for Smol LoRA vs MCP projector in this run.

## 7) Quick Quantitative Snapshot from Checkpoints

Comparing step1000 -> step1400:
- Smol LoRA relative parameter change: ~`3.40%` of initial norm
- DiT LoRA relative parameter change: ~`3.33%`
- MCP projector relative parameter change: ~`0.74%`

Interpretation:
- Smol LoRA and DiT LoRA are moving substantially more than MCP in this interval.
- Even with smaller relative change in MCP, Smol separability still degrades.

## 8) Current Hypotheses

1. Objective is under-constrained for semantic separation.
   - Only diffusion loss active; no explicit anti-collapse semantic regularization active.
2. Joint update of Smol LoRA + MCP + DiT LoRA allows a shortcut that reduces dependence on prompt-specific directions.
3. Resume without optimizer/scheduler state may worsen drift trajectory.
4. CFG dropout at 0.1 is insufficient alone to preserve prompt geometry under this joint setup.

## 9) Questions to Ask Expert

1. Should Smol LoRA be frozen initially while training only MCP (+ maybe DiT LoRA), then unfrozen later?
2. Should bridge optimizer be split into separate LR groups:
   - very low LR for Smol LoRA
   - higher LR for MCP projector?
3. Should semantic anti-collapse loss be enabled from early stage (not just monitoring)?
4. Is two-phase training preferred here:
   - phase A: student-side alignment with DiT mostly fixed
   - phase B: controlled DiT LoRA unfreeze?
5. How important is restoring optimizer/scheduler state for resumed sharded runs in this setup?

## 10) Repro/Resume Command Used

```bash
CUDA_VISIBLE_DEVICES=1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=3 --master_port=29672 \
tools/train_stage1_teacher_free.py \
  --config configs/stage1_teacher_free_msrvtt_train_stage3_crossattn_deepspeed_3gpu.yaml \
  --max-gpus 3 \
  --resume-from output/stage1_teacher_free_msrvtt_real/20260218_233150/checkpoint_step1000.pt
```

## 11) Artifacts

- Base run checkpoints:
  - `output/stage1_teacher_free_msrvtt_real/20260218_233150/checkpoint_step{200,400,600,800,1000}.pt`
- Resume run checkpoints:
  - `output/stage1_teacher_free_msrvtt_real/20260219_023052/checkpoint_step{1200,1400,...}.pt`
- Inference comparison folder (from earlier triage):
  - `output/infer_compare_msrvtt_steps100_500_1100_20260219_020703`

