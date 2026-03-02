# Stage-3 Note: Timestep-Aware and Chunk-Causal Training (MSRVTT, 2026-02-21)

## Scope of this note
This README only explains the two mechanisms you asked about:
1. `timestep-aware` timestep generation.
2. `chunk-causal` timestep generation.

It does not cover the broader attention/capacity experiments.

## 1) What `timestep-aware` means in our training code
In `tools/train_stage1_teacher_free.py`, we now support SANA-style timestep generation through:
- `process_timesteps(...)` from `nets/third_party/sana/diffusion/model/respace.py`.

When enabled, timesteps are sampled with density-based schemes (for example `logit_normal`) and can be produced in frame-aware form.

Key control flags in training:
- `train.use_sana_process_timesteps`
- `scheduler.weighting_scheme` (must be one of `logit_normal`, `stretched_logit_normal`, `mode` for this path)
- `train.same_timestep_prob`
- `train.chunk_sampling_strategy`

Reference points:
- `tools/train_stage1_teacher_free.py:1167`
- `tools/train_stage1_teacher_free.py:1732`
- `nets/third_party/sana/diffusion/model/respace.py:320`

## 2) What `chunk-causal` means
`chunk-causal` is activated when `chunk_index` is provided.

Conceptually:
- Temporal latent frames `T` are divided into chunks by `chunk_index`.
- Each chunk gets its own timestep value(s).
- Timesteps are then repeated inside each chunk.
- This gives time-structured corruption instead of a single global timestep for all frames.

In upstream SANA logic:
- If `chunk_index` exists and random `< same_timestep_prob`, it falls back to same timestep for all frames.
- Else, it performs chunk-causal sampling using:
  - `chunk_sampling_strategy="uniform"` or
  - `chunk_sampling_strategy="incremental"`.

Reference points:
- `nets/third_party/sana/diffusion/model/respace.py:347`
- `nets/third_party/sana/diffusion/model/respace.py:358`
- `nets/third_party/sana/diffusion/model/respace.py:379`

## 3) What we changed in our Stage-3 training path
In `tools/train_stage1_teacher_free.py`:
1. Parse and normalize `chunk_index` from config.
2. If `chunk_index` is set, force `use_sana_process_timesteps=True`.
3. Call `process_timesteps(...)` with:
   - `num_frames=t`
   - `chunk_index`
   - `chunk_sampling_strategy`
   - `same_timestep_prob`
4. Pass `chunk_index` into DiT forward through `model_kwargs`.

Reference points:
- `tools/train_stage1_teacher_free.py:1171`
- `tools/train_stage1_teacher_free.py:1189`
- `tools/train_stage1_teacher_free.py:1725`
- `tools/train_stage1_teacher_free.py:1760`

## 4) Current run status (important)
From current log:
- `output/logs/stage3_msrvtt_bridge_fulldit_diffusion_3gpu_20260221_122105.log:283`

It shows:
- `use_process_timesteps=True`
- `chunk_index=None`

So the current run is:
- using timestep-aware sampling,
- but **not** using chunk-causal sampling yet (because `chunk_index` is `None`).

## 5) How to enable chunk-causal correctly
You need to set `chunk_index` in config (SANA model config or `cfg.model.dit.chunk_index`).

Important:
- `chunk_index` is in **latent frame space**, not raw video frame space.
- Example: if latent `T=13`, valid boundaries must be within `[0, 13]`.

Example idea for `T=13`:
- `chunk_index: [0, 4, 8]`
- implied chunks: `[0..3]`, `[4..7]`, `[8..12]`

Also set:
- `train.chunk_sampling_strategy`: `uniform` or `incremental`
- `train.same_timestep_prob`: usually small (for example `0.0` to `0.2`)

## 6) Train/Infer parity caveat
Training currently can pass `chunk_index` to DiT (`model_kwargs["chunk_index"]`).

In `tools/inference/test_q1_student_video.py`, `model_kwargs` currently includes:
- `data_info`
- `mask`

but does not currently include `chunk_index`.

If you train with chunk-causal and want strict parity, inference should pass the same `chunk_index` (and compatible temporal setup).

Reference point:
- `tools/inference/test_q1_student_video.py:451`

## 7) Practical verification checklist
Before long runs, verify:
1. Startup log prints:
   - `use_process_timesteps=True`
   - expected `chunk_index` value (not `None` if chunk-causal is intended)
2. Latent temporal length `T` matches your chunk design.
3. Inference path uses matching temporal setup and (if needed) `chunk_index`.

## 8) Bottom line
- We already integrated SANA-style timestep-aware sampling into Stage-3 training.
- Chunk-causal support is also integrated in training code.
- Your current run is not chunk-causal yet because `chunk_index=None`.
- To truly test chunk-causal behavior, set `chunk_index` in latent-frame coordinates and mirror it in inference.
