# MobileOV Experiment Report (2026-03-25)

## Scope

This report summarizes the recent `SmolVLM2/Qwen -> bridge -> SANA-Video`
experiments that were run between `2026-03-15` and `2026-03-25`.

The goal is to preserve:

- the shared architecture and dataset assumptions,
- the exact loss families we tried,
- representative log evidence from the real runs,
- the inference artifacts we used to judge success/failure,
- and a concrete explanation of what likely failed and what should be tested next.

This document is written to be presentation-friendly. It is intended to be read
by someone who was not present for all experiments.

## Executive Summary

The short version is:

1. The current system can remain numerically stable without obviously hard
   collapsing, but still fail badly at prompt-grounded visual generation.
2. `semantic stability != correct text-to-visual alignment` in this project.
3. The most trustworthy historical family is still the `SmolVLM2-500M + bridge
   + full DiT + online teacher distill + semantic_probe + 1V:1I` regime.
4. `bridge-only` was not sufficient for `SmolVLM2-2.2B`.
5. `Qwen3-VL` did not solve the problem. It often looked semantically healthier
   than Smol in bridge-only settings, but still failed visually.
6. Opening full DiT on top of the Qwen bridge checkpoint caused severe prompt
   manifold drift and made inference worse, not better.
7. The current Smol `online_teacher_30k` reproduction remains the closest thing
   to a plausible recipe family, but it is still not visually aligned enough at
   `5k` to count as a successful reproduction.

## Shared Architecture

### Text-conditioning pipeline

All recent experiments use the same high-level structure:

1. A text backbone produces prompt-side hidden states.
2. A prompt bridge/projector maps those hidden states into the token space that
   SANA expects.
3. SANA-Video consumes the projected prompt tokens as conditioning.
4. Wan VAE decodes video latents into frames.

Core implementation files:

- `tools/train_stage1_teacher_free.py`
- `nets/omni/modules/sana_prompt_bridge.py`
- `nets/omni/modules/sana_prompt_bridge_qwen3vl.py`
- `configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp_nocfgdrop.yaml`

### Projector shape

The common conditioning target is:

- `out_seq_len = 300`
- `out_dim = 2304`

For the main Smol reproduction run:

- projector type: `mcp_full`
- `mcp_hidden_dim = 1536`
- `mcp_num_fuse_layers = 4`
- `mcp_use_refine = true`

This means the prompt bridge is not a single linear head. It is a multi-stage
sequence projector that produces the final `300 x 2304` prompt tokens consumed
by SANA cross-attention.

### Video latent contract

The training and inference pipeline is using:

- `frame_num = 81`
- `latent_t = 21`
- `vae_stride_t = 4`

This is visible in both training/inference hints and inference logs. In other
words, the model generates full temporal latents first and then decodes them to
`81` frames.

### SANA objective

The SANA objective in these runs is not classic DDPM epsilon prediction.
It is the flow-style objective configured as:

- `predict_flow_v: true`
- `noise_schedule: linear_flow`

from:

- `configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp_nocfgdrop.yaml`

So when logs print `diff=...`, that is the flow-matching diffusion loss term.

## Shared Dataset Setup

### Main training manifest family

Most of the recent runs use the mixed OpenVid + current LAION/COYO manifest
family, for example:

- `data/mix/manifests/joint_openvid_current_laion_coyo_20260315.csv`
- `data/mix/manifests/joint_openvid_current_laion_coyo_20260315_video.csv`
- `data/mix/manifests/joint_openvid_current_laion_coyo_20260315_image.csv`

### Modalities

The training pipeline supports joint video+image training via:

- `video_modality = video`
- `image_modality = image`

### Schedules we used

Two schedules matter in this report:

- `1V:1I`
  - configured with `image_per_video: 1`, `interval: 0`
- `5V:1I`
  - configured with `image_per_video: null`, `interval: 5`

The scheduler logic lives in `tools/train_stage1_teacher_free.py`.

### Prompt preprocessing

Shared preprocessing knobs used in most runs:

- `normalize_whitespace = true`
- `strip = true`
- `remove_double_newlines = true`
- `use_chi_prompt = false`
- `use_prompt_templates = false`

One important caveat discovered during the Qwen work:

- if `max_prompt_tokens` is set, prompt truncation happens using the student
  tokenizer before the prompt is passed to the online teacher.

This is less problematic for Smol-to-Gemma-style distillation than for
Qwen-to-Gemma-style distillation, but it is still an architectural caveat worth
remembering.

## Loss Families Used

### 1. Diffusion / flow loss

Present in all runs:

- `loss.diff.weight = 1.0`

This is the only loss directly tied to the SANA denoising objective.

### 2. Token-level online teacher distillation

When enabled, the teacher path adds:

- `token_mse_weight`
- `token_cos_weight`
- `pooled_cos_weight`

In the canonical Smol online-teacher family, this was:

- `token_mse_weight = 1.0`
- `token_cos_weight = 0.5`
- `pooled_cos_weight = 0.2`

### 3. Semantic anti-collapse probe

When enabled, this adds:

- variance term (`sem_var`)
- covariance term (`sem_cov`)
- geometry term (`sem_geom`)

Important nuance:

- `sem_var`, `sem_cov`, and `sem_geom` in logs are loss terms, not raw feature
  statistics.
- Lower is usually better, but only within context.

### 4. Representation-only distillation variant

For one Qwen bridge-only run, tokenwise losses were disabled and only pooled
teacher alignment was kept:

- `token_mse_weight = 0.0`
- `token_cos_weight = 0.0`
- `pooled_cos_weight = 0.3`
- `semantic_probe.geom_source = teacher`

This was an attempt to avoid ill-posed token-to-token matching between Qwen and
Gemma teacher representations.

## Experiment Matrix

| ID | Date | Backbone | Trainable Modules | Schedule | Loss Family | Main Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| A | 2026-03-15/16 | SmolVLM2-500M | bridge + full DiT | 1V:1I | online distill + semantic_probe, then nodistill phase 2 | historical best family |
| B | 2026-03-23 | SmolVLM2-2.2B | bridge only | 1V:1I | online distill + semantic_probe | no visual payoff |
| C | 2026-03-24 | Qwen3-VL 2B | bridge only | 5V:1I | online distill | better loss than Qwen 1V:1I, still poor visuals |
| D | 2026-03-24 | Qwen3-VL 2B | bridge only | 5V:1I | pooled-only distill + teacher-geometry semantic_probe | semantically healthier, still visually poor |
| E | 2026-03-24 | Qwen3-VL 2B | bridge + full DiT | 5V:1I | diffusion only | numerically stable, no visual breakthrough |
| F | 2026-03-24 | Qwen3-VL 2B | bridge + full DiT, init from D@10k | 5V:1I | diffusion + semantic_probe | severe manifold drift, worse inference |
| G | 2026-03-24/25 | SmolVLM2-500M | bridge + full DiT | 1V:1I | online distill + semantic_probe | healthiest current family, but still not aligned enough by 5k |

## Detailed Experiment Notes

## A. Historical Visual Baseline Family (2026-03-15 / 2026-03-16)

### What it was

This is the family we keep referring back to when we say "the old visually
strong line".

It consisted of two related phases:

1. `2026-03-15 online_teacher_30k`
   - `SmolVLM2-500M`
   - bridge trainable
   - full DiT trainable
   - online teacher distill
   - semantic probe on
   - `1V:1I`
2. `2026-03-16 initfrom30k_nodistill`
   - initialized from the previous line
   - same backbone and full DiT regime
   - distill off
   - semantic probe still on

### Why it mattered

This family remains the strongest historical reference point because:

- it kept DiT fully trainable,
- it maintained low semantic_probe losses for a long time,
- and later visually strong outputs are most plausibly descendants of this
  family rather than of any recent bridge-only run.

### Representative log evidence: `2026-03-16 initfrom30k_nodistill`

File:

- `output/logs/train_bridge_dit_openvid_current_initfrom30k_nodistill_300k_fsdpfix_20260316.log`

Key lines:

```text
2026-03-16 14:30:30,887 - INFO - Trainable params bridge: 3.14M
2026-03-16 14:30:30,887 - INFO - Trainable params DiT: 1028.43M
2026-03-16 14:30:30,887 - INFO - Semantic anti-collapse enabled: weight=0.2000 var=1.000 cov=0.050 geom=1.000 every=5 prompts=6 target_std=1.00
```

```text
2026-03-16 14:57:51,855 - INFO - Step 1000 | mode=image loss=0.160919 diff=0.061279 d_mse=0.000000 d_cos=0.000000 d_pool=0.000000 ... sem_var=0.441992 sem_cov=0.032235 sem_geom=0.054595 ...
```

```text
2026-03-16 16:47:14,213 - INFO - Step 5000 | mode=image loss=0.202747 diff=0.108398 d_mse=0.000000 d_cos=0.000000 d_pool=0.000000 ... sem_var=0.401124 sem_cov=0.041020 sem_geom=0.068566 ...
```

```text
2026-03-16 19:04:27,541 - INFO - Step 10000 | mode=image loss=0.227218 diff=0.133789 d_mse=0.000000 d_cos=0.000000 d_pool=0.000000 ... sem_var=0.395141 sem_cov=0.042287 sem_geom=0.069891 ...
```

### What we learn from it

- This family did **not** rely on token distill forever.
- Once the upstream `30k` online-teacher stage had created a useful init,
  phase 2 could continue with `diff + semantic_probe` only.
- Crucially, even in nodistill mode, the bridge manifold did not distort
  catastrophically.

## B. SmolVLM2-2.2B Bridge-Only + Online Teacher + Anti-Collapse (2026-03-23)

### What it was

This run tested whether simply replacing `SmolVLM2-500M` with `SmolVLM2-2.2B`
would improve bridge-only distillation.

Config family:

- `configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_only_smolvlm2_2p2b_online_teacher_20k_2gpu_20260323.yaml`

### Representative log evidence

File:

- `output/logs/train_bridgeonly_2p2b_online_teacher_20k_gpu45_20260323.log`

```text
2026-03-23 12:24:39,418 - INFO - Distill enabled: mode=online_only precomputed_dir=<none> token_mse=1.000 token_cos=0.500 pooled_cos=0.200 every_steps=2 ...
2026-03-23 12:24:39,674 - INFO - Trainable params bridge: 8.93M
2026-03-23 12:24:39,674 - INFO - Trainable params DiT: 0.00M
2026-03-23 12:24:39,674 - INFO - Semantic anti-collapse enabled: weight=0.2000 var=1.000 cov=0.050 geom=1.000 every=5 prompts=6 target_std=1.00
```

```text
2026-03-23 12:25:03,053 - INFO - Step 20 | mode=image loss=2.984797 diff=0.198242 d_mse=1.965047 d_cos=0.982529 d_pool=0.964622 ... sem_var=0.645924 sem_cov=0.005996 sem_geom=0.040368 ...
```

```text
2026-03-23 12:34:08,214 - INFO - Step 500 | mode=image loss=1.437893 diff=0.077148 d_mse=0.947656 d_cos=0.473831 d_pool=0.253763 ... sem_var=0.616614 sem_cov=0.007620 sem_geom=0.010104 ...
```

```text
2026-03-23 12:43:36,187 - INFO - Step 1000 | mode=image loss=1.182952 diff=0.063965 d_mse=0.776341 d_cos=0.388172 d_pool=0.192317 ... sem_var=0.518916 sem_cov=0.018516 sem_geom=0.030643 ...
```

```text
2026-03-23 13:59:22,716 - INFO - Step 5000 | mode=image loss=1.493147 diff=0.376953 d_mse=0.787435 d_cos=0.393720 d_pool=0.159310 ... sem_var=0.441124 sem_cov=0.031448 sem_geom=0.057486 ...
```

### Inference status

Output folder:

- `output/inference_bridgeonly_2p2b_step5000_fixed_20260323`

Key files:

- `.../q1_student_20260323_140906_a_golden_retriever_running_along_a_beach.mp4`
- `.../q1_student_20260323_141018_a_chef_slicing_colorful_vegetables_in_a_.mp4`

### Interpretation

This line taught us an important lesson:

- teacher distill can improve matching losses,
- semantic collapse can remain under control,
- but bridge-only is still not enough to produce convincing visuals.

This was one of the clearest cases where optimization improved while actual
video quality remained poor.

## C. Qwen3-VL Bridge-Only + Online Teacher, 5V:1I (2026-03-24)

### What it was

This was the first Qwen run that looked better than the naive `1V:1I` Qwen run
on loss curves.

Config family:

- `configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_only_qwen3vl_2b_online_teacher_openended_5v1i_2gpu_20260324.yaml`

### Representative log evidence

File:

- `output/logs/train_bridgeonly_qwen3vl_2b_online_teacher_openended_5v1i_gpu56_20260324.log`

```text
2026-03-24 01:20:43,779 - INFO - Distill enabled: mode=online_only precomputed_dir=<none> token_mse=1.000 token_cos=0.500 pooled_cos=0.200 every_steps=2 ...
2026-03-24 01:20:44,084 - INFO - Trainable params bridge: 8.93M
2026-03-24 01:20:44,084 - INFO - Trainable params DiT: 0.00M
```

```text
2026-03-24 01:21:19,669 - INFO - Step 20 | mode=video loss=2.914873 diff=0.218750 d_mse=1.997604 d_cos=0.998808 d_pool=0.995571 ...
```

```text
2026-03-24 01:34:40,708 - INFO - Step 500 | mode=video loss=1.364605 diff=0.085449 d_mse=0.991557 d_cos=0.495782 d_pool=0.198540 ...
```

```text
2026-03-24 01:48:34,679 - INFO - Step 1000 | mode=video loss=0.961235 diff=0.100586 d_mse=0.678573 d_cos=0.339289 d_pool=0.062158 ...
```

```text
2026-03-24 03:39:44,884 - INFO - Step 5000 | mode=video loss=0.970876 diff=0.122070 d_mse=0.668867 d_cos=0.334435 d_pool=0.063607 ...
```

### Interpretation

This run looked healthier than the Qwen `1V:1I` bridge-only run in terms of the
raw loss curves. However, it still did not deliver good prompt-grounded visual
behavior.

The important lesson here was:

- Qwen can look fine numerically,
- and even benefit from a more video-heavy schedule,
- while still not mapping into a SANA-friendly conditioning manifold.

## D. Qwen3-VL Bridge-Only + Representation Distill + Teacher Geometry (2026-03-24)

### Why this variant existed

We became uncomfortable with tokenwise MSE/cosine distillation between:

- Qwen backbone outputs, and
- Gemma/SANA teacher outputs.

So we tried a representation-level variant:

- `token_mse_weight = 0.0`
- `token_cos_weight = 0.0`
- `pooled_cos_weight = 0.3`
- `semantic_probe.geom_source = teacher`

Config family:

- `configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_only_qwen3vl_2b_reprdistill_teachergeom_5v1i_2gpu_20260324.yaml`

### Representative log evidence

File:

- `output/logs/train_bridgeonly_qwen3vl_2b_reprdistill_teachergeom_5v1i_gpu56_20260324.log`

```text
2026-03-24 13:35:19,835 - INFO - Distill enabled: mode=online_only precomputed_dir=<none> token_mse=0.000 token_cos=0.000 pooled_cos=0.300 every_steps=1 ...
2026-03-24 13:35:20,111 - INFO - Trainable params bridge: 8.93M
2026-03-24 13:35:20,111 - INFO - Trainable params DiT: 0.00M
2026-03-24 13:35:20,111 - INFO - Semantic anti-collapse enabled: weight=0.1500 var=1.000 cov=0.050 geom=1.000 source=teacher every=5 prompts=6 target_std=1.00
```

```text
2026-03-24 13:49:28,602 - INFO - Step 500 | mode=video loss=0.211921 diff=0.085449 d_mse=0.000000 d_cos=0.000000 d_pool=0.120562 ... sem_var=0.555077 sem_cov=0.014601 sem_geom=0.046217 ...
```

```text
2026-03-24 14:03:35,352 - INFO - Step 1000 | mode=video loss=0.195027 diff=0.099609 d_mse=0.000000 d_cos=0.000000 d_pool=0.044461 ... sem_var=0.483504 sem_cov=0.023350 sem_geom=0.062526 ...
2026-03-24 14:03:35,428 - INFO - Step 1000 | probe_semantic mcp_offdiag(mean/min/max)=0.598090/0.496989/0.714471 smol_offdiag=0.225709 prompts=6 ...
```

```text
2026-03-24 15:56:25,130 - INFO - Step 5000 | mode=video loss=0.215518 diff=0.120605 d_mse=0.000000 d_cos=0.000000 d_pool=0.063784 ... sem_var=0.417475 sem_cov=0.034354 sem_geom=0.085993 ...
2026-03-24 15:56:25,197 - INFO - Step 5000 | probe_semantic mcp_offdiag(mean/min/max)=0.548256/0.477632/0.617437 smol_offdiag=0.225709 prompts=6 ...
```

```text
2026-03-24 18:17:32,284 - INFO - Step 10000 | mode=video loss=0.163449 diff=0.076660 d_mse=0.000000 d_cos=0.000000 d_pool=0.039573 ... sem_var=0.409945 sem_cov=0.035809 sem_geom=0.087712 ...
2026-03-24 18:17:32,352 - INFO - Step 10000 | probe_semantic mcp_offdiag(mean/min/max)=0.544702/0.479203/0.605367 smol_offdiag=0.225709 prompts=6 ...
```

### Interpretation

This run is important because it separated two ideas:

- prompt manifold stability,
- and actual video quality.

What improved:

- anti-collapse behaved much better than in later Qwen full-DiT phase 2,
- prompt-space drift was controlled,
- and the bridge stayed meaningfully spread.

What did **not** improve:

- video quality remained weak,
- and object/action alignment stayed poor.

This was the strongest evidence that "better semantic geometry alone" was still
not enough.

## E. Qwen3-VL + Full DiT + Diffusion Only (2026-03-24)

### What it was

We then tested the opposite extreme:

- keep Qwen,
- open full DiT,
- remove teacher distill,
- optimize only diffusion.

Config family:

- `configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_dit_qwen3vl_2b_diffonly_openended_5v1i_2gpu_20260324.yaml`

### Representative log evidence

File:

- `output/logs/train_bridge_dit_qwen3vl_2b_diffonly_openended_5v1i_gpu56_20260324.log`

```text
2026-03-24 04:51:52,648 - INFO - Trainable params bridge: 8.93M
2026-03-24 04:51:52,649 - INFO - Trainable params DiT: 1028.43M
```

```text
2026-03-24 04:52:40,178 - INFO - Step 20 | mode=video loss=0.218750 diff=0.218750 d_mse=0.000000 d_cos=0.000000 d_pool=0.000000 ... grad=24.4564 ...
```

```text
2026-03-24 05:10:39,817 - INFO - Step 500 | mode=video loss=0.084473 diff=0.084473 d_mse=0.000000 d_cos=0.000000 d_pool=0.000000 ... grad=0.0428 ...
```

```text
2026-03-24 05:29:23,297 - INFO - Step 1000 | mode=video loss=0.099121 diff=0.099121 d_mse=0.000000 d_cos=0.000000 d_pool=0.000000 ... grad=0.0609 ...
```

```text
2026-03-24 07:59:15,630 - INFO - Step 5000 | mode=video loss=0.119629 diff=0.119629 d_mse=0.000000 d_cos=0.000000 d_pool=0.000000 ... grad=0.1077 ...
```

```text
2026-03-24 11:06:55,009 - INFO - Step 10000 | mode=video loss=0.075684 diff=0.075684 d_mse=0.000000 d_cos=0.000000 d_pool=0.000000 ... grad=0.0361 ...
```

### Interpretation

This run was numerically clean and stable.

However, it taught us another important lesson:

- simply opening full DiT and optimizing diffusion does not guarantee good
  prompt-grounded generation,
- and removing distill entirely may allow the system to converge to a generic
  denoising solution that underuses the text conditioner.

## F. Qwen Phase 2: Init-from-10k Bridge Checkpoint + Full DiT + Semantic Probe (2026-03-24)

### What it was

This phase started from the best-looking Qwen bridge-only checkpoint and then
opened full DiT.

Config family:

- `configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_dit_qwen3vl_2b_initfrom10k_semprobe_5v1i_2gpu_20260324.yaml`

The idea was simple:

- stage 1: get a semantically healthy bridge,
- stage 2: let DiT learn visual prior on top of that bridge.

### Representative log evidence

File:

- `output/logs/train_bridge_dit_qwen3vl_2b_initfrom10k_semprobe_5v1i_gpu56_20260324.log`

```text
2026-03-24 19:22:00,130 - INFO - Trainable params bridge: 8.93M
2026-03-24 19:22:00,130 - INFO - Trainable params DiT: 1028.43M
2026-03-24 19:22:00,130 - INFO - Semantic anti-collapse enabled: weight=0.2000 var=1.000 cov=0.050 geom=1.000 source=raw every=5 prompts=6 target_std=1.00
```

```text
2026-03-24 19:22:47,644 - INFO - Step 20 | mode=video loss=0.225482 diff=0.107422 ... sem_var=0.409916 sem_cov=0.035816 sem_geom=0.178596 ...
```

```text
2026-03-24 19:59:43,491 - INFO - Step 1000 | mode=video loss=0.136091 diff=0.098633 ... sem_var=0.147498 sem_cov=0.195611 sem_geom=0.030014 ...
2026-03-24 19:59:43,565 - INFO - Step 1000 | probe_semantic mcp_offdiag(mean/min/max)=0.054095/-0.227995/0.461172 smol_offdiag=0.225709 prompts=6 ...
```

```text
2026-03-24 20:37:25,515 - INFO - Step 2000 | mode=video loss=0.169216 diff=0.133789 ... sem_var=0.128975 sem_cov=0.210710 sem_geom=0.037625 ...
2026-03-24 20:37:25,582 - INFO - Step 2000 | probe_semantic mcp_offdiag(mean/min/max)=0.031983/-0.246216/0.450344 smol_offdiag=0.225709 prompts=6 ...
```

```text
2026-03-24 22:30:29,732 - INFO - Step 5000 | mode=video loss=0.154068 diff=0.120117 ... sem_var=0.117908 sem_cov=0.220159 sem_geom=0.040838 ...
2026-03-24 22:30:29,799 - INFO - Step 5000 | probe_semantic mcp_offdiag(mean/min/max)=0.023711/-0.253800/0.438890 smol_offdiag=0.225709 prompts=6 ...
```

### Interpretation

This was one of the most important negative results in the whole sequence.

It showed that:

- opening full DiT on top of a "semantically healthy" Qwen bridge did **not**
  preserve that semantic structure,
- bridge/prompt space drifted heavily,
- `sem_cov` became very large,
- prompt similarities collapsed toward near-zero or even negative pairwise
  cosine in the probe,
- and inference got worse.

This is the clearest evidence that phase-2 bridge drift is a real failure mode.

## G. Current SmolVLM2-500M Reproduction of the Old Online-Teacher Family (2026-03-24/25)

### What it is

This is the current run we restarted in order to recover the old recipe family
as faithfully as possible.

Config:

- `configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_dit_online_teacher_30k_repro_2gpu_20260324.yaml`

This run matches the old family in the important ways:

- `SmolVLM2-500M`
- bridge trainable
- full DiT trainable
- online teacher distill enabled
- semantic anti-collapse enabled
- `1V:1I`
- `grad_accum_steps = 1`

### Representative log evidence

File:

- `output/logs/train_bridge_dit_smolvlm2_500m_online_teacher_30k_repro_gpu56_20260324.log`

```text
2026-03-24 23:13:39,015 - INFO - Distill enabled: mode=online_only precomputed_dir=<none> token_mse=1.000 token_cos=0.500 pooled_cos=0.200 every_steps=2 ...
2026-03-24 23:13:39,259 - INFO - Trainable params bridge: 3.14M
2026-03-24 23:13:39,260 - INFO - Trainable params DiT: 1028.43M
2026-03-24 23:13:39,260 - INFO - Semantic anti-collapse enabled: weight=0.2000 var=1.000 cov=0.050 geom=1.000 source=raw every=5 prompts=6 target_std=1.00
```

```text
2026-03-24 23:14:12,684 - INFO - Step 20 | mode=image loss=3.134085 diff=0.248047 d_mse=2.031505 d_cos=1.015758 d_pool=1.033111 ... sem_var=0.636340 sem_cov=0.006713 sem_geom=0.063481 ...
```

```text
2026-03-24 23:27:25,550 - INFO - Step 500 | mode=image loss=2.289654 diff=0.108887 d_mse=1.542668 d_cos=0.771338 d_pool=0.602013 ... sem_var=0.649113 sem_cov=0.005621 sem_geom=0.010745 ...
```

```text
2026-03-24 23:41:11,453 - INFO - Step 1000 | mode=image loss=1.579919 diff=0.075195 d_mse=1.053674 d_cos=0.526840 d_pool=0.323606 ... sem_var=0.606105 sem_cov=0.008717 sem_geom=0.008001 ...
```

```text
2026-03-24 23:55:12,055 - INFO - Step 1500 | mode=image loss=1.419671 diff=0.103516 d_mse=0.921794 d_cos=0.460900 d_pool=0.229577 ... sem_var=0.571951 sem_cov=0.012031 sem_geom=0.017428 ...
```

```text
2026-03-25 00:09:01,161 - INFO - Step 2000 | mode=image loss=1.413698 diff=0.130859 d_mse=0.902991 d_cos=0.451498 d_pool=0.189540 ... sem_var=0.560545 sem_cov=0.013290 sem_geom=0.019745 ...
```

```text
2026-03-25 01:04:33,599 - INFO - Step 4000 | mode=image loss=1.489145 diff=0.154297 d_mse=0.947718 d_cos=0.473862 d_pool=0.186895 ... sem_var=0.538512 sem_cov=0.016013 sem_geom=0.024789 ...
```

```text
2026-03-25 01:32:20,055 - INFO - Step 5000 | mode=image loss=1.408012 diff=0.113281 d_mse=0.916113 d_cos=0.458059 d_pool=0.186312 ... sem_var=0.535549 sem_cov=0.016386 sem_geom=0.025261 ...
```

### Interpretation

This run currently looks like the healthiest *training dynamics* among the
recent experiments:

- semantic_probe losses remain small and stable,
- the system is not showing the same bridge manifold drift seen in Qwen phase 2,
- and the overall regime is much closer to the historical baseline family.

However, inference by `5k` is still not strong enough.

The current evidence suggests:

- semantic stability is real,
- prompt diversity is real,
- but object/action grounding is still weak.

So this run is promising enough to continue, but not successful enough yet to
be declared a reproduction win.

## Quick Inference Proxy Table

The table below uses a simple CLIP text-image score on `frame000` as a weak
prompt-alignment proxy.

This metric is **not** definitive. It is included only to provide a consistent,
compact numerical comparison across runs.

| Model / run | Golden retriever | Chef |
| --- | ---: | ---: |
| Base SANA (same fixed/12-step infer regime) | 0.3620 | 0.3120 |
| Smol 2.2B bridge-only @5k | 0.2069 | 0.1933 |
| Qwen bridge-only @5k | 0.1570 | 0.2303 |
| Qwen repr-distill bridge-only @5k | 0.2047 | 0.1730 |
| Qwen phase2 full-DiT @5k | 0.1332 | 0.1146 |
| Smol 500M repro @5k | 0.0321 | 0.1753 |

### How to read this table

- Base SANA is much better than all student runs.
- Qwen bridge-only and Qwen repr-distill each looked "best" on one of the two
  prompts, but neither generalized.
- Qwen phase 2 was clearly worse.
- The current Smol repro is semantically healthier than Qwen phase 2 but still
  unstable in prompt-grounded quality at `5k`.

This table supports the qualitative judgment we already had from watching the
videos:

- many runs preserve diversity,
- but still fail at correct prompt binding.

## Main Failure Patterns Observed

## 1. Stable losses do not guarantee usable videos

Several runs improved numerically while still producing poor or weakly aligned
outputs.

This was especially clear for:

- `SmolVLM2-2.2B bridge-only`, and
- `Qwen bridge-only` variants.

## 2. Bridge-only is not enough for harder backbone swaps

This is the strongest current conclusion.

For both:

- `SmolVLM2-2.2B`, and
- `Qwen3-VL`,

bridge-only training was not sufficient to produce high-quality prompt-grounded
video generation.

The bridge can remain stable and still fail to form the exact local conditioning
structure that SANA cross-attention needs.

## 3. Qwen semantic health did not translate into visual usefulness

Qwen often looked promising because:

- collapse did not appear catastrophic,
- prompt space sometimes looked more spread,
- and representation-level losses could be stabilized.

But the actual videos showed that this did not mean the conditioner was useful
for the generator.

## 4. Full DiT can make things worse if the bridge drifts

The Qwen phase-2 experiment showed a new failure mode:

- as soon as full DiT was opened, the bridge/prompt manifold could drift so far
  that inference quality degraded further.

This suggests that later-stage training must control bridge drift explicitly.

## 5. Token-level distill is suspicious across mismatched token spaces

The Qwen experiments strongly suggest that tokenwise teacher distillation is not
well justified when:

- student and teacher tokenization differ,
- sequence structures differ,
- and the student path includes a strong MCP fusion step.

That does not prove token distill is always wrong, but it does make it a poor
first choice for Qwen-to-Gemma alignment.

## 6. CFG mismatch remains a plausible secondary issue

The main Smol reproduction config still uses:

- `cfg_dropout_prob = 0.0`

while inference often uses:

- `cfg_scale = 3.0`

This alone does not explain the failures, but it remains a plausible secondary
source of degradation and should be tested explicitly.

## What We Currently Believe

Based on the experiments above, the current working belief is:

1. The core issue is **not** simple numerical collapse.
2. The core issue is **semantic grounding failure**:
   - the model can output diverse videos,
   - while still not binding object/action/layout correctly to the prompt.
3. The most likely places where things go wrong are:
   - the bridge objective,
   - prompt/teacher alignment assumptions,
   - and bridge drift once full DiT is opened.

## Recommended Diagnostic Ladder

These are the next experiments that would most cleanly separate causes.

### 1. Prompt sensitivity test

Use the same checkpoint, same seed, and compare:

- correct prompt,
- empty prompt,
- wrong prompt,
- object-swapped prompt.

Goal:

- determine whether the model is ignoring prompts,
- or using prompts but binding them incorrectly.

### 2. CFG sweep

For the current Smol repro checkpoint, compare:

- `cfg_scale = 1.0`
- `cfg_scale = 2.0`
- `cfg_scale = 3.0`

Goal:

- determine whether inference CFG is amplifying a train/infer mismatch.

### 3. Teacher-vs-student conditioning substitution test

Hold `x_t` and `t` fixed and compare DiT behavior under:

- teacher conditioning,
- student conditioning.

Goal:

- if teacher works and student fails, the main problem is bridge/objective.
- if both fail, the main problem is DiT drift or broader training dynamics.

### 4. Tiny overfit test

Train on a tiny handpicked subset of very explicit prompts, for example:

- golden retriever on beach,
- red car on coastal road,
- barista pouring latte art,
- chef slicing vegetables.

Goal:

- if the system cannot even overfit a tiny explicit subset, the recipe or
  objective is wrong at a basic level.

### 5. Bridge drift ablation in phase 2

If we reopen full DiT again, try:

- freezing bridge for the first `1k-2k` phase-2 steps, or
- dropping bridge LR by `10x`.

Goal:

- preserve the stage-1 conditioner while letting DiT adapt first.

## Practical Conclusion

If we have to summarize the entire recent sequence in one sentence, it is this:

> We can stabilize the prompt manifold more easily than we can make SANA use it
> correctly.

The most useful operational conclusion is:

- keep the current Smol reproduction running longer,
- do not make another large recipe jump yet,
- and use the diagnostic ladder above to decide whether the next fix belongs in
  inference, distill objective, or phase-2 bridge/DiT dynamics.

## Reference Files

### Main configs

- `configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_dit_online_teacher_30k_2gpu_20260315.yaml`
- `configs/archive/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_dit_300k_initfrom30k_nodistill_2gpu_20260316.yaml`
- `configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_only_smolvlm2_2p2b_online_teacher_20k_2gpu_20260323.yaml`
- `configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_only_qwen3vl_2b_online_teacher_openended_5v1i_2gpu_20260324.yaml`
- `configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_only_qwen3vl_2b_reprdistill_teachergeom_5v1i_2gpu_20260324.yaml`
- `configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_dit_qwen3vl_2b_diffonly_openended_5v1i_2gpu_20260324.yaml`
- `configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_dit_qwen3vl_2b_initfrom10k_semprobe_5v1i_2gpu_20260324.yaml`
- `configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_dit_online_teacher_30k_repro_2gpu_20260324.yaml`

### Main logs

- `output/logs/train_bridge_dit_openvid_current_initfrom30k_nodistill_300k_fsdpfix_20260316.log`
- `output/logs/train_bridgeonly_2p2b_online_teacher_20k_gpu45_20260323.log`
- `output/logs/train_bridgeonly_qwen3vl_2b_online_teacher_openended_5v1i_gpu56_20260324.log`
- `output/logs/train_bridgeonly_qwen3vl_2b_reprdistill_teachergeom_5v1i_gpu56_20260324.log`
- `output/logs/train_bridge_dit_qwen3vl_2b_diffonly_openended_5v1i_gpu56_20260324.log`
- `output/logs/train_bridge_dit_qwen3vl_2b_initfrom10k_semprobe_5v1i_gpu56_20260324.log`
- `output/logs/train_bridge_dit_smolvlm2_500m_online_teacher_30k_repro_gpu56_20260324.log`

### Key inference folders

- `output/inference_bridgeonly_2p2b_step5000_fixed_20260323`
- `output/inference_bridgeonly_qwen3vl_2b_step5000_fixed12_20260324`
- `output/inference_bridgeonly_qwen3vl_2b_reprdistill_teachergeom_step5000_fixed12_20260324`
- `output/inference_bridge_dit_qwen3vl_2b_initfrom10k_semprobe_step5000_fixed12_20260324`
- `output/inference_bridge_dit_smolvlm2_500m_online_teacher_30k_repro_step5000_fixed12_20260325`
- `output/inference_sana_base_fixed12_20260324_prompt1`
- `output/inference_sana_base_fixed12_20260324_prompt2`
