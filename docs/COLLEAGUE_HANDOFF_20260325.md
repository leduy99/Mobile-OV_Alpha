# MobileOV Text-to-Video Experiment Handoff (2026-03-25)

## Purpose

This document is a self-contained summary of the recent MobileOV text-to-video
experiments. It is written so a teammate can understand:

- what architecture we are training,
- what data and schedules we used,
- which losses we tried,
- what actually happened in the important runs,
- what evidence we have from logs and inference,
- and what the current failure diagnosis is.

This note does not assume the reader has already read earlier notes.

## Executive Summary

The project is currently in a state where:

1. The system can be numerically stable and avoid obvious hard semantic
   collapse, yet still fail badly at prompt-grounded visual generation.
2. The most promising family remains the historical SmolVLM2-500M family:
   bridge + full DiT + online teacher distill + semantic anti-collapse +
   `1V:1I`.
3. Bridge-only training was not enough for SmolVLM2-2.2B.
4. Qwen3-VL looked semantically healthier in some bridge-only settings, but it
   did not solve the visual grounding problem.
5. Opening full DiT on top of the Qwen bridge checkpoint made the prompt
   manifold drift badly and made inference worse.
6. The current SmolVLM2-500M reproduction is the closest candidate to a stable
   recipe, but even there the main failure is still weak or misdirected prompt
   binding rather than clean object/action alignment.

The strongest current diagnosis is:

- the model is not purely ignoring text,
- the model is not in catastrophic collapse,
- the bridge is not functionally far from teacher inside DiT,
- but the effective prompt influence on DiT is too weak to reliably steer
  generation toward the requested object/action.

## System Architecture

All recent experiments use the same high-level pipeline:

1. A text backbone produces prompt-side hidden states.
2. A bridge or projector maps those hidden states into the prompt-token space
   expected by SANA-Video.
3. SANA-Video predicts flow/denoising updates conditioned on those prompt
   tokens.
4. Wan VAE decodes latents into final video frames.

The key architectural variants were:

- `SmolVLM2-500M`
- `SmolVLM2-2.2B`
- `Qwen3-VL-Embedding-2B`

The common target prompt shape for SANA conditioning is:

- sequence length: `300`
- hidden size: `2304`

In the main Smol reproduction line, the projector is a multi-stage MCP-style
projector rather than a single linear head:

- projector type: `mcp_full`
- hidden dim: `1536`
- fuse layers: `4`
- refine stage: enabled

## Video Contract

Training and inference use the same latent/video contract:

- output video frames: `81`
- latent temporal length: `21`
- temporal VAE stride: `4`

So the model predicts a full latent video of length `21` and the VAE expands it
to `81` frames.

## Diffusion Objective

The SANA objective used in these runs is flow-style diffusion, not classic DDPM
epsilon prediction.

The main configuration points are:

- `predict_flow_v = true`
- `noise_schedule = linear_flow`

So the `diff` term printed in the logs is the flow-matching diffusion loss.

## Datasets and Joint Training Schedule

The main training data family is a mixed manifest composed of:

- OpenVid video data
- current curated LAION / COYO image data

The two joint schedules used in the experiments were:

- `1V:1I`
  - one video micro-step followed by one image micro-step
- `5V:1I`
  - five video micro-steps for every one image micro-step

The schedule mattered a lot in practice:

- the historical visually stronger family used `1V:1I`
- most Qwen experiments used `5V:1I`

That means some of the regressions cannot be attributed to backbone alone,
because data schedule also changed.

## Prompt Preprocessing

Most runs used:

- whitespace normalization
- strip leading/trailing whitespace
- remove repeated newlines
- no CHI prompt template
- no prompt template expansion

One important caveat discovered during the Qwen experiments:

- if `max_prompt_tokens` is set, the prompt may be truncated using the student
  tokenizer before it is passed to the teacher.

This is not ideal, especially when student and teacher use different tokenizers.

## Loss Families

### 1. Diffusion / flow loss

Present in all runs:

- `diff.weight = 1.0`

### 2. Online teacher distillation

When enabled, distillation added:

- token MSE loss
- token cosine loss
- pooled cosine loss

The standard Smol online-teacher recipe used:

- `token_mse_weight = 1.0`
- `token_cos_weight = 0.5`
- `pooled_cos_weight = 0.2`

### 3. Semantic anti-collapse probe

When enabled, the trainer reported and optimized:

- `sem_var`
- `sem_cov`
- `sem_geom`

These are loss terms, not raw statistics.

Interpretation:

- lower `sem_var` generally means the pooled embedding standard deviation is
  closer to target,
- lower `sem_cov` means less redundant cross-dimension correlation,
- lower `sem_geom` means geometry is closer to the chosen reference.

### 4. Representation-only distillation variant

One Qwen bridge-only variant removed tokenwise distillation entirely and kept:

- `token_mse_weight = 0.0`
- `token_cos_weight = 0.0`
- `pooled_cos_weight = 0.3`
- `semantic_probe.geom_source = teacher`

This was intended to avoid forcing a bad token-to-token alignment between Qwen
and Gemma teacher embeddings.

## Experiment Timeline

The important runs can be grouped into seven families.

### Family A: Historical Smol baseline family (2026-03-15 / 2026-03-16)

This is the family that historically looked best.

Phase 1:

- SmolVLM2-500M
- bridge trainable
- full DiT trainable
- online teacher distill enabled
- semantic anti-collapse enabled
- `1V:1I`

Phase 2:

- initialized from the Phase 1 checkpoint
- same backbone
- same full DiT regime
- distill disabled
- semantic anti-collapse still enabled

This is the strongest historical reference family.

### Family B: SmolVLM2-2.2B bridge-only

- SmolVLM2-2.2B
- bridge-only
- DiT frozen
- online teacher distill
- semantic anti-collapse
- `1V:1I`

Outcome:

- stable optimization,
- no meaningful visual payoff,
- prompt-side matching improved,
- but bridge-only was not enough.

### Family C: Qwen bridge-only with standard token distill

- Qwen3-VL-Embedding-2B
- bridge-only
- DiT frozen
- online teacher distill
- mostly `5V:1I`

Outcome:

- better than the earliest Qwen settings,
- still not visually convincing,
- still poor prompt binding.

### Family D: Qwen bridge-only with pooled-only distill + teacher geometry

- Qwen3-VL-Embedding-2B
- bridge-only
- DiT frozen
- token MSE and token cosine removed
- pooled teacher cosine only
- semantic anti-collapse enabled
- geometry anchored to teacher
- `5V:1I`

Outcome:

- numerically healthy,
- looked less collapsed than earlier Qwen runs,
- still visually poor,
- prompt alignment remained weak.

### Family E: Qwen full DiT diffusion-only

- Qwen3-VL-Embedding-2B
- bridge trainable
- full DiT trainable
- no distill
- no anti-collapse
- `5V:1I`

Outcome:

- optimization was very clean,
- but there was no visual breakthrough,
- prompt binding remained weak.

### Family F: Qwen phase 2 from bridge checkpoint

- Qwen3-VL-Embedding-2B
- initialized from the bridge-only checkpoint
- bridge + full DiT trainable
- diffusion + semantic anti-collapse
- no distill
- `5V:1I`

Outcome:

- severe prompt manifold drift,
- semantic probe became suspicious,
- inference got worse, not better.

### Family G: Current SmolVLM2-500M reproduction

- SmolVLM2-500M
- bridge + full DiT trainable
- online teacher distill
- semantic anti-collapse
- `1V:1I`

Outcome so far:

- most trustworthy current run,
- no obvious hard collapse,
- different prompts produce meaningfully different images,
- but prompt binding is still too weak,
- early visual quality is not good enough to call this a successful
  reproduction yet.

## Representative Log Evidence

This section uses direct log excerpts from the real runs.

### A. Historical 2026-03-16 full-DiT Smol nodistill line

This historical line matters because it anchors what a healthy full-DiT Smol
phase looked like.

```text
2026-03-16 14:30:30,887 - INFO - Trainable params bridge: 3.14M
2026-03-16 14:30:30,887 - INFO - Trainable params DiT: 1028.43M
2026-03-16 14:30:30,887 - INFO - Semantic anti-collapse enabled: weight=0.2000 var=1.000 cov=0.050 geom=1.000 every=5 prompts=6 target_std=1.00
```

```text
2026-03-16 14:57:51,855 - INFO - Step 1000 | mode=image loss=0.160919 diff=0.061279 d_mse=0.000000 d_cos=0.000000 d_pool=0.000000 ... sem_var=0.441992 sem_cov=0.032235 sem_geom=0.054595 ...
```

```text
2026-03-16 19:04:27,541 - INFO - Step 10000 | mode=image loss=0.227218 diff=0.133789 d_mse=0.000000 d_cos=0.000000 d_pool=0.000000 ... sem_var=0.395141 sem_cov=0.042287 sem_geom=0.069891 ...
```

Takeaway:

- full DiT was active,
- anti-collapse stayed in a reasonable range,
- no sign of the severe prompt-space distortion later seen in Qwen phase 2.

### B. SmolVLM2-2.2B bridge-only failure

```text
2026-03-23 12:24:39,674 - INFO - Trainable params bridge: 8.93M
2026-03-23 12:24:39,674 - INFO - Trainable params DiT: 0.00M
2026-03-23 12:24:39,674 - INFO - Semantic anti-collapse enabled: weight=0.2000 var=1.000 cov=0.050 geom=1.000 every=5 prompts=6 target_std=1.00
```

```text
2026-03-23 12:43:36,187 - INFO - Step 1000 | mode=image loss=1.182952 diff=0.063965 d_mse=0.776341 d_cos=0.388172 d_pool=0.192317 ... sem_var=0.518916 sem_cov=0.018516 sem_geom=0.030643 ...
```

```text
2026-03-23 13:59:22,716 - INFO - Step 5000 | mode=image loss=1.493147 diff=0.376953 d_mse=0.787435 d_cos=0.393720 d_pool=0.159310 ... sem_var=0.441124 sem_cov=0.031448 sem_geom=0.057486 ...
```

Takeaway:

- prompt-side matching improved,
- norms stayed stable,
- but bridge-only did not produce good visuals.

### C. Qwen bridge-only with standard token distill

```text
2026-03-24 01:20:43,779 - INFO - Distill enabled: mode=online_only precomputed_dir=<none> token_mse=1.000 token_cos=0.500 pooled_cos=0.200 every_steps=2 ...
2026-03-24 01:20:44,084 - INFO - Trainable params bridge: 8.93M
2026-03-24 01:20:44,084 - INFO - Trainable params DiT: 0.00M
```

```text
2026-03-24 01:48:34,679 - INFO - Step 1000 | mode=video loss=0.961235 diff=0.100586 d_mse=0.678573 d_cos=0.339289 d_pool=0.062158 ... v_micro=801 i_micro=199 ...
```

```text
2026-03-24 03:39:44,884 - INFO - Step 5000 | mode=video loss=0.970876 diff=0.122070 d_mse=0.668867 d_cos=0.334435 d_pool=0.063607 ...
```

Takeaway:

- optimization looked better than the earliest Qwen attempts,
- but visual quality was still not convincing.

### D. Qwen bridge-only with pooled-only distill + teacher geometry

```text
2026-03-24 13:35:19,835 - INFO - Distill enabled: mode=online_only precomputed_dir=<none> token_mse=0.000 token_cos=0.000 pooled_cos=0.300 every_steps=1 ...
2026-03-24 13:35:20,111 - INFO - Semantic anti-collapse enabled: weight=0.1500 var=1.000 cov=0.050 geom=1.000 source=teacher every=5 prompts=6 target_std=1.00
2026-03-24 13:35:20,111 - INFO - Trainable params bridge: 8.93M
2026-03-24 13:35:20,111 - INFO - Trainable params DiT: 0.00M
```

```text
2026-03-24 14:03:35,352 - INFO - Step 1000 | mode=video loss=0.195027 diff=0.099609 d_mse=0.000000 d_cos=0.000000 d_pool=0.044461 ... sem_var=0.483504 sem_cov=0.023350 sem_geom=0.062526 ...
2026-03-24 14:03:35,428 - INFO - Step 1000 | probe_semantic mcp_offdiag(mean/min/max)=0.598090/0.496989/0.714471 smol_offdiag=0.225709 prompts=6 mcp_tok=9.83 smol_tok=9.83
```

```text
2026-03-24 18:17:32,284 - INFO - Step 10000 | mode=video loss=0.163449 diff=0.076660 d_mse=0.000000 d_cos=0.000000 d_pool=0.039573 ... sem_var=0.409945 sem_cov=0.035809 sem_geom=0.087712 ...
```

Takeaway:

- collapse looked less severe,
- but prompt-space remained very clustered,
- visual performance remained poor.

### E. Qwen phase 2 full-DiT drift

```text
2026-03-24 19:22:00,130 - INFO - Trainable params bridge: 8.93M
2026-03-24 19:22:00,130 - INFO - Trainable params DiT: 1028.43M
2026-03-24 19:22:00,130 - INFO - Semantic anti-collapse enabled: weight=0.2000 var=1.000 cov=0.050 geom=1.000 source=raw every=5 prompts=6 target_std=1.00
```

```text
2026-03-24 19:59:43,491 - INFO - Step 1000 | mode=video loss=0.136091 diff=0.098633 d_mse=0.000000 d_cos=0.000000 d_pool=0.000000 ... sem_var=0.147498 sem_cov=0.195611 sem_geom=0.030014 ...
2026-03-24 19:59:43,565 - INFO - Step 1000 | probe_semantic mcp_offdiag(mean/min/max)=0.054095/-0.227995/0.461172 smol_offdiag=0.225709 prompts=6 mcp_tok=9.83 smol_tok=9.83
```

```text
2026-03-24 22:30:29,732 - INFO - Step 5000 | mode=video loss=0.154068 diff=0.120117 d_mse=0.000000 d_cos=0.000000 d_pool=0.000000 ... sem_var=0.117908 sem_cov=0.220159 sem_geom=0.040838 ...
```

Takeaway:

- this was the clearest sign of prompt manifold distortion,
- `sem_cov` exploded compared with the healthier runs,
- `mcp_offdiag` dropped toward zero and even negative values,
- inference got worse rather than better.

### F. Current SmolVLM2-500M reproduction

This is the most important current run.

```text
2026-03-24 23:13:39,015 - INFO - Distill enabled: mode=online_only precomputed_dir=<none> token_mse=1.000 token_cos=0.500 pooled_cos=0.200 every_steps=2 ...
2026-03-24 23:13:39,259 - INFO - Trainable params bridge: 3.14M
2026-03-24 23:13:39,260 - INFO - Trainable params DiT: 1028.43M
2026-03-24 23:13:39,260 - INFO - Semantic anti-collapse enabled: weight=0.2000 var=1.000 cov=0.050 geom=1.000 source=raw every=5 prompts=6 target_std=1.00
```

```text
2026-03-24 23:41:11,453 - INFO - Step 1000 | mode=image loss=1.579919 diff=0.075195 d_mse=1.053674 d_cos=0.526840 d_pool=0.323606 ... cond_uncond_dloss=0.257812 ... sem_var=0.606105 sem_cov=0.008717 sem_geom=0.008001 ...
```

```text
2026-03-25 01:04:33,599 - INFO - Step 4000 | mode=image loss=1.489145 diff=0.154297 d_mse=0.947718 d_cos=0.473862 d_pool=0.186895 ... cond_uncond_dloss=0.562500 ... sem_var=0.538512 sem_cov=0.016013 sem_geom=0.024789 ...
```

```text
2026-03-25 01:32:20,055 - INFO - Step 5000 | mode=image loss=1.408012 diff=0.113281 d_mse=0.916113 d_cos=0.458059 d_pool=0.186312 ... cond_uncond_dloss=0.406250 ... sem_var=0.535549 sem_cov=0.016386 sem_geom=0.025261 ...
```

```text
2026-03-25 02:00:09,914 - INFO - Step 6000 | mode=image loss=1.452781 diff=0.087402 d_mse=0.953616 d_cos=0.476811 d_pool=0.306663 ... cond_uncond_dloss=0.718750 ... sem_var=0.533541 sem_cov=0.016636 sem_geom=0.025751 ...
```

Takeaway:

- this run is stable,
- it does not show the alarming semantic-probe drift seen in Qwen phase 2,
- but stable geometry alone has not produced strong prompt-grounded visuals.

## Inference Observations

### SANA base sanity check

We verified the inference pipeline separately using the original SANA-Video
model under the same fixed inference setup:

- fixed backend
- 12 denoise steps
- 81 frames
- 832x480
- same seed family

The base SANA model produced much stronger prompt alignment than the trained
student checkpoints.

This is important because it suggests the inference wrapper is not the primary
problem. The main problem is in the trained conditioning path and/or how DiT has
adapted to it.

### Smol reproduction checkpoint behavior

The current Smol reproduction was manually checked at `1k`, `4k`, and `5k`.

Observed pattern:

- different prompts do produce different images,
- the run is not obviously dead,
- but the generated object/action often does not match the prompt well enough.

This means the failure is not simply "all prompts map to the same frame", but
the prompt-conditioned direction is still too weak or too generic.

## Ladder Test on the Current Smol Reproduction

To understand why the current Smol run still fails, a ladder test was run on
the `step6000` checkpoint.

The ladder test asked four questions:

1. Does the model react to prompt changes at all?
2. Is CFG mismatch a major cause?
3. Is student conditioning functionally far from teacher conditioning?
4. Is the main failure collapse, prompt ignore, or weak binding?

### Rung 1: Prompt sensitivity

Prompts tested:

- a golden retriever running along a beach at sunset
- a chef slicing colorful vegetables in a busy kitchen
- a barista pouring latte art in a cozy cafe
- an astronaut floating above earth in space
- an empty prompt

CLIP text-image score on frame 0:

- golden: `0.0317`
- chef: `0.1997`
- latte: `0.1662`
- astronaut: `0.1687`

Image-image similarity between outputs:

- golden vs chef: `0.6705`
- golden vs latte: `0.7042`
- golden vs astronaut: `0.6100`
- chef vs empty: `0.6816`
- astronaut vs empty: `0.6265`

Interpretation:

- outputs are not identical,
- so this is not pure hard collapse,
- but unrelated prompts still produce overly similar images.

Most important cross-score example:

- golden image -> golden prompt: `0.0317`
- golden image -> chef prompt: `0.2078`
- golden image -> latte prompt: `0.1711`
- golden image -> astronaut prompt: `0.1573`

This is strong evidence of prompt-binding failure:

- the golden output matches several wrong prompts better than its own prompt.

### Rung 2: CFG sweep

Same checkpoint, same seed, same prompts, different CFG:

- golden `cfg=1.0`: `0.0881`
- golden `cfg=2.0`: `0.0378`
- golden `cfg=3.0`: `0.0317`
- chef `cfg=1.0`: `0.2007`
- chef `cfg=2.0`: `0.2055`
- chef `cfg=3.0`: `0.1997`

Interpretation:

- the golden prompt improves substantially when CFG is reduced to `1.0`,
- the chef prompt is almost insensitive to the sweep,
- CFG mismatch is therefore a real but secondary issue.

This makes sense because current training uses:

- `cfg_dropout_prob = 0.0`

but evaluation often used:

- `cfg_scale = 3.0`

### Rung 3: Probe-only functional diagnostics

Probe-only outputs:

```text
ProbeOnly Step 1 | ... cond_uncond_dloss=0.308594 cond_pred_l2=37.519650 cond_pred_ratio=0.021121
ProbeOnly Step 1 | ... cond_uncond_dloss=0.906250 cond_pred_l2=31.936808 cond_pred_ratio=0.020438
ProbeOnly Step 1 | ... cond_uncond_dloss=1.218750 cond_pred_l2=35.928421 cond_pred_ratio=0.021791
```

Interpretation:

- removing conditioning does make diffusion loss worse,
- so the model is not fully ignoring prompt input,
- but `cond_pred_ratio` is only about `0.02`,
- meaning prompt changes produce only a small change in DiT prediction relative
  to the overall prediction magnitude.

This is a clean sign of weak conditioning strength.

### Rung 4: Teacher-vs-student substitution

Aggregate substitution results:

- `student_teacher_pred_mse = 0.0132`
- `student_teacher_pred_cos = 0.9979`
- `student_vs_uncond_pred_mse = 0.0316`
- `teacher_vs_uncond_pred_mse = 0.0207`

Per-prompt student-teacher context cosine:

- roughly `0.42 - 0.48`

Interpretation:

- student and teacher embeddings are only moderately aligned at the raw
  representation level,
- but once both pass through DiT on the same latent and timestep, their
  predictions are extremely similar,
- so the main problem is not catastrophic student-vs-teacher functional
  mismatch inside DiT.

The more likely issue is:

- the whole conditioning system is too weak,
- so DiT can satisfy diffusion loss while staying close to a generic visual
  prior.

## Current Diagnosis

The best current diagnosis is:

1. The model is not in pure hard semantic collapse.
   - Different prompts produce somewhat different outputs.
   - `cond_uncond_dloss > 0` confirms that prompt information is used.

2. The dominant failure is weak or misdirected prompt binding.
   - Some outputs match the wrong prompt better than the correct prompt.
   - Prompt changes do not alter DiT prediction strongly enough.

3. The main problem is not catastrophic bridge-vs-teacher functional mismatch.
   - Student and teacher conditioning lead to very similar DiT predictions.

4. CFG mismatch is real, but secondary.
   - Reducing CFG helps some prompts.
   - It does not explain the whole failure.

5. Semantic anti-collapse helps prevent total degeneration, but it does not by
   itself solve prompt grounding.

## Why the Qwen Experiments Likely Failed

The Qwen experiments likely failed for a combination of reasons:

1. Qwen embeddings looked semantically healthy but did not naturally serve as a
   good SANA conditioning manifold.
2. Bridge-only training improved representation health without creating strong
   visual grounding.
3. Full-DiT Qwen phase 2 let the conditioning manifold drift too far.
4. The `5V:1I` schedule likely made early visual grounding harder than the old
   `1V:1I` family.
5. Pooled-only distillation avoided bad tokenwise alignment, but it was too
   weak to teach strong prompt-local structure to the frozen DiT.

## Current Best Recipe to Keep

If we need one recipe to remember as the current best working family, it is:

- backbone: `SmolVLM2-500M`
- bridge: trainable
- DiT: full trainable
- distill: online teacher
- semantic anti-collapse: enabled
- schedule: `1V:1I`
- grad accumulation: `1`
- diffusion/flow loss active

Important loss weights:

- `diff.weight = 1.0`
- `token_mse_weight = 1.0`
- `token_cos_weight = 0.5`
- `pooled_cos_weight = 0.2`
- `semantic_probe.weight = 0.2`
- `semantic_probe.var_weight = 1.0`
- `semantic_probe.cov_weight = 0.05`
- `semantic_probe.geom_weight = 1.0`

Optimization:

- bridge learning rate: `5e-5`
- DiT learning rate: `1e-5`
- max grad norm: `0.1`

Important caveat:

- even this family is not yet a success; it is simply the most credible current
  base recipe.

## Recommended Next Experiments

The most useful next steps are not to switch backbone again immediately.
They are to diagnose and strengthen prompt influence.

### 1. Evaluate with CFG parity

For diagnosis, use:

- `cfg_scale = 1.0`

Reason:

- current training uses `cfg_dropout_prob = 0.0`
- higher CFG at inference may be exaggerating misalignment for some prompts

### 2. Add a functional prompt-separation objective

The ladder test showed that the main problem is low functional prompt effect.
So the next loss should directly encourage:

- a larger difference between conditioned and unconditioned prediction
- or a larger difference between correct-prompt prediction and wrong-prompt
  prediction on the same latent

This is more targeted than just increasing representation matching.

### 3. Keep semantic anti-collapse, but do not treat it as the solution

Anti-collapse is useful as a guardrail, but the current failure is not mainly
"the embeddings collapsed". It is "the prompt influence is too weak or too
generic".

### 4. If a phase-2 nodistill run is revisited, only do it after validating the
current Smol online-teacher line further

The historical family suggests that:

1. online-teacher phase can create a good initialization,
2. then a nodistill full-DiT phase may refine visuals.

But that only makes sense once the upstream Smol line is validated more
strongly.

## Bottom Line

The project is not blocked by a single obvious bug.

The current evidence points to a more specific failure mode:

- the model can remain stable,
- the model can avoid hard collapse,
- the model can even produce diverse outputs,
- but prompt influence on the actual denoising path is too weak and sometimes
  misdirected.

The right direction is therefore:

- keep the credible Smol recipe as the base,
- evaluate more carefully with CFG parity,
- and design the next experiment to strengthen functional prompt influence
  rather than only representation similarity.
