# Overfit And Recaption Diagnostics (2026-03-25)

This note records the two main diagnostic tracks we ran on 2026-03-25:

1. `clean16` overfit experiments on the current `SmolVLM2-500M -> bridge -> full DiT -> SANA` stack.
2. OpenVid-1M part-level recaptioning with `SmolVLM2-500M-Video-Instruct`, followed by prompt-similarity analysis.

The goal was not to find a better production recipe immediately. The goal was to answer three narrower questions:

- Can the current generation stack memorize a tiny, clean training set if we make the problem easy enough?
- If overfit still fails, is the bottleneck more likely to be data/ground-truth, the text/bridge interface, or the objective?
- Would regenerating denser captions for OpenVid make the prompt space materially less collapsed before training?

The short version is:

- The overfit tests improved training stability and prompt usage a bit, but still did **not** produce strong prompt-binding even on a tiny `16`-video subset.
- The scratch overfit run was better than the earlier `init-from-10k` variant, but it still only reached `2/4` diagonal wins on the four canonical train prompts.
- Raw Smol prompt embeddings for `clean16` were already heavily clustered.
- SmolVLM2 recaptioning on one OpenVid part reduced average prompt similarity a little, but not enough to count as a decisive fix, and some recaptions were clearly worse or hallucinated.

The evidence from these tests points more strongly to an **objective / functional prompt influence problem** than to a simple data-loading bug.

---

## 1. Common Diagnostic Setup

### 1.1 Tiny overfit dataset

We created a tiny video-only subset:

- Manifest: `data/mix/manifests/openvid_clean16_overfit_20260325_video.csv`
- Size: `16` samples
- Modality: `video` only
- Latents: precomputed WAN VAE latents from `.pkl`
- No image side, no joint alternation

The four canonical prompts used repeatedly for diagnostics and inference were:

1. `selfie_livingroom`
   - `The video shows a woman in a striped shirt taking a selfie in a living room. She is standing in front of a beige couch and appears to be speaking. The room has a warm and cozy atmosphere.`
2. `man_street`
   - `The video features a young man walking down a street. He is wearing a green scarf and glasses. The street is lined with trees and buildings. The man is walking towards the camera. The video is shot in a realistic style.`
3. `lily_bloom`
   - `Time-lapse of a white water lily blooming; beginning as a closed bud and gradually opening to full bloom. The petals separate, the central stamens and stigma become reveal, culminating in a fully open flower.`
4. `cookies_baking`
   - `Time-lapse of almond shortbread cookies baking in an oven, starting from raw dough to attaining full expansion and uniform browning, indicating a complete baking process.`

### 1.2 Inference protocol used for fair comparison

All overfit inference comparisons in this note use the same diagnostic settings unless stated otherwise:

- inference backend: `fixed`
- scheduler family: flow matching / `flow_dpm-solver`
- denoise steps: `12`
- `cfg_scale = 1.0`
- seed: `0`
- frame size: `832x480`
- frames: `81`
- CLIP proxy: `openai/clip-vit-base-patch32`
- prompt scoring: CLIP text-image similarity on `frame000`

This setup was chosen because earlier ladder tests had already shown that `cfg > 1` could make prompt adherence look worse and could obscure what the model was actually learning.

### 1.3 What was trainable vs frozen in the scratch overfit run

For the final scratch overfit run, the trainable/frozen split was:

- Trainable:
  - bridge / projector
  - full DiT (`train_modules: [all]`)
- Frozen:
  - SmolVLM2 backbone
  - student vision head
  - VAE
  - all LoRA branches disabled
- Loss:
  - diffusion only
  - no teacher distill
  - no semantic anti-collapse loss

In other words, the scratch run was intentionally stripped down to answer:

- if we remove teacher and regularization, can the generator plus bridge simply memorize the tiny set?

---

## 2. Raw Smol Prompt-Space Check On `clean16`

Before interpreting overfit results, we measured raw Smol prompt similarity directly on the `16` captions in `clean16`.

Output file:

- `output/analysis/clean16_smol_prompt_similarity_20260325.json`

### 2.1 Main statistics

Using the raw Smol text path, we measured two pooled representations:

- mean-pool over valid tokens
- last-token representation

Results:

- `mean-pool offdiag mean = 0.8027359247207642`
- `mean-pool offdiag std = 0.11356745660305023`
- `last-token offdiag mean = 0.8750001788139343`
- `last-token offdiag std = 0.07239539921283722`

These values are high. Even before bridge training, the text encoder already places many prompts very close together.

### 2.2 Example highly similar pairs

Top mean-pool pairs:

- prompt `1` vs `5`: `0.9792598485946655`
  - both are variants of `man walking down a street`
- prompt `8` vs `11`: `0.9787642955780029`
  - both are `ice melting`
- prompt `2` vs `6`: `0.9786615371704102`
  - both are women indoors

Top last-token pairs were even tighter:

- prompt `1` vs `5`: `0.9955471754074097`
- prompt `4` vs `5`: `0.9953550696372986`
- prompt `2` vs `14`: `0.9891476631164551`

### 2.3 Interpretation

This does **not** prove the text encoder is unusable. It does mean the burden on bridge + DiT is heavy:

- the raw text space is already anisotropic and clustered,
- several prompts with different visual targets start out very close,
- so a weak conditioning objective can easily produce generic or partially shared outputs.

This finding was one of the motivations for trying recaptioning later in the day.

---

## 3. Overfit Experiment A: `init-from-10k` With Distill + Semantic Probe

### 3.1 Purpose

This first overfit run asked:

- if we start from the current stronger Smol checkpoint and keep the existing online-teacher losses, can the model memorize the `16` clean videos quickly?

### 3.2 Recipe

Config:

- `configs/stage1_teacher_free_openvid_clean16_overfit_initfrom10k_1gpu_20260325.yaml`

Key settings:

- init from the current Smol repro `step10000`
- `SmolVLM2-500M` frozen text backbone
- bridge trainable
- full DiT trainable
- `16` clean videos, no joint image branch
- `total_steps = 2000`
- `save_every_steps = 200`
- `batch_size = 1`, `grad_accum_steps = 1`
- `cfg_dropout_prob = 0.0`
- online teacher distill enabled
- semantic probe enabled

Losses:

- `diff.weight = 1.0`
- `token_mse_weight = 1.0`
- `token_cos_weight = 0.5`
- `pooled_cos_weight = 0.2`
- `semantic_probe.weight = 0.2`

### 3.3 Training behavior

Representative log lines:

- `Step 50`
  - `loss=1.602420 diff=0.078125 d_mse=1.047134 d_cos=0.523570 d_pool=0.308337 ... cond_uncond_dloss=0.441406 ... cond_pred_ratio=0.033011 sem_var=0.754085 sem_cov=0.003179 sem_geom=0.014301`
- `Step 1000`
  - `loss=0.893388 diff=0.066406 d_mse=0.544925 d_cos=0.272464 d_pool=0.052200 ... cond_uncond_dloss=0.193359 ... cond_pred_ratio=0.037737 sem_var=0.673258 sem_cov=0.013734 sem_geom=0.002981`
- `Step 1000 probe_semantic`
  - `mcp_offdiag(mean/min/max)=0.802436/0.678262/0.985303 smol_offdiag=0.767632`
- `Step 2000`
  - `loss=0.798430 diff=0.087891 d_mse=0.457762 d_cos=0.228882 d_pool=0.026958 ... cond_uncond_dloss=0.367188 ... cond_pred_ratio=0.014273 sem_var=0.661590 sem_cov=0.015398 sem_geom=0.002365`
- `Step 2000 probe_semantic`
  - `mcp_offdiag(mean/min/max)=0.787429/0.656274/0.985752 smol_offdiag=0.767632`

The optimization part was superficially healthy:

- distill losses dropped strongly,
- pooled distill dropped to near zero,
- total loss decreased.

But the functional prompt-effect probe was not healthy:

- `cond_pred_ratio` did **not** improve steadily,
- by `step2000` it had fallen to `0.014273`,
- that is weaker than what we would want for an intentional memorization test.

### 3.4 Inference result at `step2000`

Cross-score file:

- `output/analysis/overfit_clean16_step2000_clip_crossscores_20260325.json`

This run failed in a particularly revealing way:

- all `4/4` frames were scored highest against `cookies_baking`

Detailed examples:

- selfie frame:
  - `selfie_livingroom = 0.1777`
  - `cookies_baking = 0.2239`
  - best prompt: `cookies_baking`
- man street frame:
  - `man_street = 0.1979`
  - `cookies_baking = 0.2638`
  - best prompt: `cookies_baking`
- lily frame:
  - `lily_bloom = 0.2479`
  - `cookies_baking = 0.2919`
  - best prompt: `cookies_baking`
- cookies frame:
  - `cookies_baking = 0.3133`
  - best prompt: `cookies_baking`

### 3.5 Interpretation

This experiment showed that:

- merely shrinking to a tiny dataset was not enough,
- keeping the old regularization and teacher terms did not force useful memorization,
- the run learned some generic denoising behavior and some representation matching,
- but it collapsed functionally toward one dominant prompt family.

This is why we abandoned the `init-from-10k + distill + semantic probe` direction for the next overfit test.

---

## 4. Overfit Experiment B: Scratch, No Distill, No Regularization

### 4.1 Purpose

The second overfit run asked a cleaner question:

- if we start from scratch and strip away teacher losses and semantic regularizers, can `bridge + full DiT` memorize the `16` videos using only diffusion loss?

This is the most important overfit test of the day.

### 4.2 Recipe

Config:

- `configs/stage1_teacher_free_openvid_clean16_overfit_nodistill_noreg_scratch_1gpu_20260325.yaml`

Key settings:

- init from scratch
- `SmolVLM2-500M` text backbone frozen
- bridge trainable
- full DiT trainable
- no teacher distill
- no semantic probe
- `max_prompt_tokens = null`
- `shuffle = false`
- `drop_last = false`
- `total_steps = 6000`
- `save_every_steps = 1000`
- `batch_size = 1`, `grad_accum_steps = 1`

This run was intentionally constrained to avoid wasting compute on a muddy setup.

### 4.3 Training behavior

Representative log lines:

- `Step 50`
  - `loss=0.065430 diff=0.065430 ... cond_uncond_dloss=0.175781 ... cond_pred_ratio=0.016252`
- `Step 1000`
  - `loss=0.063477 diff=0.063477 ... cond_uncond_dloss=0.261719 ... cond_pred_ratio=0.024384`
- `Step 1000 probe_semantic`
  - `mcp_offdiag(mean/min/max)=0.596118/0.488504/0.850967 smol_offdiag=0.741030`
- `Step 5350`
  - `loss=0.109863 diff=0.109863 ... cond_uncond_dloss=0.414062 ... cond_pred_ratio=0.084370`
- `Step 6000`
  - `loss=0.059814 diff=0.059814 ... cond_uncond_dloss=0.217773 ... cond_pred_ratio=0.049096`
- `Step 6000 probe_semantic`
  - `mcp_offdiag(mean/min/max)=0.573058/0.464669/0.829546 smol_offdiag=0.741030`

What improved relative to the previous overfit run:

- optimization remained simple and stable,
- prompt effect increased above the very low early values,
- `mcp_offdiag` was much better than raw Smol prompt similarity,
- there was no evidence of a single catastrophic drift toward one teacher-shaped attractor.

What still did **not** happen:

- `cond_pred_ratio` never became large,
- the final prompt effect remained weak in absolute terms,
- the model still did not cleanly memorize all four canonical train prompts.

The maximum `cond_pred_ratio` observed in the run was:

- `0.084370` at `step5350`

The final value was:

- `0.049096` at `step6000`

That is directionally better than the start, but still too small for a convincing memorization story.

### 4.4 Inference result at `step1000`

Recomputed CLIP file:

- `output/analysis/overfit_clean16_nodistill_noreg_scratch_step1000_clip_crossscores_recomputed_20260325.json`

At `step1000`, the model achieved only `1/4` diagonal wins.

Prompt-level result:

- `selfie_livingroom`
  - diagonal `0.1463`
  - best prompt `lily_bloom = 0.1966`
- `man_street`
  - diagonal `0.1918`
  - best prompt `selfie_livingroom = 0.2023`
- `lily_bloom`
  - diagonal `0.2496`
  - best prompt `cookies_baking = 0.2610`
- `cookies_baking`
  - diagonal `0.2857`
  - best prompt `cookies_baking = 0.2857`

Average diagonal score at `step1000`:

- `0.2184`

Image-image off-diagonal mean at `step1000`:

- `0.6712`

### 4.5 Inference result at final checkpoint (`step6000` / `checkpoint_final`)

Cross-score file:

- `output/analysis/overfit_clean16_nodistill_noreg_scratch_final_clip_crossscores_20260325.json`

Final summary:

- diagonal wins: `2/4`
- diagonal win rate: `0.5`
- mean diagonal CLIP score: `0.26234907656908035`
- mean best-per-row score: `0.27825919911265373`

Detailed prompt-level result:

- `selfie_livingroom`
  - diagonal `0.2927`
  - best prompt `selfie_livingroom`
  - clear improvement from `0.1463`
- `man_street`
  - diagonal `0.2076`
  - best prompt `selfie_livingroom = 0.2523`
  - still wrong
- `lily_bloom`
  - diagonal `0.2132`
  - best prompt `cookies_baking = 0.2321`
  - still wrong
- `cookies_baking`
  - diagonal `0.3359`
  - best prompt `cookies_baking`
  - best case in the set

Image-image off-diagonal mean at final:

- `0.6916`

This is slightly **higher** than at `step1000`, meaning the outputs are still fairly similar to each other overall.

### 4.6 What improved from `step1000` to final

Per-prompt diagonal score deltas:

- `selfie_livingroom`: `0.1463 -> 0.2927` (`+0.1464`)
- `man_street`: `0.1918 -> 0.2076` (`+0.0157`)
- `lily_bloom`: `0.2496 -> 0.2132` (`-0.0363`)
- `cookies_baking`: `0.2857 -> 0.3359` (`+0.0502`)

So the model did learn **something**:

- selfie got much better,
- cookies got better,
- total diagonal wins improved from `1/4` to `2/4`.

But the final outcome still fails the test we wanted:

- if the system truly overfit cleanly, we would expect nearly all train prompts to win on the diagonal,
- especially in such a tiny dataset.

### 4.7 Interpretation

This is the most important finding of the overfit track.

The scratch run says:

- removing teacher and regularization **helps** relative to the previous overfit recipe,
- but even then, the system still does not strongly memorize prompt-to-video mapping,
- therefore the problem is **not just** that distill or semantic regularization were over-constraining the run.

The model is learning denoising and some prompt influence, but not enough prompt discrimination.

That points more strongly to:

1. objective weakness
   - the flow-matching objective rewards denoising correctness, but does not explicitly reward correct-prompt vs wrong-prompt separation strongly enough,
2. conditioning interface weakness
   - bridge + DiT may still allow a solution that uses prompt only weakly,
3. raw text-space clustering as an upstream amplifier
   - since Smol prompt embeddings are already highly clustered.

What it points away from:

- a simple data-loading bug,
- a purely numerical instability explanation,
- or the idea that just training longer on a tiny set will automatically solve prompt binding.

---

## 5. OpenVid Part-0 Recaption Experiment With SmolVLM2

### 5.1 Purpose

The recaption experiment was designed to test a different hypothesis:

- perhaps the prompt contract itself is too weak or too clustered,
- and maybe generating denser, more literal captions for OpenVid would improve prompt separation before training.

### 5.2 Data slice used

We downloaded and extracted OpenVid-1M `part 0` and built a manifest for that part.

Manifest summary:

- `download_data/data/openvid/manifests/openvid_part0.summary.json`

Key numbers:

- `total_csv_rows = 1019957`
- `matched_rows = 33117`
- `missing_rows = 986840`
- selected parts: `[0]`

For the recaption analysis itself we sampled:

- `64` videos
- random sampling
- `seed = 0`

### 5.3 Recaption model and prompt

Recaption model:

- `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`

Exact recaption query used:

- `Describe this video with one dense factual caption. Name the main subject, action, scene, and salient objects. Avoid speculation, evaluation, and storytelling.`

Analysis outputs:

- `output/analysis/openvid_part0_smol_recaption_20260325/summary.json`
- `output/analysis/openvid_part0_smol_recaption_20260325/recaption_rows.csv`

### 5.4 Similarity results

Main statistics:

- original caption offdiag mean: `0.9006277322769165`
- recap caption offdiag mean: `0.8741124868392944`
- paired original-vs-recap mean: `0.8630544543266296`
- cross global mean: `0.8269507884979248`

Interpretation:

- recaptioning reduced prompt clustering slightly,
- but only modestly,
- and not enough to count as a major structural improvement.

In relative terms, the reduction in mean off-diagonal similarity was only about `0.0265` cosine.

### 5.5 Length effect

The regenerated captions were usually **shorter**, not richer.

From the sampled CSV we observed:

- original captions often looked like already-dense OpenVid descriptions,
- Smol recaptions often compressed them into shorter summaries.

This matters because the initial hypothesis was “regenerate clearer captions”. In practice, the model often produced “shorter captions” rather than “better captions”.

### 5.6 Good examples

Example: `celebv_zHLfQpIw1N8_1_0.mp4`

Original:

- `The video features a man with a beard and mustache, wearing a blue cap with the word "Farm Rescue" on it. He is dressed in a blue shirt and is standing in a field with a green background. ...`

Recaption:

- `A man wearing a baseball cap with the words "Farm Rescue" on it is standing in a field.`

This is shorter, but still factual and close enough.

Example: `celebv_oPCMkHDUz6Y_0.mp4`

Original:

- `The video features a man with a beard and glasses, wearing a green and black checkered shirt. He is seated in a room with a blurred background ...`

Recaption:

- `A man in a green shirt is speaking to the camera in a room with a window.`

This is acceptable, but clearly flatter and less specific than the original.

### 5.7 Bad examples and failure modes

Example: `celebv_FZlm1ledK-I_0.mp4`

Original:

- serious man in a colorful patterned shirt, looking downward indoors

Recaption:

- `A man in a colorful shirt is playing a game of poker in a room with a fireplace and a window.`

This is a clear hallucination.

Example: `celebv_4sNyWmYN-rM_1.mp4`

Original:

- man in blue/silver futuristic suit in an urban setting

Recaption:

- `The video features a scene from the movie "The Hunger Games: Catching Fire," ...`

This injects unsupported named-entity speculation.

Example: `celebv_q6V-CtYY8D8_24_0.mp4`

Original:

- woman with curly hair in pink blouse speaking outdoors

Recaption:

- `A man in a pink shirt is speaking to the camera in front of a building.`

This flips gender incorrectly.

Example: `celebv_39Br2A7lxac_31.mp4`

Original:

- young woman in purple sweater smiling in front of a white wall

Recaption:

- repeated text loop about `The Office`

This is a degenerate generation failure.

### 5.8 Interpretation

The recaption test gives a mixed answer:

What looks promising:

- there is a small but real decrease in prompt similarity,
- so the data-contract hypothesis is not crazy.

What looks weak:

- the gain is modest,
- many recaptions are shorter and flatter than the original,
- some are clearly hallucinated or templated,
- therefore this exact recaption pipeline is **not** ready to become the new training data contract.

The result is useful diagnostically, but not yet actionable as a dataset replacement.

---

## 6. Combined Interpretation

Putting the overfit and recaption results together:

### 6.1 What now looks unlikely

It is now less likely that the main failure is a simple data bug such as:

- loading the wrong latent file,
- training on random garbage latents,
- or using captions that are completely disconnected from the videos.

The overfit experiments were too controlled for that explanation to remain the most plausible one.

### 6.2 What now looks more likely

The stronger explanation is a combination of three factors:

1. The raw text encoder space is already heavily clustered.
2. The current objective still allows low-prompt-influence solutions.
3. The bridge + DiT interface is not forcing enough functional separation between `correct prompt` and `wrong prompt` on the denoising path.

### 6.3 Why the overfit tests matter so much

The scratch overfit result is especially important because it rules out an easy excuse.

If the system cannot strongly memorize `16` clean videos even when:

- using only video data,
- training bridge + full DiT,
- removing teacher distill,
- removing semantic regularization,
- and training for `6000` steps,

then we should not expect full-data training with the same objective to suddenly produce strong prompt grounding.

### 6.4 What the recaption test changes

The recaption test does **not** overturn the overfit conclusion.

Instead, it adds one more nuance:

- data contract may indeed matter,
- but a naive Smol recaption pass is not enough by itself,
- and the main bottleneck still appears to be how prompt information is used inside the generative path.

---

## 7. Practical Takeaways For Next Steps

Based on these diagnostics, the next steps that make the most sense are:

1. Add a **functional prompt-separation loss** in overfit first.
   - We now have evidence that diffusion-only is too weak.
   - The next loss should directly reward larger prediction differences between `correct prompt`, `wrong prompt`, and optionally `unconditional` on the same noisy latent.

2. Run a **teacher-conditioned overfit baseline** on the same `clean16` set.
   - If teacher conditioning can overfit but student conditioning cannot, the bottleneck is strongly in the student/bridge interface.
   - If teacher conditioning also fails badly, the objective / DiT prompt path is the larger issue.

3. If we revisit recaptioning, do not use the current prompt as-is.
   - The current recap prompt is too free-form.
   - A stricter rewrite should require literal subject/action/scene/object extraction and forbid named-entity speculation.

4. Do not treat teacher-distill reduction alone as the answer.
   - Overfit scratch improved over the distill-heavy run, but it still did not solve prompt binding.
   - So the core problem is not simply “too much distill”.

---

## 8. Final Bottom Line

The most important conclusion from 2026-03-25 is this:

- We now have stronger evidence that the main failure is **not** pure semantic collapse and **not** an obvious data-loading error.
- The current stack can learn denoising and some prompt influence, but that prompt influence remains too weak and too entangled.
- Even in a tiny overfit setting, the model does not memorize prompt-video mapping cleanly.
- Recaptioning with SmolVLM2 reduces prompt clustering a little, but the improvement is too small and too noisy to count as a reliable fix.

So the current priority should be:

- change the **functional training objective** before scaling further,
- use overfit as the fast diagnostic loop,
- and treat recaptioning as a secondary data-contract experiment, not the primary fix.
