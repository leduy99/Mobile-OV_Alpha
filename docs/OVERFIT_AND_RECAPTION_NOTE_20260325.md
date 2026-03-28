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

### 4.2.1 Exact loss stack used in the scratch run

The scratch overfit run used the simplest loss stack we had tried that day.

Direct config settings:

- `loss.diff.weight = 1.0`
- `loss.distill.enabled = false`
- `loss.semantic_probe.enabled = false`
- `loss.norm.enabled = false`
- `loss.gate.enabled = false`
- `run.cfg_dropout_prob = 0.0`
- `train.flow_shift = 3.0`

So in practice:

- the total optimization target was just the diffusion loss,
- there was no token-level teacher supervision,
- there was no pooled-teacher supervision,
- there was no semantic anti-collapse regularizer,
- there was no auxiliary norm or gate penalty.

The diffusion loss itself inherits the SANA-video flow-matching formulation:

- noisy latent is sampled along a linear flow path between clean latent and Gaussian noise,
- the model predicts `flow_v`,
- the target is `noise - x_start`,
- and the loss is MSE on that target.

This matters because it means the run was **not** regularized toward prompt discrimination in any direct way. The objective only rewarded denoising correctness.

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

### 4.6.1 Case study: why `selfie` and `cookies` worked better than `man` and `lily`

Looking only at aggregate metrics hides an important nuance. The final run did not fail uniformly.

There were two relatively successful cases:

- `selfie_livingroom`
- `cookies_baking`

And two persistent failure cases:

- `man_street`
- `lily_bloom`

This asymmetry turned out to be informative.

#### Raw Smol prompt similarity already predicts the confusion pattern

From the raw Smol prompt-similarity analysis on the same four prompts:

- `selfie_livingroom` vs `man_street` = `0.8853`
- `lily_bloom` vs `cookies_baking` = `0.8256`
- `selfie_livingroom` vs `lily_bloom` = `0.7305`
- `selfie_livingroom` vs `cookies_baking` = `0.7753`
- `man_street` vs `lily_bloom` = `0.5936`
- `man_street` vs `cookies_baking` = `0.6414`

The two pairs that are most strongly confused by the generator are also the two pairs that start out closest in the raw Smol text space:

- `man_street` gets pulled toward `selfie_livingroom`
- `lily_bloom` gets pulled toward `cookies_baking`

So the model is not confusing prompts randomly. It is confusing prompts along directions that were already close upstream.

#### Ground-truth quality is also uneven across the four cases

Using WAN decode on the stored ground-truth latents, we measured CLIP text-image agreement for the actual training targets:

- `selfie_livingroom`: `0.3396 / 0.3534 / 0.3364` on `first / mid / last`
- `man_street`: `0.2356 / 0.2298 / 0.2407`
- `lily_bloom`: `0.3453 / 0.3646 / 0.3532`
- `cookies_baking`: `0.3664 / 0.3558 / 0.3410`

This is important:

- `man_street` is the weakest target even in the ground-truth decode proxy,
- `selfie`, `lily`, and `cookies` are all materially clearer.

So `man_street` is doubly disadvantaged:

- it is already very close to `selfie` in text space,
- and its target video appears less semantically sharp in the latent decode proxy.

#### Final overfit errors are structured, not arbitrary

At the final checkpoint:

- `selfie_livingroom`
  - diagonal `0.2927`
  - best prompt `selfie_livingroom`
  - margin vs best wrong prompt: `+0.1117`
- `cookies_baking`
  - diagonal `0.3359`
  - best prompt `cookies_baking`
  - margin vs best wrong prompt: `+0.0633`
- `man_street`
  - diagonal `0.2076`
  - best prompt `selfie_livingroom = 0.2523`
  - margin vs best wrong prompt: `-0.0448`
- `lily_bloom`
  - diagonal `0.2132`
  - best prompt `cookies_baking = 0.2321`
  - margin vs best wrong prompt: `-0.0189`

These margins tell a coherent story:

- `selfie` is clearly learned,
- `cookies` is learned but less cleanly,
- `lily` is close to working but still loses to its neighboring prompt family,
- `man` remains too generic and is still absorbed by the indoor-person prototype.

#### Final outputs remain clustered, but not equally so

Mean image-image off-diagonal similarity at final:

- `selfie_livingroom`: `0.6369`
- `man_street`: `0.7572`
- `lily_bloom`: `0.6962`
- `cookies_baking`: `0.6761`

This again lines up with qualitative inspection:

- `selfie` is the most visually distinct output,
- `man_street` is the least distinct and looks the most generic,
- `cookies` and `lily` sit in between.

#### Positive control interpretation

This section is important because it changes how we should talk about the overfit result.

The run is **not** a uniform failure. It is better described as:

- capable of learning some coarse prototypes or families,
- capable of genuinely improving at least one prompt substantially (`selfie_livingroom`),
- but still not discriminative enough to separate nearby prompts reliably.

Human visual inspection agreed with this reading:

- the final `selfie_livingroom` sample looked materially better than the `step1000` and `step2000` samples,
- and it looked plausibly closer to the ground-truth video semantics than the early checkpoints.

So the scratch run does show real learning. The problem is that the learning remains selective and entangled.

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

It also suggests a more precise qualitative diagnosis:

- the system can learn **family-level** semantics,
- but it still struggles with **instance-level discrimination** when two prompts live close together in the upstream text space or when one target is less semantically sharp than another.

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

---

## 9. Follow-Up Image Overfit Positive Control (`clean10`)

Later the same day we ran a complementary image-only overfit test because the video overfit result was mixed:

- the stack clearly learned something,
- but the learning was uneven,
- and it was still possible to argue that video generation was simply too hard a setting for a clean diagnosis.

So we switched to a much easier positive-control test:

- image-only,
- `10` carefully chosen `LAION/COYO` samples,
- same `SmolVLM2-500M -> bridge -> full DiT -> SANA` structure,
- same diffusion-only objective,
- no teacher distill,
- no semantic regularizer.

The goal here was narrower:

- can the current architecture actually learn clean prompt-image mappings at all when the problem is easy enough?

### 9.1 Dataset and recipe

Manifest:

- `data/mix/manifests/laion_coyo_clean10_image_overfit_20260325.csv`

Training config:

- `configs/stage1_teacher_free_laion_coyo_clean10_image_overfit_nodistill_noreg_scratch_1gpu_20260325.yaml`

Run directory:

- `output/stage1_overfit_laion_coyo_clean10_image_smolvlm2_500m_nodistill_noreg_scratch_20260325_1gpu/20260325_223156`

Key recipe:

- `SmolVLM2-500M` frozen
- bridge trainable
- full DiT trainable
- image-only (`latent_t = 1`)
- init from scratch
- `batch_size = 1`
- `grad_accum_steps = 1`
- `shuffle = false`
- `drop_last = false`
- `total_steps = 6000`
- `save_every_steps = 500`

Exact loss stack:

- `loss.diff.weight = 1.0`
- `loss.distill.enabled = false`
- `loss.semantic_probe.enabled = false`
- `loss.norm.enabled = false`
- `loss.gate.enabled = false`
- `run.cfg_dropout_prob = 0.0`

So this run is directly comparable in spirit to the clean16 scratch video overfit run, except that:

- the modality is image-only,
- the training set is smaller (`10` samples),
- and the task is much easier than video generation.

### 9.2 The ten selected prompts

The clean10 image subset intentionally avoided obvious product-listing garbage, slogans, or text-heavy meme captions.

Representative prompts in the subset:

1. `a man and a woman stand by the car he kisses her on the neck`
2. `Woman cooking rice with vegetables in kitchen`
3. `A fresh baked Earl Grey Honey Lemon Mascarpone Cheesecake`
4. `An aerial image of Vancouver, showing Stanley Park, downtown, Granville Bridge, Burrard Street Bridge and the waterfront.`

The remaining six prompts were selected with the same spirit:

- concrete subjects,
- visible actions or scenes,
- short enough to be tractable,
- image-grounded rather than slogan-like.

### 9.3 Training behavior

Representative log points:

- `Step 100`
  - `loss=0.086914 diff=0.086914 ... cond_uncond_dloss=0.240234 ... cond_pred_ratio=0.064863`
- `Step 500`
  - `loss=0.099121 diff=0.099121 ... cond_uncond_dloss=0.152344 ... cond_pred_ratio=0.096768`
- `Step 500 probe_semantic`
  - `mcp_offdiag(mean/min/max)=0.446789/... smol_offdiag=0.451428`
- `Step 3000`
  - `loss=0.105469 diff=0.105469 ... cond_uncond_dloss=0.271484 ... cond_pred_ratio=0.096546`
- `Step 3000 probe_semantic`
  - `mcp_offdiag(mean/min/max)=0.444347/... smol_offdiag=0.451428`
- `Step 6000`
  - `loss=0.056885 diff=0.056885 ... cond_uncond_dloss=0.203125 ... cond_pred_ratio=0.079331`
- `Step 6000 probe_semantic`
  - `mcp_offdiag(mean/min/max)=0.440565/0.362253/0.570518 smol_offdiag=0.451428`

There are two important observations here.

First:

- bridge separation does **not** need to become dramatically better for the run to improve visually.
- `mcp_offdiag` only moved from about `0.4468` to `0.4406`.

Second:

- `cond_pred_ratio` stayed materially higher than in the failing full image-only run.
- it hovered roughly in the `0.08 - 0.10` band instead of decaying toward `~0.02`.

That is already a strong signal that this small clean subset is a very different optimization regime from the noisy full image manifest.

### 9.4 Inference progression

We inferred three checkpoints with the same fair-comparison protocol:

- backend `fixed`
- `12` denoise steps
- `cfg_scale = 1.0`
- `num_frames = 1`
- seed `0`

Outputs:

- `step500`
  - `output/inference_laion_coyo_clean10_image_overfit_step500_fixed12_cfg1_f1_20260325`
- `step3000`
  - `output/inference_laion_coyo_clean10_image_overfit_step3000_fixed12_cfg1_f1_20260325`
- `final`
  - `output/inference_laion_coyo_clean10_image_overfit_final_fixed12_cfg1_f1_20260325`

Qualitative summary:

- `step500`
  - still looked semantically weak and partially collapsed
  - very similar to the earlier failure mode seen in the full image-only run
- `step3000`
  - improved clearly
  - prompt differences became much easier to see
- `step6000 / final`
  - looked noticeably better again
  - overall prompt-image alignment became much more believable

This progression matters a lot. It shows that the architecture can move from:

- early generic / weakly conditioned outputs

to:

- distinctly better prompt-conditioned outputs

under a clean small-scale image-only training setup.

### 9.5 Why this result changes the interpretation

This clean10 run is not a production recipe. It is a diagnostic control.

What it tells us:

1. The current architecture is **capable of learning**.
   - The model is not fundamentally broken.
   - Bridge + full DiT can learn useful prompt conditioning under the right conditions.

2. Diffusion-only is **not inherently hopeless**.
   - The same diffusion-only family that looked terrible on the full noisy image manifest becomes workable on a tiny clean subset.

3. The earlier full image-only failure should not be read as “the model cannot learn image generation”.
   - It is more accurate to read it as:
   - the current stack does not cope well with the scale and noise of the full image manifest on the timescale we trained.

### 9.6 What this does and does not prove

What it proves:

- the stack can learn on clean image-only data,
- prompt conditioning is not dead by design,
- there is no need to abandon the current architecture purely because of the failing full image-only run.

What it does **not** prove:

- that full-data training only needs more wall-clock time,
- that the noisy full image manifest is good enough as-is,
- or that any large-scale image run will eventually become clean10-like if left running indefinitely.

The reason we should be careful is that the full image-only run showed a very different signature:

- raw step count looked large,
- but epoch count was still low because the dataset had `43,387` samples,
- and, more importantly, `cond_pred_ratio` decayed strongly over training.

That is not the same pattern as a healthy run that is simply learning slowly.

### 9.7 Operational decision taken after this result

The clean10 positive control changed the next-step plan.

Before this run, it was still plausible that:

- the architecture itself might be too weak,
- or that diffusion-only was fundamentally the wrong family for image alignment.

After this run, the more productive interpretation became:

- architecture is probably acceptable,
- small clean subsets can work,
- so the next experiment should scale the clean subset upward in a controlled way.

This is why the next planned image experiment is:

- build a cleaner `1000`-image subset from the same larger image pool,
- train it for many epochs with no practical step cap,
- save checkpoints every `5000` steps,
- and use the best resulting checkpoint later as the image-side initialization for image+video joint training.

### 9.8 Revised bottom line after adding clean10

The updated view after adding the clean10 image positive control is:

- the current stack still has a real prompt-conditioning problem at scale,
- but it is **not** an “architecture cannot learn” problem,
- and it is **not** enough to say only “diffusion-only is bad”.

The more precise statement is:

- the architecture can learn on clean small-scale image data,
- the stack fails much more easily when data becomes noisier and larger,
- and the next most useful question is how far we can scale a clean subset before the same prompt-collapse pattern returns.

---

## 10. Architecture Update On 2026-03-27: Add A Lexical Branch From `hidden_0`

After the large-scale `clean10k` image runs continued to produce weak prompt binding, we paused and measured where SmolVLM2 was losing diversity across layers on the exact prompt probes we had been using.

This produced a new architectural decision:

- keep the existing `hidden_last / last-K` semantic stream,
- add a lexical stream from `hidden_0`,
- merge that lexical signal **inside the bridge**,
- and prefer the simple `gated add, no bottleneck` design over a compressed lexical branch.

### 10.1 Motivation: early-layer similarity measurement

Artifact:

- `output/analysis/smol_early_layer_similarity_clean10k_probe_20260327/summary.json`

We measured prompt similarity on the four `clean10k` probe prompts using:

- raw token embedding pool,
- `hidden_0`,
- several intermediate hidden layers,
- `hidden_last`.

Main results:

- `token_embedding_raw offdiag mean = 0.275702`
- `hidden_0 offdiag mean = 0.275702`
- `hidden_1 offdiag mean = 0.929846`
- `hidden_2 offdiag mean = 0.940681`
- `hidden_4 offdiag mean = 0.997026`
- `hidden_8 offdiag mean = 0.994812`
- `hidden_last offdiag mean = 0.660336`

Interpretation:

- SmolVLM2 still has strong lexical diversity at `token_embedding_raw` / `hidden_0`.
- That diversity is lost extremely early. By `hidden_1` and `hidden_2`, prompts are already highly clustered.
- Mid layers are almost fully collapsed on this probe set.
- `hidden_last` is more useful semantically than the mid layers, but it is still much less diverse than `hidden_0`.

This was the turning point.

The bridge had been consuming only the late semantic stream. That meant it was being fed a representation that was already much more clustered than the raw lexical signal available one layer earlier.

### 10.2 Design constraint: do not throw away `hidden_last`

We did **not** want to replace `hidden_last` entirely.

Reason:

- future work still needs a semantic stream that is likely useful for editing and general instruction-following,
- so the right design is not “use only `hidden_0`,”
- the right design is “preserve the semantic trunk, but inject lexical diversity back into the bridge.”

### 10.3 Two bridge ablations that were implemented

We implemented two bridge variants on top of the current MCP path.

Common idea:

- semantic stream = current `last-K` fused MCP input,
- lexical stream = `hidden_0`,
- final bridge output contract unchanged:
  - still `300 x 2304`,
  - still fed into the same DiT path.

The two variants were:

1. `mcp_lexical_gated`
   - lexical stream taken from `hidden_0`
   - normalized and added back into the semantic stream with a small learnable gate
   - **no bottleneck**

2. `mcp_lexical_bottleneck`
   - lexical stream taken from `hidden_0`
   - projected through a bottleneck MLP `960 -> 256 -> 960`
   - then added back with a learnable gate

The bottleneck version was included because it is more edge-friendly and more regularized, but it was only an ablation, not the default assumption.

### 10.4 Clean10 ablation protocol

To compare bridge designs cleanly, we trained both variants on the same `clean10` image-only overfit subset.

Configs:

- `configs/stage1_teacher_free_laion_coyo_clean10_image_overfit_lexical_gated_1gpu_20260327.yaml`
- `configs/stage1_teacher_free_laion_coyo_clean10_image_overfit_lexical_bottleneck_1gpu_20260327.yaml`

Run logs:

- `output/logs/train_laion_coyo_clean10_image_overfit_lexical_gated_gpu1_20260327.log`
- `output/logs/train_laion_coyo_clean10_image_overfit_lexical_bottleneck_gpu2_20260327.log`

Recipe:

- same `clean10` subset as the earlier positive-control image overfit,
- `SmolVLM2-500M` frozen,
- bridge trainable,
- full DiT trainable,
- diffusion-only,
- `save_every_steps = 1000`,
- `total_steps = 6000`,
- same four probe prompts,
- same inference setup used for fair comparison:
  - fixed backend,
  - `12` denoise steps,
  - `cfg_scale = 1.0`,
  - `num_frames = 1`.

When this note was updated, both runs had already completed normally:

- both reached `step6000`,
- both saved `checkpoint_step6000.pt`,
- both saved `checkpoint_final.pt`,
- no active train process remained.

### 10.5 Step4000 inference comparison

Artifacts:

- gated images:
  - `output/inference_laion_coyo_clean10_image_overfit_lexical_gated_step4000_fixed12_cfg1_f1_20260327`
- bottleneck images:
  - `output/inference_laion_coyo_clean10_image_overfit_lexical_bottleneck_step4000_fixed12_cfg1_f1_20260327`

Train-side snapshot at `step4000`:

- gated:
  - `cond_pred_ratio = 0.078495`
  - `mcp_offdiag = 0.442785`
- bottleneck:
  - `cond_pred_ratio = 0.042352`
  - `mcp_offdiag = 0.426468`

Interpretation:

- the bottleneck variant did produce slightly lower `mcp_offdiag`,
- but the gated-no-bottleneck variant had much stronger **functional prompt influence** already at `step4000`,
- and subjective image quality looked clearly better for the gated variant.

The qualitative judgment from direct image inspection was:

- both lexical variants looked better than the earlier non-lexical large-scale image runs,
- but `mcp_lexical_gated` looked noticeably stronger than `mcp_lexical_bottleneck`,
- especially in preserving prompt-specific visual details.

### 10.6 Final step6000 comparison

Final train-side numbers:

- gated:
  - `Step 6000 diff = 0.057373`
  - `cond_pred_ratio = 0.093590`
  - `mcp_offdiag = 0.445160`
- bottleneck:
  - `Step 6000 diff = 0.225586`
  - `cond_pred_ratio = 0.043350`
  - `mcp_offdiag = 0.436139`

The pattern stayed the same:

- bottleneck remained slightly more compressed in prompt space,
- but it also remained much weaker on the actual prompt-effect probe,
- and its images still looked worse.

This is an important result because it separates two ideas that could easily be confused:

- lower prompt-space similarity inside the bridge is **not** sufficient by itself,
- what matters more for generation is whether the lexical branch improves **functional prompt influence** at the DiT output.

In this ablation, the gated no-bottleneck branch won on the criterion that matters most.

### 10.7 Architecture decision taken from this ablation

We are dropping the bottleneck branch and keeping the simpler lexical-gated design.

Decision:

- selected architecture: `mcp_lexical_gated`
- rejected architecture: `mcp_lexical_bottleneck`

Reasoning:

- the similarity study showed that the most useful missing signal lives at `hidden_0`,
- the clean10 ablation showed that adding that signal back helps,
- the no-bottleneck version preserves more of that lexical signal,
- and in practice it gave better prompt binding than the bottlenecked variant.

So the updated bridge design principle is:

- preserve the late semantic trunk,
- inject `hidden_0` as a lexical residual inside the bridge,
- do **not** bottleneck that lexical stream unless later edge deployment constraints force a re-tradeoff.

### 10.8 Updated takeaway after the lexical ablation

This update matters because it changes the diagnosis again in a productive way.

Before the lexical ablation, it was still plausible that:

- the current bridge family might just be too weak,
- or that the text-conditioning path was fundamentally not recoverable.

After the similarity measurement plus clean10 lexical ablation, the better interpretation is:

- SmolVLM2 still contains useful prompt-diverse signal,
- but that signal is lost very early in the text stack,
- feeding a lexical branch from `hidden_0` into the bridge is a valid way to recover it,
- and the simplest gated-add version currently looks like the best design choice.

This does **not** solve the full large-scale image problem by itself.

But it is a meaningful architecture improvement, and it gives a much better-conditioned starting point for the next controlled training runs.

## 11. Bridge-only online distill on clean10k: why the target space had to change

After the lexical bridge update, the next hypothesis was that clean10k might still be failing because the **teacher distillation target itself was misaligned** with the native SANA conditioning path.

The important architectural observation was:

- native `Sana-video` does **not** condition directly on raw Gemma decoder hidden states,
- the teacher text path first produces Gemma decoder hidden states,
- then those hidden states are passed through `y_proj`,
- then through `attention_y_norm` when `y_norm=true`,
- and only that tensor is actually consumed by the DiT cross-attention blocks.

So distilling bridge outputs against raw teacher hidden states, even with an extra ad-hoc layer norm on both sides, was not the clean target.

The bridge was being asked to imitate a representation that the native SANA model does not itself condition on directly.

That design mismatch matters because:

- it can reward tokenwise alignment in the wrong space,
- it can overwrite useful lexical diversity coming from `hidden_0`,
- and it can still allow a collapsed solution as long as the raw-space loss goes down.

### 11.1 Distill target change that was made

The bridge-only online teacher run was therefore changed to distill inside the native SANA conditioning path:

- teacher target space: `Gemma decoder hidden -> y_proj -> attention_y_norm`
- student target space: `bridge output -> same frozen y_proj -> same frozen attention_y_norm`
- conditioner weights are frozen for the distill computation,
- so gradients still go into the bridge, but the target space itself does not drift just to satisfy distill.

This became the new baseline teacher target:

- `distill.target_space = sana_post_ynorm`
- `distill.freeze_sana_conditioner = true`

This change was motivated by **native-path fidelity**, not by a guess about which space might be more diverse.
The goal was simply to make the student imitate the exact conditioning tensor that SANA uses.

## 12. Why a second v2 update was still needed after post-ynorm distill

The post-ynorm target fixed the largest conceptual error, but the run still plateaued too early.

The key symptoms from the first bridge-only online-teacher run were:

- it was clearly healthier than the older bridge-only distill formulation,
- but prompt-space similarity still stayed too high,
- and image quality still lagged behind what the training metrics initially suggested.

The later diagnosis was that the remaining issue was no longer “wrong target space”, but a combination of:

- lexical signal being injected too weakly,
- too much normalization around the bridge output,
- no explicit in-batch anti-collapse term,
- no explicit geometry-preservation term tied to `hidden_0`,
- and a hard text truncation policy that could still throw away useful lexical cues before the bridge ever saw them.

### 12.1 Concrete reasons for each new change

#### A. Stronger lexical residual gate

Measured directly from the previous run checkpoint, the lexical gate was still effectively tiny, around the initial `~0.05` scale.
That meant the most prompt-diverse signal in the system, `hidden_0`, was only entering the bridge as a very small residual.

So the gate init was increased:

- old: `mcp_lexical_gate_init = 0.05`
- new: `mcp_lexical_gate_init = 0.2`

This is a small architectural nudge, but it is important because it directly increases the usable amplitude of the lexical branch.

#### B. Remove the extra pre-DiT layer norm

The old path effectively normalized conditioning multiple times:

- bridge output norm,
- additional trainer-side layer norm,
- and then native SANA `attention_y_norm` during target projection.

That much normalization risks flattening the geometry we actually want to preserve.

So the extra trainer-side norm before DiT was made optional and disabled in `v2`:

- `model.student.projector.pre_dit_layernorm = false`

The intent is to preserve a little more structure before the final native SANA conditioning norm.

#### C. Add in-batch contrastive distill

Token MSE and cosine losses only say:

- “make student prompt `i` close to teacher prompt `i`.”

They do **not** strongly say:

- “make student prompt `i` different from teacher prompt `j != i`.”

That is a classic gap when collapse is a concern.

So an in-batch contrastive loss was added on pooled `post_ynorm` features:

- positive: student `i` vs teacher `i`
- negatives: student `i` vs teacher `j != i`

This directly discourages the trivial “all prompts end up nearby” solution.

#### D. Add hidden0 geometry preservation

The whole lexical-bridge idea came from the earlier similarity study:

- `hidden_0` stayed very diverse,
- middle Smol layers collapsed strongly,
- `hidden_last` recovered only partial diversity.

So if the bridge is supposed to recover lost lexical distinctions, it is not enough to merely align to the teacher pointwise.

We also want the bridge output to preserve some of the **relative prompt geometry** already present in `hidden_0`.

That motivated the new geometry loss:

- pool student conditioning in the distill space,
- pool `hidden_0`,
- match their in-batch similarity structure.

This is meant to protect prompt separation rather than only semantic alignment.

#### E. Add functional distill through the frozen DiT

Pointwise conditioning similarity is still only a proxy.
What we actually care about is whether the student conditioning produces the **same denoiser behavior** as the teacher conditioning.

So a small functional loss was enabled:

- run the frozen DiT with teacher conditioning,
- compare prediction against the same frozen DiT with student conditioning,
- add light MSE / cosine penalties on those predictions.

This is valuable because it supervises the bridge in the same functional space that matters for generation.

#### F. Replace hard tail-only truncation with an edge-friendly full-window selector

This was the most subtle update.

The original strict SANA-parity path kept only:

- BOS token,
- plus the final tail tokens needed to fit the `300` token budget.

That is simple, but it assumes the most useful lexical information is always concentrated at the end of the sequence.
For long captions, that is often wrong.

Important object words or rare attributes can appear near the start or middle, while the tail may contain generic stylistic or descriptive residue.

A full learned resampler could address this, but that would add parameters and make the bridge heavier.
The edge-friendly alternative that was implemented instead is:

- enable a full text window first,
- then select back down to `300` tokens using a lightweight deterministic strategy,
- with no new learned parameters.

The selected strategy for `v2` is:

- `strict_sana_use_full_text_window = true`
- `strict_sana_token_select_strategy = head_uniform_tail`
- `strict_sana_head_tokens = 96`
- `strict_sana_tail_tokens = 96`

Meaning:

- keep BOS,
- keep a sizable head chunk,
- keep a sizable tail chunk,
- and sample the middle region uniformly for the remaining budget.

This is attractive because it is:

- zero-parameter,
- cheap,
- easy to ablate,
- and much more likely to preserve lexical cues than pure tail-only truncation.

## 13. The clean10k bridge-only online-teacher v2 recipe

The new development recipe keeps the overall high-level setup unchanged:

- dataset: `clean10k`
- `batch_size = 16`
- bridge-only training
- frozen DiT weights
- diffusion loss retained
- online teacher retained

But the conditioning/auxiliary objective is now explicitly stronger:

- distill in `sana_post_ynorm` space
- frozen SANA conditioner for target projection
- stronger lexical residual init
- no trainer-side pre-DiT layer norm
- in-batch contrastive distill
- hidden0 geometry loss
- functional distill
- full-window `head_uniform_tail` token selection

This is intentionally still edge-conscious:

- the bridge is still small,
- the token-selection update adds no learned module,
- and all new behavior is toggleable for clean ablation study.

## 14. Early v2 evidence

The `v2` run is still ongoing at the time of this note update, so the evidence is early.
But the early trend is already meaningful.

Representative early comparison against the previous bridge-only online-teacher run:

- previous run, `step200`:
  - `loss = 1.4748`
  - `offdiag_cos = 0.8761`
  - `mcp_offdiag = 0.9112`
  - `cond_pred_ratio = 0.1141`
- `v2`, `step200`:
  - `loss = 0.7484`
  - `offdiag_cos = 0.5042`
  - `mcp_offdiag = 0.6024`
  - `cond_pred_ratio = 0.0412`

This should not be over-interpreted, but it is still a very important signal:

- the bridge prompt space is **much less collapsed** in `v2`,
- and that was the main objective of the redesign.

Later early-run `v2` numbers also stayed in that healthier regime:

- `step400`:
  - `loss = 0.6246`
  - `offdiag_cos = 0.5419`
  - `mcp_offdiag = 0.6011`
  - `cond_pred_ratio = 0.1371`
- `step500`:
  - `loss = 0.6031`
  - `offdiag_cos = 0.5100`
  - `d_nce = 1.1815`
  - `d_h0geom = 0.1617`

This is not yet evidence that generation quality is solved.
But it is evidence that the updated bridge objective is behaving much more like the intended design.

## 15. Current architecture decision after the v2 update

The working hypothesis going forward is now:

- keep the lexical-gated bridge,
- keep the native-path teacher target (`post_ynorm`),
- keep bridge-only distill as the most controlled setup,
- and continue treating prompt geometry preservation as a first-class objective rather than a side effect.

The key design lesson from this phase is:

- choosing a more diverse input layer such as `hidden_0` is necessary,
- but it is not sufficient,
- because the bridge can still collapse that diversity later unless the target space and the auxiliary losses explicitly protect it.

That is the real motivation behind the latest update series.
