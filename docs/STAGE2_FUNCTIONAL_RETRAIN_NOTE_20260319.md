# Stage 2 Functional Retrain Note (2026-03-19)

This note records why we introduced the 3-stage training plan, what we observed in practice, and why Stage 2 was modified and relaunched on 2026-03-19.

## Why we moved to a 3-stage plan

The original direct diffusion training path repeatedly showed a soft failure mode:
- training stayed numerically stable,
- checkpoints loaded correctly,
- inference eventually became reproducible after removing the buggy legacy backend,
- but prompt sensitivity and video quality remained weak.

The key working hypothesis became:
- the conditioning stack was not strong enough,
- and asking `SmolVLM2 -> bridge -> DiT` to learn everything only through diffusion loss was too hard.

That led to the following 3-stage plan.

### Stage 1: prompt-only teacher distillation
Goal:
- teach the student text path and bridge to imitate the SANA / Gemma conditioning space before involving diffusion.

Why:
- this isolates the conditioner problem,
- gives us a teacher anchor,
- and avoids the generator compensating too early for a weak student manifold.

### Stage 2: bridge reinjection with frozen DiT
Goal:
- feed the distilled student conditioning back into diffusion,
- but keep the DiT frozen so we can test whether the student manifold is already compatible with what the pretrained generator expects.

Why:
- if DiT is unfrozen too early, it can learn to work around a flawed conditioner,
- which hides the real bottleneck and makes diagnosis much harder.

### Stage 3: bridge + DiT joint training
Goal:
- only after the student-conditioned manifold is reasonably compatible with the frozen generator,
- allow DiT to adapt jointly and improve generation quality.

Why:
- Stage 3 should be refinement,
- not a rescue path for a still-misaligned conditioner.

## What Stage 1 told us

Stage 1 looked directionally successful.

Observed signs:
- prompt-only distillation converged cleanly,
- prompt diversity improved compared with the older no-distill and weak-distill runs,
- semantic collapse looked noticeably reduced during inference,
- easy prompts stopped collapsing into nearly identical outputs as often.

Interpretation:
- Stage 1 was probably not placebo,
- teacher-guided conditioner distillation is likely the correct direction,
- the student-side prompt manifold can be improved in isolation.

## What the first Stage 2 told us

The first Stage 2 setup kept DiT frozen and already used teacher-token losses, but it still produced weak results.

Symptoms:
- easy prompts looked less bad than hard prompts,
- hard prompts remained very weak,
- outputs still looked like a weakly-conditioned denoiser rather than a strongly prompt-grounded generator,
- generic semantic probe metrics became hard to interpret.

The important shift in diagnosis was:
- the problem was no longer simply “the student prompt space is too collapsed”,
- the problem became “the student-conditioned manifold is not yet functionally equivalent to the teacher-conditioned manifold for the frozen DiT”.

That distinction matters.

A student token tensor can be:
- closer to teacher tokens under MSE,
- closer under cosine,
- still not trigger the same behavior inside the frozen DiT.

In practice, the frozen DiT cares about more than absolute token closeness:
- interface statistics,
- covariance structure,
- token organization,
- cross-attention usage,
- and the effective response of the denoiser under the same noisy latent and timestep.

## Why the old Stage 2 was not enough

The previous Stage 2 was still under-constrained for the real task.

It mostly optimized:
- token MSE,
- token cosine,
- pooled cosine,
- and a generic semantic regularizer based on variance / covariance / geometry heuristics.

What it did **not** optimize directly:
- whether `DiT(x_t, student_condition)` behaves like `DiT(x_t, teacher_condition)`.

That became the central issue.

We also became less confident that the generic semantic probe should dominate Stage 2 decisions.

Reason:
- even if those regularizers improve some generic geometry notion,
- they may still push the student away from the actual Gemma / SANA manifold that the frozen DiT expects.

So the Stage 2 question changed from:
- “is the student embedding space well-behaved in general?”

to:
- “is the student embedding space correct for the exact teacher-relative interface used by the frozen DiT?”

## Why we decided to modify Stage 2 instead of continuing or jumping to Stage 3

We did **not** want to:
- keep training the old Stage 2 unchanged for a long time,
- or jump into Stage 3 too early.

Why not continue old Stage 2 as-is:
- it could keep improving token losses without improving actual generation behavior,
- which would waste time and give misleading optimism.

Why not jump to Stage 3:
- if DiT is unfrozen too early, it can learn to compensate for a flawed student manifold,
- which makes the system look better without actually solving conditioner-to-generator compatibility.

The practical decision was:
- keep Stage 1,
- revise Stage 2,
- retrain Stage 2 from the best Stage 1 checkpoint.

## What changed in the new Stage 2 (v2)

The revised Stage 2 keeps the original spirit of reinjection, but adds a stronger functional constraint.

### Kept from the earlier Stage 2
- start from the best Stage 1 prompt-distill checkpoint,
- keep DiT frozen,
- keep teacher-token supervision,
- keep the `5 video : 1 image` schedule,
- keep the same student adaptation family.

### Changed in Stage 2 v2
- add a **functional DiT-response distillation loss**,
- keep teacher-token losses as support,
- disable the generic semantic probe in this run,
- focus the objective on teacher-relative functional matching instead of generic embedding geometry.

### New idea
For the same noisy latent `x_t` and timestep `t`, compare:
- `DiT(x_t, teacher_condition)`
- `DiT(x_t, student_condition)`

Then penalize the difference directly.

This makes Stage 2 care about:
- how the frozen generator actually reacts,
- not only whether student tokens look close to teacher tokens.

### New losses added
- functional prediction MSE
- functional prediction cosine loss

These are tracked in logs as:
- `f_pred_mse`
- `f_pred_cos`

## Why this change is important

This change makes Stage 2 much closer to the real bottleneck.

We are no longer only asking:
- “does the student look like the teacher in token space?”

We are now also asking:
- “does the student make the frozen DiT behave like it would under teacher conditioning?”

That is the actual compatibility problem we care about before entering Stage 3.

## Current run to remember

Stage 2 v2 was relaunched on 2026-03-19 from the best available Stage 1 checkpoint.

Important run facts:
- DiT remains frozen,
- the run starts from Stage 1 step 50k,
- functional distillation is enabled from the start,
- the new log fields are `f_pred_mse` and `f_pred_cos`.

Early signal from the new run was encouraging enough to continue:
- the run passed a real one-step boot test,
- then launched cleanly on 2 GPUs,
- and both token-space losses and functional losses started decreasing early.

This is **not** enough to claim success yet,
but it is enough to justify continuing the revised Stage 2 rather than reverting immediately.

## What we should monitor next

The next decisions should be based on whether Stage 2 v2 closes the teacher-student gap in a way that matters for the frozen DiT.

Priority checks:
- teacher vs student substitution test under the same frozen DiT,
- trend of `f_pred_mse` and `f_pred_cos`,
- same-seed prompt swap behavior,
- easy vs hard prompt gap,
- whether student-conditioned outputs move closer to teacher-conditioned outputs.

## Decision rule before Stage 3

We should only move to Stage 3 when Stage 2 gives at least one strong sign that the interface problem is improving.

Examples of acceptable signs:
- functional DiT-response gap decreases clearly,
- teacher-vs-student substitution gap narrows,
- same-seed prompt swaps begin to produce cleaner semantic deltas,
- easy prompts become meaningfully closer to teacher-conditioned behavior.

If those signals do not improve, Stage 3 should be delayed.

## What we saw by step 10k

By the time the revised Stage 2 reached roughly 10k steps, the run still looked numerically healthy.

Stable signs:
- no NaNs or hard divergence,
- gradients stayed finite,
- embedding scale stayed stable,
- the run continued cleanly on 2 GPUs.

Loss behavior was mixed rather than uniformly good.

What improved:
- token-space distillation losses improved compared with the early part of the run,
- functional DiT-response losses also improved,
- total loss became lower on average than in the earliest training window.

What did **not** improve:
- diffusion loss did not show a clean downward trend,
- it stayed noisy and in some windows looked slightly worse than earlier,
- so better total loss mostly came from better teacher-matching, not better denoising quality.

This matters because the revised Stage 2 objective is still dominated by teacher-relative matching terms.
That means:
- the run can look better in loss space,
- while final video quality still fails to improve enough.

## What inference said at 5k and 10k

We directly compared checkpoint 5k and checkpoint 10k using the same clean fixed inference setup:
- same backend,
- same seed,
- same resolution,
- same number of frames,
- same sampling settings.

Two anchor prompts were used first:
- `a golden retriever running along a beach at sunset, cinematic motion, gentle ocean waves`
- `a chef slicing colorful vegetables in a bright kitchen, realistic hand motion, shallow depth of field`

Direct qualitative result:
- checkpoint 10k was **not** a convincing improvement over checkpoint 5k,
- both checkpoints still produced largely abstract, weakly grounded outputs,
- the chef prompt at 10k looked slightly less empty than at 5k,
- but the golden retriever prompt still failed badly at both checkpoints,
- there was no clear qualitative jump that matched the better loss curves.

We also ran 4 additional prompts at checkpoint 10k:
- cherry blossom blooming time-lapse,
- snowy forest aerial view,
- astronaut outside a space station,
- violinist performing under a spotlight.

Result:
- all 4 were still poor,
- the snowy forest prompt was the least bad because it retained the most scene-like structure,
- but none of the 4 showed reliable prompt grounding.

So by 10k, the practical reading is:
- Stage 2 v2 is probably learning something useful about teacher-relative matching,
- but that improvement has **not yet transferred** into clear video-quality gains.

## Current interpretation after the 10k checkpoint

The revised Stage 2 should **not** be judged a failure yet, but it also should **not** be treated as solved.

Most likely interpretation:
- the new objective is closer to the real bottleneck than the old Stage 2 objective,
- but it is still not sufficient by itself to guarantee good generation quality by 5k or 10k.

This is still compatible with the main diagnosis:
- Stage 1 improved the conditioner,
- Stage 2 is an interface-calibration problem,
- and functional matching is the right direction,
- but the current formulation may still be too weak or too local to fully fix generation behavior.

## Current decision

The working decision after reviewing losses and 5k / 10k inference is:
- keep the revised Stage 2 running until **20k**,
- do not move to Stage 3 yet,
- and do not over-read the current loss improvements as proof that generation quality is fixed.

Why 20k:
- 5k was too early,
- 10k still looked inconclusive,
- 20k is a reasonable cap for this Stage 2 recipe before deciding whether it needs another redesign.

What 20k should answer:
- whether the qualitative gap between teacher-matching losses and actual inference quality begins to close,
- whether easy prompts improve beyond the current abstract failure mode,
- and whether Stage 2 v2 is genuinely progressing or simply optimizing a still-incomplete proxy objective.

## Short summary

The 3-stage plan was introduced because direct diffusion training kept producing weak prompt grounding.

Stage 1 improved the conditioner in isolation.

The original Stage 2 showed that token-space matching alone was not enough.

So on 2026-03-19 we changed Stage 2 to focus on **functional manifold matching**:
- keep DiT frozen,
- keep teacher-token losses,
- add direct teacher-vs-student DiT-response matching,
- and retrain Stage 2 from the best Stage 1 checkpoint.

That is the current working strategy.
