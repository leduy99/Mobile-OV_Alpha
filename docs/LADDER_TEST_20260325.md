# Ladder Test - SmolVLM2 500M Reproduction (Step 6000)

Target checkpoint:
- `output/stage1_bridge_dit_openvid_current_laion_coyo_online_teacher_30k_repro_20260324_2gpu/20260324_231255/checkpoint_step6000.pt`

Target recipe:
- `SmolVLM2-500M`
- `bridge + full DiT`
- `online distill`
- `semantic_probe`
- `1V:1I`
- `cfg_dropout_prob = 0.0`

## Goal

Use a short diagnostic ladder to answer four questions:
1. Does the model react to prompt changes at all?
2. Is the failure mainly caused by inference CFG mismatch?
3. Is student conditioning functionally far from teacher conditioning?
4. Is the dominant failure mode semantic collapse, prompt ignore, or weak prompt binding?

## Artifacts

Prompt sensitivity:
- outputs: `output/ladder_smol_repro_step6000_prompt_sensitivity_20260325`
- logs: `output/logs/ladder_prompt_sensitivity_step6000_*_20260325.log`
- CLIP summary: `output/analysis/ladder_prompt_sensitivity_step6000_clip_20260325.json`
- CLIP cross-scores: `output/analysis/ladder_prompt_sensitivity_step6000_clip_crossscores_20260325.json`

CFG sweep:
- outputs: `output/ladder_smol_repro_step6000_cfg_sweep_20260325`
- logs: `output/logs/ladder_cfg_sweep_step6000_*_20260325.log`
- CLIP summary: `output/analysis/ladder_cfg_sweep_step6000_clip_20260325.json`

Probe-only diagnostics:
- config: `configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_dit_online_teacher_30k_repro_probeonly_step6000_1gpu_20260325.yaml`
- log: `output/logs/ladder_probeonly_step6000_20260325.log`

Teacher-vs-student substitution:
- script: `tools/analysis/teacher_student_substitution_test.py`
- JSON: `output/analysis/teacher_student_substitution_step6000_20260325.json`
- log: `output/logs/teacher_student_substitution_step6000_20260325.log`

## Rung 1 - Prompt Sensitivity

Prompts used:
- `a golden retriever running along a beach at sunset`
- `a chef slicing colorful vegetables in a busy kitchen`
- `a barista pouring latte art in a cozy cafe`
- `an astronaut floating above earth in space`
- `""` (via `__EMPTY__` sentinel)

### CLIP text-image score on `frame000`
- `golden`: `0.0317`
- `chef`: `0.1997`
- `latte`: `0.1662`
- `astronaut`: `0.1687`

### CLIP image-image similarity between outputs
- `golden vs chef`: `0.6705`
- `golden vs latte`: `0.7042`
- `golden vs astronaut`: `0.6100`
- `chef vs empty`: `0.6816`
- `astronaut vs empty`: `0.6265`

Interpretation:
- Outputs are not identical, so this is not a pure hard-collapse case.
- However, similarity is still fairly high across unrelated prompts.
- The model changes mode somewhat, but not enough to reliably bind the requested object/action.

### Cross-score matrix (image vs all prompts)
A useful failure pattern appears in `output/analysis/ladder_prompt_sensitivity_step6000_clip_crossscores_20260325.json`:
- `golden image -> golden prompt`: `0.0317`
- `golden image -> chef prompt`: `0.2078`
- `golden image -> latte prompt`: `0.1711`
- `golden image -> astronaut prompt`: `0.1573`

Interpretation:
- The golden-retriever output is not just weakly aligned.
- It is aligned more strongly with several wrong prompts than with its own prompt.
- This is strong evidence for prompt-binding failure, not merely weak visual quality.

## Rung 2 - CFG Sweep

Prompts tested:
- `golden`
- `chef`

### CLIP text-image score on `frame000`
- `golden cfg=1.0`: `0.0881`
- `golden cfg=2.0`: `0.0378`
- `golden cfg=3.0`: `0.0317`
- `chef cfg=1.0`: `0.2007`
- `chef cfg=2.0`: `0.2055`
- `chef cfg=3.0`: `0.1997`

Interpretation:
- `golden` improves noticeably when CFG is reduced to `1.0`.
- `chef` is mostly insensitive to the sweep.
- Therefore CFG mismatch is not the whole problem, but it is a real secondary issue for at least some prompts.
- Since training uses `cfg_dropout_prob = 0.0` while inference uses CFG > 1, the mismatch is plausible and should be treated as real.

## Rung 3 - Probe-Only Functional Diagnostics

Probe-only run uses the actual training pipeline, initialized from `step6000`, but exits before backward.

Relevant lines from `output/logs/ladder_probeonly_step6000_20260325.log`:

> `ProbeOnly Step 1 | ... cond_uncond_dloss=0.308594 cond_pred_l2=37.519650 cond_pred_ratio=0.021121`
>
> `ProbeOnly Step 1 | ... cond_uncond_dloss=0.906250 cond_pred_l2=31.936808 cond_pred_ratio=0.020438`
>
> `ProbeOnly Step 1 | ... cond_uncond_dloss=1.218750 cond_pred_l2=35.928421 cond_pred_ratio=0.021791`

Interpretation:
- Replacing conditioning with unconditioned context makes diffusion loss worse (`cond_uncond_dloss > 0`).
- So the model is not fully ignoring the prompt.
- But `cond_pred_ratio` is only around `~0.02`, which means changing the prompt causes only a small change in the DiT prediction relative to prediction magnitude.
- This points to weak conditioning strength rather than total prompt blindness.

## Rung 4 - Teacher-vs-Student Substitution

Aggregate results from `output/analysis/teacher_student_substitution_step6000_20260325.json`:
- `student_teacher_pred_mse`: `0.0132`
- `student_teacher_pred_cos`: `0.9979`
- `student_vs_uncond_pred_mse`: `0.0316`
- `teacher_vs_uncond_pred_mse`: `0.0207`

Per-prompt context cosine is much lower:
- `student_teacher_ctx_cos`: roughly `0.42 - 0.48`

Interpretation:
- Representation-level alignment between student and teacher embeddings is only moderate.
- But functionally, once both pass through DiT on the same `x_t, t`, their predictions are still extremely close.
- This means the main failure is probably not that the bridge sends DiT into a completely wrong regime relative to teacher.
- The more likely problem is that the overall prompt-conditioning effect is weak, and the generator can satisfy diffusion loss while staying close to a generic visual prior.

## Combined Diagnosis

The ladder test suggests the following:

1. The model is **not** in pure semantic collapse.
- Different prompts do produce somewhat different outputs.
- `cond_uncond_dloss > 0` also confirms prompt information is being used.

2. The dominant failure is **weak or misdirected prompt binding**.
- Cross-scores show some outputs match wrong prompts better than their own prompt.
- Prompt changes do not move DiT prediction strongly enough (`cond_pred_ratio ~ 0.02`).

3. The dominant failure is **not** catastrophic student-teacher functional mismatch.
- `student_teacher_pred_cos ~ 0.998` is too high for that story.
- Student and teacher contexts differ at embedding level, but DiT reacts to them very similarly.

4. CFG mismatch is a **real secondary issue**.
- `golden` is clearly better at `cfg=1.0` than `cfg=3.0`.
- This should be fixed or at least controlled in evaluation.

## Practical Next Steps

Most promising next experiments:

1. Train/evaluate with CFG parity.
- Short term: prefer `cfg_scale=1.0` for diagnosis.
- Medium term: train with non-zero `cfg_dropout_prob` if inference will use CFG > 1.

2. Increase prompt effect strength rather than only representation alignment.
- Current results suggest the model already uses prompt information, but too weakly.
- The next loss should encourage larger functional separation between prompt-conditioned and unconditioned predictions.

3. Add a prompt-contrastive functional objective.
- Example: explicitly maximize difference between predictions under correct prompt and unrelated prompt on the same `x_t, t`.
- This targets the weak `cond_pred_ratio` failure directly.

4. Keep using semantic anti-collapse, but do not expect it alone to solve binding.
- The failure here is not mainly “collapse”.
- It is more “prompt influence too small / too generic”.

## Bottom Line

This checkpoint is failing mostly because:
- prompt information is present,
- student conditioning is not drastically worse than teacher conditioning inside DiT,
- but the **effective conditioning signal is too weak** to reliably steer generation toward the requested object/action.

That means the next fix should focus first on **strengthening functional prompt influence**, not on changing backbone again or only adding more teacher-representation matching.
