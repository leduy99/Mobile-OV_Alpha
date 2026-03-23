# Stage 1 Sem+Vis Restart Note (2026-03-21)

## Why we stopped the hybrid continuation line

We stopped the `stage25_hybrid_semvis_funcdistill_stage1init` line after checking both `5k` and `10k` inference.
The line remained numerically stable, but the qualitative outputs did not improve in a meaningful way.
The `5k -> 10k` comparison stayed in the same failure family:

- `golden retriever`: abstract sepia blobs, still not forming `dog / beach / sunset`
- `chef`: still blown-out, weak structure, no reliable `chef / vegetables / kitchen`

This strongly suggests the problem is upstream of the hybrid continuation itself.
Our current interpretation is that the earlier semantic-only Stage 1 objective likely learned a manifold that is cleaner in prompt space, but still not functionally aligned enough with the generator interface. By the time we try to repair it downstream with Stage 2 / Stage 2.5, the line is already too biased toward a poor interface.

## New decision

Instead of continuing to patch later stages, we will restart from pretrained base and change the recipe from the beginning.

This new recipe is meant to learn both:

- stronger prompt semantics than the older bridge-only / online-teacher line
- better visual compatibility than the prompt-only distillation line

## New 500M recipe

Config:
- `configs/stage1_teacher_free_joint_openvid_current_laion_coyo_semvis_frombase_5v1i_2gpu_20260321.yaml`

Launcher:
- `scripts/train_openvid_current_laion_coyo_stage1_semvis_frombase_5v1i.sh`

Key design choices:
- restart from pretrained base, not from any Stage 1 or Stage 2 checkpoint
- keep the semantic student path trainable:
  - bridge full
  - SmolVLM2 text LoRA
  - top-2 text layers
  - final norm
- allow a small amount of generator-side adaptation from step 0:
  - DiT cross-attention LoRA only
- keep the teacher anchoring losses:
  - token MSE
  - token cosine
  - pooled cosine
  - functional DiT-response matching
- reduce proxy-loss dominance relative to the previous Stage 2 / Stage 2.5 lines so diffusion quality matters more
- keep `5V:1I`
- use `grad_accum_steps=2` to reduce diffusion-side noise a bit without increasing memory much

## Why this is different from the failed continuation line

The failed continuation line inherited a student/bridge state that was already shaped by the earlier Stage 1 manifold. The new line removes that inheritance.

This new run asks a cleaner question:

Can we learn semantic alignment and visual compatibility together from the pretrained base, instead of learning them in separate stages and trying to splice them later?

## Prepared 2.2B follow-up

We also prepared a parallel config and launcher for a larger text model if the new 500M restart still fails:

Config:
- `configs/stage1_teacher_free_joint_openvid_current_laion_coyo_semvis_frombase_smolvlm2_2p2b_5v1i_2gpu_20260321.yaml`

Launcher:
- `scripts/train_openvid_current_laion_coyo_stage1_semvis_frombase_2p2b_5v1i.sh`

The 2.2B version keeps the same recipe but switches to:
- `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- a converted checkpoint path under `omni_ckpts/smolvlm2_2.2b/`
- slightly lighter student adaptation (`top-1 layer`, smaller student LoRA rank) to control memory

## Rollout order

1. Run the new `500M` sem+vis-aware restart from pretrained base.
2. Judge the line early at `5k` and `10k` with the same clean fixed-backend inference protocol.
3. If the new 500M recipe still fails qualitatively, move to the prepared `2.2B` variant.
