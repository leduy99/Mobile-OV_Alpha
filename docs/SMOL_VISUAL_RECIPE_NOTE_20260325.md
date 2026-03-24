# Smol Visual Recipe Note (2026-03-25)

## Summary

This note captures the `SmolVLM2-500M` recipe family that currently looks the
most trustworthy after the failed Qwen experiments.

This is **not** a claim that the line is already solved.
It is a reminder that this family:

- stays numerically stable,
- does not show the same geometry drift seen in the Qwen phase-2 runs,
- keeps semantic anti-collapse metrics in a reasonable range,
- and still looks like a plausible visual recipe worth waiting on.

## Why this note exists

We wanted to verify whether the older visually stronger family could still be
reproduced on the current codebase.

The original `2026-03-16 initfrom30k_nodistill` config depended on:

- `output/stage1_bridge_dit_openvid_current_laion_coyo_online_teacher_30k_20260315_2gpu/20260315_201712/checkpoint_final.pt`

That upstream init checkpoint was no longer on disk.
So instead of pretending we could reproduce the downstream line directly, we
restarted from the upstream ancestor recipe first.

## Canonical reproduction run

### Config

- [stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_dit_online_teacher_30k_repro_2gpu_20260324.yaml](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/configs/stage1_teacher_free_joint_openvid_current_laion_coyo_bridge_dit_online_teacher_30k_repro_2gpu_20260324.yaml)

### Launcher

- [train_openvid_current_laion_coyo_bridge_dit_online_teacher_30k_repro_gpu56_20260324.sh](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/scripts/train_openvid_current_laion_coyo_bridge_dit_online_teacher_30k_repro_gpu56_20260324.sh)

### Log

- [train_bridge_dit_smolvlm2_500m_online_teacher_30k_repro_gpu56_20260324.log](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/output/logs/train_bridge_dit_smolvlm2_500m_online_teacher_30k_repro_gpu56_20260324.log)

### Output dir

- `output/stage1_bridge_dit_openvid_current_laion_coyo_online_teacher_30k_repro_20260324_2gpu`

## Exact recipe to remember

This is the recipe shape that currently looks worth keeping:

- backbone: `SmolVLM2-500M`
- bridge: trainable
- DiT: full trainable
- distill: `online_only`
- semantic anti-collapse: enabled
- joint schedule: `1V:1I`
- `grad_accum_steps=1`
- `flow_shift=3.0`
- `save_every_steps=1000`

### Important loss settings

- `diff.weight = 1.0`
- `distill.token_mse_weight = 1.0`
- `distill.token_cos_weight = 0.5`
- `distill.pooled_cos_weight = 0.2`
- `semantic_probe.weight = 0.2`
- `semantic_probe.var_weight = 1.0`
- `semantic_probe.cov_weight = 0.05`
- `semantic_probe.geom_weight = 1.0`

### Learning rates

- `bridge lr = 5e-5`
- `dit lr = 1e-5`
- `max_grad_norm = 0.1`

## Why this family looks healthier than the Qwen runs

Compared with the recent Qwen bridge-only and Qwen phase-2 runs, this Smol line
does **not** show the same alarming probe behavior.

What looked encouraging:

- `sem_cov` stayed low instead of exploding,
- `sem_geom` stayed small,
- distill losses remained active without destabilizing the run,
- and the line stayed in a familiar, older training regime that had previously
  produced the best visual family we saw.

Example from the reproduction log around `4k`:

- `diff` around `0.15`
- `d_mse` around `0.95`
- `d_cos` around `0.47`
- `d_pool` around `0.19`
- `sem_var` around `0.538`
- `sem_cov` around `0.016`
- `sem_geom` around `0.025`

This is very different from the Qwen phase-2 behavior, where `sem_cov` climbed
into the `~0.20` range and prompt-space geometry looked badly distorted.

## Early inference status

Latest early checkpoint manually checked:

- [checkpoint_step4000.pt](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/output/stage1_bridge_dit_openvid_current_laion_coyo_online_teacher_30k_repro_20260324_2gpu/20260324_231255/checkpoint_step4000.pt)

Inference output folder:

- [inference_bridge_dit_smolvlm2_500m_online_teacher_30k_repro_step4000_fixed12_20260325](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/output/inference_bridge_dit_smolvlm2_500m_online_teacher_30k_repro_step4000_fixed12_20260325)

Direct prompt files:

- golden video: [mp4](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/output/inference_bridge_dit_smolvlm2_500m_online_teacher_30k_repro_step4000_fixed12_20260325/q1_student_20260325_011514_a_golden_retriever_running_along_a_beach.mp4)
- golden frame: [png](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/output/inference_bridge_dit_smolvlm2_500m_online_teacher_30k_repro_step4000_fixed12_20260325/q1_student_20260325_011514_a_golden_retriever_running_along_a_beach_frame000.png)
- chef video: [mp4](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/output/inference_bridge_dit_smolvlm2_500m_online_teacher_30k_repro_step4000_fixed12_20260325/q1_student_20260325_011628_a_chef_slicing_colorful_vegetables_in_a_.mp4)
- chef frame: [png](/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/output/inference_bridge_dit_smolvlm2_500m_online_teacher_30k_repro_step4000_fixed12_20260325/q1_student_20260325_011628_a_chef_slicing_colorful_vegetables_in_a__frame000.png)

Interpretation of `4k`:

- still too early to call it a successful reproduction,
- not yet visually strong enough to declare victory,
- but the line still looks more credible than the recent Qwen attempts,
- so this is a run to **wait on**, not a run to kill early.

## What to do with this family later

If this line improves by `5k`, `10k`, or `20k`, this family should be treated
as a candidate base recipe again.

The most natural future branch is:

1. finish or at least validate this upstream `online_teacher_30k` family,
2. if it looks good enough, recreate the old downstream idea:
   - `initfrom30k_nodistill`
   - same `SmolVLM2-500M`
   - same `1V:1I`
   - same full-DiT regime
   - but with distill disabled in phase 2

## Practical takeaway

If we get lost later, this is the recipe family to revisit first:

- `SmolVLM2-500M`
- `bridge + full DiT`
- `online distill`
- `semantic_probe`
- `1V:1I`
- `grad_acc=1`

It is currently the closest thing we have to a plausible "stable visual recipe"
after the recent Qwen detours.
