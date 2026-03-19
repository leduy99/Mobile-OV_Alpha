# DiT Sharding Fail-Safe Notes (2026-02-28)

## Why this was added

We hit repeated OOM during full `T=21` training even with large GPUs.  
Root cause from logs: DiT was running in **manual gradient all-reduce mode** instead of being wrapped by FSDP/DDP/DeepSpeed.

Evidence from old OOM run:
- `output/logs/train_joint_v4plus_openvid_partial_allavail_3gpu_ncclp2poff_20260227_175524.log:46`
- `output/logs/train_joint_v4plus_openvid_partial_allavail_3gpu_ncclp2poff_20260227_175524.log:55`
- `output/logs/train_joint_v4plus_openvid_partial_allavail_3gpu_ncclp2poff_20260227_175524.log:68`
  - `Rank X using manual gradient all-reduce for DiT (FSDP/DDP disabled)`
- Same run then OOM:
  - `output/logs/train_joint_v4plus_openvid_partial_allavail_3gpu_ncclp2poff_20260227_175524.log:121`
  - `output/logs/train_joint_v4plus_openvid_partial_allavail_3gpu_ncclp2poff_20260227_175524.log:165`
  - `output/logs/train_joint_v4plus_openvid_partial_allavail_3gpu_ncclp2poff_20260227_175524.log:224`


## Code changes

### 1) Multi-GPU fail-fast for DiT sharding path

File:
- `tools/train_stage1_teacher_free.py:1811`

New behavior:
- In multi-GPU, if DiT is **not** wrapped by `FSDP/DDP/DeepSpeed`, training now fails by default.
- Manual DiT sync fallback is allowed **only** when explicitly set:
  - `run.allow_manual_dit_sync: true`

Relevant checks:
- `tools/train_stage1_teacher_free.py:1677` reads `run.require_dit_sharding`
- `tools/train_stage1_teacher_free.py:1811` reads `run.allow_manual_dit_sync`
- `tools/train_stage1_teacher_free.py:1819` raises `RuntimeError` when unsafe fallback is not allowed

Intent:
- Prevent accidental launch in memory-heavy manual DiT sync mode.
- Force explicit choice instead of silent fallback.

### 1.1) Before vs After (code-level)

File:
- `tools/train_stage1_teacher_free.py`

Before (effective behavior):
- Manual DiT fallback was possible when:
  - `world_size > 1`
  - `model.dit.fsdp=false`, `model.dit.ddp=false`, `model.dit.deepspeed=false`
  - and `run.require_dit_sharding` was not set/false
- In that case, launch continued and logs showed:
  - `Rank X using manual gradient all-reduce for DiT (FSDP/DDP disabled)`

After:
- Added an explicit gate:
  - `run.allow_manual_dit_sync` (default false behavior expected in production configs)
- New rule in multi-GPU:
  - if DiT is not wrapped by FSDP/DDP/DeepSpeed, raise runtime error unless `run.allow_manual_dit_sync=true`
- This prevents silent/manual fallback by default.

Representative post-fix code path:
```python
dit_is_wrapped = bool(use_fsdp or use_dit_ddp or use_dit_deepspeed)
allow_manual_dit_sync = bool(getattr(cfg.run, "allow_manual_dit_sync", False))
if not dit_is_wrapped:
    if require_dit_sharding or (not allow_manual_dit_sync):
        raise RuntimeError(...)
```


### 2) Existing positive signal to confirm sharding

File:
- `tools/train_stage1_teacher_free.py:1122`

Expected startup log when correct:
- `Wrapping DiT with FSDP (use_orig_params=True, sync_module_states=False, sharding=full_shard)`


## Config changes used for stable debug baseline

File:
- `configs/stage1_teacher_free_joint_openvid_partial_msrvtt_laion_coyo_ivjoint_3gpu_20260227.yaml`

Key flags:
- `run.require_dit_sharding: true` (`:18`)
- `run.allow_manual_dit_sync: false` (`:19`)
- `model.dit.fsdp: true`
- `model.dit.fsdp_sharding_strategy: full_shard`
- `train.latent_window.frames: null` (full latent `T=21`)
- `model.student.lora.enable: false` (freeze SmolVLM2 LoRA during OOM debugging)


## Example: bad vs good startup

### Bad (unsafe fallback, likely OOM)

You will see:
- `Rank X using manual gradient all-reduce for DiT (FSDP/DDP disabled)`

Example:
- `output/logs/train_joint_v4plus_openvid_partial_allavail_3gpu_ncclp2poff_20260227_175524.log:46`


### Good (sharded DiT)

You should see:
- `Wrapping DiT with FSDP ... sharding=full_shard`
- `Observed train effective latent_t=21 (post-window)` (for full-T run)

Example:
- `output/logs/oom_debug_fullt21_nowindow_nosmoltrain_3gpu_20260228_054136.log:289`
- `output/logs/oom_debug_fullt21_nowindow_nosmoltrain_3gpu_20260228_054136.log:326`


## How the bug was debugged

### Step A) Confirm symptom from old failing runs

Search old logs for both markers:
- manual DiT sync marker
- CUDA OOM marker

Examples found together in the same run:
- Manual DiT sync:
  - `output/logs/train_joint_v4plus_openvid_partial_allavail_3gpu_ncclp2poff_20260227_175524.log:46`
  - `output/logs/train_joint_v4plus_openvid_partial_allavail_3gpu_ncclp2poff_20260227_175524.log:55`
  - `output/logs/train_joint_v4plus_openvid_partial_allavail_3gpu_ncclp2poff_20260227_175524.log:68`
- OOM:
  - `output/logs/train_joint_v4plus_openvid_partial_allavail_3gpu_ncclp2poff_20260227_175524.log:121`
  - `output/logs/train_joint_v4plus_openvid_partial_allavail_3gpu_ncclp2poff_20260227_175524.log:165`
  - `output/logs/train_joint_v4plus_openvid_partial_allavail_3gpu_ncclp2poff_20260227_175524.log:224`


### Step B) Reproduce with controlled settings

Controlled run config:
- full latent `T=21` (`train.latent_window.frames=null`)
- Smol LoRA disabled (`model.student.lora.enable=false`)
- DiT FSDP on (`model.dit.fsdp=true`, `full_shard`)

Result:
- No OOM in short controlled run.
- Confirmed sharded path:
  - `output/logs/oom_debug_fullt21_nowindow_nosmoltrain_3gpu_20260228_054136.log:289`


### Step C) A/B sanity check (attention config suspicion)

A/B run with SANA chunk config vs non-chunk config while keeping DiT FSDP sharded.
- No immediate OOM in short runs in either A/B case.
- This reduced confidence that `attn_type` was the primary trigger for the specific OOM incident.
- The strongest correlated trigger remained manual DiT fallback.


### Step D) Headroom probe

Run a longer short-test (`--total-steps 20`) with full `T=21`, no window, no Smol LoRA.
- Observed stable memory around ~30GB/GPU on 3 GPUs.
- This confirmed full `T=21` is feasible under the sharded path for this setup.


### Commands used during debug (examples)

```bash
# Find OOM and manual-DiT-sync in logs
rg -n -uu "manual gradient all-reduce for DiT|OutOfMemoryError|CUDA out of memory" output/logs -g "*.log"

# Reproduce controlled run (full T=21)
CUDA_VISIBLE_DEVICES=1,2,3 \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=3 tools/train_stage1_teacher_free.py \
  --config configs/stage1_teacher_free_joint_openvid_partial_msrvtt_laion_coyo_ivjoint_3gpu_20260227.yaml \
  --max-gpus 3 --total-steps 5

# Check that startup is sharded and no manual DiT fallback
rg -n "Wrapping DiT with FSDP|manual gradient all-reduce for DiT" output/logs/<run>.log
```


## Practical checklist before long runs

1. Confirm startup log has `Wrapping DiT with FSDP`.
2. Ensure there is no `manual gradient all-reduce for DiT`.
3. Confirm expected latent temporal length (`effective latent_t`).
4. Keep `run.allow_manual_dit_sync: false` unless intentionally debugging.
5. If you intentionally test manual mode, set:
   - `run.allow_manual_dit_sync: true`
   - and accept higher OOM risk.


## Minimal config snippet (recommended)

```yaml
run:
  require_dit_sharding: true
  allow_manual_dit_sync: false
  dit_sync: false

model:
  dit:
    fsdp: true
    fsdp_sharding_strategy: full_shard
```
