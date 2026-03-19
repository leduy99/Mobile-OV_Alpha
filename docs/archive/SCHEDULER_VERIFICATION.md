# Scheduler/Solver Verification - MobileOV vs OmniVideo

## Date
2025-01-17

## Verification Results

### ✅ Scheduler Type
- **OmniVideo**: `FlowUniPCMultistepScheduler` ✅
- **MobileOV**: `FlowUniPCMultistepScheduler` ✅
- **Status**: **MATCH!**

### ✅ Scheduler Configuration

| Parameter | OmniVideo | MobileOV | Match? |
|-----------|----------|----------|--------|
| **Scheduler Class** | `FlowUniPCMultistepScheduler` | `FlowUniPCMultistepScheduler` | ✅ |
| **Init shift** | `1` | `1` | ✅ |
| **Set shift** | `5.0` (default) | `5.0` (default) | ✅ |
| **use_dynamic_shifting** | `False` | `False` | ✅ |
| **num_train_timesteps** | `1000` | `1000` | ✅ |

### ✅ Scheduler.step() Signature

**OmniVideo** (`omni_video_unified_gen.py`, line 371-376):
```python
temp_x0 = sample_scheduler.step(
    noise_pred.unsqueeze(0),
    t,
    latents[0].unsqueeze(0),
    return_dict=False,
    generator=seed_g)[0]
```

**MobileOV** (`inference_trained_extract_frames.py`, line 657-663):
```python
temp_x0 = sample_scheduler.step(
    noise_pred.unsqueeze(0),  # model_output
    t,  # timestep
    latents[0].unsqueeze(0),  # sample
    return_dict=False,
    generator=seed_g
)[0]  # FlowUniPC returns tuple, take first element
```

**Status**: ✅ **MATCH!**

### ✅ Timesteps

- **OmniVideo**: `tensor([999, 995, 991, 987, 982, ...])`
- **MobileOV**: `tensor([999, 995, 991, 987, 982, ...])`
- **Status**: ✅ **MATCH!**

## Classifier-Free Guidance (CFG)

### OmniVideo
- **Uses CFG**: Yes
- **guide_scale**: `5.0` (default)
- **Implementation**: 
  ```python
  noise_pred_cond = self.model(..., **arg_c)[0]
  noise_pred_uncond = self.model(..., **arg_null)[0]
  noise_pred = noise_pred_cond + guide_scale * (noise_pred_cond - noise_pred_uncond)
  ```

### MobileOV
- **Uses CFG**: ❌ **NO** (currently not implemented)
- **Implementation**: Direct model output, no CFG

**Note**: MobileOV training doesn't use CFG, so inference without CFG is consistent with training.

## Summary

### ✅ What Matches
1. **Scheduler Type**: Both use `FlowUniPCMultistepScheduler` ✅
2. **Shift Configuration**: Both use `shift=5.0` ✅
3. **Scheduler.step()**: Same signature and usage ✅
4. **Timesteps**: Identical ✅

### ⚠️ Difference
1. **CFG**: OmniVideo uses CFG (`guide_scale=5.0`), MobileOV does not
   - **Reason**: MobileOV training doesn't use CFG, so inference without CFG is consistent
   - **Impact**: OmniVideo may have better quality due to CFG, but MobileOV is consistent with its training

## Conclusion

✅ **Scheduler/Solver đã GIỐNG NHAU!**

MobileOV đã sử dụng:
- ✅ `FlowUniPCMultistepScheduler` (UniPC)
- ✅ `shift=5.0` (match OmniVideo)
- ✅ Same `step()` signature

**Ready for retrain!** 🚀
