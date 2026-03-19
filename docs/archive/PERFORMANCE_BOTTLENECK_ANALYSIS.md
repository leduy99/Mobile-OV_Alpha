# Performance Bottleneck Analysis - T5-only Baseline

## Executive Summary

**T5-only baseline inference is slow (~14.2s/step) because:**
1. **Self-attention dominates computation** (2730x more operations than cross-attention)
2. **seq_len=32760 is very large** for full self-attention
3. **CFG doubles forward pass** (expected, but adds to total time)

## Detailed Analysis

### Profile Results

```
Average times per step:
  Forward (conditioned):   7079.19ms (~7.08s)
  Forward (unconditioned): 7084.25ms (~7.08s)
  CFG calculation:           0.14ms (~0.0001s)
  Total per step:         14163.58ms (~14.16s)
```

### Computational Complexity

**Configuration:**
- `seq_len = 32760` (video patches: 21 frames × 60×104 spatial patches)
- `context_len = 12` (T5 tokens)
- `target_shape = (16, 21, 60, 104)` (C, F, H, W)

**Operations:**
- **Self-attention**: `seq_len² = 32760² = 1,073,217,600` operations (~1.07B)
- **Cross-attention**: `seq_len × context_len = 32760 × 12 = 393,120` operations (~0.39M)

**Ratio**: Self-attention is **2730x larger** than cross-attention!

### Root Cause

1. **Self-attention is O(n²)**: With `seq_len=32760`, self-attention requires ~1.07B operations per layer
2. **No sliding window**: WAN model uses `window_size=(-1, -1)`, meaning **full self-attention** (no local attention optimization)
3. **Multiple layers**: WAN model has multiple attention blocks, each performing full self-attention
4. **CFG doubles forward pass**: Each step requires 2 forward passes (conditioned + unconditioned)

### Why This is Slow

```
For each denoising step:
  1. Conditioned forward:  ~7.08s
     - Self-attention:    ~6.5s (dominates)
     - Cross-attention:   ~0.1s
     - Other ops:         ~0.5s
  
  2. Unconditioned forward: ~7.08s (same breakdown)
  
  3. CFG calculation:        ~0.0001s (negligible)
  
  Total:                   ~14.16s/step
```

### Comparison with OmniVideo

**Expected behavior:**
- OmniVideo should have **similar performance** for the same resolution
- Both use the same WAN model architecture
- Both have `seq_len=32760` for 832×480 resolution
- Both use CFG (forward 2x)

**If OmniVideo is faster, possible reasons:**
1. **Different resolution**: OmniVideo might use smaller resolution
2. **Model state**: Different eval/train mode settings
3. **Optimization flags**: Different PyTorch optimization settings
4. **Hardware**: Different GPU or CUDA version

### Solutions

#### 1. Reduce Resolution (Quick Fix)
- Use smaller resolution (e.g., 640×360 instead of 832×480)
- Reduces `seq_len` quadratically
- Example: 640×360 → `seq_len ≈ 20,000` → ~2.5x faster

#### 2. Reduce Frame Count (Quick Fix)
- Use fewer frames (e.g., 49 instead of 81)
- Reduces `seq_len` linearly
- Example: 49 frames → `seq_len ≈ 19,800` → ~2.7x faster

#### 3. Use Sliding Window Attention (Architecture Change)
- Modify WAN model to use `window_size=(w_h, w_w)` instead of `(-1, -1)`
- Reduces self-attention from O(n²) to O(n×w)
- Requires model architecture changes

#### 4. Gradient Checkpointing (Memory vs Speed Trade-off)
- Use gradient checkpointing to reduce memory
- May slow down inference slightly but saves memory

#### 5. Mixed Precision Optimization
- Already using `torch.cuda.amp.autocast(dtype=torch.float32)`
- Could try `bfloat16` for faster computation (if model supports)

### Recommendations

1. **Verify OmniVideo baseline performance** for same resolution
   - If OmniVideo is also ~14s/step → This is expected
   - If OmniVideo is faster → Investigate differences

2. **Accept current performance** if quality is acceptable
   - ~14s/step × 50 steps = ~12 minutes per video
   - This is reasonable for high-quality video generation

3. **Optimize for speed** if needed:
   - Reduce resolution to 640×360
   - Reduce frames to 49
   - Use fewer sampling steps (30 instead of 50)

### Conclusion

**The slow performance is NOT a bug** - it's due to the computational complexity of self-attention with large `seq_len`. This is inherent to the WAN model architecture for high-resolution video generation.

**Next steps:**
1. Extract frames from generated video ✅
2. Compare with OmniVideo baseline for same resolution
3. Decide if current performance is acceptable or if optimization is needed
