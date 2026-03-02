# Setup and Training Guide

Hướng dẫn setup và training cho MobileOV.

## Table of Contents
1. [Model Setup](#model-setup)
2. [Checkpoint Structure](#checkpoint-structure)
3. [Training Guide](#training-guide)
4. [Training Fixes](#training-fixes)

---

## Model Setup

### Checkpoint Structure

```
omni_ckpts/
├── wan/
│   └── wanxiang1_3b/          # WAN model checkpoint directory
│       ├── config.json
│       ├── diffusion_pytorch_model.safetensors
│       ├── models_t5_umt5-xxl-enc-bf16.pth
│       ├── Wan2.1_VAE.pth
│       └── adapter/            # Adapter checkpoint
│           └── adapter_pytorch_model.bin
└── smolvlm2_500m/
    └── smolvlm2_500m.pt        # SmolVLM2 checkpoint
```

### Environment Setup

See `SETUP_MODELS.md` for detailed environment setup instructions.

---

## Checkpoint Structure

### Training Checkpoints

MobileOV training saves:
- **WAN model**: In DeepSpeed checkpoint
- **Adapter**: In DeepSpeed checkpoint
- **SmolVLM2VisionHead**: Saved separately (not in DeepSpeed checkpoint)
  - Path: `checkpoint_dir/smolvlm2_vision_head/vision_head.pt`

### Loading Checkpoints

```python
model = MobileOVModel.from_pretrained(
    wan_ckpt_dir="omni_ckpts/wan/wanxiang1_3b",
    adapter_ckpt_dir="output/training_xxx/checkpoint_epoch_X_step_Y/epoch_X_step_Y",
    smolvlm2_ckpt_path="omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt",
    vision_head_ckpt_path="output/training_xxx/checkpoint_epoch_X_step_Y/smolvlm2_vision_head/vision_head.pt",
    adapter_query_length=64,  # Match training config
    ...
)
```

---

## Training Guide

### Quick Start (OpenVid-1M Test Subset)

#### 1. Prepare Data
```bash
python tools/data_prepare/create_openvid_test_subset.py \
    --output_dir data/openvid_test \
    --num_samples 100 \
    --create_dummy
```

#### 2. Train
```bash
python finetune_model.py \
    --config configs/mobile_ov_openvid_overfit.yaml \
    --ckpt_dir omni_ckpts/wan/wanxiang1_3b \
    --output_dir output/training_$(date +%Y%m%d_%H%M%S)
```

### Training Configuration

**Key Settings** (`configs/mobile_ov_openvid_overfit.yaml`):
- `adapter.query_length: 64` - Match OmniVideo context size
- `use_precomputed_features: false` - Must be false to train VisionHead
- `disable_t5_context: false` - Enable T5 to match OmniVideo
- `train_smolvlm2_vision_head: true` - Train VisionHead
- `train_adapter: true` - Train Adapter
- `train_wan_model: false` - Freeze WAN
- `train_smolvlm2: false` - Freeze SmolVLM2

### Training Process

1. **Phase 1**: Train VisionHead + Adapter with flow matching loss
2. **Monitor**: Loss should decrease, context shapes should be correct
3. **Checkpoint**: Saved every `save_interval` steps
4. **VisionHead**: Saved separately in each checkpoint directory

---

## Training Fixes

### Fixes Applied

1. **VisionHead Creation**: Fixed to create even when `use_precomputed_features=True`
2. **Checkpoint Saving**: Fixed path logic for VisionHead saving
3. **Gradient Flow**: Fixed to preserve gradients for trainable components
4. **Context Processing**: Fixed to match OmniVideo logic

### Known Issues

1. **VisionHead Size**: 151M params (normal but large)
2. **Memory**: May need gradient checkpointing for large batches
3. **Checkpoint Size**: ~28GB per checkpoint (DeepSpeed format)

---

*Last updated: 2025-01-17*
