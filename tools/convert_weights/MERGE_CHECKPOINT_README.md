# Merge Checkpoints Guide

Hướng dẫn merge checkpoints từ SmolVLM2 và Omni-Video thành unified checkpoint cho `MobileOVModel`.

## Tổng quan

Script này sẽ:
1. ✅ Load SmolVLM2 checkpoint (đã convert từ HuggingFace)
2. ✅ Load Omni-Video checkpoint (WAN + Adapter, **bỏ VisionHead**)
3. ✅ Initialize projection layer (random/xavier/zeros/kaiming)
4. ✅ Save thành unified checkpoint structure

## Cách sử dụng

### Bước 1: Convert SmolVLM2 checkpoint (nếu chưa có)

```bash
python tools/convert_weights/convert_smolvlm2_weight.py \
    --model_id "HuggingFaceTB/SmolVLM2-500M-Instruct" \
    --output_path omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt
```

### Bước 2: Merge checkpoints

```bash
python tools/convert_weights/merge_mobile_ov_checkpoint.py \
    --smolvlm2_ckpt omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt \
    --omnivideo_ckpt_dir omni_ckpts/omnivideo_ckpt \
    --output_dir omni_ckpts/mobile_ov_merged \
    --adapter_in_channels 1152 \
    --adapter_out_channels 4096 \
    --adapter_query_length 64 \
    --init_projection xavier
```

### Arguments

- `--smolvlm2_ckpt`: Path đến SmolVLM2 checkpoint (.pt file đã convert)
- `--omnivideo_ckpt_dir`: Directory chứa Omni-Video checkpoint (cần có `wan_model/` và `adapter/`)
- `--output_dir`: Directory output cho merged checkpoint
- `--adapter_in_channels`: Adapter input channels (default: 1152)
- `--adapter_out_channels`: Adapter output channels (default: 4096)
- `--adapter_query_length`: Adapter query length (default: 64)
- `--init_projection`: Cách init projection layer: `random`, `zeros`, `xavier`, `kaiming` (default: `xavier`)
- `--device`: Device để load checkpoints (default: `cpu`)

## Output Structure

Sau khi merge, output directory sẽ có cấu trúc:

```
mobile_ov_merged/
├── smolvlm2_model.pt              # SmolVLM2 model (full object)
├── wan_model/                      # WAN model checkpoint
│   ├── ...
├── adapter/                        # Adapter checkpoint
│   └── adapter_pytorch_model.bin
├── visual_context_adapter/         # (optional) Visual context adapter
│   └── visual_context_adapter_pytorch_model.bin
├── smolvlm2_projection/            # Projection layer
│   └── smolvlm2_projection_pytorch_model.bin
└── metadata.json                   # Metadata về merge process
```

## Load merged checkpoint

Sau khi merge, có thể load như sau:

```python
from nets.omni.modules.mobile_ov_model import MobileOVModel

model = MobileOVModel.from_pretrained(
    wan_ckpt_dir="omni_ckpts/mobile_ov_merged/wan_model",
    adapter_ckpt_dir="omni_ckpts/mobile_ov_merged/adapter",
    smolvlm2_ckpt_path="omni_ckpts/mobile_ov_merged/smolvlm2_model.pt",
    adapter_in_channels=1152,
    adapter_out_channels=4096,
    adapter_query_length=64,
    use_precomputed_features=False,  # Use on-the-fly encoding
)
```

**Lưu ý**: `MobileOVModel` sẽ tự động load projection layer từ `smolvlm2_projection/` nếu có.

## Training từ merged checkpoint

Có thể train từ merged checkpoint ngay:

```yaml
# config.yaml
training:
  model_settings:
    model_type: "mobile_ov"
    smolvlm2_ckpt_path: "omni_ckpts/mobile_ov_merged/smolvlm2_model.pt"
    use_precomputed_features: false
    train_smolvlm2: false  # Freeze SmolVLM2
    train_smolvlm2_projection: true  # Train projection layer
```

Và trong training code, set `args.ckpt_dir` trỏ đến merged checkpoint directory:

```python
# Training sẽ load:
# - WAN từ: args.ckpt_dir/wan_model/
# - Adapter từ: args.ckpt_dir/adapter/
# - SmolVLM2 từ: args.training.model_settings.smolvlm2_ckpt_path
```

## So sánh với train from scratch

### Option 1: Merge checkpoints (Recommended)
- ✅ Sử dụng pretrained weights từ cả 2 models
- ✅ Chỉ cần train projection layer (nhỏ, nhanh)
- ✅ Có thể fine-tune SmolVLM2 nếu cần
- ⚠️ Cần merge script

### Option 2: Train from scratch
- ❌ Phải train toàn bộ từ đầu
- ❌ Tốn thời gian và resources hơn
- ✅ Không cần merge script

## Tips

1. **Projection layer initialization**:
   - `xavier`: Recommended, thường cho kết quả tốt
   - `kaiming`: Cũng tốt cho deep networks
   - `random`: Có thể thử nếu xavier không tốt
   - `zeros`: Không recommended (gradient flow kém)

2. **Memory**: Merge trên CPU để tránh OOM, sau đó load vào GPU khi train

3. **Verify**: Sau khi merge, nên test load model để đảm bảo không có lỗi

## Troubleshooting

### Lỗi: "Adapter checkpoint not found"
- Kiểm tra path: `omnivideo_ckpt_dir/adapter/adapter_pytorch_model.bin`
- Hoặc: `omnivideo_ckpt_dir/adapter_pytorch_model.bin`

### Lỗi: "Could not detect SmolVLM2 hidden_size"
- Script sẽ dùng default 1024 (đúng cho SmolVLM2-500M)
- Có thể check trong `metadata.json` sau khi merge

### Lỗi khi load projection layer
- Đảm bảo `smolvlm2_projection/` nằm cùng directory với `smolvlm2_model.pt`
- Hoặc load manually sau khi init model



