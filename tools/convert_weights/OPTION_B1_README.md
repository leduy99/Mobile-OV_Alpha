# Option B1 Implementation: State Dict Format

## Tổng quan

Đã implement Option B1 để save checkpoint dưới dạng `state_dict` thay vì full object, cho phép load mà không cần transformers runtime.

## Cấu trúc Checkpoint mới

Checkpoint bây giờ chứa:

```python
{
    # Format 1: Safe (no transformers needed)
    "state_dict": {...},           # Model weights
    "config_dict": {...},          # Config as dict
    "tokenizer_vocab": {...},      # Tokenizer vocabulary
    "tokenizer_config": {...},      # Tokenizer config
    
    # Format 2: Fallback (requires transformers)
    "model": <model_object>,       # Full object for backward compat
    "tokenizer": <tokenizer_object>,
    
    "model_id": "...",
    "checkpoint_format": "state_dict"
}
```

## Các bước đã làm

### 1. ✅ Modified Convert Script

**File**: `tools/convert_weights/convert_smolvlm2_weight.py`

- Save `state_dict` thay vì chỉ full object
- Save `config_dict` (serializable)
- Save `tokenizer_vocab` và `tokenizer_config`
- Giữ full object làm fallback

### 2. ✅ Updated Load Function

**File**: `nets/smolvlm2/modeling_smolvlm2.py`

- Ưu tiên load từ `state_dict` (không cần transformers)
- Fallback về full object nếu không có architecture code
- Tự động detect format và chọn cách load phù hợp

### 3. ✅ Updated Config

**File**: `nets/smolvlm2/config_smolvlm2.py`

- Thêm `from_dict()` method để recreate config từ dict
- Thêm `to_dict()` method để serialize config

### 4. ⚠️ Architecture Code (Cần copy)

**File**: `nets/smolvlm2/architecture_smolvlm2.py`

- Placeholder hiện tại
- Cần copy architecture code từ transformers

## Cách copy Architecture Code

### Option A: Dùng helper script

```bash
python tools/convert_weights/copy_smolvlm_architecture.py \
    --output nets/smolvlm2/architecture_smolvlm2.py
```

### Option B: Copy manual

1. Tìm file: `transformers/models/smolvlm/modeling_smolvlm.py`
2. Copy các classes cần thiết:
   - `SmolVLMModel`
   - `SmolVLMPreTrainedModel`
   - `SmolVLMEncoder`, `SmolVLMEncoderLayer`
   - `SmolVLMVisionTransformer`, `SmolVLMVisionAttention`
   - Helper classes: `SmolVLMRMSNorm`, `SmolVLMSimpleMLP`, etc.
3. Adjust imports:
   - `from transformers.xxx` → local imports hoặc implement lại
4. Save vào `nets/smolvlm2/architecture_smolvlm2.py`

## Testing

### Test convert script:

```bash
python tools/convert_weights/convert_smolvlm2_weight.py \
    --model-id HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
    --output-path test_ckpt.pt
```

### Test load (với architecture code):

```python
from nets.smolvlm2 import load_smolvlm2_from_ckpt

model = load_smolvlm2_from_ckpt("test_ckpt.pt", device="cpu")
# Should load from state_dict if architecture code exists
```

### Test load (fallback):

```python
# Nếu chưa có architecture code, sẽ fallback về full object
# (vẫn cần transformers runtime)
model = load_smolvlm2_from_ckpt("test_ckpt.pt", device="cpu")
```

## Lưu ý

1. **Architecture code phức tạp**: File có thể rất dài (1000+ lines) và có nhiều dependencies
2. **Import adjustments**: Cần adjust imports từ `transformers.xxx` sang local hoặc implement lại
3. **Version compatibility**: Architecture code có thể thay đổi giữa các version transformers
4. **Fallback vẫn hoạt động**: Nếu chưa copy architecture code, vẫn có thể dùng fallback (cần transformers)

## Next Steps

1. ✅ Convert script đã save state_dict
2. ✅ Load function đã support state_dict
3. ⚠️ **Cần copy architecture code** để hoàn thành Option B1
4. Test end-to-end sau khi có architecture code

## Benefits

- ✅ **Safe**: Không cần transformers runtime (sau khi có architecture code)
- ✅ **Backward compatible**: Vẫn có fallback cho full object
- ✅ **Smaller checkpoint**: state_dict thường nhỏ hơn full object
- ✅ **Version independent**: Không phụ thuộc transformers version



