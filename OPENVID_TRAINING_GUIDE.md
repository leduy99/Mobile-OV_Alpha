# OpenVid-1M Training Guide

Hướng dẫn training MobileOVModel với OpenVid-1M dataset (test subset).

## Quick Start (Test với 100 samples)

### 1. Tạo test subset (đã chạy xong)
```bash
python tools/data_prepare/create_openvid_test_subset.py \
    --output_dir data/openvid_test \
    --num_samples 100 \
    --create_dummy
```

Điều này sẽ:
- Download CSV file từ HuggingFace (~800MB)
- Tạo subset 100 samples
- Tạo dummy preprocessed data (100 pickle files)

### 2. Chạy training test
```bash
bash scripts/test_train_openvid.sh <path_to_wan_checkpoints>
```

Hoặc manual:
```bash
python finetune_model.py \
    --config configs/mobile_ov_openvid_test.yaml \
    --ckpt_dir omni_ckpts/wan/wanxiang1_3b \
    --output_dir output/openvid_test_$(date +%Y%m%d_%H%M%S)
```

## Config Details

### Test Config (`configs/mobile_ov_openvid_test.yaml`)
- **Epochs**: 3 (lâu hơn test nhanh)
- **Batch size**: 2
- **Gradient accumulation**: 2 (effective batch = 4)
- **Learning rate**: 1e-4 (base), với separate param groups:
  - Projection: 2e-4 (2x)
  - Adapter: 5e-5 (0.5x)
- **Samples**: 100 từ OpenVid-1M
- **Preprocessed**: Dùng dummy data (không cần video files thật)

### Long Training Config (`configs/mobile_ov_openvid_long.yaml`)
- **Epochs**: 5
- **Batch size**: 4
- **Gradient accumulation**: 4 (effective batch = 16)
- **Samples**: Full dataset (cần download và preprocess videos)

## Dataset Structure

```
data/openvid_test/
├── OpenVid-1M.csv                    # Full CSV (818MB)
├── OpenVid-1M_test_subset.csv       # Test subset (100 samples)
└── preprocessed/                     # Dummy preprocessed features
    ├── ---_iRTHryQ_13_0to241_features.pkl
    ├── ---agFLYkbY_7_0to303_features.pkl
    └── ...
```

## Preprocessed Data Format

Mỗi pickle file chứa:
```python
{
    'latent_feature': torch.Tensor,  # [16, 21, 32, 32] - VAE latent
    'prompt': str,                    # Text caption
    'text_emb': List[torch.Tensor],   # T5 embeddings
    'video_path': str,                # Path to video (dummy for test)
    'frame_num': int,                 # Number of frames
}
```

## Training với Real Videos (Khi sẵn sàng)

### 1. Download videos (optional, chỉ khi cần)
```bash
python tools/data_prepare/download_openvid.py \
    --output_dir data/openvid \
    --num_parts 10  # Download 10 parts đầu tiên
```

### 2. Extract features từ videos
```bash
python tools/data_prepare/extract_openvid_features.py \
    --csv_path data/openvid/OpenVid-1M.csv \
    --video_dir data/openvid/videos \
    --output_dir data/openvid/preprocessed \
    --ckpt_dir omni_ckpts/wan/wanxiang1_3b \
    --frame_num 21 \
    --target_size 512,512 \
    --max_samples 1000  # Start with 1000 samples
```

### 3. Train với real data
```bash
python finetune_model.py \
    --config configs/mobile_ov_openvid_long.yaml \
    --ckpt_dir omni_ckpts/wan/wanxiang1_3b \
    --output_dir output/openvid_long_training
```

## Monitoring Training

- **TensorBoard**: `tensorboard --logdir output/openvid_test_*/tensorboard`
- **Checkpoints**: Saved every 100 steps in `output/openvid_test_*/checkpoint_*`
- **Logs**: Check console output for loss, LR, gradient norms

## Expected Results

Với test subset (100 samples, 3 epochs):
- **Total steps**: ~150 steps (100 samples / 2 batch_size * 3 epochs)
- **Training time**: ~30-60 phút (tùy GPU)
- **Loss**: Nên giảm dần từ ~1.0 xuống ~0.5-0.8

## Troubleshooting

### Lỗi: "No valid dataloaders were created"
- Check: CSV file path đúng không?
- Check: Preprocessed directory có files không?
- Check: Video names trong CSV match với preprocessed filenames

### Lỗi: "CUDA out of memory"
- Giảm `batch_size` trong config
- Enable `gradient_checkpointing: true`
- Giảm `num_workers` trong dataloader config

### Loss không giảm
- Check: Gradient norms > 0 (xem logs)
- Check: Learning rates đúng không (projection 2x, adapter 0.5x)
- Check: Attention mask được apply (đã fix)
- Check: T5 context disabled khi dùng SmolVLM2 (đã fix)

## Next Steps

Sau khi test thành công:
1. Download và preprocess nhiều videos hơn (1000-10000 samples)
2. Train với config `mobile_ov_openvid_long.yaml`
3. Monitor loss và sample quality
4. Tune hyperparameters nếu cần

## Stage1 Teacher-Free (SANA + FSDP 4 GPU)

Script đang dùng cho stage1 teacher-free:
- `tools/train_stage1_teacher_free.py`
- config mặc định: `configs/stage1_teacher_free_openvid_train.yaml`

### Trạng thái hiện tại (đã sẵn sàng train multi-GPU)
- Có hỗ trợ FSDP cho DiT (`model.dit.fsdp: true`)
- Student chạy DDP
- Gradient accumulation đã được tách đúng `micro_step` và `update_step`
- Checkpoint lưu theo `update_step` (không còn lệch do accumulation)
- Lưu thêm `dit_trainable_state` để giữ phần DiT đang fine-tune
- Có `checkpoint_final.pt` khi kết thúc

### Chạy 4 GPU

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master_port=29501 \
  tools/train_stage1_teacher_free.py \
  --config configs/stage1_teacher_free_openvid_train.yaml \
  --max-gpus 4
```

### Smoke test nhanh (1 GPU, ít step)

```bash
python tools/train_stage1_teacher_free.py \
  --config configs/stage1_teacher_free_openvid_train.yaml \
  --max-gpus 1 \
  --total-steps 5 \
  --save-every 5 \
  --log-every 1
```
