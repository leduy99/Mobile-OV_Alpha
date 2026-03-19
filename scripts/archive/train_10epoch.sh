#!/bin/bash
# Training script for MobileOVModel with OpenVid-1M subset - 10 epochs

set -e  # Exit on error

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate omnivideo

# Reduce CUDA fragmentation for large models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set paths
PROJECT_DIR="/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2"
cd "$PROJECT_DIR"

# Configuration
CONFIG_FILE="configs/mobile_ov_openvid_10epoch.yaml"
WAN_CKPT_DIR="omni_ckpts/wan/wanxiang1_3b"
OUTPUT_DIR="output/training_10epoch_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "=========================================="
echo "Training Configuration:"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "WAN checkpoint: $WAN_CKPT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Epochs: 10"
echo "Dataset: OpenVid-1M subset (100 samples)"
echo "=========================================="
echo ""

# Check if checkpoints exist
if [ ! -d "$WAN_CKPT_DIR" ]; then
    echo "ERROR: WAN checkpoint directory not found: $WAN_CKPT_DIR"
    exit 1
fi

if [ ! -f "omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt" ]; then
    echo "ERROR: SmolVLM2 checkpoint not found: omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt"
    exit 1
fi

if [ ! -f "data/openvid_test/OpenVid-1M_test_subset.csv" ]; then
    echo "ERROR: Dataset CSV not found: data/openvid_test/OpenVid-1M_test_subset.csv"
    exit 1
fi

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. GPU may not be available."
else
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Start training
echo "Starting training..."
echo ""

python finetune_model.py \
    --config "$CONFIG_FILE" \
    --ckpt_dir "$WAN_CKPT_DIR" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Training completed!"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

