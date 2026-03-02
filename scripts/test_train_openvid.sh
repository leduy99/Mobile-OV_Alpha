#!/bin/bash
# Quick test training script with OpenVid-1M test subset

set -e

# Configuration
CONFIG="configs/mobile_ov_openvid_test.yaml"
CKPT_DIR="${1:-omni_ckpts/wan/wanxiang1_3b}"  # Default WAN checkpoint directory
OUTPUT_DIR="output/openvid_test_$(date +%Y%m%d_%H%M%S)"

echo "=== OpenVid-1M Test Training ==="
echo "Config: $CONFIG"
echo "Checkpoint dir: $CKPT_DIR"
echo "Output dir: $OUTPUT_DIR"

# Check if test data exists
if [ ! -f "data/openvid_test/OpenVid-1M_test_subset.csv" ]; then
    echo "Error: Test subset not found. Please run:"
    echo "  python tools/data_prepare/create_openvid_test_subset.py --output_dir data/openvid_test --num_samples 100 --create_dummy"
    exit 1
fi

# Check if preprocessed data exists
if [ ! -d "data/openvid_test/preprocessed" ]; then
    echo "Error: Preprocessed data not found. Please run:"
    echo "  python tools/data_prepare/create_openvid_test_subset.py --output_dir data/openvid_test --num_samples 100 --create_dummy"
    exit 1
fi

# Activate conda environment if needed
# conda activate omnivideo

# Run training
python finetune_model.py \
    --config "$CONFIG" \
    --ckpt_dir "$CKPT_DIR" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== Training Complete ==="
echo "Output directory: $OUTPUT_DIR"

