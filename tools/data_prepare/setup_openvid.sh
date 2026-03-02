#!/bin/bash
# Setup script for OpenVid-1M dataset

set -e

# Configuration
OUTPUT_DIR="${1:-data/openvid}"
NUM_PARTS="${2:-10}"  # Download first 10 parts for testing

echo "=== OpenVid-1M Dataset Setup ==="
echo "Output directory: $OUTPUT_DIR"
echo "Number of parts to download: $NUM_PARTS"

# Step 1: Download CSV file
echo ""
echo "Step 1: Downloading OpenVid-1M.csv..."
python tools/data_prepare/download_openvid.py \
    --output_dir "$OUTPUT_DIR" \
    --csv_only

# Step 2: Download video parts
echo ""
echo "Step 2: Downloading video parts (this may take a while)..."
python tools/data_prepare/download_openvid.py \
    --output_dir "$OUTPUT_DIR" \
    --num_parts "$NUM_PARTS" \
    --start_part 0

# Step 3: Extract videos from zip files
echo ""
echo "Step 3: Extracting videos from zip files..."
VIDEO_DIR="$OUTPUT_DIR/videos"
mkdir -p "$VIDEO_DIR"

for zip_file in "$OUTPUT_DIR"/OpenVid_part*.zip; do
    if [ -f "$zip_file" ]; then
        echo "Extracting $zip_file..."
        unzip -j -q "$zip_file" -d "$VIDEO_DIR" || echo "Warning: Failed to extract $zip_file"
    fi
done

echo ""
echo "=== Setup Complete ==="
echo "CSV file: $OUTPUT_DIR/OpenVid-1M.csv"
echo "Video directory: $VIDEO_DIR"
echo ""
echo "Next steps:"
echo "1. Extract features: python tools/data_prepare/extract_openvid_features.py \\"
echo "     --csv_path $OUTPUT_DIR/OpenVid-1M.csv \\"
echo "     --video_dir $VIDEO_DIR \\"
echo "     --output_dir $OUTPUT_DIR/preprocessed \\"
echo "     --ckpt_dir <path_to_wan_checkpoints> \\"
echo "     --max_samples 1000  # Start with small number for testing"
echo ""
echo "2. Train: python finetune_model.py \\"
echo "     --config configs/mobile_ov_openvid_long.yaml \\"
echo "     --ckpt_dir <path_to_wan_checkpoints> \\"
echo "     --output_dir output/openvid_training"

