#!/bin/bash
# Training script for MobileOVModel with OpenVid-1M subset - Overfit Training
# This script will run training in a tmux session so you can detach and close terminal

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
CONFIG_FILE="configs/mobile_ov_openvid_overfit.yaml"
WAN_CKPT_DIR="omni_ckpts/wan/wanxiang1_3b"
OUTPUT_DIR="output/training_overfit_$(date +%Y%m%d_%H%M%S)"
TMUX_SESSION_NAME="mobile_ov_overfit_train"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "=========================================="
echo "Training Configuration:"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "WAN checkpoint: $WAN_CKPT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Epochs: 100 (full clue training, no gamma gating, use_precomputed_features=false)"
echo "Dataset: OpenVid-1M subset (100 samples)"
echo "Tmux session: $TMUX_SESSION_NAME"
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

# Check GPU availability and find empty GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. GPU may not be available."
    SELECTED_GPU=0
else
    echo "Checking GPU availability..."
    echo ""
    
    # Find GPU with most free memory (at least 10GB free)
    SELECTED_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
        awk -F', ' '{if ($2 > 10000) print $1 " " $2}' | \
        sort -k2 -rn | head -1 | awk '{print $1}')
    
    if [ -z "$SELECTED_GPU" ]; then
        # If no GPU has 10GB free, use GPU with most free memory
        SELECTED_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
            awk -F', ' '{print $1 " " $2}' | \
            sort -k2 -rn | head -1 | awk '{print $1}')
        echo "⚠️  WARNING: No GPU with >10GB free. Using GPU $SELECTED_GPU (may be busy)"
    else
        echo "✅ Selected GPU $SELECTED_GPU (sufficient free memory)"
    fi
    
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader | \
        awk -F', ' '{printf "GPU %s (%s): %s/%s MB (%.1f%% used, %.1f%% free), Util: %s%%\n", $1, $2, $3, $4, ($3/$4)*100, (($4-$3)/$4)*100, $6}'
    echo ""
    echo "Using GPU: $SELECTED_GPU"
    echo ""
    
    # Set CUDA_VISIBLE_DEVICES to use selected GPU
    export CUDA_VISIBLE_DEVICES=$SELECTED_GPU
fi

# Check if tmux session already exists
if tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null; then
    echo "WARNING: Tmux session '$TMUX_SESSION_NAME' already exists!"
    echo "You can attach to it with: tmux attach -t $TMUX_SESSION_NAME"
    echo "Or kill it with: tmux kill-session -t $TMUX_SESSION_NAME"
    echo ""
    read -p "Do you want to kill the existing session and start a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$TMUX_SESSION_NAME"
        echo "Killed existing session."
    else
        echo "Aborted. Please handle the existing session manually."
        exit 1
    fi
fi

# Create tmux session and run training
echo "Creating tmux session '$TMUX_SESSION_NAME' and starting training..."
echo ""
echo "To attach to the session later, run:"
echo "  tmux attach -t $TMUX_SESSION_NAME"
echo ""
echo "To detach from tmux (keep training running):"
echo "  Press Ctrl+B, then D"
echo ""

# Create tmux session and run training command
tmux new-session -d -s "$TMUX_SESSION_NAME" -c "$PROJECT_DIR" \
    "bash -c '
        source $(conda info --base)/etc/profile.d/conda.sh && \
        conda activate omnivideo && \
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
        export CUDA_VISIBLE_DEVICES=$SELECTED_GPU && \
        cd $PROJECT_DIR && \
        echo \"==========================================\" && \
        echo \"Starting training in tmux session...\" && \
        echo \"Session: $TMUX_SESSION_NAME\" && \
        echo \"Output: $OUTPUT_DIR\" && \
        echo \"GPU: $SELECTED_GPU (CUDA_VISIBLE_DEVICES=$SELECTED_GPU)\" && \
        echo \"==========================================\" && \
        echo \"\" && \
        python finetune_model.py \
            --config $CONFIG_FILE \
            --ckpt_dir $WAN_CKPT_DIR \
            --output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/training.log && \
        echo \"\" && \
        echo \"==========================================\" && \
        echo \"Training completed!\" && \
        echo \"Output directory: $OUTPUT_DIR\" && \
        echo \"==========================================\" && \
        echo \"Press any key to close this window...\" && \
        read -n 1
    '"

# Wait a moment for tmux to start
sleep 2

# Check if session was created successfully
if tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null; then
    echo "✅ Tmux session created successfully!"
    echo ""
    echo "To view training progress:"
    echo "  tmux attach -t $TMUX_SESSION_NAME"
    echo ""
    echo "To detach (keep training running):"
    echo "  Press Ctrl+B, then D"
    echo ""
    echo "Training is now running in the background."
    echo "You can safely close this terminal."
else
    echo "❌ Failed to create tmux session!"
    exit 1
fi
