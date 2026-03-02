#!/bin/bash
# Test inference với multiple gamma values trong tmux

set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate omnivideo

# Set paths
PROJECT_DIR="/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2"
cd "$PROJECT_DIR"

# Configuration
CHECKPOINT_DIR="output/training_overfit_20260115_184949/checkpoint_latest/latest"
CSV_PATH="data/openvid_test/OpenVid-1M_test_subset.csv"
NUM_SAMPLES=3
SIZE="832*480"
OUTPUT_DIR="output/inference_gamma_test_latest_$(date +%Y%m%d_%H%M%S)"
TMUX_SESSION_NAME="test_gamma_inference"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Test Inference với Multiple Gamma Values"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "CSV: $CSV_PATH"
echo "Samples: $NUM_SAMPLES"
echo "Output: $OUTPUT_DIR"
echo "Tmux session: $TMUX_SESSION_NAME"
echo "=========================================="
echo ""

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

# Create tmux session and run test
echo "Creating tmux session '$TMUX_SESSION_NAME' and starting gamma test..."
echo ""
echo "To attach to the session later, run:"
echo "  tmux attach -t $TMUX_SESSION_NAME"
echo ""

tmux new-session -d -s "$TMUX_SESSION_NAME" -c "$PROJECT_DIR" \
    "bash -c '
        source $(conda info --base)/etc/profile.d/conda.sh && \
        conda activate omnivideo && \
        cd $PROJECT_DIR && \
        echo \"==========================================\" && \
        echo \"Starting gamma test in tmux session...\" && \
        echo \"Session: $TMUX_SESSION_NAME\" && \
        echo \"Output: $OUTPUT_DIR\" && \
        echo \"==========================================\" && \
        echo \"\" && \
        python test_inference_multiple_gamma.py \
            --checkpoint_dir $CHECKPOINT_DIR \
            --csv_path $CSV_PATH \
            --num_samples $NUM_SAMPLES \
            --size $SIZE \
            --output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/gamma_test.log && \
        echo \"\" && \
        echo \"==========================================\" && \
        echo \"Gamma test completed!\" && \
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
    echo "To view progress:"
    echo "  tmux attach -t $TMUX_SESSION_NAME"
    echo ""
    echo "To detach (keep running):"
    echo "  Press Ctrl+B, then D"
    echo ""
    echo "Test is now running in the background."
    echo "You can safely close this terminal."
else
    echo "❌ Failed to create tmux session!"
    exit 1
fi
