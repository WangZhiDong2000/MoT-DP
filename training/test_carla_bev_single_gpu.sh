#!/bin/bash
# Single GPU Testing Script for CARLA BEV Diffusion Policy
# Usage: bash test_carla_bev_single_gpu.sh [checkpoint_path] [gpu_id] [test_split]

CHECKPOINT_PATH="${1:-checkpoints/carla_dit_best/best_model.pth}"
GPU_ID="${2:-0}"
TEST_SPLIT="${3:-test}"
CONFIG_PATH="config/pdm_server.yaml"

echo "=========================================="
echo "Single GPU Model Testing"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "GPU ID: $GPU_ID"
echo "Test Split: $TEST_SPLIT"
echo "Config: $CONFIG_PATH"
echo "=========================================="

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Run testing
python training/test_carla_bev_multi_gpu.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --test_split "$TEST_SPLIT"

echo ""
echo "=========================================="
echo "Testing completed!"
echo "=========================================="
