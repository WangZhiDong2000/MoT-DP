#!/bin/bash
# Multi-GPU Testing Script for CARLA BEV Diffusion Policy
# Usage: bash test_carla_bev_multi_gpu.sh [checkpoint_path] [num_gpus] [test_split]

# Default values
CHECKPOINT_PATH="${1:-checkpoints/carla_dit_best/best_model.pth}"
NUM_GPUS="${2:-4}"
TEST_SPLIT="${3:-test}"
CONFIG_PATH="config/pdm_server.yaml"

echo "=========================================="
echo "Multi-GPU Model Testing"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Number of GPUs: $NUM_GPUS"
echo "Test Split: $TEST_SPLIT"
echo "Config: $CONFIG_PATH"
echo "=========================================="

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Set environment variables for better performance
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run with torchrun (recommended for PyTorch 1.9+)
echo "Starting testing with $NUM_GPUS GPUs..."
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    training/test_carla_bev_multi_gpu.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --test_split "$TEST_SPLIT"

echo ""
echo "=========================================="
echo "Testing completed!"
echo "=========================================="
