#!/usr/bin/env python3
"""
Test script to measure Transfuser forward pass time.

This script loads a model and measures the time it takes for a single forward pass
with different batch sizes and configurations.
"""

import torch
import numpy as np
import time
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dataset.preprocess_transfuser_features import (
    TransfuserConfig, load_transfuser_model
)


def create_dummy_batch(batch_size=1, device='cuda'):
    """Create dummy RGB and LiDAR tensors for testing."""
    # RGB: (B, 3, 384, 1024)
    image = torch.randn(batch_size, 3, 384, 1024, dtype=torch.float32, device=device)
    # LiDAR BEV: (B, 1, 256, 256)
    lidar_bev = torch.randn(batch_size, 1, 256, 256, dtype=torch.float32, device=device)
    return image, lidar_bev


def benchmark_forward(model, batch_size=1, num_iterations=10, device='cuda', warmup=3):
    """
    Benchmark the forward pass time.
    
    Args:
        model: TransfuserBackbone model
        batch_size: Batch size for the forward pass
        num_iterations: Number of iterations to measure
        device: Device (cuda or cpu)
        warmup: Number of warmup iterations
    
    Returns:
        times: List of forward pass times (in milliseconds)
        avg_time: Average forward pass time (in milliseconds)
        std_time: Standard deviation (in milliseconds)
    """
    print(f"\nBenchmarking with batch_size={batch_size}")
    print(f"  Warmup iterations: {warmup}")
    print(f"  Measured iterations: {num_iterations}")
    
    # Warmup
    print("  Running warmup...")
    for _ in range(warmup):
        image, lidar_bev = create_dummy_batch(batch_size, device)
        with torch.no_grad():
            _ = model(image, lidar_bev)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    times = []
    print("  Measuring forward pass times...")
    
    with torch.no_grad():
        for i in range(num_iterations):
            image, lidar_bev = create_dummy_batch(batch_size, device)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(image, lidar_bev)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            forward_time = (end_time - start_time) * 1000  # Convert to milliseconds
            times.append(forward_time)
            
            if (i + 1) % max(1, num_iterations // 5) == 0:
                print(f"    Progress: {i+1}/{num_iterations}")
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    variance = sum((t - avg_time) ** 2 for t in times) / len(times)
    std_time = np.sqrt(variance)
    
    return times, avg_time, std_time


def main():
    parser = argparse.ArgumentParser(description='Measure Transfuser forward pass time')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/home/wang/Project/carla_garage/leaderboard/leaderboard/pretrained_models/all_towns',
                        help='Directory containing pretrained Transfuser checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1],
                        help='Batch sizes to test (e.g., 1 2 4 8)')
    parser.add_argument('--num_iterations', type=int, default=20,
                        help='Number of iterations to measure')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Number of warmup iterations')
    args = parser.parse_args()
    
    print("="*70)
    print("Transfuser Forward Pass Time Benchmark")
    print("="*70)
    
    # Load model
    print("\nLoading Transfuser model...")
    config = TransfuserConfig()
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = load_transfuser_model(args.checkpoint_dir, config, device)
    print("✓ Model loaded successfully")
    
    # Check GPU memory if using CUDA
    if device == 'cuda':
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Run benchmarks
    all_results = {}
    
    for batch_size in args.batch_sizes:
        times, avg_time, std_time = benchmark_forward(
            model,
            batch_size=batch_size,
            num_iterations=args.num_iterations,
            device=device,
            warmup=args.warmup
        )
        all_results[batch_size] = {
            'times': times,
            'avg': avg_time,
            'std': std_time
        }
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for batch_size in sorted(all_results.keys()):
        results = all_results[batch_size]
        avg_time = results['avg']
        std_time = results['std']
        fps = 1000 / avg_time
        
        print(f"\nBatch Size: {batch_size}")
        print(f"  Average forward time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  Throughput: {fps:.1f} fps")
        print(f"  Time per sample: {avg_time/batch_size:.2f} ms")
    
    # Overall comparison
    if len(all_results) > 1:
        print("\n" + "-"*70)
        print("Comparison:")
        for batch_size in sorted(all_results.keys()):
            avg_time = all_results[batch_size]['avg']
            fps = 1000 / avg_time
            print(f"  Batch {batch_size}: {fps:.1f} fps ({avg_time:.2f} ms/batch)")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
