#!/usr/bin/env python3
"""
Preprocess Senna nuScenes dataset for training.

This script loads pre-processed Senna pkl files and merges them with
inference hidden states (vit features, decision features, anchor trajectory).

Input:
    - Senna processed data: /mnt/data2/nuscenes/processed_senna/senna_train_4obs.pkl
    - Senna processed data: /mnt/data2/nuscenes/processed_senna/senna_val_4obs.pkl
    - Hidden states: /mnt/data2/nuscenes/processed_senna/inference_hidden_states.pkl
    
Output:
    - Training samples with all features merged
    
Features from inference_hidden_states.pkl (keyed by sample_token):
    - vit_hidden_states: (128, 2560) - ViT visual features
    - action_hidden_states: (4, 2560) - Decision/action features
    - reasoning_hidden_states: (20, 2560) - Reasoning features
    - pred_traj: (1, 6, 2) - Anchor trajectory prediction
"""

import os
import sys
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pickle
import numpy as np
import torch
from tqdm import tqdm
import argparse
from dataset.config_loader import load_config


def load_senna_data(senna_path):
    """Load Senna preprocessed data file"""
    print(f"Loading Senna data from {senna_path}...")
    
    # Handle numpy compatibility
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    
    with open(senna_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} samples")
    return data


def load_hidden_states(hidden_states_path):
    """Load inference hidden states file"""
    print(f"Loading hidden states from {hidden_states_path}...")
    
    with open(hidden_states_path, 'rb') as f:
        hidden_states = pickle.load(f)
    
    print(f"Loaded hidden states for {len(hidden_states)} samples")
    return hidden_states


def merge_sample_with_hidden_states(sample, hidden_states_dict, convert_to_numpy=True):
    """
    Merge a single sample with its corresponding hidden states.
    
    Args:
        sample: Dict containing sample data from Senna pkl
        hidden_states_dict: Dict mapping sample_token to hidden states
        convert_to_numpy: Whether to convert torch tensors to numpy arrays
        
    Returns:
        Merged sample dict or None if hidden states not found
    """
    sample_token = sample['sample_token']
    
    # Check if hidden states exist for this sample
    if sample_token not in hidden_states_dict:
        return None
    
    hs = hidden_states_dict[sample_token]
    
    # Create merged sample
    merged = sample.copy()
    
    # Add hidden states features
    if convert_to_numpy:
        # Convert torch tensors to numpy arrays for storage efficiency
        # Note: bfloat16 doesn't have native numpy support, so we convert via float32
        def tensor_to_numpy(t, dtype=np.float16):
            """Convert torch tensor to numpy, handling bfloat16 properly"""
            if isinstance(t, torch.Tensor):
                # bfloat16 must be converted to float32 first
                if t.dtype == torch.bfloat16:
                    return t.float().numpy().astype(dtype)
                else:
                    return t.numpy()
            return t
        
        merged['vit_hidden_states'] = tensor_to_numpy(hs['vit_hidden_states'], np.float16)
        merged['action_hidden_states'] = tensor_to_numpy(hs['action_hidden_states'], np.float16)
        merged['reasoning_hidden_states'] = tensor_to_numpy(hs['reasoning_hidden_states'], np.float16)
        merged['anchor_trajectory'] = tensor_to_numpy(hs['pred_traj'], np.float32)  # Keep float32 for trajectory
    else:
        merged['vit_hidden_states'] = hs['vit_hidden_states']
        merged['action_hidden_states'] = hs['action_hidden_states']
        merged['reasoning_hidden_states'] = hs['reasoning_hidden_states']
        merged['anchor_trajectory'] = hs['pred_traj']
    
    return merged


def process_senna_dataset(senna_path, hidden_states_path, output_dir, 
                          convert_to_numpy=True, skip_missing=True):
    """
    Process Senna dataset by merging with hidden states.
    
    Args:
        senna_path: Path to Senna pkl file (train or val)
        hidden_states_path: Path to inference_hidden_states.pkl
        output_dir: Directory to save processed samples
        convert_to_numpy: Whether to convert torch tensors to numpy
        skip_missing: Whether to skip samples without hidden states
        
    Returns:
        List of merged samples
    """
    # Load data
    senna_data = load_senna_data(senna_path)
    hidden_states = load_hidden_states(hidden_states_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process samples
    valid_samples = []
    skipped_count = 0
    
    print("Merging samples with hidden states...")
    for sample in tqdm(senna_data):
        merged = merge_sample_with_hidden_states(
            sample, hidden_states, convert_to_numpy=convert_to_numpy
        )
        
        if merged is not None:
            valid_samples.append(merged)
        else:
            if skip_missing:
                skipped_count += 1
            else:
                # Keep sample without hidden states (with None values)
                merged = sample.copy()
                merged['vit_hidden_states'] = None
                merged['action_hidden_states'] = None
                merged['reasoning_hidden_states'] = None
                merged['anchor_trajectory'] = None
                valid_samples.append(merged)
    
    print(f"\nProcessed {len(valid_samples)} valid samples")
    if skip_missing:
        print(f"Skipped {skipped_count} samples (no hidden states found)")
    
    # Determine output filename
    senna_filename = os.path.basename(senna_path)
    output_filename = senna_filename.replace('.pkl', '_with_features.pkl')
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump({
            'samples': valid_samples,
            'config': {
                'source_file': senna_path,
                'hidden_states_file': hidden_states_path,
                'convert_to_numpy': convert_to_numpy,
                'skip_missing': skip_missing,
            }
        }, f)
    
    print(f"✓ Saved {len(valid_samples)} samples to {output_path}")
    
    return valid_samples


def inspect_sample(sample):
    """Print detailed information about a sample for debugging"""
    print("\n" + "="*70)
    print("Sample inspection:")
    print("="*70)
    
    for key, value in sample.items():
        print(f"\n  {key}:")
        if isinstance(value, np.ndarray):
            print(f"    Type: numpy.ndarray")
            print(f"    Shape: {value.shape}")
            print(f"    Dtype: {value.dtype}")
        elif isinstance(value, torch.Tensor):
            print(f"    Type: torch.Tensor")
            print(f"    Shape: {value.shape}")
            print(f"    Dtype: {value.dtype}")
        elif isinstance(value, list):
            print(f"    Type: list")
            print(f"    Length: {len(value)}")
            if len(value) > 0:
                print(f"    First element type: {type(value[0])}")
        elif isinstance(value, (str, int, float)):
            print(f"    Type: {type(value).__name__}")
            print(f"    Value: {value}")
        else:
            print(f"    Type: {type(value)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess Senna nuScenes dataset')
    parser.add_argument('--senna_dir', type=str, default=None,
                        help='Directory containing Senna pkl files')
    parser.add_argument('--hidden_states_path', type=str, default=None,
                        help='Path to inference_hidden_states.pkl')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for processed samples')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'both'], default=None,
                        help='Which split to process')
    parser.add_argument('--convert_to_numpy', action='store_true', default=True,
                        help='Convert torch tensors to numpy arrays')
    parser.add_argument('--keep_missing', action='store_true', default=False,
                        help='Keep samples without hidden states (with None values)')
    parser.add_argument('--inspect', action='store_true', default=False,
                        help='Print detailed sample information after processing')

    args = parser.parse_args()

    defaults = {
        'senna_dir': '/mnt/data2/nuscenes/processed_senna',
        'hidden_states_path': '/mnt/data2/nuscenes/processed_senna/inference_hidden_states.pkl',
        'output_dir': '/mnt/data2/nuscenes/processed_data',
        'split': 'both',
    }

    cfg = load_config('preprocess_senna', vars(args), defaults=defaults)

    # Determine which files to process
    senna_files = {
        'train': 'senna_train_4obs.pkl',
        'val': 'senna_val_4obs.pkl',
    }
    
    splits_to_process = []
    if cfg['split'] in ['train', 'both']:
        splits_to_process.append('train')
    if cfg['split'] in ['val', 'both']:
        splits_to_process.append('val')

    # Load hidden states once (it's large, ~21GB)
    print("Loading hidden states (this may take a while)...")
    hidden_states = load_hidden_states(cfg['hidden_states_path'])
    
    for split in splits_to_process:
        print(f"\n{'='*70}")
        print(f"Processing {split} split")
        print(f"{'='*70}")

        senna_path = os.path.join(cfg['senna_dir'], senna_files[split])

        if not os.path.exists(senna_path):
            print(f"Warning: {senna_path} not found, skipping {split} split")
            continue

        # Load Senna data
        senna_data = load_senna_data(senna_path)
        
        # Create output directory
        os.makedirs(cfg['output_dir'], exist_ok=True)
        
        # Process samples
        valid_samples = []
        skipped_count = 0
        
        print("Merging samples with hidden states...")
        for sample in tqdm(senna_data):
            merged = merge_sample_with_hidden_states(
                sample, hidden_states, convert_to_numpy=args.convert_to_numpy
            )
            
            if merged is not None:
                valid_samples.append(merged)
            else:
                if not args.keep_missing:
                    skipped_count += 1
                else:
                    # Keep sample without hidden states
                    merged = sample.copy()
                    merged['vit_hidden_states'] = None
                    merged['action_hidden_states'] = None
                    merged['reasoning_hidden_states'] = None
                    merged['anchor_trajectory'] = None
                    valid_samples.append(merged)
        
        print(f"\nProcessed {len(valid_samples)} valid samples")
        if not args.keep_missing:
            print(f"Skipped {skipped_count} samples (no hidden states found)")
        
        # Inspect first sample if requested
        if args.inspect and len(valid_samples) > 0:
            inspect_sample(valid_samples[0])
        
        # Determine output filename
        output_filename = senna_files[split].replace('.pkl', '_with_features.pkl')
        output_path = os.path.join(cfg['output_dir'], output_filename)
        
        print(f"Saving to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump({
                'samples': valid_samples,
                'config': {
                    'source_file': senna_path,
                    'hidden_states_file': cfg['hidden_states_path'],
                    'convert_to_numpy': args.convert_to_numpy,
                    'skip_missing': not args.keep_missing,
                }
            }, f)
        
        print(f"✓ Saved {len(valid_samples)} samples to {output_path}")

    print(f"\n{'='*70}")
    print("✓ All processing complete!")
    print(f"{'='*70}")
