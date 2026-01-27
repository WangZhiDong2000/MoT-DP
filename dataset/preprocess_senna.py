#!/usr/bin/env python3
"""
Preprocess Senna nuScenes dataset for training.

This script loads pre-processed Senna pkl files and merges them with:
1. Inference hidden states (action features, reasoning features, anchor trajectory)
2. SparseDrive sparse instance features (det/map features)

Input:
    - Senna processed data: /mnt/data2/nuscenes/processed_senna/senna_train_4obs.pkl
    - Senna processed data: /mnt/data2/nuscenes/processed_senna/senna_val_4obs.pkl
    - Hidden states: /mnt/data2/nuscenes/processed_senna/inference_hidden_states.pkl
    - SparseDrive features: /mnt/data2/nuscenes/samples/SPARSEDRIVE_FEATURE/sample_{idx}.npz
    - nuScenes infos: data/infos/nuscenes_infos_train.pkl, nuscenes_infos_val.pkl (for sample_token to idx mapping)
    
Output:
    - Training samples with all features merged
    
Features from inference_hidden_states.pkl (keyed by sample_token):
    - action_hidden_states: (4, 2560) - Decision/action features
    - reasoning_hidden_states: (20, 2560) - Reasoning features
    - pred_traj: (1, 6, 2) - Anchor trajectory prediction

SparseDrive features from .npz files (indexed by sample idx from nuscenes_infos):
    - det_instance_feature: (50, 256) - Detection instance features
    - det_anchor_embed: (50, 256) - Detection anchor embeddings
    - det_classification: (50, 10) - Detection classification scores
    - det_prediction: (50, 11) - Detection box predictions
    - map_instance_feature: (10, 256) - Map instance features  
    - map_anchor_embed: (10, 256) - Map anchor embeddings
    - ego_feature: (1, 256) - Ego vehicle feature
    
NOTE: vit_hidden_states and lidar BEV tokens are NO LONGER used.
      SparseDrive sparse features replace them as the perception backbone.
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


# SparseDrive feature directory
SPARSEDRIVE_FEATURE_DIR = '/mnt/data2/nuscenes/samples/SPARSEDRIVE_FEATURE'

# nuScenes infos directory (for sample_token to idx mapping)
NUSCENES_INFOS_DIR = os.path.join(project_root, 'data', 'infos')


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


def load_nuscenes_infos(split='train'):
    """
    Load nuScenes infos file and build sample_token to idx mapping.
    
    Args:
        split: 'train' or 'val'
        
    Returns:
        Dict mapping sample_token to idx in the infos list
    """
    infos_path = os.path.join(NUSCENES_INFOS_DIR, f'nuscenes_infos_{split}.pkl')
    print(f"Loading nuScenes infos from {infos_path}...")
    
    with open(infos_path, 'rb') as f:
        infos = pickle.load(f)
    
    # Build sample_token -> idx mapping
    token_to_idx = {}
    for idx, info in enumerate(infos['infos']):
        token_to_idx[info['token']] = idx
    
    print(f"Built mapping for {len(token_to_idx)} samples ({split})")
    return token_to_idx


def load_sparsedrive_features(sample_token, token_to_idx, feature_dir=SPARSEDRIVE_FEATURE_DIR):
    """
    Load SparseDrive sparse instance features for a sample.
    
    Args:
        sample_token: nuScenes sample token
        token_to_idx: Dict mapping sample_token to file index
        feature_dir: Directory containing SparseDrive .npz files
        
    Returns:
        Dict with SparseDrive features or None if not found
    """
    # Get the index for this sample_token
    if sample_token not in token_to_idx:
        return None
    
    idx = token_to_idx[sample_token]
    feature_path = os.path.join(feature_dir, f"sample_{idx}.npz")
    
    if not os.path.exists(feature_path):
        return None
    
    try:
        data = np.load(feature_path)
        features = {
            'det_instance_feature': data['det_instance_feature'].astype(np.float16),  # (50, 256)
            'det_anchor_embed': data['det_anchor_embed'].astype(np.float16),          # (50, 256)
            'det_classification': data['det_classification'].astype(np.float16),      # (50, 10)
            'det_prediction': data['det_prediction'].astype(np.float32),              # (50, 11) - keep higher precision
            'map_instance_feature': data['map_instance_feature'].astype(np.float16),  # (10, 256)
            'map_anchor_embed': data['map_anchor_embed'].astype(np.float16),          # (10, 256)
            'ego_feature': data['ego_feature'].astype(np.float16),                    # (1, 256)
        }
        return features
    except Exception as e:
        print(f"Warning: Failed to load SparseDrive features for {sample_token}: {e}")
        return None


def merge_sample_with_features(sample, hidden_states_dict, token_to_idx, 
                               sparsedrive_feature_dir=SPARSEDRIVE_FEATURE_DIR, 
                               convert_to_numpy=True):
    """
    Merge a single sample with its corresponding hidden states and SparseDrive features.
    
    Args:
        sample: Dict containing sample data from Senna pkl
        hidden_states_dict: Dict mapping sample_token to hidden states
        token_to_idx: Dict mapping sample_token to SparseDrive file index
        sparsedrive_feature_dir: Directory containing SparseDrive .npz files
        convert_to_numpy: Whether to convert torch tensors to numpy arrays
        
    Returns:
        Merged sample dict or None if required features not found
    """
    sample_token = sample['sample_token']
    
    # Check if hidden states exist for this sample
    if sample_token not in hidden_states_dict:
        return None
    
    # Load SparseDrive features (using token_to_idx mapping)
    sparsedrive_features = load_sparsedrive_features(sample_token, token_to_idx, sparsedrive_feature_dir)
    if sparsedrive_features is None:
        return None
    
    hs = hidden_states_dict[sample_token]
    
    # Create merged sample - remove lidar BEV related fields
    merged = {}
    for key, value in sample.items():
        # Skip old lidar BEV token paths - no longer needed
        if key in ['hist_bev_token_paths', 'hist_bev_token_global_paths', 'lidar_token']:
            continue
        merged[key] = value
    
    # Add hidden states features (but NOT vit_hidden_states - replaced by SparseDrive)
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
        
        # NOTE: vit_hidden_states is NO LONGER saved - replaced by SparseDrive features
        merged['action_hidden_states'] = tensor_to_numpy(hs['action_hidden_states'], np.float16)
        merged['reasoning_hidden_states'] = tensor_to_numpy(hs['reasoning_hidden_states'], np.float16)
        merged['anchor_trajectory'] = tensor_to_numpy(hs['pred_traj'], np.float32)  # Keep float32 for trajectory
    else:
        merged['action_hidden_states'] = hs['action_hidden_states']
        merged['reasoning_hidden_states'] = hs['reasoning_hidden_states']
        merged['anchor_trajectory'] = hs['pred_traj']
    
    # Add SparseDrive features (already in numpy format with proper dtype)
    merged['det_instance_feature'] = sparsedrive_features['det_instance_feature']    # (50, 256)
    merged['det_anchor_embed'] = sparsedrive_features['det_anchor_embed']            # (50, 256)
    merged['det_classification'] = sparsedrive_features['det_classification']        # (50, 10)
    merged['det_prediction'] = sparsedrive_features['det_prediction']                # (50, 11)
    merged['map_instance_feature'] = sparsedrive_features['map_instance_feature']    # (10, 256)
    merged['map_anchor_embed'] = sparsedrive_features['map_anchor_embed']            # (10, 256)
    merged['sparse_ego_feature'] = sparsedrive_features['ego_feature']               # (1, 256)
    
    return merged


def process_senna_dataset(senna_path, hidden_states_path, output_dir,
                          split='train',
                          sparsedrive_feature_dir=SPARSEDRIVE_FEATURE_DIR,
                          convert_to_numpy=True, skip_missing=True):
    """
    Process Senna dataset by merging with hidden states.
    
    Args:
        senna_path: Path to Senna pkl file (train or val)
        hidden_states_path: Path to inference_hidden_states.pkl
        output_dir: Directory to save processed samples
        split: 'train' or 'val' (for loading correct nuScenes infos)
        convert_to_numpy: Whether to convert torch tensors to numpy
        skip_missing: Whether to skip samples without hidden states
        
    Returns:
        List of merged samples
    """
    # Load data
    senna_data = load_senna_data(senna_path)
    hidden_states = load_hidden_states(hidden_states_path)
    
    # Load nuScenes infos for sample_token -> idx mapping
    token_to_idx = load_nuscenes_infos(split)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process samples
    valid_samples = []
    skipped_count = 0
    missing_hidden_states = 0
    missing_sparsedrive = 0
    
    print("Merging samples with hidden states and SparseDrive features...")
    print(f"SparseDrive feature directory: {sparsedrive_feature_dir}")
    for sample in tqdm(senna_data):
        sample_token = sample['sample_token']
        
        # Check if hidden states exist
        if sample_token not in hidden_states:
            missing_hidden_states += 1
            if skip_missing:
                skipped_count += 1
                continue
        
        # Check if SparseDrive features exist
        if sample_token not in token_to_idx:
            missing_sparsedrive += 1
            if skip_missing:
                skipped_count += 1
                continue
        
        merged = merge_sample_with_features(
            sample, hidden_states, token_to_idx, sparsedrive_feature_dir, convert_to_numpy=convert_to_numpy
        )
        
        if merged is not None:
            valid_samples.append(merged)
        else:
            if skip_missing:
                skipped_count += 1
            else:
                # Keep sample without features (with None values)
                merged = sample.copy()
                merged['action_hidden_states'] = None
                merged['reasoning_hidden_states'] = None
                merged['anchor_trajectory'] = None
                merged['det_instance_feature'] = None
                merged['det_anchor_embed'] = None
                merged['det_classification'] = None
                merged['det_prediction'] = None
                merged['map_instance_feature'] = None
                merged['map_anchor_embed'] = None
                merged['sparse_ego_feature'] = None
                valid_samples.append(merged)
    
    print(f"\nProcessed {len(valid_samples)} valid samples")
    if skip_missing:
        print(f"Skipped {skipped_count} samples total:")
        print(f"  - Missing hidden states: {missing_hidden_states}")
        print(f"  - Missing SparseDrive features (not in infos): {missing_sparsedrive}")
    
    # Determine output filename - use new suffix to indicate SparseDrive features
    senna_filename = os.path.basename(senna_path)
    output_filename = senna_filename.replace('.pkl', '_with_sparse_features.pkl')
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump({
            'samples': valid_samples,
            'config': {
                'source_file': senna_path,
                'hidden_states_file': hidden_states_path,
                'sparsedrive_feature_dir': sparsedrive_feature_dir,
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
    parser = argparse.ArgumentParser(description='Preprocess Senna nuScenes dataset with SparseDrive features')
    parser.add_argument('--senna_dir', type=str, default=None,
                        help='Directory containing Senna pkl files')
    parser.add_argument('--hidden_states_path', type=str, default=None,
                        help='Path to inference_hidden_states.pkl')
    parser.add_argument('--sparsedrive_feature_dir', type=str, default=None,
                        help='Directory containing SparseDrive .npz files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for processed samples')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'both'], default=None,
                        help='Which split to process')
    parser.add_argument('--convert_to_numpy', action='store_true', default=True,
                        help='Convert torch tensors to numpy arrays')
    parser.add_argument('--keep_missing', action='store_true', default=False,
                        help='Keep samples without features (with None values)')
    parser.add_argument('--inspect', action='store_true', default=False,
                        help='Print detailed sample information after processing')

    args = parser.parse_args()

    defaults = {
        'senna_dir': '/mnt/data2/nuscenes/processed_senna',
        'hidden_states_path': '/mnt/data2/nuscenes/processed_senna/inference_hidden_states.pkl',
        'sparsedrive_feature_dir': SPARSEDRIVE_FEATURE_DIR,
        'output_dir': '/mnt/data2/nuscenes/processed_data',
        'split': 'both',
    }

    cfg = load_config('preprocess_senna', vars(args), defaults=defaults)
    
    # Get SparseDrive feature directory from config or use default
    sparsedrive_dir = cfg.get('sparsedrive_feature_dir', SPARSEDRIVE_FEATURE_DIR)

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
    
    print(f"SparseDrive feature directory: {sparsedrive_dir}")
    
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
        
        # Load token_to_idx mapping for this split
        print(f"Loading nuScenes infos for {split} split to get token->idx mapping...")
        token_to_idx = load_nuscenes_infos(split)
        print(f"Loaded {len(token_to_idx)} token-to-idx mappings")
        
        # Create output directory
        os.makedirs(cfg['output_dir'], exist_ok=True)
        
        # Process samples
        valid_samples = []
        skipped_count = 0
        
        print("Merging samples with hidden states and SparseDrive features...")
        for sample in tqdm(senna_data):
            merged = merge_sample_with_features(
                sample, hidden_states, token_to_idx, sparsedrive_dir, convert_to_numpy=args.convert_to_numpy
            )
            
            if merged is not None:
                valid_samples.append(merged)
            else:
                if not args.keep_missing:
                    skipped_count += 1
                else:
                    # Keep sample without features
                    merged = sample.copy()
                    merged['action_hidden_states'] = None
                    merged['reasoning_hidden_states'] = None
                    merged['anchor_trajectory'] = None
                    merged['det_instance_feature'] = None
                    merged['det_anchor_embed'] = None
                    merged['det_classification'] = None
                    merged['det_prediction'] = None
                    merged['map_instance_feature'] = None
                    merged['map_anchor_embed'] = None
                    merged['sparse_ego_feature'] = None
                    valid_samples.append(merged)
        
        print(f"\nProcessed {len(valid_samples)} valid samples")
        if not args.keep_missing:
            print(f"Skipped {skipped_count} samples (missing hidden states or SparseDrive features)")
        
        # Inspect first sample if requested
        if args.inspect and len(valid_samples) > 0:
            inspect_sample(valid_samples[0])
        
        # Determine output filename - new suffix for SparseDrive features
        output_filename = senna_files[split].replace('.pkl', '_with_sparse_features.pkl')
        output_path = os.path.join(cfg['output_dir'], output_filename)
        
        print(f"Saving to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump({
                'samples': valid_samples,
                'config': {
                    'source_file': senna_path,
                    'hidden_states_file': cfg['hidden_states_path'],
                    'sparsedrive_feature_dir': sparsedrive_dir,
                    'convert_to_numpy': args.convert_to_numpy,
                    'skip_missing': not args.keep_missing,
                }
            }, f)
        
        print(f"✓ Saved {len(valid_samples)} samples to {output_path}")

    print(f"\n{'='*70}")
    print("✓ All processing complete!")
    print(f"{'='*70}")
