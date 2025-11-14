#!/usr/bin/env python3
"""
Preprocess nuScenes dataset for training.

This script processes nuScenes infos data and generates training samples with:
- BEV features from LIDAR
- Historical waypoints
- Future trajectories
- Navigation commands

Input:
    - nuScenes infos: /home/wang/Project/MoT-DP/data/infos/mini/nuscenes_infos_train.pkl
    - BEV features: /home/wang/Dataset/v1.0-mini/samples/LIDAR_TOP_BEV_FEATURES/
    
Output:
    - Training samples: /home/wang/Dataset/v1.0-mini/processed_data/
"""

import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import argparse
from pathlib import Path
from pyquaternion import Quaternion

def load_nuscenes_infos(infos_path):
    """Load nuScenes infos file"""
    # Compatibility fix for numpy version mismatch
    import sys
    import numpy
    sys.modules['numpy._core'] = numpy.core
    sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
    
    with open(infos_path, 'rb') as f:
        data = pickle.load(f)
    return data['infos'], data['metadata']

def get_bev_feature_path(dataset_root, lidar_token):
    """Get path to BEV feature file from lidar token"""
    # Extract token from lidar path
    # lidar_path format: ../data/nuscenes/samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin
    feature_dir = os.path.join(dataset_root, 'samples', 'LIDAR_TOP_BEV_FEATURES')
    
    # Token is the full filename without extension
    token_path = os.path.join(feature_dir, f'{lidar_token}_token.pt')
    token_global_path = os.path.join(feature_dir, f'{lidar_token}_token_global.pt')
    
    return token_path, token_global_path

def extract_lidar_token_from_path(lidar_path):
    """Extract token from lidar path"""
    # lidar_path: ../data/nuscenes/samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin
    filename = os.path.basename(lidar_path)
    # Remove .pcd.bin extension
    token = filename.replace('.pcd.bin', '')
    return token

def ego_to_local_transform(ego2global_translation, ego2global_rotation):
    """
    Create transformation matrix from ego vehicle frame to local (current) frame.
    
    Args:
        ego2global_translation: [x, y, z] translation from ego to global
        ego2global_rotation: [w, x, y, z] quaternion from ego to global
        
    Returns:
        4x4 transformation matrix from global to ego (inverse of ego2global)
    """
    # Create ego2global matrix
    ego2global = np.eye(4)
    ego2global[:3, 3] = ego2global_translation
    ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
    
    # Return global2ego (inverse)
    global2ego = np.linalg.inv(ego2global)
    return global2ego

def transform_waypoint_to_ego(waypoint_global, current_global2ego):
    """Transform a waypoint from global frame to current ego frame"""
    waypoint_4d = np.array([waypoint_global[0], waypoint_global[1], 0.0, 1.0])
    waypoint_ego = current_global2ego @ waypoint_4d
    return waypoint_ego[:2]  # Return only x, y

def get_history_waypoints(infos, current_idx, scene_token, obs_horizon=4):
    """
    Extract historical waypoints in current ego frame.
    
    Args:
        infos: List of all info dicts
        current_idx: Index of current frame
        scene_token: Token of current scene
        obs_horizon: Number of historical frames
        
    Returns:
        waypoints_hist: List of historical waypoints [[x, y], ...], length = obs_horizon
    """
    current_info = infos[current_idx]
    current_global2ego = ego_to_local_transform(
        current_info['ego2global_translation'],
        current_info['ego2global_rotation']
    )
    
    waypoints_hist = []
    
    # Go backwards from current frame
    for offset in range(obs_horizon):
        hist_idx = current_idx - offset
        
        # Check if we're still in the same scene and within bounds
        if hist_idx >= 0 and infos[hist_idx]['scene_token'] == scene_token:
            hist_info = infos[hist_idx]
            # Get historical ego position in global frame
            hist_translation = hist_info['ego2global_translation']
            # Transform to current ego frame
            waypoint_ego = transform_waypoint_to_ego(hist_translation, current_global2ego)
            waypoints_hist.insert(0, waypoint_ego)
        else:
            # Pad with first available waypoint or zeros
            if waypoints_hist:
                waypoints_hist.insert(0, waypoints_hist[0].copy())
            else:
                waypoints_hist.insert(0, np.array([0.0, 0.0]))
    
    return waypoints_hist

def get_future_waypoints(infos, current_idx, scene_token, action_horizon=6):
    """
    Extract future waypoints in current ego frame.
    
    Args:
        infos: List of all info dicts
        current_idx: Index of current frame
        scene_token: Token of current scene
        action_horizon: Number of future frames
        
    Returns:
        waypoints_fut: List of future waypoints [[x, y], ...], length = action_horizon
        valid_mask: Boolean mask indicating valid waypoints
    """
    current_info = infos[current_idx]
    current_global2ego = ego_to_local_transform(
        current_info['ego2global_translation'],
        current_info['ego2global_rotation']
    )
    
    waypoints_fut = []
    valid_mask = []
    
    # Go forward from next frame
    for offset in range(1, action_horizon + 1):
        fut_idx = current_idx + offset
        
        # Check if we're still in the same scene and within bounds
        if fut_idx < len(infos) and infos[fut_idx]['scene_token'] == scene_token:
            fut_info = infos[fut_idx]
            # Get future ego position in global frame
            fut_translation = fut_info['ego2global_translation']
            # Transform to current ego frame
            waypoint_ego = transform_waypoint_to_ego(fut_translation, current_global2ego)
            waypoints_fut.append(waypoint_ego)
            valid_mask.append(True)
        else:
            # Pad with last valid waypoint or zeros
            if waypoints_fut:
                waypoints_fut.append(waypoints_fut[-1].copy())
            else:
                waypoints_fut.append(np.array([0.0, 0.0]))
            valid_mask.append(False)
    
    return waypoints_fut, valid_mask

def process_sample(infos, idx, dataset_root, obs_horizon=4, action_horizon=6):
    """
    Process a single sample.
    
    Returns:
        sample_dict or None if sample should be skipped
    """
    info = infos[idx]
    scene_token = info['scene_token']
    
    # Check if we have enough future frames in the same scene
    num_future = 0
    for offset in range(1, action_horizon + 1):
        fut_idx = idx + offset
        if fut_idx < len(infos) and infos[fut_idx]['scene_token'] == scene_token:
            num_future += 1
        else:
            break
    
    # Skip if we don't have at least half of the required future frames
    if num_future < action_horizon // 2:
        return None
    
    # Extract lidar token
    lidar_token = extract_lidar_token_from_path(info['lidar_path'])
    
    # Get BEV feature paths
    token_path, token_global_path = get_bev_feature_path(dataset_root, lidar_token)
    
    # Check if BEV features exist
    if not (os.path.exists(token_path) and os.path.exists(token_global_path)):
        print(f"Warning: BEV features not found for {lidar_token}")
        return None
    
    # Get historical waypoints
    hist_waypoints = get_history_waypoints(infos, idx, scene_token, obs_horizon)
    
    # Get future waypoints
    fut_waypoints, fut_valid_mask = get_future_waypoints(infos, idx, scene_token, action_horizon)
    
    # Get ego status (velocity, acceleration, etc.)
    ego_status = info['ego_status']  # Shape: (10,)
    
    # Get navigation command
    nav_command = info['gt_ego_fut_cmd']  # Shape: (3,) one-hot
    
    # Create sample dictionary
    sample = {
        'scene_token': scene_token,
        'sample_token': info['token'],
        'timestamp': info['timestamp'],
        'lidar_token': lidar_token,
        'bev_token_path': token_path,
        'bev_token_global_path': token_global_path,
        'hist_waypoints': np.array(hist_waypoints, dtype=np.float32),  # (obs_horizon, 2)
        'fut_waypoints': np.array(fut_waypoints, dtype=np.float32),    # (action_horizon, 2)
        'fut_valid_mask': np.array(fut_valid_mask, dtype=bool),        # (action_horizon,)
        'ego_status': ego_status.astype(np.float32),                   # (10,)
        'nav_command': nav_command.astype(np.float32),                 # (3,)
    }
    
    return sample

def process_dataset(infos_path, dataset_root, output_dir, obs_horizon=4, action_horizon=6):
    """
    Process entire dataset.
    
    Args:
        infos_path: Path to nuScenes infos pkl file
        dataset_root: Root directory of nuScenes dataset
        output_dir: Directory to save processed samples
        obs_horizon: Number of historical frames
        action_horizon: Number of future frames
    """
    # Load infos
    print(f"Loading infos from {infos_path}...")
    infos, metadata = load_nuscenes_infos(infos_path)
    print(f"Loaded {len(infos)} samples from {metadata['version']}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process samples
    valid_samples = []
    skipped_count = 0
    
    print("Processing samples...")
    for idx in tqdm(range(len(infos))):
        sample = process_sample(infos, idx, dataset_root, obs_horizon, action_horizon)
        if sample is not None:
            valid_samples.append(sample)
        else:
            skipped_count += 1
    
    print(f"\nProcessed {len(valid_samples)} valid samples")
    print(f"Skipped {skipped_count} samples")
    
    # Save processed samples
    output_path = os.path.join(output_dir, os.path.basename(infos_path).replace('_infos_', '_processed_'))
    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump({
            'samples': valid_samples,
            'metadata': metadata,
            'config': {
                'obs_horizon': obs_horizon,
                'action_horizon': action_horizon,
            }
        }, f)
    
    print(f"✓ Saved {len(valid_samples)} samples to {output_path}")
    
    return valid_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess nuScenes dataset for training')
    parser.add_argument('--infos_dir', type=str, 
                       default='/home/wang/Project/MoT-DP/data/infos/mini',
                       help='Directory containing nuScenes infos files')
    parser.add_argument('--dataset_root', type=str,
                       default='/home/wang/Dataset/v1.0-mini',
                       help='Root directory of nuScenes dataset')
    parser.add_argument('--output_dir', type=str,
                       default='/home/wang/Dataset/v1.0-mini/processed_data',
                       help='Output directory for processed samples')
    parser.add_argument('--obs_horizon', type=int, default=4,
                       help='Number of historical frames')
    parser.add_argument('--action_horizon', type=int, default=6,
                       help='Number of future frames to predict')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'both'], default='both',
                       help='Which split to process')
    
    args = parser.parse_args()
    
    # Process train and/or val splits
    splits_to_process = []
    if args.split in ['train', 'both']:
        splits_to_process.append('train')
    if args.split in ['val', 'both']:
        splits_to_process.append('val')
    
    for split in splits_to_process:
        print(f"\n{'='*70}")
        print(f"Processing {split} split")
        print(f"{'='*70}")
        
        infos_path = os.path.join(args.infos_dir, f'nuscenes_infos_{split}.pkl')
        
        if not os.path.exists(infos_path):
            print(f"Warning: {infos_path} not found, skipping {split} split")
            continue
        
        process_dataset(
            infos_path=infos_path,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            obs_horizon=args.obs_horizon,
            action_horizon=args.action_horizon
        )
    
    print(f"\n{'='*70}")
    print("✓ All processing complete!")
    print(f"{'='*70}")
