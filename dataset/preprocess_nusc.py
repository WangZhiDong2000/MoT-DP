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
import sys
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pickle
import numpy as np
import torch
from tqdm import tqdm
import argparse
from pyquaternion import Quaternion
from dataset.config_loader import load_config


def load_nuscenes_infos(infos_path):
    """Load nuScenes infos file"""
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    
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
        waypoints_hist: List of historical waypoints [[x, y], ...], length = obs_horizon or None if insufficient
        has_full_history: Boolean indicating if we have full obs_horizon frames
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
            # Not enough history - return None to indicate skip
            return None, False
    
    return waypoints_hist, True

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
    
    # Get historical waypoints (now returns tuple)
    hist_waypoints, has_full_history = get_history_waypoints(infos, idx, scene_token, obs_horizon)
    
    # Skip if we don't have full history
    if not has_full_history or hist_waypoints is None:
        return None
    
    # Get BEV feature paths for all historical frames
    hist_bev_token_paths = []
    hist_bev_token_global_paths = []
    hist_lidar_tokens = []
    hist_ego_status = []
    hist_nav_command = []
    
    for offset in range(obs_horizon):
        hist_idx = idx - (obs_horizon - 1 - offset)  # From oldest to newest
        hist_info = infos[hist_idx]
        hist_lidar_token = extract_lidar_token_from_path(hist_info['lidar_path'])
        hist_token_path, hist_token_global_path = get_bev_feature_path(dataset_root, hist_lidar_token)
        
        # Check if historical BEV features exist
        if not (os.path.exists(hist_token_path) and os.path.exists(hist_token_global_path)):
            print(f"Warning: Historical BEV features not found for {hist_lidar_token}")
            return None
        
        hist_lidar_tokens.append(hist_lidar_token)
        hist_bev_token_paths.append(hist_token_path)
        hist_bev_token_global_paths.append(hist_token_global_path)
        
        # Get historical ego status and navigation command
        hist_ego_status.append(hist_info['ego_status'])  # (10,)
        hist_nav_command.append(hist_info['gt_ego_fut_cmd'])  # (3,)
    
    # Get future waypoints
    fut_waypoints, fut_valid_mask = get_future_waypoints(infos, idx, scene_token, action_horizon)
    
    # Get current ego status and navigation command (for backward compatibility)
    ego_status = info['ego_status']  # Shape: (10,)
    nav_command = info['gt_ego_fut_cmd']  # Shape: (3,) one-hot
    
    # Get obstacle information for future frames (action_horizon frames)
    fut_obstacles_list = []  # List of obstacle dicts for each future frame
    
    # Build transformation from current ego to global (for transforming obstacles)
    current_ego2global = np.eye(4)
    current_ego2global[:3, 3] = info['ego2global_translation']
    current_ego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
    current_global2ego = np.linalg.inv(current_ego2global)
    current_global2ego_rot = current_global2ego[:3, :3]
    
    for offset in range(1, action_horizon + 1):
        fut_idx = idx + offset
        
        # Check if we're still in the same scene and within bounds
        if fut_idx < len(infos) and infos[fut_idx]['scene_token'] == scene_token:
            fut_info = infos[fut_idx]
            
            # Get obstacle information from future frame
            # NOTE: gt_boxes are in LIDAR coordinate system!
            gt_boxes = fut_info.get('gt_boxes', np.array([])).copy()  # (N, 7) [x, y, z, l, w, h, yaw] in LIDAR frame
            gt_names = fut_info.get('gt_names', np.array([])).copy()  # (N,) obstacle class names
            gt_velocity = fut_info.get('gt_velocity', np.array([])).copy()  # (N, 2) [vx, vy] in LIDAR frame
            valid_flag = fut_info.get('valid_flag', np.array([])).copy()  # (N,) valid mask
            
            # Future lidar to future ego
            fut_lidar2ego = np.eye(4)
            fut_lidar2ego[:3, 3] = fut_info['lidar2ego_translation']
            fut_lidar2ego[:3, :3] = Quaternion(fut_info['lidar2ego_rotation']).rotation_matrix
            
            # Future ego to global
            fut_ego2global = np.eye(4)
            fut_ego2global[:3, 3] = fut_info['ego2global_translation']
            fut_ego2global[:3, :3] = Quaternion(fut_info['ego2global_rotation']).rotation_matrix
            
            # Combined transformation: future_lidar -> global -> current_ego
            fut_lidar2current_ego = current_global2ego @ fut_ego2global @ fut_lidar2ego
            fut_lidar2current_ego_rot = fut_lidar2current_ego[:3, :3]
            
            # Filter valid obstacles
            if len(valid_flag) > 0 and len(gt_boxes) > 0:
                valid_mask = valid_flag.astype(bool)
                gt_boxes = gt_boxes[valid_mask]
                gt_names = gt_names[valid_mask]
                gt_velocity = gt_velocity[valid_mask]
                
                # Transform box positions and orientations from future lidar frame to CURRENT ego frame
                for i in range(len(gt_boxes)):
                    # Box position in future lidar frame
                    box_pos_fut_lidar = np.array([gt_boxes[i, 0], gt_boxes[i, 1], gt_boxes[i, 2], 1.0])
                    # Transform to CURRENT ego frame
                    box_pos_current_ego = fut_lidar2current_ego @ box_pos_fut_lidar
                    gt_boxes[i, 0] = box_pos_current_ego[0]
                    gt_boxes[i, 1] = box_pos_current_ego[1]
                    gt_boxes[i, 2] = box_pos_current_ego[2]
                    
                    # Transform yaw angle by rotating the heading vector
                    # Create heading vector in future LIDAR frame
                    heading_lidar = np.array([np.cos(gt_boxes[i, 6]), np.sin(gt_boxes[i, 6]), 0.0])
                    # Transform to CURRENT ego frame (only rotation, no translation)
                    heading_current_ego = fut_lidar2current_ego_rot @ heading_lidar
                    # Calculate new yaw angle
                    gt_boxes[i, 6] = np.arctan2(heading_current_ego[1], heading_current_ego[0])
                    
                    # Transform velocity vector (only rotation, no translation)
                    if len(gt_velocity) > i:
                        vel_lidar = np.array([gt_velocity[i, 0], gt_velocity[i, 1], 0.0])
                        vel_current_ego = fut_lidar2current_ego_rot @ vel_lidar
                        gt_velocity[i, 0] = vel_current_ego[0]
                        gt_velocity[i, 1] = vel_current_ego[1]
            
            fut_obstacles_list.append({
                'gt_boxes': gt_boxes.astype(np.float32) if len(gt_boxes) > 0 else np.array([], dtype=np.float32).reshape(0, 7),
                'gt_names': gt_names,
                'gt_velocity': gt_velocity.astype(np.float32) if len(gt_velocity) > 0 else np.array([], dtype=np.float32).reshape(0, 2),
            })
        else:
            # No valid future frame, append empty obstacles
            fut_obstacles_list.append({
                'gt_boxes': np.array([], dtype=np.float32).reshape(0, 7),
                'gt_names': np.array([]),
                'gt_velocity': np.array([], dtype=np.float32).reshape(0, 2),
            })
    
    # Create sample dictionary
    sample = {
        'scene_token': scene_token,
        'sample_token': info['token'],
        'timestamp': info['timestamp'],
        'lidar_token': lidar_token,  # Current frame token
        'bev_token_path': token_path,  # Current frame BEV path (for backward compatibility)
        'bev_token_global_path': token_global_path,  # Current frame BEV global path
        'hist_lidar_tokens': hist_lidar_tokens,  # (obs_horizon,) historical lidar tokens
        'hist_bev_token_paths': hist_bev_token_paths,  # (obs_horizon,) historical BEV feature paths
        'hist_bev_token_global_paths': hist_bev_token_global_paths,  # (obs_horizon,) historical BEV global paths
        'hist_ego_status': np.array(hist_ego_status, dtype=np.float32),  # (obs_horizon, 10)
        'hist_nav_command': np.array(hist_nav_command, dtype=np.float32),  # (obs_horizon, 3)
        'hist_waypoints': np.array(hist_waypoints, dtype=np.float32),  # (obs_horizon, 2)
        'fut_waypoints': np.array(fut_waypoints, dtype=np.float32),    # (action_horizon, 2)
        'fut_valid_mask': np.array(fut_valid_mask, dtype=bool),        # (action_horizon,)
        'ego_status': ego_status.astype(np.float32),                   # (10,) - current frame (backward compatibility)
        'nav_command': nav_command.astype(np.float32),                 # (3,) - current frame (backward compatibility)
        'fut_obstacles': fut_obstacles_list,  # List of action_horizon dicts, each with gt_boxes, gt_names, gt_velocity
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
    parser.add_argument('--infos_dir', type=str, default=None,
                        help='Directory containing nuScenes infos files')
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='Root directory of nuScenes dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for processed samples')
    parser.add_argument('--obs_horizon', type=int, default=None,
                        help='Number of historical frames')
    parser.add_argument('--action_horizon', type=int, default=None,
                        help='Number of future frames to predict')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'both'], default=None,
                        help='Which split to process')

    args = parser.parse_args()

    defaults = {
        'infos_dir': os.path.join(project_root, 'data', 'infos'),
        'dataset_root': '/mnt/data2/nuscenes',
        'output_dir': '/mnt/data2/nuscenes/processed_data',
        'obs_horizon': 4,
        'action_horizon': 6,
        'split': 'both',
    }

    cfg = load_config('preprocess_nusc', vars(args), defaults=defaults)

    # Process train and/or val splits
    splits_to_process = []
    if cfg['split'] in ['train', 'both']:
        splits_to_process.append('train')
    if cfg['split'] in ['val', 'both']:
        splits_to_process.append('val')

    for split in splits_to_process:
        print(f"\n{'='*70}")
        print(f"Processing {split} split")
        print(f"{'='*70}")

        infos_path = os.path.join(cfg['infos_dir'], f'nuscenes_infos_{split}.pkl')

        if not os.path.exists(infos_path):
            print(f"Warning: {infos_path} not found, skipping {split} split")
            continue

        process_dataset(
            infos_path=infos_path,
            dataset_root=cfg['dataset_root'],
            output_dir=cfg['output_dir'],
            obs_horizon=cfg['obs_horizon'],
            action_horizon=cfg['action_horizon']
        )

    print(f"\n{'='*70}")
    print("✓ All processing complete!")
    print(f"{'='*70}")
