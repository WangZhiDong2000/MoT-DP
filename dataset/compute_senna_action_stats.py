#!/usr/bin/env python3
"""
计算Senna nuScenes数据集中的action_stats (agent_pos/fut_waypoints统计信息)
适用于预处理后的Senna数据，统计fut_waypoints的min, max, mean, std
"""
import os
import sys
import pickle
import numpy as np
import torch
from tqdm import tqdm
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def compute_action_stats_from_senna_pkl(pkl_path, max_samples=None):
    """
    从预处理的Senna pkl文件中统计action (fut_waypoints) 的统计信息
    
    Args:
        pkl_path: Senna pkl文件路径
        max_samples: 最多处理多少个样本（None表示处理全部）
    
    Returns:
        stats: 包含min, max, mean, std的字典
    """
    print(f"\n{'='*60}")
    print(f"Computing action statistics from Senna pkl...")
    print(f"PKL path: {pkl_path}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading Senna data...")
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    samples = data['samples']
    print(f"✓ Loaded {len(samples)} samples")
    
    if max_samples is not None:
        samples = samples[:max_samples]
        print(f"⚠ Limiting to {len(samples)} samples for statistics computation")
    
    all_actions = []
    all_anchors = []
    failed_samples = 0
    
    print("\nProcessing samples...")
    for sample in tqdm(samples, desc="Processing"):
        try:
            # Get fut_waypoints (action / agent_pos) - these are in offset format
            fut_waypoints = sample.get('fut_waypoints')
            
            if fut_waypoints is None:
                failed_samples += 1
                continue
            
            if isinstance(fut_waypoints, torch.Tensor):
                fut_waypoints = fut_waypoints.cpu().numpy()
            elif not isinstance(fut_waypoints, np.ndarray):
                fut_waypoints = np.array(fut_waypoints)
            
            # Convert offset to absolute coordinates using cumsum
            fut_waypoints = np.cumsum(fut_waypoints, axis=0)
            
            all_actions.append(fut_waypoints)
            
            # Also collect anchor trajectories if available
            anchor_traj = sample.get('anchor_trajectory')
            if anchor_traj is not None:
                if isinstance(anchor_traj, torch.Tensor):
                    anchor_traj = anchor_traj.cpu().numpy()
                elif not isinstance(anchor_traj, np.ndarray):
                    anchor_traj = np.array(anchor_traj)
                # anchor_trajectory shape: (1, 6, 2) -> (6, 2)
                if anchor_traj.ndim == 3:
                    anchor_traj = anchor_traj[0]
                all_anchors.append(anchor_traj)
            
        except Exception as e:
            print(f"\n⚠ Error processing sample: {e}")
            failed_samples += 1
            continue
    
    if failed_samples > 0:
        print(f"\n⚠ Failed to process {failed_samples} samples")
    
    if len(all_actions) == 0:
        raise ValueError("No valid actions found in dataset!")
    
    print(f"\n✓ Successfully processed {len(all_actions)} samples")
    print("Computing statistics...")
    
    all_actions_flat = np.concatenate(all_actions, axis=0)
    
    print(f"\nTotal waypoints: {len(all_actions_flat)}")
    print(f"Action shape: {all_actions_flat.shape}")
    
    stats = {
        'min': torch.tensor(np.min(all_actions_flat, axis=0), dtype=torch.float32),
        'max': torch.tensor(np.max(all_actions_flat, axis=0), dtype=torch.float32),
        'mean': torch.tensor(np.mean(all_actions_flat, axis=0), dtype=torch.float32),
        'std': torch.tensor(np.std(all_actions_flat, axis=0), dtype=torch.float32),
    }
    
    print(f"\n{'='*60}")
    print("Action Statistics (fut_waypoints / agent_pos):")
    print(f"{'='*60}")
    print(f"Dimensions: {all_actions_flat.shape[1]} (x, y)")
    print(f"\nMin:  {stats['min'].numpy()}")
    print(f"Max:  {stats['max'].numpy()}")
    print(f"Mean: {stats['mean'].numpy()}")
    print(f"Std:  {stats['std'].numpy()}")
    
    # Also print anchor trajectory stats if available
    if len(all_anchors) > 0:
        all_anchors_flat = np.concatenate(all_anchors, axis=0)
        print(f"\n{'='*60}")
        print("Anchor Trajectory Statistics:")
        print(f"{'='*60}")
        print(f"Total anchor waypoints: {len(all_anchors_flat)}")
        print(f"Anchor min:  {np.min(all_anchors_flat, axis=0)}")
        print(f"Anchor max:  {np.max(all_anchors_flat, axis=0)}")
        print(f"Anchor mean: {np.mean(all_anchors_flat, axis=0)}")
        print(f"Anchor std:  {np.std(all_anchors_flat, axis=0)}")
    
    print(f"\n{'='*60}")
    print("Python code format (copy to your training script):")
    print("-" * 60)
    print("action_stats = {")
    print(f"    'min': torch.tensor({stats['min'].tolist()}),")
    print(f"    'max': torch.tensor({stats['max'].tolist()}),")
    print(f"    'mean': torch.tensor({stats['mean'].tolist()}),")
    print(f"    'std': torch.tensor({stats['std'].tolist()}),")
    print("}")
    print("-" * 60)
    
    return stats


def compute_combined_stats(train_pkl_path, val_pkl_path=None, max_samples=None):
    """
    从训练集和验证集合并计算统计信息
    
    Args:
        train_pkl_path: 训练集pkl路径
        val_pkl_path: 验证集pkl路径（可选）
        max_samples: 每个集合的最大样本数
    
    Returns:
        stats: 合并后的统计信息
    """
    print(f"\n{'='*60}")
    print("Computing combined action statistics...")
    print(f"{'='*60}\n")
    
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    
    all_actions = []
    all_anchors = []
    
    # Process train set
    print(f"Loading train data from {train_pkl_path}...")
    with open(train_pkl_path, 'rb') as f:
        train_data = pickle.load(f)
    train_samples = train_data['samples']
    if max_samples:
        train_samples = train_samples[:max_samples]
    print(f"✓ Loaded {len(train_samples)} train samples")
    
    for sample in tqdm(train_samples, desc="Processing train"):
        fut_waypoints = sample.get('fut_waypoints')
        if fut_waypoints is not None:
            if isinstance(fut_waypoints, torch.Tensor):
                fut_waypoints = fut_waypoints.cpu().numpy()
            # Convert offset to absolute coordinates using cumsum
            fut_waypoints = np.cumsum(fut_waypoints, axis=0)
            all_actions.append(fut_waypoints)
        
        anchor_traj = sample.get('anchor_trajectory')
        if anchor_traj is not None:
            if isinstance(anchor_traj, torch.Tensor):
                anchor_traj = anchor_traj.cpu().numpy()
            if anchor_traj.ndim == 3:
                anchor_traj = anchor_traj[0]
            all_anchors.append(anchor_traj)
    
    # Process val set if provided
    if val_pkl_path and os.path.exists(val_pkl_path):
        print(f"\nLoading val data from {val_pkl_path}...")
        with open(val_pkl_path, 'rb') as f:
            val_data = pickle.load(f)
        val_samples = val_data['samples']
        if max_samples:
            val_samples = val_samples[:max_samples]
        print(f"✓ Loaded {len(val_samples)} val samples")
        
        for sample in tqdm(val_samples, desc="Processing val"):
            fut_waypoints = sample.get('fut_waypoints')
            if fut_waypoints is not None:
                if isinstance(fut_waypoints, torch.Tensor):
                    fut_waypoints = fut_waypoints.cpu().numpy()
                # Convert offset to absolute coordinates using cumsum
                fut_waypoints = np.cumsum(fut_waypoints, axis=0)
                all_actions.append(fut_waypoints)
            
            anchor_traj = sample.get('anchor_trajectory')
            if anchor_traj is not None:
                if isinstance(anchor_traj, torch.Tensor):
                    anchor_traj = anchor_traj.cpu().numpy()
                if anchor_traj.ndim == 3:
                    anchor_traj = anchor_traj[0]
                all_anchors.append(anchor_traj)
    
    # Compute statistics
    all_actions_flat = np.concatenate(all_actions, axis=0)
    
    print(f"\n{'='*60}")
    print(f"Total samples: {len(all_actions)}")
    print(f"Total waypoints: {len(all_actions_flat)}")
    print(f"Action shape: {all_actions_flat.shape}")
    print(f"{'='*60}")
    
    stats = {
        'min': torch.tensor(np.min(all_actions_flat, axis=0), dtype=torch.float32),
        'max': torch.tensor(np.max(all_actions_flat, axis=0), dtype=torch.float32),
        'mean': torch.tensor(np.mean(all_actions_flat, axis=0), dtype=torch.float32),
        'std': torch.tensor(np.std(all_actions_flat, axis=0), dtype=torch.float32),
    }
    
    print(f"\nAction Statistics (combined train+val):")
    print(f"Min:  {stats['min'].numpy()}")
    print(f"Max:  {stats['max'].numpy()}")
    print(f"Mean: {stats['mean'].numpy()}")
    print(f"Std:  {stats['std'].numpy()}")
    
    # Anchor trajectory stats
    if len(all_anchors) > 0:
        all_anchors_flat = np.concatenate(all_anchors, axis=0)
        print(f"\nAnchor Trajectory Statistics:")
        print(f"Total anchor waypoints: {len(all_anchors_flat)}")
        print(f"Anchor min:  {np.min(all_anchors_flat, axis=0)}")
        print(f"Anchor max:  {np.max(all_anchors_flat, axis=0)}")
        print(f"Anchor mean: {np.mean(all_anchors_flat, axis=0)}")
        print(f"Anchor std:  {np.std(all_anchors_flat, axis=0)}")
    
    print(f"\n{'='*60}")
    print("Python code format:")
    print("-" * 60)
    print("action_stats = {")
    print(f"    'min': torch.tensor({stats['min'].tolist()}),")
    print(f"    'max': torch.tensor({stats['max'].tolist()}),")
    print(f"    'mean': torch.tensor({stats['mean'].tolist()}),")
    print(f"    'std': torch.tensor({stats['std'].tolist()}),")
    print("}")
    print("-" * 60)
    
    return stats


def save_stats_to_config(stats, config_path, output_path=None):
    """
    保存统计信息到YAML配置文件
    """
    if output_path is None:
        output_path = config_path
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'action_stats' not in config:
            config['action_stats'] = {}
        
        config['action_stats']['min'] = stats['min'].tolist()
        config['action_stats']['max'] = stats['max'].tolist()
        config['action_stats']['mean'] = stats['mean'].tolist()
        config['action_stats']['std'] = stats['std'].tolist()
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✓ Action stats saved to config: {output_path}")
        
    except Exception as e:
        print(f"\n⚠ Failed to save stats to config: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute action statistics from Senna nuScenes dataset')
    parser.add_argument(
        '--train_pkl',
        type=str,
        default='/mnt/data2/nuscenes/processed_data/senna_train_4obs_with_features.pkl',
        help='Path to train pkl file'
    )
    parser.add_argument(
        '--val_pkl',
        type=str,
        default='/mnt/data2/nuscenes/processed_data/senna_val_4obs_with_features.pkl',
        help='Path to val pkl file (optional)'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='/root/z_projects/code/MoT-DP-1/config/nuscences_server.yaml',
        help='Path to config file to save stats'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process per split (None = all)'
    )
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Do NOT save computed stats to config file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Output path for config file (default: overwrite original)'
    )
    parser.add_argument(
        '--train_only',
        action='store_true',
        help='Only use train set for statistics (ignore val set)'
    )
    
    args = parser.parse_args()
    
    # Check paths
    if not os.path.exists(args.train_pkl):
        print(f"❌ Train pkl not found: {args.train_pkl}")
        return
    
    try:
        if args.train_only or not os.path.exists(args.val_pkl):
            # Train only
            stats = compute_action_stats_from_senna_pkl(
                pkl_path=args.train_pkl,
                max_samples=args.max_samples
            )
        else:
            # Combined train + val
            stats = compute_combined_stats(
                train_pkl_path=args.train_pkl,
                val_pkl_path=args.val_pkl,
                max_samples=args.max_samples
            )
        
        # Save to config
        if not args.no_save:
            if os.path.exists(args.config_path):
                save_stats_to_config(stats, args.config_path, args.output_path)
            else:
                print(f"⚠ Config file not found: {args.config_path}")
                print("  Stats not saved to config.")
                print("  You can specify a custom config path using --config_path")
        
        print("\n✓ Statistics computation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error computing statistics: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
