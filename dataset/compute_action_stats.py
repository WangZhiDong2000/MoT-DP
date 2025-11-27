#!/usr/bin/env python3
"""
计算数据集中的action_stats (agent_pos统计信息)
适用于CARLA数据集，统计ego_waypoints/agent_pos的min, max, mean, std
"""
import os
import sys
import pickle
import glob
import numpy as np
import torch
from tqdm import tqdm
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def compute_action_stats_from_dataset(dataset_path, max_samples=None):
    """
    从预处理的pkl文件中统计action (agent_pos) 的统计信息
    
    Args:
        dataset_path: 数据集路径，包含train/val子目录或直接包含pkl文件
        max_samples: 最多处理多少个样本（None表示处理全部）
    
    Returns:
        stats: 包含min, max, mean, std的字典
    """
    print(f"\n{'='*60}")
    print(f"Computing action statistics from dataset...")
    print(f"Dataset path: {dataset_path}")
    print(f"{'='*60}\n")
    
    train_files = glob.glob(os.path.join(dataset_path, "train", "*.pkl"))
    val_files = glob.glob(os.path.join(dataset_path, "val", "*.pkl"))
    direct_files = glob.glob(os.path.join(dataset_path, "*.pkl"))
    
    if train_files or val_files:
        all_files = sorted(train_files + val_files)
        print(f"✓ Found {len(all_files)} samples ({len(train_files)} train, {len(val_files)} val)")
    elif direct_files:
        all_files = sorted(direct_files)
        print(f"✓ Found {len(all_files)} samples")
    else:
        raise FileNotFoundError(f"No pkl files found in {dataset_path} or its train/val subdirectories")
    
    if max_samples is not None:
        all_files = all_files[:max_samples]
        print(f"⚠ Limiting to {len(all_files)} samples for statistics computation")
    
    all_actions = []
    failed_samples = 0
    
    print("\nLoading samples...")
    for pkl_file in tqdm(all_files, desc="Processing"):
        try:
            with open(pkl_file, 'rb') as f:
                sample = pickle.load(f)
            
            # 获取ego_waypoints (agent_pos)
            ego_waypoints = sample.get('ego_waypoints')
            
            if ego_waypoints is None:
                failed_samples += 1
                continue
            
            if isinstance(ego_waypoints, torch.Tensor):
                ego_waypoints = ego_waypoints.cpu().numpy()
            elif not isinstance(ego_waypoints, np.ndarray):
                ego_waypoints = np.array(ego_waypoints)
            
            # if len(ego_waypoints) > 1:
            #     ego_waypoints = ego_waypoints[1:]
            
            all_actions.append(ego_waypoints)
            
        except Exception as e:
            print(f"\n⚠ Error loading {pkl_file}: {e}")
            failed_samples += 1
            continue
    
    if failed_samples > 0:
        print(f"\n⚠ Failed to load {failed_samples} samples")
    
    if len(all_actions) == 0:
        raise ValueError("No valid actions found in dataset!")
    

    print(f"\n✓ Successfully loaded {len(all_actions)} samples")
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
    print("Action Statistics (agent_pos):")
    print(f"{'='*60}")
    print(f"Dimensions: {all_actions_flat.shape[1]} (x, y)")
    print(f"\nMin:  {stats['min'].numpy()}")
    print(f"Max:  {stats['max'].numpy()}")
    print(f"Mean: {stats['mean'].numpy()}")
    print(f"Std:  {stats['std'].numpy()}")
    print(f"{'='*60}\n")
    
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


def save_stats_to_config(stats, config_path, output_path=None):
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
    
    parser = argparse.ArgumentParser(description='Compute action statistics from CARLA dataset')
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='/share-data/pdm_lite/tmp_data/train',
        help='Path to processed dataset (containing train/val folders or pkl files)'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='/root/z_projects/code/MoT-DP-1/config/pdm_server.yaml',
        help='Path to config file to save stats'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (None = all)'
    )
    parser.add_argument(
        '--save_to_config',
        action='store_true',
        help='Save computed stats to config file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Output path for config file (default: overwrite original)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset path not found: {args.dataset_path}")
        print("\nPlease specify the correct path using --dataset_path")
        return
    
    try:
        stats = compute_action_stats_from_dataset(
            dataset_path=args.dataset_path,
            max_samples=args.max_samples
        )
        
        if args.save_to_config:
            if os.path.exists(args.config_path):
                save_stats_to_config(stats, args.config_path, args.output_path)
            else:
                print(f"⚠ Config file not found: {args.config_path}")
                print("  Stats not saved to config.")
        
        print("\n✓ Statistics computation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error computing statistics: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
