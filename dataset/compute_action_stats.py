#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import torch
from tqdm import tqdm
import yaml
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def compute_action_stats_from_processed_data(pkl_path, max_samples=None):
    """
    Args:
        pkl_path: 处理后的pkl文件路径 (如 nuscenes_processed_train.pkl)
        max_samples: 最多处理多少个样本（None表示处理全部）
    
    Returns:
        stats: 包含min, max, mean, std的字典
    """
    print(f"\n{'='*60}")
    print(f"Computing action statistics from processed nuScenes data...")
    print(f"Dataset file: {pkl_path}")
    print(f"{'='*60}\n")
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    with open(pkl_path, 'rb') as f:
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
        data = pickle.load(f)
    
    samples = data['samples']
    print(f"✓ Found {len(samples)} samples")
    
    if max_samples is not None:
        samples = samples[:max_samples]
        print(f"⚠ Limiting to {len(samples)} samples for statistics computation")
    
    all_actions = []
    failed_samples = 0
    
    print("\nProcessing samples...")
    for sample in tqdm(samples, desc="Processing"):
        try:
            fut_waypoints = sample.get('fut_waypoints')
            
            if fut_waypoints is None:
                failed_samples += 1
                continue
            
            if isinstance(fut_waypoints, torch.Tensor):
                fut_waypoints = fut_waypoints.cpu().numpy()
            elif not isinstance(fut_waypoints, np.ndarray):
                fut_waypoints = np.array(fut_waypoints)
            
            all_actions.append(fut_waypoints)
            
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
    print("Action Statistics (fut_waypoints):")
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


def save_stats_to_config(stats, config_path):
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        if 'action_stats' not in config:
            config['action_stats'] = {}
        
        config['action_stats']['min'] = stats['min'].tolist()
        config['action_stats']['max'] = stats['max'].tolist()
        config['action_stats']['mean'] = stats['mean'].tolist()
        config['action_stats']['std'] = stats['std'].tolist()
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✓ Action stats saved to config: {config_path}")
        
    except Exception as e:
        print(f"\n⚠ Failed to save stats to config: {e}")


def main():
    default_train_pkl = '/home/wang/Dataset/v1.0-mini/processed_data/nuscenes_processed_train.pkl'
    default_config = os.path.join(project_root, 'config', 'nuscenes.yaml')
    parser = argparse.ArgumentParser(
        description='Compute action statistics from preprocessed nuScenes data'
    )
    parser.add_argument(
        '--pkl_path',
        type=str,
        default=default_train_pkl,
        help=f'Path to processed pkl file (default: {default_train_pkl})'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default=default_config,
        help=f'Path to config file to save stats (default: {default_config})'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (None = all)'
    )
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Do not save stats to config file'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pkl_path):
        print(f"❌ Processed data file not found: {args.pkl_path}")
        print("\nPlease run preprocess_nusc_new.py first to generate processed data.")
        print("Or specify a different path using --pkl_path")
        return
    
    try:
        stats = compute_action_stats_from_processed_data(
            pkl_path=args.pkl_path,
            max_samples=args.max_samples
        )
        
        if not args.no_save:
            config_dir = os.path.dirname(args.config_path)
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
                print(f"✓ Created config directory: {config_dir}")
            save_stats_to_config(stats, args.config_path)
        print("\n✓ Statistics computation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error computing statistics: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
