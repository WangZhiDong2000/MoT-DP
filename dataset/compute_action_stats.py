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
sys.path.insert(0, project_root)

from dataset.config_loader import load_config


def compute_action_stats_from_processed_data(pkl_paths, max_samples=None):
    """
    Args:
        pkl_paths: 处理后的pkl文件路径或路径列表 (如 nuscenes_processed_train.pkl 或 [train.pkl, val.pkl])
        max_samples: 最多处理多少个样本（None表示处理全部）
    
    Returns:
        stats: 包含min, max, mean, std的字典
    """
    # 支持单个路径或路径列表
    if isinstance(pkl_paths, str):
        pkl_paths = [pkl_paths]
    
    print(f"\n{'='*60}")
    print(f"Computing action statistics from processed nuScenes data...")
    print(f"Dataset files:")
    for pkl_path in pkl_paths:
        print(f"  - {pkl_path}")
    print(f"{'='*60}\n")
    
    all_samples = []
    total_loaded = 0
    
    # Load preprocessed data from all files
    for pkl_path in pkl_paths:
        print(f"Loading preprocessed data from: {pkl_path}...")
        if not os.path.exists(pkl_path):
            print(f"⚠ File not found: {pkl_path}, skipping...")
            continue
            
        with open(pkl_path, 'rb') as f:
            sys.modules['numpy._core'] = np.core
            sys.modules['numpy._core.multiarray'] = np.core.multiarray
            data = pickle.load(f)
        
        samples = data['samples']
        print(f"✓ Found {len(samples)} samples in this file")
        all_samples.extend(samples)
        total_loaded += len(samples)
    
    print(f"\n✓ Total loaded {total_loaded} samples from all files")
    
    samples = all_samples
    
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
    default_train_pkl = '/mnt/data2/nuscenes/processed_data/nuscenes_processed_train.pkl'
    default_val_pkl = '/mnt/data2/nuscenes/processed_data/nuscenes_processed_val.pkl'
    default_config = os.path.join(project_root, 'config', 'nuscences_server.yaml')
    parser = argparse.ArgumentParser(
        description='Compute action statistics from preprocessed nuScenes data (train+val combined)'
    )
    parser.add_argument('--pkl_path', type=str, default=None,
                        help=f'Path to single processed pkl file (overrides config)')
    parser.add_argument('--train_pkl', type=str, default=None,
                        help=f'Path to training pkl file')
    parser.add_argument('--val_pkl', type=str, default=None,
                        help=f'Path to validation pkl file')
    parser.add_argument('--config_path', type=str, default=None,
                        help=f'Path to config file to save stats (overrides config)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (None = all)')
    parser.add_argument('--no_save', action='store_true', default=None,
                        help='Do not save stats to config file (overrides config)')

    args = parser.parse_args()

    # Load merged config: defaults <- config/dataset.yaml <- CLI
    defaults = {
        'pkl_path': None,
        'train_pkl': default_train_pkl,
        'val_pkl': default_val_pkl,
        'config_path': default_config,
        'max_samples': None,
        'no_save': False,
    }
    cfg = load_config('compute_action_stats', vars(args), defaults=defaults)
    
    # 确定要使用的pkl文件路径
    if cfg.get('pkl_path'):
        # 如果指定了单一文件，只用那个
        pkl_paths = [cfg['pkl_path']]
        print("Using single pkl file from --pkl_path")
    else:
        # 否则使用 train + val 合并
        pkl_paths = []
        if cfg.get('train_pkl') and os.path.exists(cfg['train_pkl']):
            pkl_paths.append(cfg['train_pkl'])
        if cfg.get('val_pkl') and os.path.exists(cfg['val_pkl']):
            pkl_paths.append(cfg['val_pkl'])
        
        if not pkl_paths:
            print(f"❌ No valid pkl files found:")
            print(f"   Train: {cfg.get('train_pkl')}")
            print(f"   Val: {cfg.get('val_pkl')}")
            print("\nPlease run preprocess_nusc.py first to generate processed data.")
            print("Or specify files via --pkl_path, --train_pkl, --val_pkl or config/dataset.yaml")
            return
    
    if not all(os.path.exists(p) for p in pkl_paths):
        missing = [p for p in pkl_paths if not os.path.exists(p)]
        print(f"❌ Processed data files not found:")
        for p in missing:
            print(f"   - {p}")
        print("\nPlease run preprocess_nusc.py first to generate processed data.")
        print("Or specify different paths in config/dataset.yaml or via CLI arguments")
        return
    
    try:
        stats = compute_action_stats_from_processed_data(
            pkl_paths=pkl_paths,
            max_samples=cfg.get('max_samples')
        )

        if not cfg.get('no_save', False):
            config_path = cfg.get('config_path', default_config)
            config_dir = os.path.dirname(config_path)
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
                print(f"✓ Created config directory: {config_dir}")
            save_stats_to_config(stats, config_path)
        print("\n✓ Statistics computation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error computing statistics: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
