#!/usr/bin/env python3
"""
计算数据集中的action_stats (agent_pos和anchor统计信息)
适用于CARLA数据集，统计ego_waypoints/agent_pos和anchor(pred_traj)的min, max, mean, std
最后取两者的最小min和最大max，并更新yaml中truncated_diffusion的归一化参数
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


def compute_action_stats_from_dataset(dataset_path, image_data_root, max_samples=None):
    """
    从预处理的pkl文件中统计action (agent_pos和anchor) 的统计信息
    
    Args:
        dataset_path: 数据集路径，包含train/val子目录或直接包含pkl文件
        image_data_root: 图像数据根目录，用于加载vqa特征文件
        max_samples: 最多处理多少个样本（None表示处理全部）
    
    Returns:
        stats: 包含min, max, mean, std的字典（合并agent_pos和anchor范围）
    """
    print(f"\n{'='*60}")
    print(f"Computing action statistics from dataset...")
    print(f"Dataset path: {dataset_path}")
    print(f"Image data root: {image_data_root}")
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
    
    all_agent_pos = []
    all_anchor = []
    failed_samples = 0
    
    print("\nLoading samples...")
    for pkl_file in tqdm(all_files, desc="Processing"):
        try:
            with open(pkl_file, 'rb') as f:
                sample = pickle.load(f)
            
            # 获取ego_waypoints (agent_pos)
            ego_waypoints = sample.get('ego_waypoints')
            
            if ego_waypoints is not None:
                if isinstance(ego_waypoints, torch.Tensor):
                    # Convert BFloat16 to Float32 if necessary
                    if ego_waypoints.dtype == torch.bfloat16:
                        ego_waypoints = ego_waypoints.float()
                    ego_waypoints = ego_waypoints.cpu().numpy()
                elif not isinstance(ego_waypoints, np.ndarray):
                    ego_waypoints = np.array(ego_waypoints)
                
                # 跳过第一个点 (ego_waypoints[1:])
                if len(ego_waypoints) > 1:
                    agent_pos = ego_waypoints[1:]
                    all_agent_pos.append(agent_pos)
            
            # 获取anchor (from vqa feature: pred_traj)
            vqa_path = sample.get('vqa', None)
            if vqa_path is not None and image_data_root is not None:
                full_vqa_path = os.path.join(image_data_root, vqa_path)
                if os.path.exists(full_vqa_path):
                    try:
                        # Try loading with weights_only=True first
                        vqa_feature = torch.load(full_vqa_path, weights_only=True, map_location='cpu')
                    except Exception as e:
                        # Fall back to loading without weights_only if BFloat16 issues occur
                        if 'BFloat16' in str(e):
                            vqa_feature = torch.load(full_vqa_path, weights_only=False, map_location='cpu')
                        else:
                            raise
                    
                    if 'pred_traj' in vqa_feature:
                        anchor = vqa_feature['pred_traj']
                        if isinstance(anchor, torch.Tensor):
                            # Convert BFloat16 to Float32 if necessary
                            if anchor.dtype == torch.bfloat16:
                                anchor = anchor.float()
                            anchor = anchor.cpu().numpy()
                        all_anchor.append(anchor)
            
        except Exception as e:
            print(f"\n⚠ Error loading {pkl_file}: {e}")
            failed_samples += 1
            continue
    
    if failed_samples > 0:
        print(f"\n⚠ Failed to load {failed_samples} samples")
    
    if len(all_agent_pos) == 0 and len(all_anchor) == 0:
        raise ValueError("No valid actions found in dataset!")
    
    print(f"\n✓ Successfully loaded {len(all_agent_pos)} agent_pos samples")
    print(f"✓ Successfully loaded {len(all_anchor)} anchor samples")
    
    print("Computing statistics...")
    
    # Compute agent_pos stats
    agent_pos_stats = None
    if len(all_agent_pos) > 0:
        all_agent_pos_flat = np.concatenate(all_agent_pos, axis=0)
        print(f"\nTotal agent_pos waypoints: {len(all_agent_pos_flat)}")
        print(f"agent_pos shape: {all_agent_pos_flat.shape}")
        
        agent_pos_stats = {
            'min': np.min(all_agent_pos_flat, axis=0),
            'max': np.max(all_agent_pos_flat, axis=0),
            'mean': np.mean(all_agent_pos_flat, axis=0),
            'std': np.std(all_agent_pos_flat, axis=0),
        }
        print(f"\nagent_pos stats:")
        print(f"  Min:  {agent_pos_stats['min']}")
        print(f"  Max:  {agent_pos_stats['max']}")
        print(f"  Mean: {agent_pos_stats['mean']}")
        print(f"  Std:  {agent_pos_stats['std']}")
    
    # Compute anchor stats
    anchor_stats = None
    if len(all_anchor) > 0:
        all_anchor_concat = np.concatenate(all_anchor, axis=0)
        print(f"\nTotal anchor samples: {len(all_anchor_concat)}")
        print(f"anchor shape before reshape: {all_anchor_concat.shape}")
        
        # If anchor has 3 dimensions (e.g., [N, seq_len, 2]), flatten to 2D [N*seq_len, 2]
        # This way stats will be computed across ALL waypoints, same as agent_pos
        if all_anchor_concat.ndim == 3:
            original_shape = all_anchor_concat.shape
            all_anchor_flat = all_anchor_concat.reshape(-1, all_anchor_concat.shape[-1])
            print(f"anchor shape after reshape: {all_anchor_flat.shape} (flattened from {original_shape})")
        else:
            all_anchor_flat = all_anchor_concat
            print(f"anchor shape: {all_anchor_flat.shape}")
        
        print(f"Total anchor waypoints (after flatten): {len(all_anchor_flat)}")
        
        anchor_stats = {
            'min': np.min(all_anchor_flat, axis=0),
            'max': np.max(all_anchor_flat, axis=0),
            'mean': np.mean(all_anchor_flat, axis=0),
            'std': np.std(all_anchor_flat, axis=0),
        }
        print(f"\nanchor stats:")
        print(f"  Min:  {anchor_stats['min']}")
        print(f"  Max:  {anchor_stats['max']}")
        print(f"  Mean: {anchor_stats['mean']}")
        print(f"  Std:  {anchor_stats['std']}")
    
    # Combine stats: take min of mins and max of maxs
    if agent_pos_stats is not None and anchor_stats is not None:
        combined_min = np.minimum(agent_pos_stats['min'], anchor_stats['min'])
        combined_max = np.maximum(agent_pos_stats['max'], anchor_stats['max'])
        # For mean and std, use the combined data
        # Now both should have the same number of dimensions (2D)
        all_combined = np.concatenate([all_agent_pos_flat, all_anchor_flat], axis=0)
        combined_mean = np.mean(all_combined, axis=0)
        combined_std = np.std(all_combined, axis=0)
    elif agent_pos_stats is not None:
        combined_min = agent_pos_stats['min']
        combined_max = agent_pos_stats['max']
        combined_mean = agent_pos_stats['mean']
        combined_std = agent_pos_stats['std']
    else:
        combined_min = anchor_stats['min']
        combined_max = anchor_stats['max']
        combined_mean = anchor_stats['mean']
        combined_std = anchor_stats['std']
    
    stats = {
        'min': torch.tensor(combined_min, dtype=torch.float32),
        'max': torch.tensor(combined_max, dtype=torch.float32),
        'mean': torch.tensor(combined_mean, dtype=torch.float32),
        'std': torch.tensor(combined_std, dtype=torch.float32),
    }
    
    print(f"\n{'='*60}")
    print("Combined Action Statistics (agent_pos + anchor):")
    print(f"{'='*60}")
    print(f"Dimensions: {combined_min.shape[0]} (x, y)")
    print(f"\nCombined Min:  {stats['min'].numpy()}")
    print(f"Combined Max:  {stats['max'].numpy()}")
    print(f"Combined Mean: {stats['mean'].numpy()}")
    print(f"Combined Std:  {stats['std'].numpy()}")
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
        
        # Update truncated_diffusion normalization parameters
        # Normalization: 2*(x + offset)/range - 1, mapping data to [-1, 1]
        # offset = -x_min (with margin), range = x_max - x_min (with margin)
        x_min = stats['min'][0].item()
        x_max = stats['max'][0].item()
        y_min = stats['min'][1].item()
        y_max = stats['max'][1].item()
        
        # Add margin (round to nice numbers)
        margin = 1.0
        x_min_margin = np.floor(x_min - margin)
        x_max_margin = np.ceil(x_max + margin)
        y_min_margin = np.floor(y_min - margin)
        y_max_margin = np.ceil(y_max + margin)
        
        # Calculate normalization parameters
        norm_x_offset = -x_min_margin  # offset to make min become 0
        norm_x_range = x_max_margin - x_min_margin
        norm_y_offset = -y_min_margin
        norm_y_range = y_max_margin - y_min_margin
        
        if 'truncated_diffusion' not in config:
            config['truncated_diffusion'] = {}
        
        config['truncated_diffusion']['norm_x_offset'] = float(norm_x_offset)
        config['truncated_diffusion']['norm_x_range'] = float(norm_x_range)
        config['truncated_diffusion']['norm_y_offset'] = float(norm_y_offset)
        config['truncated_diffusion']['norm_y_range'] = float(norm_y_range)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✓ Action stats saved to config: {output_path}")
        print(f"\ntruncated_diffusion normalization parameters updated:")
        print(f"  Data range: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")
        print(f"  Extended range: x=[{x_min_margin}, {x_max_margin}], y=[{y_min_margin}, {y_max_margin}]")
        print(f"  norm_x_offset: {norm_x_offset}")
        print(f"  norm_x_range: {norm_x_range}")
        print(f"  norm_y_offset: {norm_y_offset}")
        print(f"  norm_y_range: {norm_y_range}")
        
    except Exception as e:
        print(f"\n⚠ Failed to save stats to config: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute action statistics from CARLA dataset')
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='/share-data/pdm_lite/tmp_data',
        help='Path to processed dataset (containing train/val folders or pkl files)'
    )
    parser.add_argument(
        '--image_data_root',
        type=str,
        default='/share-data/pdm_lite/',
        help='Root directory for image data (to load vqa feature files for anchor)'
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
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset path not found: {args.dataset_path}")
        print("\nPlease specify the correct path using --dataset_path")
        return
    
    try:
        stats = compute_action_stats_from_dataset(
            dataset_path=args.dataset_path,
            image_data_root=args.image_data_root,
            max_samples=args.max_samples
        )
        
        # Automatically save to config unless --no_save is specified
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
