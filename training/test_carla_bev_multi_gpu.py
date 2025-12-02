#!/usr/bin/env python3
"""
Multi-GPU Testing Script for CARLA BEV Diffusion Policy
Evaluates model performance on test set using multiple GPUs in parallel
"""
import os
import sys
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
from collections import defaultdict
from torch.distributed.elastic.multiprocessing.errors import record

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from dataset.unified_carla_dataset import CARLAImageDataset
from policy.diffusion_dit_carla_policy import DiffusionDiTCarlaPolicy


def load_config(config_path=None):
    """Load configuration from yaml file"""
    if config_path is None:
        config_path = os.path.join(project_root, "config", "pdm_server.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_driving_metrics(predicted_trajectories, target_trajectories, fut_obstacles=None):
    """
    计算驾驶性能指标
    
    Args:
        predicted_trajectories: (B, T, 2) 预测轨迹
        target_trajectories: (B, T, 2) 真实轨迹
        fut_obstacles: Optional obstacle information
    
    Returns:
        metrics: 包含L2误差和碰撞率的字典
    """
    predicted_trajectories = predicted_trajectories.detach().cpu().numpy()
    target_trajectories = target_trajectories.detach().cpu().numpy()
    
    B, T, _ = predicted_trajectories.shape
    
    l2_errors = np.linalg.norm(
        predicted_trajectories - target_trajectories, axis=-1
    )
    
    metrics = {}
    
    # === L2 误差指标 ===
    if T >= 2:  
        metrics['L2_1s'] = np.mean(l2_errors[:, 1])
    
    if T >= 4:  
        metrics['L2_2s'] = np.mean(l2_errors[:, 3])
    
    if T >= 6: 
        metrics['L2_3s'] = np.mean(l2_errors[:, 5])
    
    # L2_avg: 只计算1s, 2s, 3s时间步的平均
    l2_avg_values = []
    if T >= 2:
        l2_avg_values.append(l2_errors[:, 1])
    if T >= 4:
        l2_avg_values.append(l2_errors[:, 3])
    if T >= 6:
        l2_avg_values.append(l2_errors[:, 5])
    
    if len(l2_avg_values) > 0:
        metrics['L2_avg'] = np.mean(np.concatenate(l2_avg_values))
    else:
        metrics['L2_avg'] = 0.0
    
    # Clean NaN and Inf values
    safe_metrics = {}
    for key, value in metrics.items():
        if np.isnan(value):
            print(f"Warning: computed NaN for metric '{key}'")
            safe_metrics[key] = 0.0
        elif np.isinf(value):
            print(f"Warning: computed Inf for metric '{key}'")
            safe_metrics[key] = 1e10 if value > 0 else -1e10
        else:
            safe_metrics[key] = value
    
    return safe_metrics


def test_model(policy, test_loader, device, rank=0, world_size=1):
    """
    Test function for distributed evaluation
    
    Args:
        policy: Model to evaluate
        test_loader: DataLoader for test set
        device: Device to run on
        rank: Process rank
        world_size: Total number of processes
    
    Returns:
        test_metrics: Dictionary of averaged metrics
    """
    policy.eval()
    
    # Get the actual model (unwrap DDP if needed)
    model_for_inference = policy.module if world_size > 1 else policy
    
    test_metrics = defaultdict(list)
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(test_loader, desc="Testing", dynamic_ncols=True)
        else:
            pbar = test_loader
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Compute loss
            try:
                loss = model_for_inference.compute_loss(batch)
                test_metrics['loss'].append(loss.item())
            except Exception as e:
                print(f"Warning: Error computing loss at batch {batch_idx}: {e}")
                continue
            
            # Prepare observation dict
            obs_dict = {
                'lidar_token': batch['lidar_token'][:, :model_for_inference.n_obs_steps],
                'lidar_token_global': batch['lidar_token_global'][:, :model_for_inference.n_obs_steps],
                'ego_status': batch['ego_status'][:, :model_for_inference.n_obs_steps],  
                'gen_vit_tokens': batch['gen_vit_tokens'],
                'reasoning_query_tokens': batch['reasoning_query_tokens']
            }
            target_actions = batch['agent_pos']
            
            # Predict actions
            try:
                result = model_for_inference.predict_action(obs_dict)
                predicted_actions = torch.from_numpy(result['action']).to(device)
                
                # Align target dimensions
                if target_actions.dim() == 3:  # (B, T, 2)
                    target_actions = target_actions[:, :predicted_actions.shape[1]]
                elif target_actions.dim() == 2:  # (B, 2) 
                    target_actions = target_actions.unsqueeze(1)  # (B, 1, 2)
                
                fut_obstacles = batch.get('fut_obstacles', None)
                
                # Compute driving metrics
                driving_metrics = compute_driving_metrics(
                    predicted_actions, 
                    target_actions, 
                    fut_obstacles=fut_obstacles 
                )
                
                for key, value in driving_metrics.items():
                    test_metrics[key].append(value)
                
                if rank == 0:
                    current_metrics = {
                        'loss': f'{loss.item():.4f}',
                        'L2_avg': f'{driving_metrics.get("L2_avg", 0.0):.4f}'
                    }
                    pbar.set_postfix(current_metrics)
                    
            except Exception as e:
                print(f"Warning: Error in action prediction at batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if rank == 0:
        pbar.close()
    
    # Average metrics across all batches
    averaged_metrics = {}
    for key, values in test_metrics.items():
        if values:
            mean_value = np.mean(values)
            if np.isnan(mean_value):
                print(f"Warning: computed NaN for metric 'test_{key}'")
                averaged_metrics[f'test_{key}'] = 0.0
            elif np.isinf(mean_value):
                print(f"Warning: computed Inf for metric 'test_{key}'")
                averaged_metrics[f'test_{key}'] = 1e10 if mean_value > 0 else -1e10
            else:
                averaged_metrics[f'test_{key}'] = mean_value
    
    return averaged_metrics


@record
def main(config_path, checkpoint_path, test_split='test'):
    """
    Main testing function with multi-GPU support
    
    Args:
        config_path: Path to config yaml
        checkpoint_path: Path to model checkpoint
        test_split: Which split to test on ('test' or 'val')
    """
    torch.cuda.empty_cache()
    
    print("=" * 80)
    print("Multi-GPU Model Testing")
    print("=" * 80)
    
    config = load_config(config_path=config_path)
    
    # Initialize distributed training
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Single GPU fallback
    if world_size == 1:
        print("Running in single GPU mode")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        print(f"Running in multi-GPU mode: Rank {rank}/{world_size-1}")
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    
    if rank == 0:
        print(f"\n✓ Config loaded from: {config_path}")
        print(f"✓ Checkpoint: {checkpoint_path}")
        print(f"✓ Test split: {test_split}")
        print(f"✓ Device: {device}")
        print(f"✓ World size: {world_size}")
    
    # Load dataset
    dataset_path_root = config.get('training', {}).get('dataset_path')
    test_dataset_path = os.path.join(dataset_path_root, test_split)
    image_data_root = config.get('training', {}).get('image_data_root')
    
    if rank == 0:
        print(f"\n✓ Loading {test_split} dataset from: {test_dataset_path}")
    
    test_dataset = CARLAImageDataset(
        dataset_path=test_dataset_path,
        image_data_root=image_data_root
    )
    
    if rank == 0:
        print(f"✓ {test_split.capitalize()} samples: {len(test_dataset)}")
    
    # Create DataLoader with distributed sampler
    batch_size = config.get('dataloader', {}).get('batch_size', 32)
    num_workers = config.get('dataloader', {}).get('num_workers', 4)
    
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset,
            shuffle=False,
            num_replicas=world_size,
            rank=rank,
            drop_last=False
        )
    else:
        sampler = None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    if rank == 0:
        print(f"✓ DataLoader created with batch_size={batch_size}, num_workers={num_workers}")
    
    # Initialize policy
    action_stats = None
    if 'action_stats' in config:
        action_stats = {
            'min': torch.tensor(config['action_stats']['min'], dtype=torch.float32),
            'max': torch.tensor(config['action_stats']['max'], dtype=torch.float32),
            'mean': torch.tensor(config['action_stats']['mean'], dtype=torch.float32),
            'std': torch.tensor(config['action_stats']['std'], dtype=torch.float32),
        }
        if rank == 0:
            print("✓ Loaded action_stats from config")
    
    policy = DiffusionDiTCarlaPolicy(config, action_stats=action_stats)
    
    # Load checkpoint
    if rank == 0:
        print(f"\n✓ Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DDP training)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    policy.load_state_dict(new_state_dict, strict=False)
    
    if rank == 0:
        print("✓ Checkpoint loaded successfully")
        if 'epoch' in checkpoint:
            print(f"  - Epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"  - Val Loss: {checkpoint['val_loss']:.4f}")
    
    # Move to device
    policy = policy.to(device)
    
    # Wrap with DDP for multi-GPU
    if world_size > 1:
        policy = torch.nn.parallel.DistributedDataParallel(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
        if rank == 0:
            print("✓ Model wrapped with DistributedDataParallel")
    
    # Run testing
    if rank == 0:
        print("\n" + "=" * 80)
        print("Starting Testing")
        print("=" * 80 + "\n")
    
    test_metrics = test_model(policy, test_loader, device, rank, world_size)
    
    # Gather metrics from all GPUs
    if world_size > 1:
        # Convert metrics to tensors for all_reduce
        metric_keys = sorted(test_metrics.keys())
        metric_values = torch.tensor([test_metrics[k] for k in metric_keys], device=device)
        
        # Average across all processes
        torch.distributed.all_reduce(metric_values, op=torch.distributed.ReduceOp.SUM)
        metric_values /= world_size
        
        # Update metrics
        test_metrics = {k: v.item() for k, v in zip(metric_keys, metric_values)}
    
    # Print results (only rank 0)
    if rank == 0:
        print("\n" + "=" * 80)
        print("Test Results")
        print("=" * 80)
        
        # Print in organized format
        print(f"\n{'Metric':<20} {'Value':>15}")
        print("-" * 40)
        
        # Loss first
        if 'test_loss' in test_metrics:
            print(f"{'Loss':<20} {test_metrics['test_loss']:>15.4f}")
        
        # Then L2 metrics
        l2_metrics = ['test_L2_1s', 'test_L2_2s', 'test_L2_3s', 'test_L2_avg']
        for metric in l2_metrics:
            if metric in test_metrics:
                metric_name = metric.replace('test_', '')
                print(f"{metric_name:<20} {test_metrics[metric]:>15.4f}")
        
        # Other metrics
        for key, value in sorted(test_metrics.items()):
            if key not in ['test_loss'] + l2_metrics:
                metric_name = key.replace('test_', '')
                print(f"{metric_name:<20} {value:>15.4f}")
        
        print("=" * 80)
        
        # Save results to file
        results_dir = os.path.join(project_root, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pth', '')
        results_file = os.path.join(results_dir, f'test_results_{checkpoint_name}_{test_split}.json')
        
        results = {
            'checkpoint': checkpoint_path,
            'test_split': test_split,
            'config': config_path,
            'metrics': test_metrics,
            'world_size': world_size
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n✓ Results saved to: {results_file}")
    
    # Cleanup
    if world_size > 1:
        torch.distributed.destroy_process_group()
    
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test CARLA BEV Diffusion Policy on multiple GPUs')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config yaml file (default: config/pdm_server.yaml)')
    parser.add_argument('--checkpoint', type=str,default='/root/z_projects/code/MoT-DP-1/checkpoints/carla_dit_best/carla_policy_best.pt', required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--test_split', type=str, default='val', choices=['test', 'val'],
                        help='Which dataset split to test on (default: test)')
    
    args = parser.parse_args()
    
    main(args.config, args.checkpoint, args.test_split)

# bash training/test_carla_bev_multi_gpu.sh /root/z_projects/code/MoT-DP-1/checkpoints/carla_dit_best/carla_policy_best.pt 4 val