#!/usr/bin/env python3
"""
轨迹预测可视化脚本
根据训练脚本加载模型权重，随机选取4个样本进行可视化
每个图片展示4个子图，绘制历史轨迹、真实未来轨迹和模型预测轨迹
"""
import os
import sys
import torch
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dataset.unified_carla_dataset import CARLAImageDataset
from policy.diffusion_dit_carla_policy import DiffusionDiTCarlaPolicy


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config, checkpoint_path, device):
    """加载模型和权重"""
    print(f"Loading model from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get action stats from checkpoint config if available
    checkpoint_config = checkpoint['config']
    action_stats = {
    'min': torch.tensor([-11.77335262298584, -59.26432800292969]),
    'max': torch.tensor([98.34003448486328, 55.585079193115234]),
    'mean': torch.tensor([9.755727767944336, 0.03559679538011551]),
    'std': torch.tensor([14.527670860290527, 3.224050521850586]),
    }
    
   
    
    # Initialize policy
    policy = DiffusionDiTCarlaPolicy(config, action_stats=action_stats).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model weights from checkpoint")
        if 'epoch' in checkpoint:
            print(f"  Checkpoint epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    else:
        print("⚠ No model_state_dict found in checkpoint")
    
    policy.eval()
    return policy




def main():
    # Configuration
    config_path = os.path.join(project_root, "config", "pdm_server.yaml")
    checkpoint_path = os.path.join(project_root, "checkpoints", "carla_dit_best", "carla_policy_best.pt")
    
    # Load config
    config = load_config(config_path)
    
    # Dataset paths
    dataset_path = os.path.join(
        config.get('training', {}).get('dataset_path', ''), 'val'
    )
    image_data_root = config.get('training', {}).get('image_data_root', '')
    
    print(f"Dataset path: {dataset_path}")
    print(f"Image data root: {image_data_root}")
    
    # Check if paths exist
    if not os.path.exists(dataset_path):
        print(f"⚠ Dataset path does not exist: {dataset_path}")
        print("Please update the config file or specify the correct path")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        image_data_root=image_data_root,
        mode='val'
    )
    print(f"✓ Loaded {len(dataset)} validation samples")
    
    if len(dataset) < 4:
        print(f"⚠ Dataset has only {len(dataset)} samples, need at least 4")
        return
    
    # Load model
    if not os.path.exists(checkpoint_path):
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        print("Please specify the correct checkpoint path")
        return
    
    policy = load_model(config, checkpoint_path, device)
    
    # Randomly select 4 samples
    random.seed(42)  # For reproducibility
    sample_indices = random.sample(range(len(dataset)), 4)
    print(f"Selected sample indices: {sample_indices}")
    
    samples = []
    predictions = []
    
    # Process each sample
    with torch.no_grad():
        for idx in sample_indices:
            sample = dataset[idx]
            
            # Prepare observation dict for model
            obs_dict = {
                'lidar_token': sample['lidar_token'].unsqueeze(0).to(device),  # (1, obs_horizon, seq_len, 512)
                'lidar_token_global': sample['lidar_token_global'].unsqueeze(0).to(device),  # (1, obs_horizon, 1, 512)
                'ego_status': sample['ego_status'].unsqueeze(0).to(device),  # (1, obs_horizon, feature_dim)
                'gen_vit_tokens': sample['gen_vit_tokens'].unsqueeze(0).to(device),  # (1, ...)
                'reasoning_query_tokens': sample['reasoning_query_tokens'].unsqueeze(0).to(device),  # (1, ...)
            }
            
            # Predict
            result = policy.predict_action(obs_dict)
            pred = result['action'][0]  # (pred_horizon, 2)
            
            samples.append(sample)
            predictions.append(pred)
            
            print(f"  Sample {idx}: Prediction shape {pred.shape}")
    
    # Save data first before visualization
    save_dir = os.path.join(project_root, "image")
    os.makedirs(save_dir, exist_ok=True)
    data_save_path = os.path.join(save_dir, "trajectory_data.npz")
    
    # Save trajectory data
    np.savez(data_save_path,
             sample_indices=sample_indices,
             predictions=[p for p in predictions],
             waypoints_hist=[s['waypoints_hist'].cpu().numpy() if isinstance(s['waypoints_hist'], torch.Tensor) else s['waypoints_hist'] for s in samples],
             agent_pos=[s['agent_pos'].cpu().numpy() if isinstance(s['agent_pos'], torch.Tensor) else s['agent_pos'] for s in samples])
    print(f"✓ Saved trajectory data to: {data_save_path}")
    

    



if __name__ == "__main__":
    main()
