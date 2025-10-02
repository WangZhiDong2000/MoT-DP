#!/usr/bin/env python3

import os
import sys
import torch
import yaml
import wandb
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from dataset.generate_pdm_dataset import CARLAImageDataset
from policy.diffusion_dit_pusht_policy import DiffusionDiTPushTPolicy

def create_carla_config():
    """从yaml文件加载CARLA驾驶任务的配置"""
    import yaml
    
    config_path = "/home/wang/projects/diffusion_policy_z/config/carla.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 确保action shape匹配CARLA任务
    config['shape_meta']['action'] = {'shape': [2]}  # throttle, steer
    
    return config

def compute_driving_metrics(predicted_trajectories, target_trajectories):
    """
    计算轨迹预测相关指标，仿照TCP test.py的评估方式
    
    Args:
        predicted_trajectories: (B, T, 2) - 预测的轨迹点 [x, y] (世界坐标)
        target_trajectories: (B, T, 2) - 真实轨迹点 [x, y] (世界坐标)
    
    Returns:
        dict: 包含各种评估指标
    """
    predicted_trajectories = predicted_trajectories.detach().cpu().numpy()
    target_trajectories = target_trajectories.detach().cpu().numpy()
    
    # L2距离误差 (欧几里德距离，单位：米)
    l2_errors = np.linalg.norm(predicted_trajectories - target_trajectories, axis=-1)  # (B, T)
    
    # 不同时间步的L2误差（仿照TCP的0.5s, 1s, 1.5s, 2s评估）
    metrics = {}
    time_steps = [0, 1, 2, 3]  # 对应不同的时间点（4个预测步）
    time_labels = ['step_1', 'step_2', 'step_3', 'step_4']
    
    for i, (step, label) in enumerate(zip(time_steps, time_labels)):
        if step < l2_errors.shape[1]:
            metrics[f'trajectory_error_{label}'] = np.mean(l2_errors[:, step])
    
    # 平均轨迹误差 (单位：米)
    metrics['trajectory_error_mean'] = np.mean(l2_errors)
    
    # 分别计算x和y坐标的误差 (单位：米)
    x_error = np.mean(np.abs(predicted_trajectories[:, :, 0] - target_trajectories[:, :, 0]))
    y_error = np.mean(np.abs(predicted_trajectories[:, :, 1] - target_trajectories[:, :, 1]))
    
    metrics['x_coordinate_mae'] = x_error
    metrics['y_coordinate_mae'] = y_error
    
    # 轨迹变化平滑性评估 (单位：米)
    pred_diff = np.diff(predicted_trajectories, axis=1)
    target_diff = np.diff(target_trajectories, axis=1)
    smoothness_error = np.mean(np.linalg.norm(pred_diff - target_diff, axis=-1))
    metrics['trajectory_smoothness_error'] = smoothness_error
    
    return metrics

def validate_model(policy, val_loader, device):
    """验证模型并计算评估指标"""
    policy.eval()
    val_metrics = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            # 移动数据到设备
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # 计算损失
            loss = policy.compute_loss(batch)
            val_metrics['loss'].append(loss.item())
            

            obs_dict = {
                'image': batch['image'][:, :policy.n_obs_steps],  # (B, obs_horizon, C, H, W)
                'agent_pos': batch['agent_pos'][:, :policy.n_obs_steps],
                'speed': batch['speed'][:, :policy.n_obs_steps],
                'target_point': batch['target_point'][:, :policy.n_obs_steps],
                'next_command': batch['next_command'][:, :policy.n_obs_steps],  
            }
            
            # 预测动作
            try:
                result = policy.predict_action(obs_dict)
                predicted_actions = torch.from_numpy(result['action']).to(device)
                print(predicted_actions.shape)
                action_horizon = predicted_actions.shape[1]
                ego_waypoints = batch['ego_waypoints']
                if len(ego_waypoints.shape) == 4:
                    last_obs_waypoints = ego_waypoints[:, -1, :, :]
                    target_actions = last_obs_waypoints[:, :action_horizon, :]
                else:
                    target_actions = ego_waypoints[:, :action_horizon]
                driving_metrics = compute_driving_metrics(predicted_actions, target_actions)
                for key, value in driving_metrics.items():
                    val_metrics[key].append(value)
            except Exception as e:
                print(f"Warning: Error in action prediction during validation: {e}")
                continue
    
    # 计算平均指标
    averaged_metrics = {}
    for key, values in val_metrics.items():
        if values:  # 确保列表不为空
            averaged_metrics[f'val_{key}'] = np.mean(values)
        
    return averaged_metrics

def train_carla_policy():
    """训练CARLA驾驶策略"""
    print("Initializing CARLA driving policy training...")
    
    # 创建配置
    config = create_carla_config()
    
    # 初始化wandb
    wandb.init(
        project=config.get('logging', {}).get('wandb_project', "carla-diffusion-policy"),
        name=config.get('logging', {}).get('run_name', "carla_dit_full_validation"),
        config={
            "learning_rate": config.get('optimizer', {}).get('lr', 5e-5),
            "epochs": config.get('training', {}).get('num_epochs', 50),
            "batch_size": config.get('dataloader', {}).get('batch_size', 16),
            "obs_horizon": config.get('obs_horizon', 2),
            "action_horizon": config.get('action_horizon', 4),
            "pred_horizon": config.get('pred_horizon', 8),
            "dataset_path": config.get('training', {}).get('dataset_path', ""),
            "max_files": 10,
            "train_split": 0.8,
            "weight_decay": config.get('optimizer', {}).get('weight_decay', 1e-5),
            "num_workers": config.get('dataloader', {}).get('num_workers', 4)
        }
    )
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据集设置
    dataset_path = config.get('training', {}).get('dataset_path', "/home/wang/projects/diffusion_policy_z/data/tmp_data")
    pred_horizon = config.get('pred_horizon', 8)
    obs_horizon = config.get('obs_horizon', 2)
    action_horizon = config.get('action_horizon', 4)
    max_files = 10  # 适当增加文件数量
    
    print(f"Loading CARLA dataset from {dataset_path}")
    
    # 创建训练和验证数据集
    train_dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon, 
        action_horizon=action_horizon,
        max_files=max_files,
        train_split=0.8,
        mode='train'
    )
    
    val_dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        max_files=max_files,
        train_split=0.8,
        mode='val'
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 创建数据加载器
    batch_size = config.get('dataloader', {}).get('batch_size', 16)
    num_workers = config.get('dataloader', {}).get('num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print("Initializing policy model...")
    # 创建策略模型
    policy = DiffusionDiTPushTPolicy(config).to(device)
    
    # 不再设置数据标准化器
    
    # 优化器
    lr = config.get('optimizer', {}).get('lr', 5e-5)
    weight_decay = config.get('optimizer', {}).get('weight_decay', 1e-5)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    
    print("Setting up normalizers...")
    print("Starting training loop...")
    
    # 训练循环
    num_epochs = config.get('training', {}).get('num_epochs', 50)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        policy.train()
        train_losses = []
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Training...")
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            # 移动数据到设备
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # 前向传播
            optimizer.zero_grad()
            loss = policy.compute_loss(batch)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # 记录训练指标
            if batch_idx % 10 == 0:
                step = epoch * len(train_loader) + batch_idx
                wandb.log({
                    "train/loss_step": loss.item(),
                    "train/epoch": epoch,
                    "train/step": step,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/batch_idx": batch_idx
                })
                
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # 计算epoch训练指标
        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")
        
        # 记录epoch级别的训练指标到wandb
        wandb.log({
            "train/loss_epoch": avg_train_loss,
            "train/epoch": epoch,
            "train/samples_processed": (epoch + 1) * len(train_dataset)
        })

        # 验证阶段
        print("Validating...")
        val_metrics = validate_model(policy, val_loader, device)
        
        # 记录所有指标到wandb，分类组织
        log_dict = {
            "epoch": epoch,
            "train/loss": avg_train_loss,
        }
        
        # 添加验证损失
        if 'val_loss' in val_metrics:
            log_dict["val/loss"] = val_metrics['val_loss']
        
        # 添加轨迹预测指标
        trajectory_metrics = {}
        for key, value in val_metrics.items():
            if 'trajectory_error' in key:
                new_key = f"val/trajectory/{key.replace('val_', '')}"
                trajectory_metrics[new_key] = value
            elif 'coordinate' in key:
                new_key = f"val/coordinate/{key.replace('val_', '')}"
                trajectory_metrics[new_key] = value
            elif 'smoothness' in key:
                new_key = f"val/smoothness/{key.replace('val_', '')}"
                trajectory_metrics[new_key] = value
            else:
                # 其他验证指标
                new_key = f"val/{key.replace('val_', '')}"
                trajectory_metrics[new_key] = value
        
        log_dict.update(trajectory_metrics)
        wandb.log(log_dict)
        
        # 打印验证指标
        print(f"Validation metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # 保存最佳模型
        val_loss = val_metrics.get('val_loss', float('inf'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_dir = "/home/wang/projects/diffusion_policy_z/checkpoints/carla_dit"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            torch.save({
                'model_state_dict': policy.state_dict(),
                'config': config,
                'epoch': epoch,
                'val_loss': val_loss,
                'train_loss': avg_train_loss,
                'val_metrics': val_metrics
            }, os.path.join(checkpoint_dir, "carla_policy_best.pt"))
            
            print(f"✓ New best model saved with val_loss: {val_loss:.4f}")
            
            # 记录最佳模型信息到wandb
            wandb.log({
                "best_model/epoch": epoch,
                "best_model/val_loss": val_loss,
                "best_model/train_loss": avg_train_loss
            })
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # 记录训练完成信息到wandb
    wandb.log({
        "training/completed": True,
        "training/total_epochs": num_epochs,
        "training/best_val_loss": best_val_loss,
        "training/final_train_loss": avg_train_loss
    })
    
    # 关闭wandb
    wandb.finish()

if __name__ == "__main__":
    train_carla_policy()