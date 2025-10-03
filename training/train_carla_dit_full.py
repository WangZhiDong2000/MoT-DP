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
from policy.diffusion_dit_carla_policy import DiffusionDiTCarlaPolicy
import yaml

def create_carla_config():
    config_path = "/home/wang/projects/diffusion_policy_z/config/carla.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def compute_action_stats(train_dataset):
    """
    计算训练数据集中action的统计信息，包含outlier过滤
    """
    print("Computing action statistics from training dataset...")
    all_actions = []
    
    # 随机采样一部分数据来计算统计信息（避免内存问题）
    sample_size = min(1000, len(train_dataset))
    indices = np.random.choice(len(train_dataset), sample_size, replace=False)
    
    for i in tqdm(indices, desc="Collecting action samples"):
        sample = train_dataset[i]
        agent_pos = sample['agent_pos']  # shape: (sequence_length, 2)
        
        # 提取action部分（obs_horizon之后的部分）
        obs_horizon = train_dataset.obs_horizon
        actions = agent_pos[obs_horizon:]  # shape: (action_horizon, 2)
        all_actions.append(actions)
    
    # 转换为numpy数组并计算统计信息
    all_actions = np.concatenate(all_actions, axis=0)  # shape: (N * action_horizon, 2)
    
    print(f"Raw action statistics from {len(all_actions)} samples:")
    raw_min = np.min(all_actions, axis=0)
    raw_max = np.max(all_actions, axis=0)
    print(f"  Raw range: X=[{raw_min[0]:.4f}, {raw_max[0]:.4f}], Y=[{raw_min[1]:.4f}, {raw_max[1]:.4f}]")
    
    # 使用percentile来过滤极端outliers
    percentile_low = 1  # 过滤掉最低1%
    percentile_high = 99  # 过滤掉最高1%
    
    action_min = np.percentile(all_actions, percentile_low, axis=0)
    action_max = np.percentile(all_actions, percentile_high, axis=0)
    action_mean = np.mean(all_actions, axis=0)
    action_std = np.std(all_actions, axis=0)
    
    print(f"Filtered action statistics (using {percentile_low}-{percentile_high} percentile):")
    print(f"  Min: {action_min}")
    print(f"  Max: {action_max}")
    print(f"  Mean: {action_mean}")
    print(f"  Std: {action_std}")
    print(f"  Range: {action_max - action_min}")
    
    # 进一步验证：确保范围合理（驾驶任务中相对位移很少超过10米）
    reasonable_max_range = 10.0
    if np.any(action_max - action_min > reasonable_max_range):
        print(f"  ⚠️ Range still seems large, applying conservative clipping...")
        # 使用更保守的范围
        conservative_min = np.maximum(action_min, np.array([-3.0, -3.0]))
        conservative_max = np.minimum(action_max, np.array([8.0, 4.0]))
        action_min = conservative_min
        action_max = conservative_max
        print(f"  Conservative range: X=[{action_min[0]:.4f}, {action_max[0]:.4f}], Y=[{action_min[1]:.4f}, {action_max[1]:.4f}]")
    
    action_stats = {
        'min': torch.from_numpy(action_min).float(),
        'max': torch.from_numpy(action_max).float(),
        'mean': torch.from_numpy(action_mean).float(),
        'std': torch.from_numpy(action_std).float()
    }
    
    return action_stats

def compute_driving_metrics(predicted_trajectories, target_trajectories):
    predicted_trajectories = predicted_trajectories.detach().cpu().numpy()
    target_trajectories = target_trajectories.detach().cpu().numpy()

    # L2距离误差
    l2_errors = np.linalg.norm(predicted_trajectories - target_trajectories, axis=-1)  # (B, T)

    # 不同时间步的L2误差（仿照TCP的0.5s, 1s, 1.5s, 2s评估）
    metrics = {}
    time_steps = [0, 1, 2, 3]  
    time_labels = ['step_1', 'step_2', 'step_3', 'step_4']
    
    for i, (step, label) in enumerate(zip(time_steps, time_labels)):
        if step < l2_errors.shape[1]:
            metrics[f'trajectory_error_{label}'] = np.mean(l2_errors[:, step])
    
    # 平均轨迹误差 
    metrics['trajectory_error_mean'] = np.mean(l2_errors)
    
    # x和y坐标的误差
    x_error = np.mean(np.abs(predicted_trajectories[:, :, 0] - target_trajectories[:, :, 0]))
    y_error = np.mean(np.abs(predicted_trajectories[:, :, 1] - target_trajectories[:, :, 1]))
    
    metrics['x_coordinate_mae'] = x_error
    metrics['y_coordinate_mae'] = y_error
    
    # 轨迹变化平滑性评估
    pred_diff = np.diff(predicted_trajectories, axis=1)
    target_diff = np.diff(target_trajectories, axis=1)
    smoothness_error = np.mean(np.linalg.norm(pred_diff - target_diff, axis=-1))
    metrics['trajectory_smoothness_error'] = smoothness_error
    
    return metrics

def validate_model(policy, val_loader, device):
    policy.eval()
    val_metrics = defaultdict(list)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            loss = policy.compute_loss(batch)
            val_metrics['loss'].append(loss.item())
            
            obs_dict = {
                'image': batch['image'][:, :policy.n_obs_steps],  # (B, obs_horizon, C, H, W)
                'agent_pos': batch['agent_pos'][:, :policy.n_obs_steps],  # (B, obs_horizon, 2) - 观测步的agent_pos
                'speed': batch['speed'][:, :policy.n_obs_steps],
                'target_point': batch['target_point'][:, :policy.n_obs_steps],
                'next_command': batch['next_command'][:, :policy.n_obs_steps],  
            }
            
            try:
                result = policy.predict_action(obs_dict)
                predicted_actions = torch.from_numpy(result['action']).to(device)
                # 目标动作是观测步之后的位置
                target_actions = batch['agent_pos'][:, policy.n_obs_steps:policy.n_obs_steps + predicted_actions.shape[1]]
                
                # 调试信息：打印数值范围
                if batch_idx == 0:
                    print(f"\n=== Debug Info ===")
                    print(f"Predicted actions shape: {predicted_actions.shape}")
                    print(f"Predicted actions range: [{predicted_actions.min():.4f}, {predicted_actions.max():.4f}]")
                    print(f"Predicted actions mean: {predicted_actions.mean():.4f}, std: {predicted_actions.std():.4f}")
                    print(f"Target actions shape: {target_actions.shape}")
                    print(f"Target actions range: [{target_actions.min():.4f}, {target_actions.max():.4f}]")
                    print(f"Target actions mean: {target_actions.mean():.4f}, std: {target_actions.std():.4f}")
                    print(f"Sample predicted: {predicted_actions[0, :2]}")
                    print(f"Sample target: {target_actions[0, :2]}")
                    print(f"==================\n")
                
                driving_metrics = compute_driving_metrics(predicted_actions, target_actions)
                for key, value in driving_metrics.items():
                    val_metrics[key].append(value)
            except Exception as e:
                print(f"Warning: Error in action prediction during validation: {e}")
                continue
    
    averaged_metrics = {}
    for key, values in val_metrics.items():
        if values:  
            averaged_metrics[f'val_{key}'] = np.mean(values)
        
    return averaged_metrics

def train_carla_policy():
    print("Initializing CARLA driving policy training...")
    config = create_carla_config()
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_path = config.get('training', {}).get('dataset_path', "/home/wang/projects/diffusion_policy_z/data/tmp_data")
    pred_horizon = config.get('pred_horizon', 6)
    obs_horizon = config.get('obs_horizon', 2)
    action_horizon = config.get('action_horizon', 4)
    max_files = 5
    print(f"Loading CARLA dataset from {dataset_path}")
    
    # 创建训练和验证数据集
    train_dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon, 
        action_horizon=action_horizon,
        max_files=max_files,
        train_split=0.8,
        mode='train',
        device=device,
        use_gpu_processing=True
    )
    
    val_dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        max_files=max_files,
        train_split=0.8,
        mode='val',
        device=device,
        use_gpu_processing=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 计算action统计信息（如果启用normalization）
    action_stats = None
    if config.get('enable_action_normalization', False):
        action_stats = compute_action_stats(train_dataset)
        
        # 将统计信息添加到wandb配置中
        wandb.config.update({
            "action_stats": {
                "min": action_stats['min'].tolist(),
                "max": action_stats['max'].tolist(),
                "mean": action_stats['mean'].tolist(),
                "std": action_stats['std'].tolist()
            }
        })
    else:
        print("Action normalization disabled in config")
    
    batch_size = config.get('dataloader', {}).get('batch_size', 32)
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
    policy = DiffusionDiTCarlaPolicy(config, action_stats=action_stats).to(device)
    print(f"Policy action steps (n_action_steps): {policy.n_action_steps}")
    
    # 检查数据范围
    print("\n=== Checking data statistics ===")
    sample_batch = next(iter(train_loader))
    print(f"Sample agent_pos shape: {sample_batch['agent_pos'].shape}")
    print(f"Sample agent_pos range: [{sample_batch['agent_pos'].min():.4f}, {sample_batch['agent_pos'].max():.4f}]")
    print(f"Sample agent_pos mean: {sample_batch['agent_pos'].mean():.4f}, std: {sample_batch['agent_pos'].std():.4f}")
    print(f"First sample agent_pos:\n{sample_batch['agent_pos'][0]}")
    print("===================================\n")
    
    lr = config.get('optimizer', {}).get('lr', 5e-5)
    weight_decay = config.get('optimizer', {}).get('weight_decay', 1e-5)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    print("Starting training loop...")
    

    num_epochs = config.get('training', {}).get('num_epochs', 50)
    best_val_loss = float('inf') 
    for epoch in range(num_epochs):
        policy.train()
        train_losses = []

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Training...")
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            optimizer.zero_grad()
            loss = policy.compute_loss(batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
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
        

        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")
        
        wandb.log({
            "train/loss_epoch": avg_train_loss,
            "train/epoch": epoch,
            "train/samples_processed": (epoch + 1) * len(train_dataset)
        })

        if (epoch + 1) % 1 == 0:
            print("Validating...")
            val_metrics = validate_model(policy, val_loader, device)
        
            log_dict = {
                "epoch": epoch,
                "train/loss": avg_train_loss,
            }
        
            if 'val_loss' in val_metrics:
                log_dict["val/loss"] = val_metrics['val_loss']
        
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
    wandb.log({
        "training/completed": True,
        "training/total_epochs": num_epochs,
        "training/best_val_loss": best_val_loss,
        "training/final_train_loss": avg_train_loss
    })
    wandb.finish()

if __name__ == "__main__":
    train_carla_policy()