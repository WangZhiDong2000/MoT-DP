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
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from dataset.unified_carla_dataset import CARLAImageDataset
from policy.diffusion_dit_carla_policy import DiffusionDiTCarlaPolicy

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(project_root, "config", "pdm_server.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_action_stats(config):
    if 'action_stats' in config:
        action_stats = {
            'min': torch.tensor(config['action_stats']['min'], dtype=torch.float32),
            'max': torch.tensor(config['action_stats']['max'], dtype=torch.float32),
            'mean': torch.tensor(config['action_stats']['mean'], dtype=torch.float32),
            'std': torch.tensor(config['action_stats']['std'], dtype=torch.float32),
        }
        print("✓ Loaded action_stats from config file")
        return action_stats
    else:
        print("⚠ No action_stats found in config, action normalization will be disabled")
        return None


def safe_wandb_log(data, use_wandb=True):
    if not use_wandb:
        return
    try:
        cleaned_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    value = value.item()
                else:
                    continue
            
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    value = value.item()
                else:
                    continue
            
            if isinstance(value, (np.integer, np.floating)):
                value = value.item() 
            
            if isinstance(value, (int, float, np.integer, np.floating)):
                if np.isnan(value):
                    continue
                elif np.isinf(value):
                    value = 1e10 if value > 0 else -1e10
            
            if isinstance(value, np.generic):
                value = value.item()
            
            cleaned_data[key] = value
        
        if not cleaned_data:
            return
        sys.stdout.flush()
        sys.stderr.flush()
    
        wandb.log(cleaned_data)
        
        sys.stdout.flush()
        sys.stderr.flush()
        
    except Exception as e:
        import traceback
        traceback.print_exc()


def safe_wandb_finish(use_wandb=True):
    if not use_wandb:
        return
    wandb.finish(quiet=True, exit_code=0)


def check_box_collision_2d(boxes1, boxes2):
    """
    使用分离轴定理(SAT)检测2D边界框碰撞
    
    Args:
        boxes1: (N, 5) 或 (1, 5) [x, y, w, l, yaw] 边界框
        boxes2: (N, 5) 或 (M, 5) [x, y, w, l, yaw] 边界框
    
    Returns:
        collision: bool数组 (N,)，True表示发生碰撞
    """
    N1, N2 = boxes1.shape[0], boxes2.shape[0]
    
    if N1 == 0 or N2 == 0:
        return np.zeros(max(N1, N2), dtype=bool)
    
    def get_box_corners(x, y, w, l, yaw):
        corners_local = np.array([
            [-l/2, -w/2],  
            [l/2, -w/2],   
            [l/2, w/2],    
            [-l/2, w/2]    
        ])
        
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rot_mat = np.array([[cos_yaw, -sin_yaw],
                           [sin_yaw, cos_yaw]])
        corners_global = corners_local @ rot_mat.T + np.array([x, y])
        return corners_global
    
    def get_axes(corners):
        edges = np.array([
            corners[1] - corners[0],  
            corners[2] - corners[1],  
        ])
        axes = np.array([
            [-edges[0, 1], edges[0, 0]],  
            [-edges[1, 1], edges[1, 0]],  
        ])
        norm = np.linalg.norm(axes, axis=1, keepdims=True)
        norm = np.where(norm < 1e-10, 1.0, norm)
        axes = axes / norm
        return axes
    
    def project_onto_axis(corners, axis):
        projections = corners @ axis
        return np.min(projections), np.max(projections)
    
    def check_overlap_on_axis(corners1, corners2, axis):
        min1, max1 = project_onto_axis(corners1, axis)
        min2, max2 = project_onto_axis(corners2, axis)
        return max1 >= min2 and max2 >= min1
    
    def check_collision_single_pair(box1, box2):
        x1, y1, w1, l1, yaw1 = box1
        x2, y2, w2, l2, yaw2 = box2
        
        corners1 = get_box_corners(x1, y1, w1, l1, yaw1)
        corners2 = get_box_corners(x2, y2, w2, l2, yaw2)

        axes1 = get_axes(corners1)
        axes2 = get_axes(corners2)
        all_axes = np.vstack([axes1, axes2])

        for axis in all_axes:
            if not check_overlap_on_axis(corners1, corners2, axis):
                return False
        
        return True
    
    if N1 == 1 and N2 > 1:
        collision_mask = np.zeros(N2, dtype=bool)
        for i in range(N2):
            collision_mask[i] = check_collision_single_pair(boxes1[0], boxes2[i])
        return collision_mask
    
    elif N1 == N2:
        collision_mask = np.zeros(N1, dtype=bool)
        for i in range(N1):
            collision_mask[i] = check_collision_single_pair(boxes1[i], boxes2[i])
        return collision_mask
    
    else:
        raise ValueError(f"Unsupported box shapes: boxes1 {boxes1.shape}, boxes2 {boxes2.shape}. "
                        f"Expected either same shape (N, 5) or boxes1=(1, 5), boxes2=(M, 5)")


def compute_driving_metrics(predicted_trajectories, target_trajectories, fut_obstacles=None):
    """
    计算驾驶性能指标
    
    Args:
        predicted_trajectories: (B, T, 2) 预测轨迹
        target_trajectories: (B, T, 2) 真实轨迹
        fut_obstacles: List of B lists, each containing T dicts with obstacle info
                      Each dict has 'gt_boxes' (N, 7), 'gt_names' (N,), 'gt_velocity' (N, 2)
                      Optional - if None, collision metrics will not be computed
    
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
    
    safe_metrics = {}
    for key, value in metrics.items():
        if np.isnan(value):
            print(f"Warning: Metric '{key}' is NaN, replacing with 0.0")
            safe_metrics[key] = 0.0
        elif np.isinf(value):
            print(f"Warning: Metric '{key}' is Inf, replacing with large value")
            safe_metrics[key] = 1e10 if value > 0 else -1e10
        else:
            safe_metrics[key] = value
    
    return safe_metrics

def validate_model(policy, val_loader, device):

    policy.eval()
    val_metrics = defaultdict(list)
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for batch_idx, batch in enumerate(pbar):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            loss = policy.compute_loss(batch)
            val_metrics['loss'].append(loss.item())
            
            obs_dict = {
                'lidar_token': batch['lidar_token'][:, :policy.n_obs_steps],
                'lidar_token_global': batch['lidar_token_global'][:, :policy.n_obs_steps],
                'ego_status': batch['ego_status'][:, :policy.n_obs_steps],  # (B, obs_horizon, 13)
            }
            target_actions = batch['agent_pos']  
            
            try:
                result = policy.predict_action(obs_dict)
                predicted_actions = torch.from_numpy(result['action']).to(device)
                
                if target_actions.dim() == 3:  # (B, T, 2)
                    target_actions = target_actions[:, :predicted_actions.shape[1]]
                elif target_actions.dim() == 2:  # (B, 2) 
                    target_actions = target_actions.unsqueeze(1)  # (B, 1, 2)
                
                fut_obstacles = batch.get('fut_obstacles', None)

                driving_metrics = compute_driving_metrics(
                    predicted_actions, 
                    target_actions, 
                    fut_obstacles=fut_obstacles 
                )
                for key, value in driving_metrics.items():
                    val_metrics[key].append(value)
                    
                pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
            except Exception as e:
                print(f"Warning: Error in action prediction during validation: {e}")
                continue
    
    pbar.close() 
    
    averaged_metrics = {}
    for key, values in val_metrics.items():
        if values:  
            mean_value = np.mean(values)
            if np.isnan(mean_value):
                print(f"Warning: computed NaN for metric 'val_{key}'")
                averaged_metrics[f'val_{key}'] = 0.0  
            elif np.isinf(mean_value):
                print(f"Warning: computed Inf for metric 'val_{key}'")
                averaged_metrics[f'val_{key}'] = 1e10 if mean_value > 0 else -1e10 
            else:
                averaged_metrics[f'val_{key}'] = mean_value
        
    return averaged_metrics

def train_pdm_policy(config_path):
    """
    
    Args:
        config_path: 配置文件路径
    """
    print("Initializing pdm driving policy training...")
    config = load_config(config_path=config_path)

    wandb_mode = os.environ.get('WANDB_MODE', 'online') 
    use_wandb = config.get('logging', {}).get('use_wandb', True)
    
    if use_wandb:
        try:
            # 支持从config动态读取wandb账号信息并自动登录（谨慎：不要把真实api key提交到仓库）
            logging_cfg = config.get('logging', {})
            wandb_entity = logging_cfg.get('wandb_entity')
            wandb_api_key = logging_cfg.get('wandb_api_key')

            if wandb_api_key:
                # 优先通过环境变量注入，然后尝试login，这样在CI中也能工作
                os.environ['WANDB_API_KEY'] = str(wandb_api_key)
                try:
                    wandb.login(key=str(wandb_api_key))
                    print("✓ WandB login succeeded using provided api key")
                except Exception as e:
                    print(f"⚠ WandB login failed: {e}")

            init_kwargs = dict(
                project=logging_cfg.get('wandb_project', "carla-diffusion-policy"),
                name=logging_cfg.get('run_name', "carla_dit_full_validation"),
                mode=wandb_mode,
                resume='allow',
                config={
                    "learning_rate": config.get('optimizer', {}).get('lr', 5e-5),
                    "epochs": config.get('training', {}).get('num_epochs', 50),
                    "batch_size": config.get('dataloader', {}).get('batch_size', 16),
                    "obs_horizon": config.get('obs_horizon', 2),
                    "action_horizon": config.get('action_horizon', 4),
                    "pred_horizon": config.get('pred_horizon', 8),
                    "dataset_path": config.get('training', {}).get('dataset_path', ""),
                    "max_files": None,
                    "train_split": 0.8,
                    "weight_decay": config.get('optimizer', {}).get('weight_decay', 1e-5),
                    "num_workers": config.get('dataloader', {}).get('num_workers', 4)
                }
            )

            # 如果config里指定了entity（账号/组织），把它传给wandb.init
            if wandb_entity:
                init_kwargs['entity'] = wandb_entity

            wandb.init(**init_kwargs)
            print(f"✓ WandB initialized in {wandb_mode} mode")
        except Exception as e:
            print(f"⚠ WandB initialization failed: {e}")
            use_wandb = False
    else:
        use_wandb = False
    
    # 从config读取GPU ID配置
    device_config = config.get('device', {})
    gpu_ids = device_config.get('gpu_ids', [0])
    
    # 设置GPU和CUDA相关环境变量
    if torch.cuda.is_available() and gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f"✓ Using GPU(s): {gpu_ids}")
    else:
        device = torch.device('cpu')
        print("✓ Using CPU")


    # dataset
    dataset_path_root = config.get('training', {}).get('dataset_path')
    train_dataset_path = os.path.join(dataset_path_root, 'train')
    val_dataset_path = os.path.join(dataset_path_root, 'val')
    image_data_root = config.get('training', {}).get('image_data_root')
    train_dataset = CARLAImageDataset(dataset_path=train_dataset_path, image_data_root=image_data_root)
    val_dataset = CARLAImageDataset(dataset_path=val_dataset_path, image_data_root=image_data_root)

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    '''
    TODO:  how to load action stats from config file
    '''

    action_stats = {
    'min': torch.tensor([-11.77335262298584, -59.26432800292969]),
    'max': torch.tensor([98.34003448486328, 55.585079193115234]),
    'mean': torch.tensor([9.755727767944336, 0.03559679538011551]),
    'std': torch.tensor([14.527670860290527, 3.224050521850586]),
    } 
    
    batch_size = config.get('dataloader', {}).get('batch_size', 32)
    num_workers = config.get('dataloader', {}).get('num_workers', 4)
    persistent_workers = config.get('dataloader', {}).get('persistent_workers', True)
    prefetch_factor = config.get('dataloader', {}).get('prefetch_factor', 2)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    print("Initializing policy model...")
    policy = DiffusionDiTCarlaPolicy(config, action_stats=action_stats).to(device)  
    print(f"Policy action steps (n_action_steps): {policy.n_action_steps}")
    
    lr = config.get('optimizer', {}).get('lr', 5e-5)
    weight_decay = config.get('optimizer', {}).get('weight_decay', 1e-5)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)

    # 设置 checkpoint 目录
    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', "/home/wang/Project/MoT-DP/checkpoints/carla_dit")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"✓ Checkpoint directory: {checkpoint_dir}")
    
    num_epochs = config.get('training', {}).get('num_epochs', 50)
    best_val_loss = float('inf')
    val_loss = None  # 初始化验证损失
    val_metrics = {}  # 初始化验证指标  
    
    for epoch in range(num_epochs):
        policy.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for batch_idx, batch in enumerate(pbar):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            loss = policy.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{np.mean(train_losses):.4f}'
            })
            
            if batch_idx % 10 == 0:
                step = epoch * len(train_loader) + batch_idx
                safe_wandb_log({
                    "train/loss_step": loss.item(),
                    "train/epoch":  epoch ,
                    "train/step": step,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/batch_idx": batch_idx
                }, use_wandb)
        
        pbar.close() 
        
        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1}/{num_epochs} - Average training loss: {avg_train_loss:.4f}")
        
        torch.save({
                    'model_state_dict': policy.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'train_loss': avg_train_loss,
                    'val_metrics': val_metrics
                    }, os.path.join(checkpoint_dir, "carla_policy.pt"))
        
        safe_wandb_log({
            "train/loss_epoch": avg_train_loss,
            "train/epoch": 0.0,
            "train/samples_processed": (epoch + 1) * len(train_dataset)
        }, use_wandb)

        validation_freq = config.get('training', {}).get('validation_freq', 1)
        if (epoch + 1) % validation_freq == 0:
            print(f"Validating (Epoch {epoch+1}/{num_epochs})...")
            sys.stdout.flush()
            try:
                val_metrics = validate_model(policy, val_loader, device)
                print(f"✓ Validation completed")
                sys.stdout.flush()
            except Exception as e:
                print(f"✗ Error during validation: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                continue

            sys.stdout.flush()
                
            log_dict = {
                    "epoch": epoch,
                    "train/loss": avg_train_loss,
                }
        
            if 'val_loss' in val_metrics:
                log_dict["val/loss"] = val_metrics['val_loss']
        
            for key, value in val_metrics.items():
                if key == 'val_loss':
                    continue  
                elif 'L2' in key:
                    metric_name = key.replace('val_', '')
                    log_dict[f"val/L2/{metric_name}"] = value
                elif 'collision' in key:
                    metric_name = key.replace('val_', '')
                    log_dict[f"val/collision/{metric_name}"] = value
                else:
                    log_dict[f"val/{key.replace('val_', '')}"] = value
        
            print("Logging to wandb...")
            sys.stdout.flush()
            safe_wandb_log(log_dict, use_wandb)
            sys.stdout.flush()

        
            print(f"Validation metrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
        
            val_loss = val_metrics.get('val_loss', float('inf'))
            l2_3s = val_metrics.get('val_L2_3s', float('inf'))
            
            # Save best model based on L2_3s instead of val_loss
            if l2_3s < best_l2_3s:
                best_l2_3s = l2_3s
                torch.save({
                        'model_state_dict': policy.state_dict(),
                        'config': config,
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'train_loss': avg_train_loss,
                        'val_metrics': val_metrics
                        }, os.path.join(checkpoint_dir, "carla_policy_best.pt"))
                print(f"✓ New best model saved with L2_3s: {l2_3s:.4f} (val_loss: {val_loss:.4f})")
               
                safe_wandb_log({
                        "best_model/epoch": epoch,
                        "best_model/L2_3s": l2_3s,
                        "best_model/val_loss": val_loss,
                        "best_model/train_loss": avg_train_loss
                    }, use_wandb)
    
    print("Training completed!")
    print(f"Best L2_3s: {best_l2_3s:.4f}")
    safe_wandb_log({
        "training/completed": 0.0,
        "training/total_epochs": num_epochs,
        "training/best_l2_3s": best_l2_3s,
        "training/final_train_loss": avg_train_loss
    }, use_wandb)
    
    safe_wandb_finish(use_wandb)
    print("✓ Training session finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pdm Driving Policy with Diffusion DiT")
    parser.add_argument('--config_path', type=str, default="/root/z_projects/code/MoT-DP-1/config/pdm_server.yaml", 
                        help='Path to the configuration YAML file')
    args = parser.parse_args()
    train_pdm_policy(config_path=args.config_path)