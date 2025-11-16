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
from dataset.nusc_dataset import NUSCDataset, collate_fn
from policy.diffusion_dit_nusc_policy import DiffusionDiTCarlaPolicy

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(project_root, "config", "nuscenes.yaml")
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
    
    metrics['L2_avg'] = np.mean(l2_errors)
    
    # === 碰撞率指标（基于边界框重叠检测）===
    if fut_obstacles is not None:
        ego_length = 4.084  
        ego_width = 1.730   
        collisions = np.zeros((B, T), dtype=np.float32)
        
        for b in range(B):
            for t in range(T):
                pred_x = predicted_trajectories[b, t, 0]
                pred_y = predicted_trajectories[b, t, 1]
                
                if t > 0:
                    dx = predicted_trajectories[b, t, 0] - predicted_trajectories[b, t-1, 0]
                    dy = predicted_trajectories[b, t, 1] - predicted_trajectories[b, t-1, 1]

                    if np.sqrt(dx**2 + dy**2) > 0.01:  
                        pred_yaw = np.arctan2(dy, dx)
                    elif t > 1:
                        dx_prev = predicted_trajectories[b, t-1, 0] - predicted_trajectories[b, t-2, 0]
                        dy_prev = predicted_trajectories[b, t-1, 1] - predicted_trajectories[b, t-2, 1]
                        pred_yaw = np.arctan2(dy_prev, dx_prev)
                    else:
                        pred_yaw = 0.0
                else:
                    if T > 1:
                        dx = predicted_trajectories[b, 1, 0] - predicted_trajectories[b, 0, 0]
                        dy = predicted_trajectories[b, 1, 1] - predicted_trajectories[b, 0, 1]
                        if np.sqrt(dx**2 + dy**2) > 0.01:
                            pred_yaw = np.arctan2(dy, dx)
                        else:
                            pred_yaw = 0.0  
                    else:
                        pred_yaw = 0.0  
                
                pred_box = np.array([[pred_x, pred_y, ego_width, ego_length, pred_yaw]])
                
                # 获取该时刻的障碍物边界框
                obstacles_at_t = fut_obstacles[b][t]
                
                if isinstance(obstacles_at_t['gt_boxes'], torch.Tensor):
                    obs_boxes = obstacles_at_t['gt_boxes'].cpu().numpy()  # (N, 7)
                else:
                    obs_boxes = obstacles_at_t['gt_boxes']
                
                if len(obs_boxes) == 0:
                    collisions[b, t] = 0.0
                    continue
                
                obs_boxes_2d = np.zeros((len(obs_boxes), 5))
                obs_boxes_2d[:, 0] = obs_boxes[:, 0]  # x
                obs_boxes_2d[:, 1] = obs_boxes[:, 1]  # y
                obs_boxes_2d[:, 2] = obs_boxes[:, 4]  # w (width, y方向)
                obs_boxes_2d[:, 3] = obs_boxes[:, 3]  # l (length, x方向)
                obs_boxes_2d[:, 4] = obs_boxes[:, 6]  # yaw
                
                collision_mask = check_box_collision_2d(
                    pred_box,  # (1, 5)
                    obs_boxes_2d  # (N, 5)
                )  # (1, N)
                
                collisions[b, t] = 1.0 if np.any(collision_mask) else 0.0
        
        if T >= 2:
            metrics['collision_1s'] = np.mean(collisions[:, 1])
        
        if T >= 4:
            metrics['collision_2s'] = np.mean(collisions[:, 3])
        
        if T >= 6:
            metrics['collision_3s'] = np.mean(collisions[:, 5])
        
        metrics['collision_avg'] = np.mean(collisions)
    
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

def train_nusc_policy(config_path):
    """
    训练nuScenes驾驶策略模型
    
    Args:
        config_path: 配置文件路径
    """
    print("Initializing nuScenes driving policy training...")
    config = load_config(config_path=config_path)

    wandb_mode = os.environ.get('WANDB_MODE', 'online') 
    use_wandb = config.get('logging', {}).get('use_wandb', True)
    
    if use_wandb:
        try:
            # wandb.login(key="bddb5a05a0820c1157702750c9f0ce60bcac2bba", anonymous="must")
            wandb.init(
                project=config.get('logging', {}).get('wandb_project', "carla-diffusion-policy"),
                name=config.get('logging', {}).get('run_name', "carla_dit_full_validation"),
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
            print(f"✓ WandB initialized in {wandb_mode} mode")
        except Exception as e:
            print(f"⚠ WandB initialization failed: {e}")
            use_wandb = False
    else:
        use_wandb = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载nuScenes数据集
    print("\n=== Loading nuScenes Dataset ===")
    nuscenes_config = config.get('nuscenes', {})
    processed_data_root = nuscenes_config.get('processed_data_root', '/home/wang/Dataset/v1.0-mini/processed_data')
    dataset_root = nuscenes_config.get('dataset_root', '/home/wang/Dataset/v1.0-mini')
    
    train_pkl = os.path.join(processed_data_root, 'nuscenes_processed_train.pkl')
    val_pkl = os.path.join(processed_data_root, 'nuscenes_processed_val.pkl')
    
    print(f"Train data: {train_pkl}")
    print(f"Val data: {val_pkl}")
    
    train_dataset = NUSCDataset(
        processed_data_path=train_pkl,
        dataset_root=dataset_root,
        mode='train',
        load_bev_images=False 
    )
    
    val_dataset = NUSCDataset(
        processed_data_path=val_pkl,
        dataset_root=dataset_root,
        mode='val',
        load_bev_images=False  
    )
    action_stats = load_action_stats(config)
    
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
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_fn  
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_fn  
    )
    
    print("Initializing policy model...")
    policy = DiffusionDiTCarlaPolicy(config, action_stats=action_stats).to(device)  
    lr = config.get('optimizer', {}).get('lr', 5e-5)
    weight_decay = config.get('optimizer', {}).get('weight_decay', 1e-5)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', "/home/wang/Project/MoT-DP/checkpoints/carla_dit")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    num_epochs = config.get('training', {}).get('num_epochs', 50)
    best_val_loss = float('inf')
    val_loss = None  
    val_metrics = {}  
    
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
            try:
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
            except Exception as e:
                import traceback
                traceback.print_exc()
        
            print(f"Validation metrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
        
            val_loss = val_metrics.get('val_loss', float('inf'))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                        'model_state_dict': policy.state_dict(),
                        'config': config,
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'train_loss': avg_train_loss,
                        'val_metrics': val_metrics
                        }, os.path.join(checkpoint_dir, "carla_policy_best.pt"))
                print(f"✓ New best model saved with val_loss: {val_loss:.4f}")
               
                safe_wandb_log({
                        "best_model/epoch": epoch,
                        "best_model/val_loss": val_loss,
                        "best_model/train_loss": avg_train_loss
                    }, use_wandb)
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    safe_wandb_log({
        "training/completed": 0.0,
        "training/total_epochs": num_epochs,
        "training/best_val_loss": best_val_loss,
        "training/final_train_loss": avg_train_loss
    }, use_wandb)
    
    safe_wandb_finish(use_wandb)
    print("✓ Training session finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train nuScenes Driving Policy with Diffusion DiT")
    parser.add_argument('--config_path', type=str, default="/home/wang/Project/MoT-DP/config/nuscenes.yaml", 
                        help='Path to the configuration YAML file')
    args = parser.parse_args()
    train_nusc_policy(config_path=args.config_path)