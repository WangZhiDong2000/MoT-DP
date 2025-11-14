#!/usr/bin/env python3
import os
import sys

# 在导入任何库之前设置环境变量，防止wandb多进程冲突
os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['WANDB__SERVICE_WAIT'] = '300'

import torch
import yaml
import wandb
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import time
import psutil
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入nuScenes数据集类
from dataset.nusc_dataset import NUSCDataset, collate_fn

# 导入策略模型
from policy.diffusion_dit_nusc_policy import DiffusionDiTCarlaPolicy

def load_config(config_path=None):
    """加载配置文件"""
    if config_path is None:
        config_path = os.path.join(project_root, "config", "nuscenes.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_action_stats(config):
    """从配置文件加载action_stats"""
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
    """安全地记录wandb日志，处理所有异常"""
    if use_wandb:
        try:
            # 在记录前刷新输出
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            
            wandb.log(data)
            
            # 记录后再次刷新
            sys.stdout.flush()
            sys.stderr.flush()
        except:
            # 完全静默处理所有异常
            pass


def safe_wandb_finish(use_wandb=True):
    """安全地结束wandb会话，避免段错误"""
    if not use_wandb:
        return
    
    try:
        # 方法1: 使用quiet模式
        wandb.finish(quiet=True, exit_code=0)
    except:
        try:
            # 方法2: 强制终止wandb进程
            import signal
            import subprocess
            subprocess.run(['pkill', '-9', '-f', 'wandb-service'], 
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        except:
            # 最后的尝试: 什么都不做，让Python正常退出
            pass


def check_box_collision_2d(boxes1, boxes2):
    """
    使用分离轴定理(SAT)检测2D边界框碰撞
    
    分离轴定理：两个凸多边形不相交当且仅当存在一条轴，使得两个多边形在该轴上的投影不重叠。
    对于2D的OBB(Oriented Bounding Box)，只需检查4条轴：box1和box2各自的两条边的法向量。
    
    支持两种模式:
    1. boxes1和boxes2形状相同(N, 5): 逐对检测碰撞，返回(N,)
    2. boxes1为(1, 5), boxes2为(M, 5): 检测单个box1与所有box2的碰撞，返回(M,)
    
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
        """计算box的四个角点（全局坐标系）"""
        corners_local = np.array([
            [-l/2, -w/2],  # 左后
            [l/2, -w/2],   # 右后
            [l/2, w/2],    # 右前
            [-l/2, w/2]    # 左前
        ])
        
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rot_mat = np.array([[cos_yaw, -sin_yaw],
                           [sin_yaw, cos_yaw]])
        corners_global = corners_local @ rot_mat.T + np.array([x, y])
        return corners_global
    
    def get_axes(corners):
        """获取box边的法向量作为分离轴"""
        # 计算边向量
        edges = np.array([
            corners[1] - corners[0],  # 底边
            corners[2] - corners[1],  # 右边
        ])
        # 法向量（垂直于边）
        axes = np.array([
            [-edges[0, 1], edges[0, 0]],  # 底边的法向量
            [-edges[1, 1], edges[1, 0]],  # 右边的法向量
        ])
        # 归一化
        norm = np.linalg.norm(axes, axis=1, keepdims=True)
        # 避免除以零
        norm = np.where(norm < 1e-10, 1.0, norm)
        axes = axes / norm
        return axes
    
    def project_onto_axis(corners, axis):
        """将box的角点投影到轴上"""
        projections = corners @ axis
        return np.min(projections), np.max(projections)
    
    def check_overlap_on_axis(corners1, corners2, axis):
        """检查两个box在某个轴上的投影是否重叠"""
        min1, max1 = project_onto_axis(corners1, axis)
        min2, max2 = project_onto_axis(corners2, axis)
        # 投影重叠当且仅当 max1 >= min2 and max2 >= min1
        return max1 >= min2 and max2 >= min1
    
    def check_collision_single_pair(box1, box2):
        """检查单对box是否碰撞"""
        x1, y1, w1, l1, yaw1 = box1
        x2, y2, w2, l2, yaw2 = box2
        
        # 获取两个box的角点
        corners1 = get_box_corners(x1, y1, w1, l1, yaw1)
        corners2 = get_box_corners(x2, y2, w2, l2, yaw2)
        
        # 获取所有需要检查的分离轴
        axes1 = get_axes(corners1)
        axes2 = get_axes(corners2)
        all_axes = np.vstack([axes1, axes2])
        
        # 检查所有轴
        for axis in all_axes:
            if not check_overlap_on_axis(corners1, corners2, axis):
                # 找到一条分离轴，两个box不相交
                return False
        
        # 所有轴上都有重叠，两个box相交
        return True
    
    # 处理一对多的情况
    if N1 == 1 and N2 > 1:
        collision_mask = np.zeros(N2, dtype=bool)
        for i in range(N2):
            collision_mask[i] = check_collision_single_pair(boxes1[0], boxes2[i])
        return collision_mask
    
    # 处理逐对比较的情况
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
    计算驾驶性能指标，遵循DiffusionDrive的评估方式
    
    Args:
        predicted_trajectories: (B, T, 2) 预测轨迹
        target_trajectories: (B, T, 2) 真实轨迹
        fut_obstacles: List of B lists, each containing T dicts with obstacle info
                      Each dict has 'gt_boxes' (N, 7), 'gt_names' (N,), 'gt_velocity' (N, 2)
                      Optional - if None, collision metrics will not be computed
    
    Returns:
        metrics: 包含L2误差和碰撞率的字典
    
    指标说明:
        - L2误差: 预测轨迹点与真实轨迹点之间的欧几里得距离
        - 碰撞率: 预测自车边界框与真实障碍物边界框的重叠比例（DiffusionDrive方法）
        
    时间点 (2Hz频率):
        - 1s: 索引1 (第2个点)
        - 2s: 索引3 (第4个点)
        - 3s: 索引5 (第6个点)
    """
    predicted_trajectories = predicted_trajectories.detach().cpu().numpy()
    target_trajectories = target_trajectories.detach().cpu().numpy()
    
    B, T, _ = predicted_trajectories.shape
    
    # 计算L2距离误差 (B, T)
    l2_errors = np.linalg.norm(
        predicted_trajectories - target_trajectories, axis=-1
    )
    
    metrics = {}
    
    # === L2 误差指标 ===
    # nuScenes使用2Hz预测频率，action_horizon=6对应3秒
    # 索引对应: 0->0s, 1->1s, 2->未使用, 3->2s, 4->未使用, 5->3s
    
    if T >= 2:  # 至少有1s的数据 (索引1)
        metrics['L2_1s'] = np.mean(l2_errors[:, 1])
    
    if T >= 4:  # 至少有2s的数据 (索引3)
        metrics['L2_2s'] = np.mean(l2_errors[:, 3])
    
    if T >= 6:  # 有3s的数据 (索引5)
        metrics['L2_3s'] = np.mean(l2_errors[:, 5])
    
    # 平均L2误差（所有时间步）
    metrics['L2_avg'] = np.mean(l2_errors)
    
    # === 碰撞率指标（基于边界框重叠检测）===
    # 只有提供了真实障碍物数据才计算碰撞率
    if fut_obstacles is not None:
        # nuScenes ego vehicle尺寸 (根据nuScenes标准)
        ego_length = 4.084  # 车长 (meters)
        ego_width = 1.730   # 车宽 (meters)
        
        # 计算每个时间步的碰撞
        collisions = np.zeros((B, T), dtype=np.float32)
        
        for b in range(B):
            for t in range(T):
                # 构建预测自车边界框: [x, y, w, l, yaw]
                pred_x = predicted_trajectories[b, t, 0]
                pred_y = predicted_trajectories[b, t, 1]
                
                # 计算yaw（基于运动方向）
                if t > 0:
                    # 使用当前点和前一点的位置差计算运动方向
                    dx = predicted_trajectories[b, t, 0] - predicted_trajectories[b, t-1, 0]
                    dy = predicted_trajectories[b, t, 1] - predicted_trajectories[b, t-1, 1]
                    # 避免除以零：如果运动很小，使用前一时刻的yaw
                    if np.sqrt(dx**2 + dy**2) > 0.01:  # 运动超过1cm
                        pred_yaw = np.arctan2(dy, dx)
                    elif t > 1:
                        # 运动太小，使用前一时刻的yaw
                        dx_prev = predicted_trajectories[b, t-1, 0] - predicted_trajectories[b, t-2, 0]
                        dy_prev = predicted_trajectories[b, t-1, 1] - predicted_trajectories[b, t-2, 1]
                        pred_yaw = np.arctan2(dy_prev, dx_prev)
                    else:
                        pred_yaw = 0.0
                else:
                    # t=0: 使用第一段运动方向（如果有下一个点）
                    if T > 1:
                        dx = predicted_trajectories[b, 1, 0] - predicted_trajectories[b, 0, 0]
                        dy = predicted_trajectories[b, 1, 1] - predicted_trajectories[b, 0, 1]
                        if np.sqrt(dx**2 + dy**2) > 0.01:
                            pred_yaw = np.arctan2(dy, dx)
                        else:
                            pred_yaw = 0.0  # 静止或运动很小，假设朝向前方
                    else:
                        pred_yaw = 0.0  # 只有一个点，假设朝向前方
                
                # 预测自车边界框: [x, y, w, l, yaw]
                pred_box = np.array([[pred_x, pred_y, ego_width, ego_length, pred_yaw]])
                
                # 获取该时刻的障碍物边界框
                # fut_obstacles[b] 是一个包含 T 个 dict 的 list
                # 每个 dict 有 'gt_boxes' (N, 7): [x, y, z, l, w, h, yaw]
                # 注意：l=length(x方向), w=width(y方向)
                obstacles_at_t = fut_obstacles[b][t]
                
                if isinstance(obstacles_at_t['gt_boxes'], torch.Tensor):
                    obs_boxes = obstacles_at_t['gt_boxes'].cpu().numpy()  # (N, 7)
                else:
                    obs_boxes = obstacles_at_t['gt_boxes']
                
                # 如果没有障碍物，不发生碰撞
                if len(obs_boxes) == 0:
                    collisions[b, t] = 0.0
                    continue
                
                # 转换障碍物边界框格式: [x, y, z, l, w, h, yaw] -> [x, y, w, l, yaw]
                # 注意：check_box_collision_2d 需要的格式是 [x, y, w, l, yaw]
                obs_boxes_2d = np.zeros((len(obs_boxes), 5))
                obs_boxes_2d[:, 0] = obs_boxes[:, 0]  # x
                obs_boxes_2d[:, 1] = obs_boxes[:, 1]  # y
                obs_boxes_2d[:, 2] = obs_boxes[:, 4]  # w (width, y方向)
                obs_boxes_2d[:, 3] = obs_boxes[:, 3]  # l (length, x方向)
                obs_boxes_2d[:, 4] = obs_boxes[:, 6]  # yaw
                
                # 检测预测自车边界框是否与任何障碍物重叠
                collision_mask = check_box_collision_2d(
                    pred_box,  # (1, 5)
                    obs_boxes_2d  # (N, 5)
                )  # (1, N)
                
                # 如果与任何障碍物发生碰撞，标记为1
                collisions[b, t] = 1.0 if np.any(collision_mask) else 0.0
        
        # 计算各时刻的碰撞率
        if T >= 2:
            metrics['collision_1s'] = np.mean(collisions[:, 1])
        
        if T >= 4:
            metrics['collision_2s'] = np.mean(collisions[:, 3])
        
        if T >= 6:
            metrics['collision_3s'] = np.mean(collisions[:, 5])
        
        # 平均碰撞率（所有时刻的平均碰撞率）
        # 计算所有时刻、所有样本的平均碰撞率
        metrics['collision_avg'] = np.mean(collisions)
    
    return metrics

def validate_model(policy, val_loader, device):
    """
    验证模型性能
    
    Args:
        policy: 策略模型
        val_loader: 验证数据加载器
        device: 设备
    """
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
            
            # 准备观测字典（nuScenes格式）
            obs_dict = {
                'lidar_token': batch['lidar_token'][:, :policy.n_obs_steps],
                'lidar_token_global': batch['lidar_token_global'][:, :policy.n_obs_steps],
                'ego_status': batch['ego_status'][:, :policy.n_obs_steps],  # (B, obs_horizon, 13)
            }
            target_actions = batch['agent_pos']  # nuScenes的fut_waypoints已映射到agent_pos
            
            try:
                result = policy.predict_action(obs_dict)
                predicted_actions = torch.from_numpy(result['action']).to(device)
                
                # 对齐target_actions的形状
                if target_actions.dim() == 3:  # (B, T, 2)
                    target_actions = target_actions[:, :predicted_actions.shape[1]]
                elif target_actions.dim() == 2:  # (B, 2) - 单个时间步
                    target_actions = target_actions.unsqueeze(1)  # (B, 1, 2)
                
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
                
                # 获取障碍物信息用于碰撞检测
                # 注意：在验证阶段暂时跳过碰撞检测以避免潜在的段错误
                fut_obstacles = batch.get('fut_obstacles', None)
                
                # 计算驾驶性能指标（仅L2误差，不计算碰撞率）
                driving_metrics = compute_driving_metrics(
                    predicted_actions, 
                    target_actions, 
                    fut_obstacles=fut_obstacles  # 恢复碰撞检测
                )
                for key, value in driving_metrics.items():
                    val_metrics[key].append(value)
                    
                # 更新验证进度条信息
                pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
            except Exception as e:
                print(f"Warning: Error in action prediction during validation: {e}")
                continue
    
    pbar.close()  # 关闭验证进度条
    
    averaged_metrics = {}
    for key, values in val_metrics.items():
        if values:  
            averaged_metrics[f'val_{key}'] = np.mean(values)
        
    return averaged_metrics

def train_nusc_policy(config_path):
    """
    训练nuScenes驾驶策略模型
    
    Args:
        config_path: 配置文件路径
    """
    print("Initializing nuScenes driving policy training...")
    config = load_config(config_path=config_path)
    
    # 支持wandb离线模式和断网恢复
    wandb_mode = os.environ.get('WANDB_MODE', 'online')  # 可通过环境变量设置: export WANDB_MODE=offline
    use_wandb = config.get('logging', {}).get('use_wandb', True)
    
    if use_wandb:
        try:
            # wandb.login(key="bddb5a05a0820c1157702750c9f0ce60bcac2bba", anonymous="must")
            wandb.init(
                project=config.get('logging', {}).get('wandb_project', "carla-diffusion-policy"),
                name=config.get('logging', {}).get('run_name', "carla_dit_full_validation"),
                mode=wandb_mode,  # online, offline, or disabled
                resume='allow',  # 允许断点续传
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
            print("  Continuing training without WandB logging...")
            use_wandb = False
    else:
        print("⚠ WandB disabled in config")
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
        load_bev_images=False  # 训练时使用预处理特征
    )
    
    val_dataset = NUSCDataset(
        processed_data_path=val_pkl,
        dataset_root=dataset_root,
        mode='val',
        load_bev_images=False  # 验证时也使用预处理特征
    )

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 从配置文件加载action_stats
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
        collate_fn=collate_fn  # Use custom collate function
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_fn  # Use custom collate function
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
            
            # 更新进度条信息，显示当前loss
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{np.mean(train_losses):.4f}'
            })
            
            if batch_idx % 10 == 0:
                step = epoch * len(train_loader) + batch_idx
                safe_wandb_log({
                    "train/loss_step": loss.item(),
                    "train/epoch": epoch,
                    "train/step": step,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/batch_idx": batch_idx
                }, use_wandb)
        
        pbar.close()  # 关闭当前epoch的进度条
        
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
            "train/epoch": epoch,
            "train/samples_processed": (epoch + 1) * len(train_dataset)
        }, use_wandb)

        validation_freq = config.get('training', {}).get('validation_freq', 1)
        if (epoch + 1) % validation_freq == 0:
            print(f"Validating (Epoch {epoch+1}/{num_epochs})...")
            import sys
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
                print("Preparing validation metrics for logging...")
                sys.stdout.flush()
                
                log_dict = {
                    "epoch": epoch,
                    "train/loss": avg_train_loss,
                }
        
                if 'val_loss' in val_metrics:
                    log_dict["val/loss"] = val_metrics['val_loss']
        
                # 组织指标到不同的wandb类别
                for key, value in val_metrics.items():
                    if key == 'val_loss':
                        continue  # 已经处理
                    elif 'L2' in key:
                        # L2误差指标: val/L2/1s, val/L2/2s, val/L2/3s, val/L2/avg
                        metric_name = key.replace('val_', '')
                        log_dict[f"val/L2/{metric_name}"] = value
                    elif 'collision' in key:
                        # 碰撞率指标: val/collision/1s, val/collision/2s, val/collision/3s, val/collision/avg
                        metric_name = key.replace('val_', '')
                        log_dict[f"val/collision/{metric_name}"] = value
                    else:
                        # 其他指标
                        log_dict[f"val/{key.replace('val_', '')}"] = value
        
                print("Logging to wandb...")
                sys.stdout.flush()
                safe_wandb_log(log_dict, use_wandb)
                print(f"✓ Metrics logged")
                sys.stdout.flush()
            except Exception as e:
                print(f"✗ Error during metrics logging: {e}")
                import traceback
                traceback.print_exc()
        
            print(f"Validation metrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
        
            val_loss = val_metrics.get('val_loss', float('inf'))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
                try:
                    torch.save({
                        'model_state_dict': policy.state_dict(),
                        'config': config,
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'train_loss': avg_train_loss,
                        'val_metrics': val_metrics
                        }, os.path.join(checkpoint_dir, "carla_policy_best.pt"))
                    print(f"✓ New best model saved with val_loss: {val_loss:.4f}")
                except Exception as e:
                    print(f"✗ Error saving model: {e}")
                    import traceback
                    traceback.print_exc()
            
                try:
                    safe_wandb_log({
                        "best_model/epoch": epoch,
                        "best_model/val_loss": val_loss,
                        "best_model/train_loss": avg_train_loss
                    }, use_wandb)
                except Exception as e:
                    print(f"✗ Error logging best model: {e}")
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    safe_wandb_log({
        "training/completed": True,
        "training/total_epochs": num_epochs,
        "training/best_val_loss": best_val_loss,
        "training/final_train_loss": avg_train_loss
    }, use_wandb)
    
    # 使用安全的finish函数
    safe_wandb_finish(use_wandb)
    print("✓ Training session finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train nuScenes Driving Policy with Diffusion DiT")
    parser.add_argument('--config_path', type=str, default="/home/wang/Project/MoT-DP/config/nuscenes.yaml", 
                        help='Path to the configuration YAML file')
    args = parser.parse_args()
    train_nusc_policy(config_path=args.config_path)