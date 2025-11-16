#!/usr/bin/env python3
import os
import sys
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dataset.nusc_dataset import NUSCDataset, collate_fn
from policy.diffusion_dit_nusc_policy import DiffusionDiTCarlaPolicy
import torch


def load_config(config_path):
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
        return None


def load_best_model(checkpoint_path, config, device):
    print(f"Loading best model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    action_stats = load_action_stats(config)
    policy = DiffusionDiTCarlaPolicy(config, action_stats=action_stats).to(device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    print(f"✓ Model loaded successfully!")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Validation Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  - Training Loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    
    if 'val_metrics' in checkpoint:
        print(f"  - Validation Metrics:")
        for key, value in checkpoint['val_metrics'].items():
            if isinstance(value, (int, float)):
                print(f"    {key}: {value:.4f}")
    
    return policy


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
    计算驾驶性能指标 (遵循train_nusc_bev.py中的方法)
    
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



def test_model(policy, test_loader, config, device='cuda'):
    print(f"\n{'='*60}")
    print(f"Testing model on full test dataset")
    print(f"Total samples: {len(test_loader.dataset)}")
    print(f"{'='*60}\n")
    
    policy.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    driving_metrics_accumulator = defaultdict(list)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            obs_horizon = config.get('obs_horizon', 2)
            obs_dict = {
                'lidar_token': batch['lidar_token'][:, :obs_horizon],  # (B, obs_horizon, 196, 512)
                'lidar_token_global': batch['lidar_token_global'][:, :obs_horizon],  # (B, obs_horizon, 1, 512)
                'ego_status': batch['ego_status'][:, :obs_horizon],  # (B, obs_horizon, 13)
            }
            
            try:
                result = policy.predict_action(obs_dict)
                predicted_actions = torch.from_numpy(result['action']).to(device)  # (B, T, 2)
                target_actions = batch['agent_pos'][:, :predicted_actions.shape[1]]  # (B, T, 2)
                
                fut_obstacles = batch.get('fut_obstacles', None)
                
                batch_driving_metrics = compute_driving_metrics(
                    predicted_actions, 
                    target_actions,
                    fut_obstacles=fut_obstacles
                )
                for key, value in batch_driving_metrics.items():
                    driving_metrics_accumulator[key].append(value)
                
                for i in range(predicted_actions.shape[0]):
                    pred = predicted_actions[i].cpu().numpy()  # (T, 2)
                    gt = target_actions[i].cpu().numpy()  # (T, 2)
                    all_predictions.append(pred)
                    all_ground_truths.append(gt)
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # 计算所有指标
    print(f"\n{'='*60}")
    print("Computing overall metrics...")
    print(f"{'='*60}\n")
    
    driving_metrics = {}
    for key, values in driving_metrics_accumulator.items():
        if values:
            driving_metrics[key] = np.mean(values)
    
    return driving_metrics, all_predictions, all_ground_truths


def print_and_save_metrics(metrics, num_samples, save_path):
    """打印并保存指标到文件（与train_nusc_bev.py中的指标一致）"""
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60 + "\n")
    
    print(f"Number of samples: {num_samples}\n")
    
    print("Basic Metrics (按时间步的L2误差):")
    if 'L2_1s' in metrics:
        print(f"  L2 Error @ 1s: {metrics['L2_1s']:.6f}")
    if 'L2_2s' in metrics:
        print(f"  L2 Error @ 2s: {metrics['L2_2s']:.6f}")
    if 'L2_3s' in metrics:
        print(f"  L2 Error @ 3s: {metrics['L2_3s']:.6f}")
    print(f"  L2 Error (avg): {metrics.get('L2_avg', 'N/A'):.6f}")
    
    
    
    print("\nDriving Metrics (碰撞检测和L2误差):")
    if 'L2_1s' in metrics:
        print(f"  L2_1s (1 second): {metrics['L2_1s']:.6f}")
    if 'L2_2s' in metrics:
        print(f"  L2_2s (2 seconds): {metrics['L2_2s']:.6f}")
    if 'L2_3s' in metrics:
        print(f"  L2_3s (3 seconds): {metrics['L2_3s']:.6f}")
    
    if 'collision_1s' in metrics:
        print(f"  Collision Rate @ 1s: {metrics['collision_1s']:.6f}")
    if 'collision_2s' in metrics:
        print(f"  Collision Rate @ 2s: {metrics['collision_2s']:.6f}")
    if 'collision_3s' in metrics:
        print(f"  Collision Rate @ 3s: {metrics['collision_3s']:.6f}")
    if 'collision_avg' in metrics:
        print(f"  Collision Rate (avg): {metrics['collision_avg']:.6f}")
    
    # 保存到文件
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Test Results Summary\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Number of samples: {num_samples}\n\n")
        
        f.write("Basic Metrics (按时间步的L2误差):\n")
        if 'L2_1s' in metrics:
            f.write(f"  L2 Error @ 1s: {metrics['L2_1s']:.6f}\n")
        if 'L2_2s' in metrics:
            f.write(f"  L2 Error @ 2s: {metrics['L2_2s']:.6f}\n")
        if 'L2_3s' in metrics:
            f.write(f"  L2 Error @ 3s: {metrics['L2_3s']:.6f}\n")
        f.write(f"  L2 Error (avg): {metrics.get('L2_avg', 'N/A'):.6f}\n")
        
        
        f.write("Driving Metrics (碰撞检测和L2误差):\n")
        if 'L2_1s' in metrics:
            f.write(f"  L2_1s (1 second): {metrics['L2_1s']:.6f}\n")
        if 'L2_2s' in metrics:
            f.write(f"  L2_2s (2 seconds): {metrics['L2_2s']:.6f}\n")
        if 'L2_3s' in metrics:
            f.write(f"  L2_3s (3 seconds): {metrics['L2_3s']:.6f}\n")
        
        if 'collision_1s' in metrics:
            f.write(f"  Collision Rate @ 1s: {metrics['collision_1s']:.6f}\n")
        if 'collision_2s' in metrics:
            f.write(f"  Collision Rate @ 2s: {metrics['collision_2s']:.6f}\n")
        if 'collision_3s' in metrics:
            f.write(f"  Collision Rate @ 3s: {metrics['collision_3s']:.6f}\n")
        if 'collision_avg' in metrics:
            f.write(f"  Collision Rate (avg): {metrics['collision_avg']:.6f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("All Metrics (key-value pairs):\n")
        f.write("="*60 + "\n")
        for key, value in sorted(metrics.items()):
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"\n✓ Metrics saved to: {save_path}")
    
    


def main(args):
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # 加载配置
    config_path = args.config_path
    config = load_config(config_path)
    
    # 加载最优模型
    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', "/home/wang/Project/MoT-DP/checkpoints/carla_dit_best")
    checkpoint_path = os.path.join(checkpoint_dir, "carla_policy_best.pt")
    policy = load_best_model(checkpoint_path, config, device)
    
    # 加载测试数据集
    print("\nLoading test dataset...")
    nuscenes_config = config.get('nuscenes', {})
    processed_data_root = nuscenes_config.get('processed_data_root', '/home/wang/Dataset/v1.0-mini/processed_data')
    dataset_root = nuscenes_config.get('dataset_root', '/home/wang/Dataset/v1.0-mini')
    
    val_pkl = os.path.join(processed_data_root, 'nuscenes_processed_val.pkl')
    
    test_dataset = NUSCDataset(
        processed_data_path=val_pkl,
        dataset_root=dataset_root,
        mode='val',
        load_bev_images=False
    )
    
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples\n")
    
    # 创建数据加载器
    batch_size = config.get('dataloader', {}).get('batch_size', 16)
    num_workers = config.get('dataloader', {}).get('num_workers', 4)
    persistent_workers = config.get('dataloader', {}).get('persistent_workers', True)
    prefetch_factor = config.get('dataloader', {}).get('prefetch_factor', 2)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_fn
    )
    
    print(f"DataLoader created with batch_size={batch_size}, num_workers={num_workers}\n")
    
    # 运行测试
    metrics, predictions, ground_truths = test_model(
        policy=policy,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    # 打印并保存结果
    save_dir = os.path.join(project_root, 'image', 'test_results')
    metrics_path = os.path.join(save_dir, 'test_metrics.txt')
    
    print_and_save_metrics(metrics, len(predictions), metrics_path)
    
    print(f"\n{'='*60}")
    print("Testing completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test DiffusionDiTCarlaPolicy on full test dataset")
    parser.add_argument('--config_path', type=str, default=os.path.join(project_root, "config/nuscenes.yaml"), help='Path to the configuration file')
    args = parser.parse_args()
    main(args)