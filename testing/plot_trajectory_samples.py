#!/usr/bin/env python3
"""
绘制轨迹样本及预测结果
为每个样本绘制三个子图：
1. 轨迹和障碍物
2. 未来障碍物运动
3. BEV图像

按照nusc_dataset.py的visualize_sample方法
"""
import os
import sys
import torch
import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dataset.nusc_dataset import NUSCDataset, collate_fn
from policy.diffusion_dit_nusc_policy import DiffusionDiTCarlaPolicy


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_action_stats(config):
    """从config加载action统计信息"""
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


def load_best_model(checkpoint_path, config, device):
    """加载最优模型权重"""
    print(f"Loading best model from: {checkpoint_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 加载action统计信息
    action_stats = load_action_stats(config)
    
    # 初始化模型
    policy = DiffusionDiTCarlaPolicy(config, action_stats=action_stats).to(device)
    
    # 加载模型权重
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    print(f"✓ Model loaded successfully!")
    
    return policy


def get_all_samples_with_predictions(policy, test_loader, config, device='cuda'):
    """获取所有测试样本及其预测"""
    print(f"\nGenerating predictions for all samples...")
    
    policy.eval()
    samples_data = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting")):
            # 将数据移到设备上
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # 提取观测数据
            obs_horizon = config.get('obs_horizon', 2)
            obs_dict = {
                'lidar_token': batch['lidar_token'][:, :obs_horizon],
                'lidar_token_global': batch['lidar_token_global'][:, :obs_horizon],
                'ego_status': batch['ego_status'][:, :obs_horizon],
            }
            
            try:
                # 预测动作/轨迹
                result = policy.predict_action(obs_dict)
                predicted_actions = torch.from_numpy(result['action']).to(device)
                target_actions = batch['agent_pos'][:, :predicted_actions.shape[1]]
                
                # 保存样本数据
                for i in range(predicted_actions.shape[0]):
                    sample_data = {
                        'hist_waypoints': torch.zeros((obs_horizon, 2)),
                        'agent_pos': target_actions[i].cpu(),
                        'fut_valid_mask': batch.get('fut_valid_mask', torch.ones(predicted_actions.shape[1])).cpu() if i == 0 else torch.ones(predicted_actions.shape[1]),
                        'ego_status': batch['ego_status'][i, :obs_horizon].cpu() if 'ego_status' in batch else torch.zeros((obs_horizon, 13)),
                        'command': batch.get('command', torch.zeros((obs_horizon, 3)))[i, :obs_horizon].cpu() if 'command' in batch else torch.zeros((obs_horizon, 3)),
                        'fut_obstacles': batch['fut_obstacles'][i] if 'fut_obstacles' in batch else [],
                        'predicted_trajectory': predicted_actions[i].cpu(),
                        'target_trajectory': target_actions[i].cpu(),
                        'scene_token': batch.get('scene_token', [f'scene_{i}'])[i] if 'scene_token' in batch else f'scene_{i}',
                        'sample_token': batch.get('sample_token', [f'sample_{i}'])[i] if 'sample_token' in batch else f'sample_{i}',
                    }
                    
                    # 添加lidar_bev如果存在
                    if 'lidar_bev' in batch and batch['lidar_bev'] is not None:
                        sample_data['lidar_bev'] = batch['lidar_bev'][i].cpu()
                    
                    samples_data.append(sample_data)
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    return samples_data


def plot_sample_with_three_subplots(sample, save_path=None):
    """为单个sample绘制三个子图"""
    matplotlib.use('Agg')  
    fig, axes = plt.subplots(1, 3, figsize=(27, 7))
    
    ego_length = 4.084
    ego_width = 1.730
    
    # ========== Plot 1: Waypoints and Obstacles ==========
    ax1 = axes[0]
    
    # Historical waypoints (blue)
    hist_wp = sample.get('hist_waypoints', torch.zeros((2, 2)))
    hist_wp = hist_wp.cpu().numpy() if isinstance(hist_wp, torch.Tensor) else hist_wp
    if len(hist_wp) > 0:
        ax1.plot(hist_wp[:, 1], hist_wp[:, 0], 'b.-', label='Historical', markersize=10, linewidth=2)
    
    # Future waypoints - ground truth (green)
    fut_wp = sample['agent_pos'].cpu().numpy() if isinstance(sample['agent_pos'], torch.Tensor) else sample['agent_pos']
    ax1.plot(fut_wp[:, 1], fut_wp[:, 0], 'g.-', label='GT Future', markersize=10, linewidth=2)
    
    # Future waypoints - prediction (red)
    pred_wp = sample['predicted_trajectory'].cpu().numpy() if isinstance(sample['predicted_trajectory'], torch.Tensor) else sample['predicted_trajectory']
    ax1.plot(pred_wp[:, 1], pred_wp[:, 0], 'r.-', label='Pred Future', markersize=10, linewidth=2)
    
    # Mark invalid waypoints
    if 'fut_valid_mask' in sample:
        valid_mask = sample['fut_valid_mask'].cpu().numpy() if isinstance(sample['fut_valid_mask'], torch.Tensor) else sample['fut_valid_mask']
        invalid_wp = fut_wp[~valid_mask]
        if len(invalid_wp) > 0:
            ax1.scatter(invalid_wp[:, 1], invalid_wp[:, 0], c='orange', s=100, marker='x', label='Invalid', zorder=10)
    
    # Current position
    ax1.scatter(0, 0, c='black', s=200, marker='*', label='Ego (Current)', zorder=10)
    
    # Draw ego vehicle
    ego_rect = patches.Rectangle((-ego_width/2, -ego_length/2), ego_width, ego_length,
        linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.5, zorder=5)
    ax1.add_patch(ego_rect)
    
    # Visualize obstacles
    if 'fut_obstacles' in sample and len(sample['fut_obstacles']) > 0:
        obstacles_frame_0 = sample['fut_obstacles'][0] if isinstance(sample['fut_obstacles'], list) else sample['fut_obstacles']
        if isinstance(obstacles_frame_0, dict) and 'gt_boxes' in obstacles_frame_0:
            gt_boxes = obstacles_frame_0['gt_boxes'].cpu().numpy() if isinstance(obstacles_frame_0['gt_boxes'], torch.Tensor) else obstacles_frame_0['gt_boxes']
            gt_names = obstacles_frame_0['gt_names']
            
            color_map = {
                'car': 'red', 'truck': 'darkred', 'bus': 'orange', 'pedestrian': 'blue',
                'bicycle': 'cyan', 'motorcycle': 'magenta', 'traffic_cone': 'yellow',
                'barrier': 'gray', 'construction_vehicle': 'brown', 'trailer': 'pink',
            }
            
            max_dist = 60.0
            obstacles_drawn = 0
            
            for i, (box, name) in enumerate(zip(gt_boxes, gt_names)):
                x, y, z, l, w, h, yaw = box
                dist = np.sqrt(x**2 + y**2)
                if dist > max_dist:
                    continue
                
                color = color_map.get(name, 'purple')
                corners_local = np.array([[l/2,  w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2,  w/2]])
                
                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)
                rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
                
                corners_data = (rotation_matrix @ corners_local.T).T + np.array([x, y])
                corners_plot = corners_data[:, [1, 0]]
                
                polygon = patches.Polygon(corners_plot, closed=True, linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.3)
                ax1.add_patch(polygon)
                ax1.scatter(y, x, c=color, s=20, marker='o', alpha=0.7)
                
                obstacles_drawn += 1
                if obstacles_drawn >= 30:
                    break
            
            if obstacles_drawn > 0:
                ax1.scatter([], [], c='gray', s=100, marker='s', alpha=0.3, label=f'Obstacles (t+1, {obstacles_drawn} shown)')
    
    ax1.set_xlabel('Lateral (Y) [meters]', fontsize=12)
    ax1.set_ylabel('Forward (X) [meters]', fontsize=12)
    ax1.set_title('Trajectory and Obstacles (Ego Frame)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.set_xlim(-30, 30)
    ax1.set_ylim(-20, 60)
    
    # ========== Plot 2: Future Obstacle Motion ==========
    ax2 = axes[1]
    
    if len(hist_wp) > 0:
        ax2.plot(hist_wp[:, 1], hist_wp[:, 0], 'b.-', label='Historical', markersize=8, linewidth=1.5, alpha=0.7)
    ax2.plot(fut_wp[:, 1], fut_wp[:, 0], 'g.-', label='GT Future', markersize=8, linewidth=1.5, alpha=0.7)
    ax2.plot(pred_wp[:, 1], pred_wp[:, 0], 'r.-', label='Pred Future', markersize=8, linewidth=1.5, alpha=0.7)
    ax2.scatter(0, 0, c='black', s=200, marker='*', label='Ego (Current)', zorder=10)
    
    ego_rect = patches.Rectangle((-ego_width/2, -ego_length/2), ego_width, ego_length,
        linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.5, zorder=5)
    ax2.add_patch(ego_rect)
    
    if 'fut_obstacles' in sample and len(sample['fut_obstacles']) > 0:
        fut_obstacles = sample['fut_obstacles'] if isinstance(sample['fut_obstacles'], list) else []
        if len(fut_obstacles) > 0 and isinstance(fut_obstacles[0], dict):
            color_map = {
                'car': 'red', 'truck': 'darkred', 'bus': 'orange', 'pedestrian': 'blue',
                'bicycle': 'cyan', 'motorcycle': 'magenta', 'traffic_cone': 'yellow',
                'barrier': 'gray', 'construction_vehicle': 'brown', 'trailer': 'pink',
            }
            
            max_dist = 60.0
            obstacles_tracked = 0
            
            first_frame_obstacles = fut_obstacles[0]
            if isinstance(first_frame_obstacles, dict) and 'gt_boxes' in first_frame_obstacles:
                first_gt_boxes = first_frame_obstacles['gt_boxes'].cpu().numpy() if isinstance(first_frame_obstacles['gt_boxes'], torch.Tensor) else first_frame_obstacles['gt_boxes']
                first_gt_names = first_frame_obstacles['gt_names']
                
                for obs_idx, (box, name) in enumerate(zip(first_gt_boxes, first_gt_names)):
                    x, y, z, l, w, h, yaw = box
                    dist = np.sqrt(x**2 + y**2)
                    if dist > max_dist:
                        continue
                    
                    for frame_idx in range(len(fut_obstacles)):
                        if frame_idx < len(fut_obstacles):
                            frame_obstacles = fut_obstacles[frame_idx]
                            if isinstance(frame_obstacles, dict) and 'gt_boxes' in frame_obstacles:
                                obs_boxes = frame_obstacles['gt_boxes'].cpu().numpy() if isinstance(frame_obstacles['gt_boxes'], torch.Tensor) else frame_obstacles['gt_boxes']
                                if obs_idx < len(obs_boxes):
                                    box = obs_boxes[obs_idx]
                                    x, y, z, l, w, h, yaw = box
                                    
                                    corners_local = np.array([[l/2,  w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2,  w/2]])
                                    cos_yaw = np.cos(yaw)
                                    sin_yaw = np.sin(yaw)
                                    rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
                                    
                                    corners_data = (rotation_matrix @ corners_local.T).T + np.array([x, y])
                                    corners_plot = corners_data[:, [1, 0]]
                                    
                                    alpha = 0.15 + 0.25 * (frame_idx / max(1, len(fut_obstacles) - 1))
                                    polygon = patches.Polygon(corners_plot, closed=True, linewidth=1.0,
                                        edgecolor=color_map.get(name, 'purple'), facecolor=color_map.get(name, 'purple'), alpha=alpha)
                                    ax2.add_patch(polygon)
                                    
                                    if frame_idx == 0:
                                        ax2.scatter(y, x, c=color_map.get(name, 'purple'), s=30, marker='o', alpha=0.8, zorder=3)
                                    else:
                                        ax2.scatter(y, x, c=color_map.get(name, 'purple'), s=15, marker='o', alpha=0.6, zorder=3)
                    
                    obstacles_tracked += 1
                    if obstacles_tracked >= 20:
                        break
                
                num_future_frames = len(fut_obstacles)
                if obstacles_tracked > 0:
                    ax2.scatter([], [], c='gray', s=100, marker='s', alpha=0.3, label=f'Obstacles Motion ({obstacles_tracked} tracked)')
    
    ax2.set_xlabel('Lateral (Y) [meters]', fontsize=12)
    ax2.set_ylabel('Forward (X) [meters]', fontsize=12)
    num_future_frames = len(sample.get('fut_obstacles', [])) if 'fut_obstacles' in sample else 0
    ax2.set_title(f'Future Obstacle Motion ({num_future_frames} frames)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.set_xlim(-30, 30)
    ax2.set_ylim(-20, 60)
    
    # ========== Plot 3: BEV Image ==========
    ax3 = axes[2]
    if 'lidar_bev' in sample and sample['lidar_bev'] is not None:
        bev_tensor = sample['lidar_bev'][-1].cpu() if isinstance(sample['lidar_bev'], torch.Tensor) else sample['lidar_bev'][-1]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        bev_tensor = bev_tensor * std + mean
        bev_tensor = torch.clamp(bev_tensor, 0, 1)
        bev_img = bev_tensor.numpy().transpose(1, 2, 0)
        ax3.imshow(bev_img)
        ax3.set_title('BEV Image (Current Frame)', fontsize=14, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No BEV Image', ha='center', va='center', fontsize=20)
        ax3.set_title('BEV Image (Not Available)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Add info text
    if 'command' in sample:
        nav_cmd = sample['command'][0].cpu() if isinstance(sample['command'], torch.Tensor) else sample['command'][0]
        cmd_names = ['Left', 'Straight', 'Right']
        cmd_idx = torch.argmax(nav_cmd).item() if isinstance(nav_cmd, torch.Tensor) else np.argmax(nav_cmd)
    else:
        cmd_idx = 1
        cmd_names = ['Left', 'Straight', 'Right']
    
    if 'ego_status' in sample:
        ego_status = sample['ego_status'][0].cpu() if isinstance(sample['ego_status'], torch.Tensor) else sample['ego_status'][0]
        vel_x, vel_y, vel_z = ego_status[6], ego_status[7], ego_status[8]
        speed = torch.sqrt(vel_x**2 + vel_y**2 + vel_z**2).item() if isinstance(vel_x, torch.Tensor) else np.sqrt(float(vel_x)**2 + float(vel_y)**2 + float(vel_z)**2)
    else:
        speed = 0.0
    
    num_obstacles = 0
    if 'fut_obstacles' in sample and len(sample['fut_obstacles']) > 0:
        first_obs = sample['fut_obstacles'][0]
        if isinstance(first_obs, dict) and 'gt_boxes' in first_obs:
            num_obstacles = len(first_obs['gt_boxes'])
    
    if 'predicted_trajectory' in sample and 'target_trajectory' in sample:
        pred_traj = sample['predicted_trajectory'].cpu().numpy() if isinstance(sample['predicted_trajectory'], torch.Tensor) else sample['predicted_trajectory']
        target_traj = sample['target_trajectory'].cpu().numpy() if isinstance(sample['target_trajectory'], torch.Tensor) else sample['target_trajectory']
        errors = np.linalg.norm(pred_traj - target_traj, axis=1)
        ade = np.mean(errors)
        fde = errors[-1]
        mae = np.mean(np.abs(pred_traj - target_traj))
    else:
        ade, fde, mae = 0.0, 0.0, 0.0
    
    info_text = f"Command: {cmd_names[cmd_idx]} | Speed: {speed:.2f} m/s | Obstacles: {num_obstacles} | ADE: {ade:.2f} | FDE: {fde:.2f} | MAE: {mae:.2f}"
    if 'sample_token' in sample:
        info_text += f" | Sample: {str(sample['sample_token'])[:8]}..."
    
    fig.suptitle(info_text, fontsize=13, y=0.98, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_trajectory_samples_with_predictions(samples_data, num_samples=20, save_dir='./trajectory_plots'):
    """绘制轨迹样本，每个样本三个子图"""
    os.makedirs(save_dir, exist_ok=True)
    
    total_samples = min(len(samples_data), num_samples)
    sample_indices = np.random.choice(len(samples_data), total_samples, replace=False)
    sample_indices = sorted(sample_indices)
    
    print(f"\nGenerating {total_samples} plot(s) with 3 subplots each...")
    
    for idx, sample_idx in enumerate(sample_indices):
        sample = samples_data[sample_idx]
        save_path = os.path.join(save_dir, f'sample_{sample_idx:04d}.png')
        plot_sample_with_three_subplots(sample, save_path)
        
        if (idx + 1) % 5 == 0:
            print(f"  Generated {idx + 1}/{total_samples} plots...")
    
    print(f"✓ Completed generating {total_samples} plots")


def main():
    """主函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    config_path = os.path.join(project_root, "config/nuscenes.yaml")
    config = load_config(config_path)
    
    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', "/home/wang/Project/MoT-DP/checkpoints/carla_dit_best")
    checkpoint_path = os.path.join(checkpoint_dir, "carla_policy_best.pt")
    policy = load_best_model(checkpoint_path, config, device)
    
    print("\nLoading test dataset...")
    nuscenes_config = config.get('nuscenes', {})
    processed_data_root = nuscenes_config.get('processed_data_root', '/home/wang/Dataset/v1.0-mini/processed_data')
    dataset_root = nuscenes_config.get('dataset_root', '/home/wang/Dataset/v1.0-mini')
    
    val_pkl = os.path.join(processed_data_root, 'nuscenes_processed_val.pkl')
    
    val_dataset = NUSCDataset(
        processed_data_path=val_pkl,
        dataset_root=dataset_root,
        mode='val',
        load_bev_images=True
    )
    
    print(f"✓ Test dataset loaded: {len(val_dataset)} samples\n")
    
    batch_size = config.get('dataloader', {}).get('batch_size', 16)
    num_workers = config.get('dataloader', {}).get('num_workers', 4)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"DataLoader created with batch_size={batch_size}, num_workers={num_workers}\n")
    
    samples_data = get_all_samples_with_predictions(
        policy=policy,
        test_loader=val_loader,
        config=config,
        device=device
    )
    
    print(f"✓ Generated {len(samples_data)} predictions\n")
    
    save_dir = os.path.join(project_root, 'image', 'trajectory_plots')
    plot_trajectory_samples_with_predictions(
        samples_data=samples_data,
        num_samples=20,
        save_dir=save_dir
    )
    
    print(f"\n{'='*60}")
    print("Trajectory plotting completed successfully!")
    print(f"Plots saved to: {save_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
