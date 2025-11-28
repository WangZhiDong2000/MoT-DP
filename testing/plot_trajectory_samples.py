#!/usr/bin/env python3
import os
import sys
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dataset.generate_pdm_dataset import CARLAImageDataset
from policy.diffusion_dit_carla_policy import DiffusionDiTCarlaPolicy


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_best_model(checkpoint_path, config, device):
    """加载最优模型权重"""
    print(f"Loading best model from: {checkpoint_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 提取action统计信息
    action_stats = {
        'min': torch.tensor([-11.77335262298584, -59.26432800292969]),
        'max': torch.tensor([98.34003448486328, 55.585079193115234]),
        'mean': torch.tensor([10.975193977355957, 0.04004639387130737]),
        'std': torch.tensor([14.96833324432373, 3.419595956802368]),
    }
    
    # 初始化模型
    policy = DiffusionDiTCarlaPolicy(config, action_stats=action_stats).to(device)
    
    # 加载模型权重
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    print(f"✓ Model loaded successfully!")
    
    return policy


def get_all_predictions(policy, test_loader, config, device='cuda'):
    """
    获取所有测试样本的预测和真实轨迹
    
    Returns:
        all_predictions: list of (T, 2) arrays
        all_ground_truths: list of (T, 2) arrays
    """
    print(f"\nGenerating predictions for all samples...")
    
    policy.eval()
    
    all_predictions = []
    all_ground_truths = []
    
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
                'agent_pos': batch['agent_pos'][:, :obs_horizon],
                'speed': batch['speed'][:, :obs_horizon],
                'target_point': batch['target_point'][:, :obs_horizon],
                'next_command': batch['next_command'][:, :obs_horizon],  
            }
            
            try:
                # 预测动作/轨迹
                result = policy.predict_action(obs_dict)
                predicted_actions = torch.from_numpy(result['action']).to(device)  # (B, T, 2)
                target_actions = batch['agent_pos'][:, :predicted_actions.shape[1]]  # (B, T, 2)
                
                # 保存预测和真实值
                for i in range(predicted_actions.shape[0]):
                    pred = predicted_actions[i].cpu().numpy()  # (T, 2)
                    gt = target_actions[i].cpu().numpy()  # (T, 2)
                    all_predictions.append(pred)
                    all_ground_truths.append(gt)
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    return all_predictions, all_ground_truths


def plot_trajectory_samples(predictions, ground_truths, num_samples_per_page=4, 
                           num_pages=5, save_dir='./trajectory_plots'):
    """
    绘制轨迹样本
    
    Args:
        predictions: list of (T, 2) arrays - 预测轨迹
        ground_truths: list of (T, 2) arrays - 真实轨迹
        num_samples_per_page: 每页样本数（并排显示）
        num_pages: 总页数
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 总共要绘制的样本数
    total_samples = min(len(predictions), num_samples_per_page * num_pages)
    
    # 随机选择样本
    sample_indices = np.random.choice(len(predictions), total_samples, replace=False)
    sample_indices = sorted(sample_indices)
    
    print(f"\nGenerating {num_pages} plot(s) with {num_samples_per_page} samples each...")
    
    # 按页绘制
    for page_idx in range(num_pages):
        start_idx = page_idx * num_samples_per_page
        end_idx = min(start_idx + num_samples_per_page, total_samples)
        page_samples = sample_indices[start_idx:end_idx]
        
        # 创建子图
        num_cols = len(page_samples)
        fig, axes = plt.subplots(1, num_cols, figsize=(5*num_cols, 5))
        
        # 如果只有一个样本，axes不是array
        if num_cols == 1:
            axes = [axes]
        
        # 计算全局的坐标范围
        all_x_min, all_x_max = float('inf'), float('-inf')
        all_y_min, all_y_max = float('inf'), float('-inf')
        
        for page_sample_idx, sample_idx in enumerate(page_samples):
            pred = predictions[sample_idx]  # (T, 2)
            gt = ground_truths[sample_idx]   # (T, 2)
            
            # 更新坐标范围
            all_x_min = min(all_x_min, pred[:, 0].min(), gt[:, 0].min())
            all_x_max = max(all_x_max, pred[:, 0].max(), gt[:, 0].max())
            all_y_min = min(all_y_min, pred[:, 1].min(), gt[:, 1].min())
            all_y_max = max(all_y_max, pred[:, 1].max(), gt[:, 1].max())
        
        # 添加一些边距
        x_margin = (all_x_max - all_x_min) * 0.1
        y_margin = (all_y_max - all_y_min) * 0.1
        
        if x_margin == 0:
            x_margin = 10
        if y_margin == 0:
            y_margin = 10
        
        x_min = all_x_min - x_margin
        x_max = all_x_max + x_margin
        y_min = all_y_min - y_margin
        y_max = all_y_max + y_margin
        
        # 为了使用相等的x和y轴尺度，取最大范围
        max_range = max(x_max - x_min, y_max - y_min)
        
        # 重新调整范围使其对称
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        x_min = x_center - max_range / 2
        x_max = x_center + max_range / 2
        y_min = y_center - max_range / 2
        y_max = y_center + max_range / 2
        
        # 绘制每个样本
        for page_sample_idx, sample_idx in enumerate(page_samples):
            ax = axes[page_sample_idx]
            
            pred = predictions[sample_idx]  # (T, 2)
            gt = ground_truths[sample_idx]   # (T, 2)
            
            # 计算误差
            errors = np.linalg.norm(pred - gt, axis=1)
            mae = np.mean(np.abs(pred - gt))
            ade = np.mean(errors)
            fde = errors[-1]
            
            # 绘制真实轨迹
            ax.plot(gt[:, 0], gt[:, 1], 'o-', color='green', linewidth=2, 
                   markersize=6, label='Ground Truth', alpha=0.8)
            
            # 绘制预测轨迹
            ax.plot(pred[:, 0], pred[:, 1], 's--', color='red', linewidth=2, 
                   markersize=6, label='Prediction', alpha=0.8)
            
            # 连接对应点之间的误差线
            for j in range(len(pred)):
                ax.plot([gt[j, 0], pred[j, 0]], [gt[j, 1], pred[j, 1]], 
                       'k--', alpha=0.3, linewidth=0.5)
            
            # 标记起点和终点
            ax.plot(gt[0, 0], gt[0, 1], 'go', markersize=10, label='GT Start', zorder=5)
            ax.plot(gt[-1, 0], gt[-1, 1], 'g^', markersize=10, label='GT End', zorder=5)
            ax.plot(pred[0, 0], pred[0, 1], 'rs', markersize=10, label='Pred Start', zorder=5)
            ax.plot(pred[-1, 0], pred[-1, 1], 'r^', markersize=10, label='Pred End', zorder=5)
            
            # 设置轴属性
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)
            
            # 标题和标签
            title = f"Sample {sample_idx+1}\n"
            title += f"MAE={mae:.2f}, ADE={ade:.2f}, FDE={fde:.2f}"
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel('X', fontsize=9)
            ax.set_ylabel('Y', fontsize=9)
            
            # 图例
            if page_sample_idx == 0:
                ax.legend(loc='best', fontsize=8)
        
        # 调整布局（通过bbox_inches='tight'处理）
        
        # 保存
        save_path = os.path.join(save_dir, f'trajectories_page_{page_idx+1:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()


def main():
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # 加载配置
    config_path = os.path.join(project_root, "config/carla.yaml")
    config = load_config(config_path)
    
    # 加载最优模型
    checkpoint_path = "/home/wang/Project/MoT-DP/checkpoints/carla_dit/carla_policy_best.pt"
    policy = load_best_model(checkpoint_path, config, device)
    
    # 加载测试数据集
    print("\nLoading test dataset...")
    dataset_path = config.get('training', {}).get('dataset_path')
    val_dataset_path = os.path.join(dataset_path, 'val')
    image_data_root = config.get('training', {}).get('image_data_root')
    
    test_dataset = CARLAImageDataset(
        dataset_path=val_dataset_path, 
        image_data_root=image_data_root
    )
    
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples\n")
    
    # 创建数据加载器
    batch_size = config.get('dataloader', {}).get('batch_size', 16)
    num_workers = config.get('dataloader', {}).get('num_workers', 4)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"DataLoader created with batch_size={batch_size}, num_workers={num_workers}\n")
    
    # 获取所有预测
    predictions, ground_truths = get_all_predictions(
        policy=policy,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    print(f"✓ Generated {len(predictions)} predictions\n")
    
    # 绘制样本
    save_dir = os.path.join(project_root, 'image', 'trajectory_plots')
    plot_trajectory_samples(
        predictions=predictions,
        ground_truths=ground_truths,
        num_samples_per_page=4,  # 每页4个样本
        num_pages=5,              # 共5页
        save_dir=save_dir
    )
    
    print(f"\n{'='*60}")
    print("Trajectory plotting completed successfully!")
    print(f"Plots saved to: {save_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
