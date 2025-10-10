#!/usr/bin/env python3
"""
测试脚本：加载最优模型权重，从测试集抽样并可视化预测结果与ground truth
"""
import os
import sys
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random

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
        'min': torch.tensor([0, -10.5050]),
        'max': torch.tensor([24.4924,  9.9753]),
        'mean': torch.tensor([2.3079, 0.0188]),
        'std': torch.tensor([3.7443, 0.6994]),
    }
    
    # 初始化模型
    policy = DiffusionDiTCarlaPolicy(config, action_stats=action_stats).to(device)
    
    # 加载模型权重
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    # 打印checkpoint信息
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


def visualize_prediction(sample, prediction, ground_truth, sample_idx, save_dir):
    """
    可视化预测轨迹和ground truth
    
    Args:
        sample: 包含所有输入数据的字典
        prediction: 预测的agent_pos轨迹 (T, 2)
        ground_truth: 真实的agent_pos轨迹 (T, 2)
        sample_idx: 样本索引
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建两个子图：轨迹比较 + 观测图像
    fig = plt.figure(figsize=(16, 6))
    
    # 左图：轨迹可视化
    ax1 = plt.subplot(1, 2, 1)
    
    # 获取观测时间步的agent_pos
    obs_horizon = sample['agent_pos'].shape[0] - ground_truth.shape[0]
    obs_agent_pos = sample['agent_pos'][:obs_horizon].cpu().numpy()
    
    # 绘制观测轨迹
    if len(obs_agent_pos) > 0:
        ax1.plot(obs_agent_pos[:, 1], obs_agent_pos[:, 0], 'go-', 
                label='Observed', markersize=10, linewidth=2.5, alpha=0.8)
        # 标记最后一个观测点作为参考点
        ax1.plot(obs_agent_pos[-1, 1], obs_agent_pos[-1, 0], 'ks', 
                markersize=12, label='Reference (last obs)', 
                markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=2)
    
    # 绘制预测轨迹
    ax1.plot(prediction[:, 1], prediction[:, 0], 'b^--', 
            label='Prediction', markersize=8, linewidth=2, alpha=0.8)
    
    # 绘制ground truth轨迹
    ax1.plot(ground_truth[:, 1], ground_truth[:, 0], 'ro-', 
            label='Ground Truth', markersize=8, linewidth=2, alpha=0.8)
    
    # 绘制目标点
    if 'target_point' in sample and sample['target_point'] is not None:
        target_point = sample['target_point'][0].cpu().numpy()
        ax1.plot(target_point[1], target_point[0], 'g*', 
                markersize=20, label='Target Point', 
                markeredgecolor='black', markeredgewidth=1.5)
        
        # 绘制到目标点的方向线
        if len(obs_agent_pos) > 0:
            last_obs = obs_agent_pos[-1]
            ax1.plot([last_obs[1], target_point[1]], 
                    [last_obs[0], target_point[0]], 
                    'g--', alpha=0.4, linewidth=1.5)
    
    # 计算误差并在图上标注
    mse = np.mean((prediction - ground_truth) ** 2)
    mae = np.mean(np.abs(prediction - ground_truth))
    l2_dist = np.linalg.norm(prediction - ground_truth, axis=1)
    
    ax1.set_xlabel('Y (relative to last obs)', fontsize=12)
    ax1.set_ylabel('X (relative to last obs)', fontsize=12)
    ax1.set_title(f'Sample {sample_idx}: Trajectory Comparison\n'
                  f'MSE: {mse:.4f}, MAE: {mae:.4f}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 右图：显示第一帧观测图像
    ax2 = plt.subplot(1, 2, 2)
    if 'image' in sample and sample['image'] is not None:
        # 取第一帧图像
        img = sample['image'][0].cpu().numpy()  # (3, H, W)
        
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # 转换为 (H, W, 3)
        img = np.transpose(img, (1, 2, 0))
        
        ax2.imshow(img)
        ax2.set_title('First Observation Image', fontsize=13, fontweight='bold')
        ax2.axis('off')
    
    # 在图像下方添加详细误差信息
    error_text = f'Per-step L2 errors:\n'
    for i, dist in enumerate(l2_dist):
        error_text += f'Step {i+1}: {dist:.4f}  '
        if (i + 1) % 3 == 0:
            error_text += '\n'
    
    plt.figtext(0.5, 0.02, error_text, ha='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    # 保存图像
    save_path = os.path.join(save_dir, f'sample_{sample_idx}_prediction.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualization to: {save_path}")


def compute_metrics(predictions, ground_truths):
    """
    计算整体评估指标
    
    Args:
        predictions: 预测轨迹列表 [(T, 2), ...]
        ground_truths: 真实轨迹列表 [(T, 2), ...]
    
    Returns:
        metrics: 评估指标字典
    """
    all_mse = []
    all_mae = []
    all_l2_errors = []
    
    for pred, gt in zip(predictions, ground_truths):
        # MSE
        mse = np.mean((pred - gt) ** 2)
        all_mse.append(mse)
        
        # MAE
        mae = np.mean(np.abs(pred - gt))
        all_mae.append(mae)
        
        # L2 distance per timestep
        l2_dist = np.linalg.norm(pred - gt, axis=1)
        all_l2_errors.append(l2_dist)
    
    # 汇总所有样本的L2误差
    all_l2_errors = np.concatenate(all_l2_errors)
    
    metrics = {
        'MSE': np.mean(all_mse),
        'MSE_std': np.std(all_mse),
        'MAE': np.mean(all_mae),
        'MAE_std': np.std(all_mae),
        'L2_mean': np.mean(all_l2_errors),
        'L2_std': np.std(all_l2_errors),
        'L2_median': np.median(all_l2_errors),
    }
    
    return metrics


def test_model(policy, test_dataset, config, num_samples=10, visualize_samples=5, device='cuda'):
    """
    测试模型并可视化结果
    
    Args:
        policy: 训练好的模型
        test_dataset: 测试数据集
        config: 配置字典
        num_samples: 测试样本数量
        visualize_samples: 可视化的样本数量
        device: 设备
    """
    print(f"\n{'='*60}")
    print(f"Testing model on {num_samples} samples from test dataset")
    print(f"Visualizing {visualize_samples} random samples")
    print(f"{'='*60}\n")
    
    # 随机抽取测试样本
    total_samples = len(test_dataset)
    if num_samples > total_samples:
        num_samples = total_samples
        print(f"Warning: Requested {num_samples} samples but only {total_samples} available")
    
    # 随机选择样本索引
    sample_indices = random.sample(range(total_samples), num_samples)
    visualize_indices = random.sample(sample_indices, min(visualize_samples, num_samples))
    
    print(f"Selected sample indices: {sample_indices[:10]}..." if len(sample_indices) > 10 else f"Selected sample indices: {sample_indices}")
    print(f"Visualizing samples: {visualize_indices}\n")
    
    # 创建保存目录
    save_dir = os.path.join(project_root, 'image', 'test_results')
    os.makedirs(save_dir, exist_ok=True)
    
    predictions = []
    ground_truths = []
    
    policy.eval()
    with torch.no_grad():
        for idx, sample_idx in enumerate(tqdm(sample_indices, desc="Testing")):
            # 获取样本
            sample = test_dataset[sample_idx]
            
            # 准备输入数据
            batch = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.unsqueeze(0).to(device)  # 添加batch维度
            
            # 提取观测数据
            obs_horizon = config.get('obs_horizon', 2)
            obs_dict = {
                'image': batch['image'][:, :obs_horizon],
                'agent_pos': batch['agent_pos'][:, :obs_horizon],
                'speed': batch['speed'][:, :obs_horizon],
                'target_point': batch['target_point'][:, :obs_horizon],
                'next_command': batch['next_command'][:, :obs_horizon],
            }
            
            # 预测动作/轨迹
            try:
                result = policy.predict_action(obs_dict)
                predicted_actions = result['action'][0]  # 移除batch维度 (T, 2)
                print(f"Sample {sample_idx}: Predicted actions shape: {predicted_actions.shape}")
                
                # 获取ground truth
                gt_actions = batch['agent_pos'][0, obs_horizon:obs_horizon + predicted_actions.shape[0]].cpu().numpy()
                
                predictions.append(predicted_actions)
                ground_truths.append(gt_actions)
                
                # 可视化选定的样本
                if sample_idx in visualize_indices:
                    visualize_prediction(
                        sample=sample,
                        prediction=predicted_actions,
                        ground_truth=gt_actions,
                        sample_idx=sample_idx,
                        save_dir=save_dir
                    )
                    
            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}")
                continue
    
    # 计算整体指标
    print(f"\n{'='*60}")
    print("Computing overall metrics...")
    print(f"{'='*60}\n")
    
    metrics = compute_metrics(predictions, ground_truths)
    
    print("Test Results:")
    print(f"  Number of samples: {len(predictions)}")
    print(f"  MSE: {metrics['MSE']:.6f} ± {metrics['MSE_std']:.6f}")
    print(f"  MAE: {metrics['MAE']:.6f} ± {metrics['MAE_std']:.6f}")
    print(f"  L2 Distance (mean): {metrics['L2_mean']:.6f} ± {metrics['L2_std']:.6f}")
    print(f"  L2 Distance (median): {metrics['L2_median']:.6f}")
    
    # 保存指标到文件
    metrics_path = os.path.join(save_dir, 'test_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Test Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of samples: {len(predictions)}\n")
        f.write(f"MSE: {metrics['MSE']:.6f} ± {metrics['MSE_std']:.6f}\n")
        f.write(f"MAE: {metrics['MAE']:.6f} ± {metrics['MAE_std']:.6f}\n")
        f.write(f"L2 Distance (mean): {metrics['L2_mean']:.6f} ± {metrics['L2_std']:.6f}\n")
        f.write(f"L2 Distance (median): {metrics['L2_median']:.6f}\n")
        f.write(f"\nSample indices: {sample_indices}\n")
        f.write(f"Visualized indices: {visualize_indices}\n")
    
    print(f"\n✓ Metrics saved to: {metrics_path}")
    print(f"✓ Visualizations saved to: {save_dir}")
    
    return metrics, predictions, ground_truths


def main():
    """主函数"""
    # 使用时间戳作为随机种子，每次运行都不同
    import time
    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Random seed for this run: {seed}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # 加载配置
    config_path = os.path.join(project_root, "config/carla.yaml")
    config = load_config(config_path)
    
    # 加载最优模型
    checkpoint_path = "/home/wang/projects/diffusion_policy_z/checkpoints/carla_dit/carla_policy_best.pt"
    policy = load_best_model(checkpoint_path, config, device)
    
    # 加载测试数据集
    print("\nLoading test dataset...")
    dataset_path = config.get('training', {}).get('dataset_path', "/home/wang/projects/diffusion_policy_z/data")
    
    test_dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        mode='val',  # 使用验证集作为测试集
    )
    
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples\n")
    
    # 运行测试
    num_test_samples = 50  # 测试样本数量
    num_visualize = 10     # 可视化样本数量
    
    metrics, predictions, ground_truths = test_model(
        policy=policy,
        test_dataset=test_dataset,
        config=config,
        num_samples=num_test_samples,
        visualize_samples=num_visualize,
        device=device
    )
    
    print(f"\n{'='*60}")
    print("Testing completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
