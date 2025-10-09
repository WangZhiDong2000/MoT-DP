import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm


def visualize_action_stats(all_actions, action_stats, output_dir="visualization_outputs"):
    """
    使用直方图可视化动作统计数据并保存图像。
    """
    print("Visualizing action statistics...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用Seaborn以获得更好看的图形
    sns.set_theme(style="whitegrid")
    
    # 分离X和Y坐标
    actions_x = all_actions[:, 0]
    actions_y = all_actions[:, 1]
    
    # --- X坐标分布图 ---
    plt.figure(figsize=(12, 6))
    sns.histplot(actions_x, bins=100, kde=True, color='skyblue', label='X-actions Distribution')
    
    # 绘制统计线
    plt.axvline(action_stats['mean'][0].item(), color='r', linestyle='--', linewidth=2, label=f"Mean: {action_stats['mean'][0].item():.2f}")
    plt.axvline(action_stats['min'][0].item(), color='g', linestyle=':', linewidth=2, label=f"Min (1%): {action_stats['min'][0].item():.2f}")
    plt.axvline(action_stats['max'][0].item(), color='g', linestyle=':', linewidth=2, label=f"Max (99%): {action_stats['max'][0].item():.2f}")
    
    plt.title('Distribution of Actions (X-coordinate)')
    plt.xlabel('Action Value (X)')
    plt.ylabel('Frequency')
    plt.legend()
    
    x_plot_path = os.path.join(output_dir, 'action_stats_x_distribution.png')
    plt.savefig(x_plot_path)
    plt.close() # 关闭图像，防止在Jupyter等环境中直接显示
    print(f"Saved X-coordinate action distribution plot to {x_plot_path}")

    # --- Y坐标分布图 ---
    plt.figure(figsize=(12, 6))
    sns.histplot(actions_y, bins=100, kde=True, color='salmon', label='Y-actions Distribution')
    
    # 绘制统计线
    plt.axvline(action_stats['mean'][1].item(), color='r', linestyle='--', linewidth=2, label=f"Mean: {action_stats['mean'][1].item():.2f}")
    plt.axvline(action_stats['min'][1].item(), color='g', linestyle=':', linewidth=2, label=f"Min (1%): {action_stats['min'][1].item():.2f}")
    plt.axvline(action_stats['max'][1].item(), color='g', linestyle=':', linewidth=2, label=f"Max (99%): {action_stats['max'][1].item():.2f}")
    
    plt.title('Distribution of Actions (Y-coordinate)')
    plt.xlabel('Action Value (Y)')
    plt.ylabel('Frequency')
    plt.legend()
    
    y_plot_path = os.path.join(output_dir, 'action_stats_y_distribution.png')
    plt.savefig(y_plot_path)
    plt.close()
    print(f"Saved Y-coordinate action distribution plot to {y_plot_path}")


def compute_action_stats(train_dataset, obs_horizon=2, sample_n=10, visualize=False):
    print("Computing action statistics from preprocessed dataset...")
    all_actions = []
    sample_size = min(sample_n, len(train_dataset))
    indices = np.random.choice(len(train_dataset), sample_size, replace=False)
    for i in tqdm(indices, desc="Collecting action samples from preprocessed data"):
        sample = train_dataset[i]
        agent_pos = sample['agent_pos']
        if isinstance(agent_pos, torch.Tensor):
            agent_pos = agent_pos.numpy()
        actions = agent_pos[:]
        all_actions.append(actions)
    all_actions = np.concatenate(all_actions, axis=0)
    print(f"Raw action statistics from {len(all_actions)} samples:")
    raw_min = np.min(all_actions, axis=0)
    raw_max = np.max(all_actions, axis=0)
    print(f"  Raw range: X=[{raw_min[0]:.4f}, {raw_max[0]:.4f}], Y=[{raw_min[1]:.4f}, {raw_max[1]:.4f}]")
    
    percentile_low = 1.0
    percentile_high = 99.0
    filter_action_min = np.percentile(all_actions, percentile_low, axis=0)
    action_min = np.min(all_actions, axis=0)
    filter_action_max = np.percentile(all_actions, percentile_high, axis=0)
    action_max = np.max(all_actions, axis=0)
    action_mean = np.mean(all_actions, axis=0)
    action_std = np.std(all_actions, axis=0)
    
    print(f"Filtered action statistics (using {percentile_low}-{percentile_high} percentile):")
    print(f"  Min: {action_min}")
    print(f"  Max: {action_max}")
    print(f"  Mean: {action_mean}")
    print(f"  Std: {action_std}")
    
    # 返回一个包含torch.Tensor的字典
    action_stats = {
        'min': torch.from_numpy(action_min).float(),
        'max': torch.from_numpy(action_max).float(),
        'mean': torch.from_numpy(action_mean).float(),
        'std': torch.from_numpy(action_std).float(),
        'filter_min': torch.from_numpy(filter_action_min).float(),
        'filter_max': torch.from_numpy(filter_action_max).float()
    }
    
    if visualize:
        visualize_action_stats(all_actions, action_stats)
    

    return action_stats
