#!/usr/bin/env python3
"""
独立的轨迹可视化脚本
从保存的npz文件加载数据并生成可视化图片
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def visualize_from_npz(npz_path, save_path):
    """从npz文件加载数据并可视化"""
    # Load data
    data = np.load(npz_path, allow_pickle=True)
    sample_indices = data['sample_indices']
    predictions = data['predictions']
    waypoints_hist = data['waypoints_hist']
    agent_pos = data['agent_pos']
    
    print(f"Loaded data for {len(sample_indices)} samples")
    
    # Create figure
    fig = plt.figure(figsize=(16, 16))
    
    for idx in range(4):
        ax = plt.subplot(2, 2, idx + 1)
        
        sample_idx = sample_indices[idx]
        pred = predictions[idx]
        hist_traj = waypoints_hist[idx]
        gt_traj = agent_pos[idx]
        
        # 1. Plot historical trajectory (blue)
        if hist_traj is not None and len(hist_traj) > 0:
            ax.plot(hist_traj[:, 1], hist_traj[:, 0], 'b-', 
                   linewidth=2, alpha=0.6, label='Historical trajectory')
            ax.scatter(hist_traj[:, 1], hist_traj[:, 0], 
                      c='blue', s=50, alpha=0.8, zorder=3)
        
        # 2. Plot current position (origin)
        ax.scatter(0, 0, c='black', s=200, marker='o', 
                  label='Current position', zorder=5, 
                  edgecolors='yellow', linewidths=3)
        
        # 3. Plot ground truth future trajectory (green)
        if gt_traj is not None and len(gt_traj) > 0:
            gt_trajectory = np.vstack([[[0, 0]], gt_traj])
            ax.plot(gt_trajectory[:, 1], gt_trajectory[:, 0], 'g--', 
                   linewidth=2.5, alpha=0.7, label='Ground truth')
            ax.scatter(gt_traj[:, 1], gt_traj[:, 0], 
                      c='green', s=80, alpha=0.8, marker='s', zorder=4)
        
        # 4. Plot predicted trajectory (red)
        if pred is not None and len(pred) > 0:
            pred_trajectory = np.vstack([[[0, 0]], pred])
            ax.plot(pred_trajectory[:, 1], pred_trajectory[:, 0], 'r-', 
                   linewidth=2.5, alpha=0.7, label='Prediction')
            ax.scatter(pred[:, 1], pred[:, 0], 
                      c='red', s=80, alpha=0.8, marker='^', zorder=4)
        
        # Calculate metrics
        if gt_traj is not None and pred is not None:
            min_len = min(len(gt_traj), len(pred))
            if min_len > 0:
                gt_compare = gt_traj[:min_len]
                pred_compare = pred[:min_len]
                l2_errors = np.linalg.norm(gt_compare - pred_compare, axis=-1)
                avg_l2 = np.mean(l2_errors)
                ax.set_title(f'Sample {sample_idx} (Avg L2: {avg_l2:.2f}m)', 
                           fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'Sample {sample_idx}', fontsize=12, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Y (lateral / m)', fontsize=11)
        ax.set_ylabel('X (longitudinal / m)', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        
        # Add axis lines at origin
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.08, hspace=0.3, wspace=0.3)
    
    # Save
    plt.savefig(save_path, dpi=150, format='png')
    plt.close()
    
    print(f"✓ Visualization saved to: {save_path}")


if __name__ == "__main__":
    npz_path = "/root/z_projects/code/MoT-DP-1/image/trajectory_data.npz"
    save_path = "/root/z_projects/code/MoT-DP-1/image/trajectory_predictions_4samples.png"
    
    if not os.path.exists(npz_path):
        print(f"⚠ Data file not found: {npz_path}")
        print("Please run visualize_trajectory_prediction.py first to generate the data.")
    else:
        visualize_from_npz(npz_path, save_path)
