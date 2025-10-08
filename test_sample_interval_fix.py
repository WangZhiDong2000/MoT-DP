#!/usr/bin/env python3
"""
测试修复后的降采样时间对齐问题

验证预测位置是否都在参考点前方
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.generate_pdm_dataset import CARLAImageDataset
import numpy as np
import matplotlib.pyplot as plt

def test_sample_interval_fix():
    """测试不同sample_interval下的轨迹生成"""
    
    dataset_path = '/home/wang/dataset/tmp_data/tmp_data'
    pred_horizon = 7
    obs_horizon = 2
    action_horizon = 5
    
    # 测试不同的采样间隔
    test_intervals = [2, 5, 10]
    
    fig, axes = plt.subplots(1, len(test_intervals), figsize=(18, 6))
    
    for idx, sample_interval in enumerate(test_intervals):
        print(f"\n{'='*60}")
        print(f"Testing sample_interval = {sample_interval}")
        print(f"{'='*60}")
        
        dataset = CARLAImageDataset(
            dataset_path=dataset_path,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            max_files=5,
            sample_interval=sample_interval
        )
        
        # 测试多个样本
        num_negative_first_pred = 0
        num_samples_tested = min(100, len(dataset))
        
        for i in range(num_samples_tested):
            sample = dataset[i]
            agent_pos = sample['agent_pos'].numpy()
            
            # 检查第一个预测位置是否为负
            first_pred_x = agent_pos[obs_horizon, 0]  # 第一个预测点的X坐标
            if first_pred_x < 0:
                num_negative_first_pred += 1
        
        negative_ratio = num_negative_first_pred / num_samples_tested * 100
        
        print(f"\nResults for sample_interval={sample_interval}:")
        print(f"  Samples tested: {num_samples_tested}")
        print(f"  First prediction behind reference: {num_negative_first_pred}")
        print(f"  Percentage: {negative_ratio:.1f}%")
        print(f"  Status: {'✓ FIXED' if negative_ratio < 5 else '✗ STILL HAS ISSUES'}")
        
        # 可视化一个样本
        sample_idx = min(50, len(dataset) - 1)
        sample = dataset[sample_idx]
        agent_pos = sample['agent_pos'].numpy()
        target_point = sample['target_point'].numpy()[0]
        
        obs_pos = agent_pos[:obs_horizon]
        pred_pos = agent_pos[obs_horizon:]
        
        ax = axes[idx]
        ax.plot(obs_pos[:, 1], obs_pos[:, 0], 'bo-', 
                label='Observed', markersize=10, linewidth=2.5, alpha=0.7)
        ax.plot(pred_pos[:, 1], pred_pos[:, 0], 'ro--', 
                label='Predicted', markersize=8, linewidth=2, alpha=0.7)
        ax.plot(0, 0, 'ks', markersize=12, label='Reference', 
                markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=2)
        ax.plot(target_point[1], target_point[0], 'g*', 
                markersize=18, label='Target', markeredgecolor='darkgreen', markeredgewidth=2)
        
        # 标注第一个预测点
        ax.annotate(f'1st pred\n({pred_pos[0,0]:.2f}, {pred_pos[0,1]:.2f})',
                   xy=(pred_pos[0, 1], pred_pos[0, 0]), 
                   xytext=(15, 15), textcoords='offset points',
                   fontsize=9, bbox=dict(boxstyle='round', 
                   facecolor='yellow' if pred_pos[0,0] >= 0 else 'red', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('Y (Lateral) [m]', fontsize=10)
        ax.set_ylabel('X (Forward) [m]', fontsize=10)
        ax.set_title(f'sample_interval={sample_interval}\n' + 
                    f'Sample {sample_idx}\n' +
                    f'{negative_ratio:.0f}% negative first pred',
                    fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # 添加统计信息
        stats_text = f'Waypoint calc:\n'
        stats_text += f'base + (step+1)*{sample_interval}\n'
        stats_text += f'Indices: {[1 + (i+1)*sample_interval for i in range(3)]}'
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes, fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    
    plt.tight_layout()
    output_path = '/home/wang/projects/diffusion_policy_z/sample_interval_fix_verification.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: {output_path}")
    plt.show()
    
    print(f"\n{'='*60}")
    print("修复验证完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_sample_interval_fix()
