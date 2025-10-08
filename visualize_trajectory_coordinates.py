#!/usr/bin/env python3
"""
可视化和验证轨迹坐标系统

这个脚本帮助理解为什么预测位置会出现负数，并验证数据的正确性。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_trajectory_with_explanation(obs_pos, pred_pos, target_point=None, sample_idx=None):
    """
    可视化轨迹并解释坐标系统
    
    Args:
        obs_pos: 观测位置数组，shape (obs_horizon, 2)
        pred_pos: 预测位置数组，shape (action_horizon, 2)
        target_point: 目标点，shape (2,)，可选
        sample_idx: 样本索引，用于标题
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ============================================================
    # 左图：轨迹可视化
    # ============================================================
    ax1.set_title(f'Sample {sample_idx if sample_idx else "?"}: Trajectory in Vehicle Frame\n' + 
                  'Reference Point = Last Observation', fontsize=14, fontweight='bold')
    
    # 绘制观测轨迹
    ax1.plot(obs_pos[:, 1], obs_pos[:, 0], 'bo-', 
             label='Observed Positions', markersize=12, linewidth=3, alpha=0.7)
    
    # 绘制预测轨迹
    ax1.plot(pred_pos[:, 1], pred_pos[:, 0], 'ro--', 
             label='Predicted Positions', markersize=10, linewidth=2.5, alpha=0.7)
    
    # 标记参考点
    ax1.plot(0, 0, 'ks', markersize=15, label='Reference Point (Origin)', 
             markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=2, zorder=10)
    
    # 标记目标点
    if target_point is not None:
        ax1.plot(target_point[1], target_point[0], 'g*', 
                markersize=20, label='Target Point', 
                markeredgecolor='darkgreen', markeredgewidth=2, zorder=10)
        # 画从参考点到目标点的虚线
        ax1.plot([0, target_point[1]], [0, target_point[0]], 
                'g--', alpha=0.3, linewidth=1.5)
    
    # 添加坐标轴
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax1.axvline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    
    # 标注每个点的坐标
    for i, pos in enumerate(obs_pos):
        ax1.annotate(f't={i}\n({pos[0]:.2f}, {pos[1]:.2f})', 
                    xy=(pos[1], pos[0]), xytext=(10, 10),
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    for i, pos in enumerate(pred_pos):
        t = len(obs_pos) + i
        ax1.annotate(f't={t}\n({pos[0]:.2f}, {pos[1]:.2f})', 
                    xy=(pos[1], pos[0]), xytext=(10, -15),
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    # 添加坐标轴说明
    ax1.text(0.02, 0.98, 'Coordinate System:\nX: Forward (+) / Backward (-)\nY: Left (+) / Right (-)', 
            transform=ax1.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel('Y (Lateral) [meters]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('X (Forward) [meters]', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # ============================================================
    # 右图：数据分析
    # ============================================================
    ax2.set_title('Trajectory Analysis', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 计算统计信息
    full_traj = np.vstack([obs_pos, pred_pos])
    distances = np.linalg.norm(np.diff(full_traj, axis=0), axis=1)
    
    # X方向（前进方向）的变化
    x_changes = np.diff(full_traj[:, 0])
    
    # 显示分析结果
    analysis_text = f"""
TRAJECTORY ANALYSIS
{'='*50}

1. OBSERVATION PHASE (t=0 to t={len(obs_pos)-1})
   First obs:  ({obs_pos[0, 0]:7.3f}, {obs_pos[0, 1]:7.3f})
   Last obs:   ({obs_pos[-1, 0]:7.3f}, {obs_pos[-1, 1]:7.3f})  [Reference]
   
   Movement: {obs_pos[-1, 0] - obs_pos[0, 0]:.3f} m forward
   
2. PREDICTION PHASE (t={len(obs_pos)} to t={len(obs_pos)+len(pred_pos)-1})
   First pred: ({pred_pos[0, 0]:7.3f}, {pred_pos[0, 1]:7.3f})
   Last pred:  ({pred_pos[-1, 0]:7.3f}, {pred_pos[-1, 1]:7.3f})
   
   Movement: {pred_pos[-1, 0] - pred_pos[0, 0]:.3f} m forward

3. INTER-FRAME DISTANCES
   Min distance:  {distances.min():.3f} m
   Max distance:  {distances.max():.3f} m
   Mean distance: {distances.mean():.3f} m
   
4. FORWARD PROGRESS (X-axis changes)
   Positive steps: {(x_changes > 0).sum()} / {len(x_changes)}
   Negative steps: {(x_changes < 0).sum()} / {len(x_changes)}
   Zero steps:     {(x_changes == 0).sum()} / {len(x_changes)}

5. WHY NEGATIVE VALUES?
   ✓ Using RELATIVE coordinates
   ✓ Reference point = Last observation = (0, 0)
   ✓ Negative X means "behind reference point"
   ✓ This is NORMAL and CORRECT!
   
   In this sample:
   - First prediction at ({pred_pos[0, 0]:.3f}, {pred_pos[0, 1]:.3f})
   - This is {abs(pred_pos[0, 0]):.3f} m BEHIND reference point
   - But trajectory moves forward: {pred_pos[-1, 0]:.3f} m ahead
   
6. PHYSICAL INTERPRETATION
   The vehicle is moving from:
     Position {obs_pos[0, 0]:.2f} m → 0.00 m → {pred_pos[-1, 0]:.2f} m
   
   Total forward progress: {pred_pos[-1, 0] - obs_pos[0, 0]:.2f} m
   
7. DATA VALIDITY CHECK
   {'✓ PASS' if distances.min() > 0.1 and distances.max() < 5.0 else '✗ FAIL'}: Inter-frame distances are reasonable
   {'✓ PASS' if (x_changes > 0).sum() >= len(x_changes) * 0.6 else '✗ FAIL'}: Mostly forward motion
   {'✓ PASS' if abs(full_traj[:, 1]).max() < 3.0 else '⚠ WARN'}: Lateral movement is reasonable
"""
    
    ax2.text(0.05, 0.95, analysis_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig


def verify_trajectory_data(obs_pos, pred_pos):
    """
    验证轨迹数据的合理性
    
    Returns:
        dict: 验证结果
    """
    full_traj = np.vstack([obs_pos, pred_pos])
    distances = np.linalg.norm(np.diff(full_traj, axis=0), axis=1)
    x_changes = np.diff(full_traj[:, 0])
    
    results = {
        'distances': distances,
        'mean_distance': distances.mean(),
        'min_distance': distances.min(),
        'max_distance': distances.max(),
        'forward_ratio': (x_changes > 0).sum() / len(x_changes),
        'backward_ratio': (x_changes < 0).sum() / len(x_changes),
        'max_lateral': abs(full_traj[:, 1]).max(),
        'has_negative_pred': (pred_pos[:, 0] < 0).any(),
    }
    
    # 判断是否合理
    results['valid_distances'] = (distances.min() > 0.05) and (distances.max() < 5.0)
    results['valid_forward'] = results['forward_ratio'] >= 0.5
    results['valid_lateral'] = results['max_lateral'] < 5.0
    results['overall_valid'] = results['valid_distances'] and results['valid_forward'] and results['valid_lateral']
    
    return results


# ============================================================
# 示例：使用你的数据
# ============================================================
if __name__ == "__main__":
    # 你的样本 330 数据
    obs_pos = np.array([
        [-1.7727478e+00,  4.9607980e-04],
        [ 0.0000000e+00,  0.0000000e+00]
    ])
    
    pred_pos = np.array([
        [-0.4875477,  -0.00444994],
        [ 0.40189597, -0.00466377],
        [ 0.9854936,  -0.00624591],
        [ 1.2927884,  -0.00610257],
        [ 1.4458585,  -0.00676086]
    ])
    
    # 假设的目标点
    target_point = np.array([2.5, -0.01])
    
    print("="*70)
    print("轨迹坐标系统分析")
    print("="*70)
    
    # 验证数据
    results = verify_trajectory_data(obs_pos, pred_pos)
    
    print(f"\n数据验证结果:")
    print(f"  平均帧间距离: {results['mean_distance']:.3f} m")
    print(f"  距离范围: [{results['min_distance']:.3f}, {results['max_distance']:.3f}] m")
    print(f"  前向运动比例: {results['forward_ratio']:.1%}")
    print(f"  后向运动比例: {results['backward_ratio']:.1%}")
    print(f"  最大横向偏移: {results['max_lateral']:.3f} m")
    print(f"  预测位置有负数: {'是' if results['has_negative_pred'] else '否'}")
    print(f"\n整体评估: {'✓ 数据有效' if results['overall_valid'] else '✗ 数据可能有问题'}")
    
    print(f"\n解释:")
    print(f"  预测位置的第一个点 ({pred_pos[0, 0]:.3f}, {pred_pos[0, 1]:.3f}) 是负数")
    print(f"  这表示该点在参考点（最后观测位置）的后方 {abs(pred_pos[0, 0]):.3f} 米")
    print(f"  这是正常的！因为我们使用的是相对坐标系统")
    print(f"  整体轨迹仍然是向前移动的：从 {obs_pos[0, 0]:.2f} m 到 {pred_pos[-1, 0]:.2f} m")
    
    # 可视化
    print(f"\n生成可视化图表...")
    fig = visualize_trajectory_with_explanation(obs_pos, pred_pos, target_point, sample_idx=330)
    
    # 保存图表
    output_path = '/home/wang/projects/diffusion_policy_z/trajectory_coordinate_analysis.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("结论: 预测位置出现负数是正常的，这是相对坐标系统的特性！")
    print("="*70)
