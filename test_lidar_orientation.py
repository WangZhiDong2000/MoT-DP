"""
测试LiDAR BEV的坐标系方向
验证最终的 lidar_bev 张量中，X轴是否为前进方向
"""
import numpy as np
import matplotlib.pyplot as plt

# 模拟Transfuser的处理流程
def test_coordinate_system():
    print("=" * 70)
    print("测试LiDAR BEV坐标系方向")
    print("=" * 70)
    
    # 1. CARLA坐标系：x前进，y向右
    print("\n步骤1: CARLA原始坐标系")
    print("  - X轴: 前进方向 (forward)")
    print("  - Y轴: 向右方向 (right)")
    
    # 创建测试点：一条从后到前的直线 (y=0, x从-10到+10)
    test_points_carla = np.array([
        [i, 0.0, 1.0] for i in np.linspace(-10, 10, 21)  # X从后(-10)到前(+10), Y=0
    ])
    print(f"  测试点: {len(test_points_carla)} 个点在 CARLA X轴上 (前进方向)")
    print(f"  点的范围: X=[{test_points_carla[:, 0].min():.1f}, {test_points_carla[:, 0].max():.1f}], Y={test_points_carla[0, 1]:.1f}")
    
    # 2. np.histogramdd: bins=(xbins, ybins)
    print("\n步骤2: np.histogramdd voxelization")
    min_x, max_x = -32, 32
    min_y, max_y = -32, 32
    pixels_per_meter = 4.0
    hist_max_per_pixel = 5
    
    xbins = np.linspace(min_x, max_x, (max_x - min_x) * int(pixels_per_meter) + 1)
    ybins = np.linspace(min_y, max_y, (max_y - min_y) * int(pixels_per_meter) + 1)
    
    hist = np.histogramdd(test_points_carla[:, :2], bins=(xbins, ybins))[0]
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel
    overhead_splat = hist / hist_max_per_pixel
    
    print(f"  histogramdd输出形状: {overhead_splat.shape}")
    print(f"  - 第0维 (axis 0): 对应X bins (CARLA X轴, 前进方向)")
    print(f"  - 第1维 (axis 1): 对应Y bins (CARLA Y轴, 向右方向)")
    print(f"  非零元素数量: {np.count_nonzero(overhead_splat)}")
    
    # 找出非零元素的位置
    nonzero_indices = np.argwhere(overhead_splat > 0)
    if len(nonzero_indices) > 0:
        print(f"  非零位置范围: X维度=[{nonzero_indices[:, 0].min()}, {nonzero_indices[:, 0].max()}], Y维度=[{nonzero_indices[:, 1].min()}, {nonzero_indices[:, 1].max()}]")
    
    # 3. Transfuser的transpose操作
    print("\n步骤3: Transfuser transpose (overhead_splat.T)")
    overhead_splat_transposed = overhead_splat.T
    print(f"  转置后形状: {overhead_splat_transposed.shape}")
    print(f"  - 第0维 (axis 0, 图像行): 对应原Y bins → CARLA Y轴 (向右方向)")
    print(f"  - 第1维 (axis 1, 图像列): 对应原X bins → CARLA X轴 (前进方向)")
    print(f"  非零元素数量: {np.count_nonzero(overhead_splat_transposed)}")
    
    # 找出转置后非零元素的位置
    nonzero_indices_t = np.argwhere(overhead_splat_transposed > 0)
    if len(nonzero_indices_t) > 0:
        print(f"  非零位置范围: 行方向=[{nonzero_indices_t[:, 0].min()}, {nonzero_indices_t[:, 0].max()}], 列方向=[{nonzero_indices_t[:, 1].min()}, {nonzero_indices_t[:, 1].max()}]")
    
    # 4. 最终存储格式 (C, H, W)
    print("\n步骤4: 最终张量格式 (C, H, W)")
    features = np.stack([overhead_splat_transposed], axis=-1)  # (H, W, 1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)  # (1, H, W)
    
    print(f"  最终形状: {features.shape} = (C, H, W)")
    print(f"  - C: 通道数")
    print(f"  - H (height): 图像高度 → CARLA Y轴 (向右方向)")
    print(f"  - W (width): 图像宽度 → CARLA X轴 (前进方向)")
    
    # 5. 关键结论
    print("\n" + "=" * 70)
    print("关键结论:")
    print("=" * 70)
    print("在最终的 lidar_bev 张量 (C, H, W) 中:")
    print("  ✗ 第1维 (H, height): 对应CARLA Y轴 (lateral, 向右) - 不是前进方向")
    print("  ✓ 第2维 (W, width):  对应CARLA X轴 (longitudinal, 前进) - 是前进方向")
    print("\n也就是说:")
    print("  - lidar_bev[c, :, w] 访问的是一条纵向线 (Y方向, 左右)")
    print("  - lidar_bev[c, h, :] 访问的是一条横向线 (X方向, 前后)")
    print("\n因此: X轴(前进方向)对应的是张量的 WIDTH 维度,而不是 HEIGHT 维度")
    print("=" * 70)
    
    # 6. 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 转置前
    im1 = axes[0].imshow(overhead_splat, cmap='hot', vmin=0, vmax=1, 
                         origin='lower', extent=[min_y, max_y, min_x, max_x])
    axes[0].set_title('转置前 (histogramdd输出)\n第0维=X(前进), 第1维=Y(向右)', 
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Y (向右方向) [m]', fontsize=10)
    axes[0].set_ylabel('X (前进方向) [m]', fontsize=10)
    axes[0].plot(0, 0, 'c*', markersize=15, markeredgewidth=2, markeredgecolor='white')
    axes[0].axhline(y=0, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].axvline(x=0, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(im1, ax=axes[0], label='Normalized hits')
    
    # 转置后
    im2 = axes[1].imshow(features[0], cmap='hot', vmin=0, vmax=1,
                         origin='lower', extent=[min_x, max_x, min_y, max_y])
    axes[1].set_title('转置后 (final lidar_bev)\nH(行)=Y(向右), W(列)=X(前进)', 
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X (前进方向) [m]', fontsize=10)
    axes[1].set_ylabel('Y (向右方向) [m]', fontsize=10)
    axes[1].plot(0, 0, 'c*', markersize=15, markeredgewidth=2, markeredgecolor='white')
    axes[1].axhline(y=0, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].axvline(x=0, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(im2, ax=axes[1], label='Normalized hits')
    
    plt.suptitle('LiDAR BEV 坐标系转换: Transfuser Transpose', 
                fontsize=14, fontweight='bold')
    
    save_path = '/home/wang/Project/MoT-DP/image/lidar_coordinate_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n可视化已保存到: {save_path}")
    
    return features

if __name__ == "__main__":
    features = test_coordinate_system()
