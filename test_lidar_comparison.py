"""
对比检查: 我们的LiDAR处理 vs Transfuser原始实现
"""
import numpy as np
import sys
sys.path.append('/home/wang/Project/MoT-DP/Transfuser/team_code')

# ========== 我们的实现 ==========
class OurConfig:
    pixels_per_meter = 4.0
    hist_max_per_pixel = 5
    lidar_split_height = 0.2
    use_ground_plane = False
    min_x = -32
    max_x = 32
    min_y = -32
    max_y = 32
    max_height_lidar = 100.0

def our_lidar_to_histogram_features(lidar, config):
    """我们的实现 (从unified_carla_dataset.py)"""
    def splat_points(point_cloud):
        xbins = np.linspace(
            config.min_x, config.max_x,
            (config.max_x - config.min_x) * int(config.pixels_per_meter) + 1
        )
        ybins = np.linspace(
            config.min_y, config.max_y,
            (config.max_y - config.min_y) * int(config.pixels_per_meter) + 1
        )
        hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
        hist[hist > config.hist_max_per_pixel] = config.hist_max_per_pixel
        overhead_splat = hist / config.hist_max_per_pixel
        return overhead_splat.T

    lidar = lidar[lidar[..., 2] < config.max_height_lidar]
    below = lidar[lidar[..., 2] <= config.lidar_split_height]
    above = lidar[lidar[..., 2] > config.lidar_split_height]
    below_features = splat_points(below)
    above_features = splat_points(above)
    if config.use_ground_plane:
        features = np.stack([below_features, above_features], axis=-1)
    else:
        features = np.stack([above_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features


# ========== Transfuser原始实现 ==========
class TransfuserConfig:
    pixels_per_meter = 4.0
    hist_max_per_pixel = 5
    lidar_split_height = 0.2
    min_x = -32
    max_x = 32
    min_y = -32
    max_y = 32
    max_height_lidar = 100.0

def transfuser_lidar_to_histogram_features(lidar, config, use_ground_plane):
    """Transfuser原始实现 (从data.py line 916-946)"""
    def splat_points(point_cloud):
      # 256 x 256 grid
      xbins = np.linspace(config.min_x, config.max_x,
                          (config.max_x - config.min_x) * int(config.pixels_per_meter) + 1)
      ybins = np.linspace(config.min_y, config.max_y,
                          (config.max_y - config.min_y) * int(config.pixels_per_meter) + 1)
      hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
      hist[hist > config.hist_max_per_pixel] = config.hist_max_per_pixel
      overhead_splat = hist / config.hist_max_per_pixel
      # The transpose here is an efficient axis swap.
      # Comes from the fact that carla is x front, y right, whereas the image is y front, x right
      # (x height channel, y width channel)
      return overhead_splat.T

    # Remove points above the vehicle
    lidar = lidar[lidar[..., 2] < config.max_height_lidar]
    below = lidar[lidar[..., 2] <= config.lidar_split_height]
    above = lidar[lidar[..., 2] > config.lidar_split_height]
    below_features = splat_points(below)
    above_features = splat_points(above)
    if use_ground_plane:
      features = np.stack([below_features, above_features], axis=-1)
    else:
      features = np.stack([above_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features


# ========== 测试 ==========
def test_comparison():
    print("=" * 70)
    print("LiDAR BEV处理对比: 我们的实现 vs Transfuser原始实现")
    print("=" * 70)
    
    # 创建测试数据
    np.random.seed(42)
    # 模拟真实的LiDAR点云
    num_points = 10000
    lidar = np.random.randn(num_points, 3)
    lidar[:, 0] = lidar[:, 0] * 20  # X: [-60, 60] 大部分在 [-20, 20]
    lidar[:, 1] = lidar[:, 1] * 20  # Y: [-60, 60] 大部分在 [-20, 20]
    lidar[:, 2] = lidar[:, 2] * 2 + 1  # Z: [-3, 5] 大部分在 [-1, 3]
    
    print(f"\n测试点云: {num_points} 个点")
    print(f"  X 范围: [{lidar[:, 0].min():.2f}, {lidar[:, 0].max():.2f}]")
    print(f"  Y 范围: [{lidar[:, 1].min():.2f}, {lidar[:, 1].max():.2f}]")
    print(f"  Z 范围: [{lidar[:, 2].min():.2f}, {lidar[:, 2].max():.2f}]")
    
    # 处理
    our_config = OurConfig()
    transfuser_config = TransfuserConfig()
    
    our_result = our_lidar_to_histogram_features(lidar.copy(), our_config)
    transfuser_result = transfuser_lidar_to_histogram_features(lidar.copy(), transfuser_config, use_ground_plane=False)
    
    print("\n========== 参数对比 ==========")
    params = ['pixels_per_meter', 'hist_max_per_pixel', 'lidar_split_height', 
              'min_x', 'max_x', 'min_y', 'max_y', 'max_height_lidar']
    for p in params:
        our_val = getattr(our_config, p)
        tf_val = getattr(transfuser_config, p)
        match = "✓" if our_val == tf_val else "✗"
        print(f"  {p}: 我们={our_val}, Transfuser={tf_val} {match}")
    
    print("\n========== 输出对比 ==========")
    print(f"  我们的输出形状: {our_result.shape}")
    print(f"  Transfuser输出形状: {transfuser_result.shape}")
    print(f"  形状相同: {'✓' if our_result.shape == transfuser_result.shape else '✗'}")
    
    print(f"\n  我们的输出范围: [{our_result.min():.6f}, {our_result.max():.6f}]")
    print(f"  Transfuser输出范围: [{transfuser_result.min():.6f}, {transfuser_result.max():.6f}]")
    
    # 数值对比
    is_equal = np.allclose(our_result, transfuser_result)
    max_diff = np.max(np.abs(our_result - transfuser_result))
    print(f"\n  数值完全相同 (allclose): {'✓' if is_equal else '✗'}")
    print(f"  最大差异: {max_diff}")
    
    if is_equal:
        print("\n" + "=" * 70)
        print("✓ 结论: 我们的LiDAR处理与Transfuser完全一致!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("✗ 警告: 发现差异，需要检查!")
        print("=" * 70)
        
        # 找出差异位置
        diff = np.abs(our_result - transfuser_result)
        diff_positions = np.argwhere(diff > 1e-6)
        print(f"  差异位置数量: {len(diff_positions)}")
        if len(diff_positions) > 0:
            print(f"  前5个差异位置: {diff_positions[:5]}")
    
    # 附加测试：特定位置的点
    print("\n========== 特定位置测试 ==========")
    # 只在前方(x>0)放置点
    front_points = np.array([
        [10.0, 0.0, 1.0],
        [15.0, 0.0, 1.0],
        [20.0, 0.0, 1.0],
    ])
    
    our_front = our_lidar_to_histogram_features(front_points.copy(), our_config)
    tf_front = transfuser_lidar_to_histogram_features(front_points.copy(), transfuser_config, use_ground_plane=False)
    
    print(f"  前方点测试 (x=10,15,20, y=0):")
    print(f"  输出相同: {'✓' if np.allclose(our_front, tf_front) else '✗'}")
    
    # 检查非零位置
    our_nonzero = np.argwhere(our_front > 0)
    tf_nonzero = np.argwhere(tf_front > 0)
    print(f"  我们的非零位置: {our_nonzero}")
    print(f"  Transfuser非零位置: {tf_nonzero}")

if __name__ == "__main__":
    test_comparison()
