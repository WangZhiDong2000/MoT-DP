"""
测试脚本：使用真实数据测试 TransFuser Backbone Extractor
=========================================================

使用方法:
    python test_with_real_data.py --rgb_path /path/to/rgb.jpg --lidar_path /path/to/lidar.laz
    
    或不带参数，将自动在数据集中查找测试样本
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from backbone_extractor import TransFuserBackboneExtractor


def find_test_sample(dataset_path: str):
    """
    在数据集中查找一个测试样本
    
    Returns:
        tuple: (rgb_path, lidar_path) 或 (None, None)
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        return None, None
    
    # 遍历查找第一个有效样本
    for scenario_dir in dataset_path.iterdir():
        if not scenario_dir.is_dir():
            continue
        
        for route_dir in scenario_dir.iterdir():
            if not route_dir.is_dir() or route_dir.name.startswith('FAILED_'):
                continue
            
            rgb_dir = route_dir / 'rgb'
            lidar_dir = route_dir / 'lidar'
            
            if rgb_dir.exists() and lidar_dir.exists():
                # 找第一个帧
                rgb_files = sorted(rgb_dir.glob('*.jpg'))
                lidar_files = sorted(lidar_dir.glob('*.laz'))
                
                if rgb_files and lidar_files:
                    # 找到匹配的文件
                    for rgb_file in rgb_files:
                        frame_num = rgb_file.stem
                        lidar_file = lidar_dir / f"{frame_num}.laz"
                        if lidar_file.exists():
                            return str(rgb_file), str(lidar_file)
    
    return None, None


def test_with_real_data(rgb_path: str, lidar_path: str, config_path: str, device: str = 'cuda:0'):
    """
    使用真实数据测试特征提取
    """
    print("=" * 70)
    print("TransFuser Backbone Extractor - Real Data Test")
    print("=" * 70)
    
    print(f"\nInput files:")
    print(f"  RGB: {rgb_path}")
    print(f"  LiDAR: {lidar_path}")
    
    # 检查文件是否存在
    if not os.path.exists(rgb_path):
        print(f"Error: RGB file not found: {rgb_path}")
        return False
    if not os.path.exists(lidar_path):
        print(f"Error: LiDAR file not found: {lidar_path}")
        return False
    
    # 创建提取器
    print("\nInitializing extractor...")
    extractor = TransFuserBackboneExtractor(config_path=config_path, device=device)
    
    # 预处理 RGB
    print("\nPreprocessing RGB image...")
    rgb = extractor.preprocess_rgb(rgb_path)
    print(f"  RGB tensor shape: {rgb.shape}")
    print(f"  RGB value range: [{rgb.min():.2f}, {rgb.max():.2f}]")
    
    # 预处理 LiDAR
    print("\nPreprocessing LiDAR point cloud...")
    lidar_bev = extractor.preprocess_lidar(lidar_path)
    print(f"  LiDAR BEV tensor shape: {lidar_bev.shape}")
    print(f"  LiDAR BEV value range: [{lidar_bev.min():.4f}, {lidar_bev.max():.4f}]")
    print(f"  Non-zero pixels: {(lidar_bev > 0).sum().item()}")
    
    # 提取特征
    print("\nExtracting features...")
    with torch.no_grad():
        output = extractor(rgb, lidar_bev)
    
    # 打印输出信息
    print("\n" + "-" * 50)
    print("Output Features:")
    print("-" * 50)
    
    for key, value in output.items():
        if value is not None:
            print(f"\n{key}:")
            print(f"  Shape: {value.shape}")
            print(f"  Dtype: {value.dtype}")
            print(f"  Device: {value.device}")
            print(f"  Value range: [{value.min():.4f}, {value.max():.4f}]")
            print(f"  Mean: {value.mean():.4f}")
            print(f"  Std: {value.std():.4f}")
        else:
            print(f"\n{key}: None")
    
    # 保存测试结果
    output_dir = Path(current_dir) / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving test features to {output_dir}...")
    
    if output['bev_feature'] is not None:
        save_path = output_dir / "test_bev_feature.pt"
        torch.save(output['bev_feature'].cpu(), save_path)
        print(f"  Saved: {save_path}")
    
    if output['bev_feature_upscale'] is not None:
        save_path = output_dir / "test_bev_feature_upsample.pt"
        torch.save(output['bev_feature_upscale'].cpu(), save_path)
        print(f"  Saved: {save_path}")
    
    if output['fused_features'] is not None:
        save_path = output_dir / "test_fused_features.pt"
        torch.save(output['fused_features'].cpu(), save_path)
        print(f"  Saved: {save_path}")
    
    if output['image_feature_grid'] is not None:
        save_path = output_dir / "test_image_feature_grid.pt"
        torch.save(output['image_feature_grid'].cpu(), save_path)
        print(f"  Saved: {save_path}")
    
    print("\n" + "=" * 70)
    print("Real data test completed successfully!")
    print("=" * 70)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test TransFuser Backbone Extractor with real data"
    )
    parser.add_argument(
        '--rgb_path', 
        type=str, 
        default=None,
        help='Path to RGB image file (.jpg)'
    )
    parser.add_argument(
        '--lidar_path', 
        type=str, 
        default=None,
        help='Path to LiDAR file (.laz)'
    )
    parser.add_argument(
        '--dataset_path', 
        type=str, 
        default='/home/wang/Dataset/pdm_lite_mini',
        help='Path to dataset (used to find test sample if rgb_path/lidar_path not provided)'
    )
    parser.add_argument(
        '--config_path', 
        type=str, 
        default='/home/wang/Project/carla_garage/leaderboard/leaderboard/pretrained_models/all_towns',
        help='Path to TransFuser config directory'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda:0',
        help='Device to use for inference'
    )
    
    args = parser.parse_args()
    
    # 如果没有指定路径，尝试从数据集中找
    rgb_path = args.rgb_path
    lidar_path = args.lidar_path
    
    if rgb_path is None or lidar_path is None:
        print("No input files specified, searching in dataset...")
        found_rgb, found_lidar = find_test_sample(args.dataset_path)
        
        if found_rgb is None:
            print(f"Error: Could not find test sample in {args.dataset_path}")
            print("Please specify --rgb_path and --lidar_path manually")
            return
        
        rgb_path = found_rgb
        lidar_path = found_lidar
        print(f"Found test sample:")
        print(f"  RGB: {rgb_path}")
        print(f"  LiDAR: {lidar_path}")
    
    # 运行测试
    test_with_real_data(
        rgb_path=rgb_path,
        lidar_path=lidar_path,
        config_path=args.config_path,
        device=args.device
    )


if __name__ == "__main__":
    main()
