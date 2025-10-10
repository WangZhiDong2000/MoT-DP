#!/usr/bin/env python3
"""
查看测试结果的脚本
"""
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread

def display_test_results():
    """显示测试结果的可视化"""
    
    # 测试结果目录
    result_dir = '/home/wang/projects/diffusion_policy_z/image/test_results'
    
    # 读取测试指标
    metrics_file = os.path.join(result_dir, 'test_metrics.txt')
    print("="*70)
    print("测试结果摘要")
    print("="*70)
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            print(f.read())
    else:
        print("未找到测试指标文件")
        return
    
    # 获取所有可视化图像
    image_files = sorted([f for f in os.listdir(result_dir) if f.endswith('_prediction.png')])
    
    if not image_files:
        print("\n未找到可视化图像")
        return
    
    print("="*70)
    print(f"生成的可视化图像数量: {len(image_files)}")
    print("="*70)
    
    # 显示前4个样本的可视化结果
    num_to_show = min(4, len(image_files))
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for i in range(num_to_show):
        img_path = os.path.join(result_dir, image_files[i])
        img = imread(img_path)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Sample: {image_files[i].replace('sample_', '').replace('_prediction.png', '')}", 
                         fontsize=14, fontweight='bold')
    
    # 隐藏未使用的子图
    for i in range(num_to_show, 4):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # 保存组合图像
    combined_path = os.path.join(result_dir, 'combined_results.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 组合结果图已保存到: {combined_path}")
    
    plt.show()
    
    print("\n所有可视化图像文件:")
    for img_file in image_files:
        print(f"  - {os.path.join(result_dir, img_file)}")
    
    print(f"\n{'='*70}")
    print("测试完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    display_test_results()
