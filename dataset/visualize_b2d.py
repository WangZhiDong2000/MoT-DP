# -*- coding: utf-8 -*-
"""
随机可视化预处理后的轨迹数据
"""
import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

PREPROCESS_DIR = '/home/wang/Dataset/b2d_10scene/smoothed_data_b2d'


def load_pkl_file(filepath):
    """加载pickle文件"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def group_trajectories_by_scene(pkl_dir):
    """按场景分组轨迹数据，构建完整轨迹"""
    trajectories_by_scene = defaultdict(list)
    
    # 列出所有pkl文件
    pkl_files = sorted([f for f in os.listdir(pkl_dir) if f.endswith('.pkl')])
    
    for filename in pkl_files:
        # 解析文件名：Accident_Town03_Route101_Weather23_00000_train.pkl
        # 需要找到最后一个下划线前的数字作为frameID
        parts = filename.replace('.pkl', '').rsplit('_', 1)
        if len(parts) == 2:
            scene_and_frame = parts[0]
            split_parts = scene_and_frame.rsplit('_', 1)
            if len(split_parts) == 2 and split_parts[1].isdigit():
                scene_id = split_parts[0]
                frame_id = int(split_parts[1])
            else:
                continue
        else:
            continue
            
        filepath = os.path.join(pkl_dir, filename)
        data = load_pkl_file(filepath)
        
        trajectories_by_scene[scene_id].append({
            'filename': filename,
            'frame_id': frame_id,
            'ego_world_smooth': data.get('ego_world_smooth'),
            'speed': data.get('speed'),
            'theta': data.get('theta'),
            'data': data
        })
    
    # 按frame_id排序每个场景的轨迹点
    for scene_id in trajectories_by_scene:
        trajectories_by_scene[scene_id].sort(key=lambda x: x['frame_id'])
    
    return trajectories_by_scene


def visualize_trajectories(trajectories_by_scene, num_samples=5):
    """随机可视化轨迹"""
    scene_ids = list(trajectories_by_scene.keys())
    
    # 随机选择场景
    sampled_scenes = random.sample(scene_ids, min(num_samples, len(scene_ids)))
    
    num_plots = len(sampled_scenes)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
    
    if num_plots == 1:
        axes = [axes]
    
    for idx, scene_id in enumerate(sampled_scenes):
        traj_points = trajectories_by_scene[scene_id]
        
        # 提取有效的轨迹点
        valid_points = [
            p['ego_world_smooth'] 
            for p in traj_points 
            if p['ego_world_smooth'] is not None
        ]
        
        if len(valid_points) < 2:
            print(f"Scene {scene_id} has less than 2 valid points, skipping...")
            continue
        
        valid_points = np.array(valid_points)
        
        ax = axes[idx]
        
        # 绘制轨迹
        ax.plot(valid_points[:, 0], valid_points[:, 1], 'b-', linewidth=2, label='Trajectory')
        ax.scatter(valid_points[0, 0], valid_points[0, 1], color='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(valid_points[-1, 0], valid_points[-1, 1], color='red', s=100, marker='s', label='End', zorder=5)
        
        # 绘制方向箭头（每N个点绘制一个）
        step = max(1, len(valid_points) // 5)
        for i in range(0, len(valid_points) - 1, step):
            dx = valid_points[i+1, 0] - valid_points[i, 0]
            dy = valid_points[i+1, 1] - valid_points[i, 1]
            ax.arrow(valid_points[i, 0], valid_points[i, 1], dx, dy, 
                    head_width=0.5, head_length=0.3, fc='orange', ec='orange', alpha=0.6)
        
        # 获取速度信息
        speeds = [p['speed'] for p in traj_points if p['speed'] is not None]
        avg_speed = np.mean(speeds) if speeds else 0
        max_speed = np.max(speeds) if speeds else 0
        
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_title(f'Scene: {scene_id}\nFrames: {len(valid_points)} | Avg Speed: {avg_speed:.2f} m/s | Max Speed: {max_speed:.2f} m/s', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.axis('equal')
        
        print(f"Visualizing scene: {scene_id}")
        print(f"  - Number of trajectory points: {len(valid_points)}")
        print(f"  - Average speed: {avg_speed:.2f} m/s")
        print(f"  - Trajectory length: {np.sum(np.linalg.norm(np.diff(valid_points, axis=0), axis=1)):.2f} m")
        print()
    
    plt.tight_layout()
    return fig


def main():
    print("Loading trajectories from preprocessed data...")
    trajectories_by_scene = group_trajectories_by_scene(PREPROCESS_DIR)
    
    print(f"Total scenes: {len(trajectories_by_scene)}")
    for scene_id, traj_list in list(trajectories_by_scene.items())[:5]:
        valid_count = sum(1 for t in traj_list if t['ego_world_smooth'] is not None)
        print(f"  - {scene_id}: {len(traj_list)} frames ({valid_count} valid)")
    print()
    
    # 随机可视化5个场景
    print("Visualizing 5 random trajectories...\n")
    fig = visualize_trajectories(trajectories_by_scene, num_samples=5)
    
    # 保存图片
    output_path = '/home/wang/Project/MoT-DP/visualization_trajectories.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    main()