
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import io
import sys
import pickle
import glob
from tqdm import tqdm  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import textwrap


class CARLAImageDataset(torch.utils.data.Dataset):
    """
    Unified Dataset for preprocessed PDM Lite and Bench2Drive (B2D) data in 'frame' mode.
    
    Each pickle file is a complete training sample with image paths.
    Images are loaded dynamically in __getitem__.
    
    Supports both:
    - PDM Lite: Basic driving data with waypoints and sensors
    - Bench2Drive (B2D): Extended driving data with VQA and meta-actions
    
    Args:
        dataset_path (str): Directory containing individual frame pkl files
        image_data_root (str): Base path for images
    """
    
    def __init__(self,
                 dataset_path: str,
                 image_data_root: str,
                 mode: str = 'train'        # train or val
                 ):

        self.image_data_root = image_data_root
        self.dataset_path = dataset_path
        self.mode = mode

        self.image_transform = transforms.Compose([
            transforms.Resize((256, 928)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.lidar_bev_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Processed data directory not found: {dataset_path}")

        # Load pkl files - support both direct directory and train/val subdirectories
        train_files = glob.glob(os.path.join(dataset_path, "train", "*.pkl"))
        val_files = glob.glob(os.path.join(dataset_path, "val", "*.pkl"))
        direct_files = glob.glob(os.path.join(dataset_path, "*.pkl"))
        
        if train_files or val_files:
            self.sample_files = sorted(train_files + val_files)
            print(f"Found {len(self.sample_files)} preprocessed samples in '{dataset_path}' "
                  f"({len(train_files)} train, {len(val_files)} val).")
        elif direct_files:
            self.sample_files = sorted(direct_files)
            print(f"Found {len(self.sample_files)} preprocessed samples in '{dataset_path}'.")
        else:
            raise FileNotFoundError(f"No pkl files found in {dataset_path} or its train/val subdirectories.")

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        # Load the pickle file
        sample_path = self.sample_files[idx]
        with open(sample_path, 'rb') as f:
            sample = pickle.load(f)

        # load image data for visualization 
        if self.mode == 'val':
            image_paths = sample.get('rgb_hist_jpg', [])
            images_tensor = self.load_image(image_paths, sample_path)


        # Load LiDAR BEV features
        lidar_bev_paths = sample.get('lidar_bev_hist', [])
        if self.mode == 'val':
            lidar_bev_tensor = self.load_lidar_bev(lidar_bev_paths, sample_path)
        
        lidar_tokens = []
        lidar_tokens_global = []
        for bev_path in lidar_bev_paths:
            if bev_path is None:
                continue
            
            # Infer feature paths from BEV image paths
            # Example: Town01/data/Route_0/lidar_bev/0000.png 
            #       -> Town01/data/Route_0/lidar_bev_features/0000_token.pt
            bev_dir = os.path.dirname(bev_path)
            frame_id = os.path.splitext(os.path.basename(bev_path))[0]
            feature_dir = bev_dir.replace('lidar_bev', 'lidar_bev_features')
                
            token_path = os.path.join(self.image_data_root, feature_dir, f'{frame_id}_token.pt')
            token_global_path = os.path.join(self.image_data_root, feature_dir, f'{frame_id}_token_global.pt')

            try:
                lidar_token = torch.load(token_path, weights_only=True)
                lidar_token_global = torch.load(token_global_path, weights_only=True)
                lidar_tokens.append(lidar_token)
                lidar_tokens_global.append(lidar_token_global)
            except FileNotFoundError:
                print(f"Warning: LiDAR feature files not found for {bev_path}")
                continue
        

        lidar_token_tensor = torch.stack(lidar_tokens)  # (obs_horizon, seq_len, 512)
        lidar_token_global_tensor = torch.stack(lidar_tokens_global)  # (obs_horizon, 1, 512)
        
        # Load VQA feature from pt file
        vqa_path = sample.get('vqa', None)
        vqa_feature = {}
        if vqa_path is not None:
            full_vqa_path = os.path.join(self.image_data_root, vqa_path)
            try:
                vqa_feature_data = torch.load(full_vqa_path, weights_only=True)
                # Check if loaded data is a dict or tensor
                if isinstance(vqa_feature_data, dict):
                    vqa_feature = vqa_feature_data
                else:
                    # If it's a tensor, assume it's gen_vit_tokens
                    vqa_feature = {'gen_vit_tokens': vqa_feature_data}
                    print(f"Warning: VQA feature at {full_vqa_path} is a tensor, expected dict. Stored as 'gen_vit_tokens'.")
            except FileNotFoundError:
                print(f"Warning: VQA feature file not found at {full_vqa_path}")
                vqa_feature = {}
  
        
        # Convert sample data
        final_sample = dict()
        for key, value in sample.items():
            if key == 'rgb_hist_jpg' and self.mode == 'val':
                final_sample['rgb_hist_jpg'] = image_paths  
                final_sample['image'] = images_tensor
            elif key == 'lidar_bev_hist':
                if self.mode == 'val':
                    final_sample['lidar_bev'] = lidar_bev_tensor
                    final_sample['lidar_bev_hist'] = lidar_bev_paths
                if lidar_token_tensor is not None:
                    final_sample['lidar_token'] = lidar_token_tensor
                    final_sample['lidar_token_global'] = lidar_token_global_tensor
            elif key == 'speed_hist':
                speed_data = sample['speed_hist']
                final_sample['speed'] = torch.from_numpy(speed_data).float()
            elif key == 'ego_waypoints':
                final_sample['agent_pos'] = torch.from_numpy(sample['ego_waypoints'][1:]).float()
            elif key == 'vqa':
                # Add the VQA features if available
                if isinstance(vqa_feature, dict):
                    if 'gen_vit_tokens' in vqa_feature:
                        final_sample['gen_vit_tokens'] = vqa_feature['gen_vit_tokens']
                    if 'reasoning_query_tokens' in vqa_feature:
                        final_sample['reasoning_query_tokens'] = vqa_feature['reasoning_query_tokens']
                    # Handle variable-length answer_token_indexes by padding to fixed size
                    if 'answer_token_indexes' in vqa_feature:
                        answer_tokens = vqa_feature['answer_token_indexes']
                        max_answer_tokens = 8  # Fixed maximum length for padding
                        if answer_tokens.shape[0] < max_answer_tokens:
                            # Pad with -1 (or any invalid token index)
                            padding = torch.full((max_answer_tokens - answer_tokens.shape[0],), -1, dtype=answer_tokens.dtype)
                            final_sample['answer_token_indexes'] = torch.cat([answer_tokens, padding])
                        elif answer_tokens.shape[0] > max_answer_tokens:
                            # Truncate if longer than max
                            final_sample['answer_token_indexes'] = answer_tokens[:max_answer_tokens]
                        else:
                            final_sample['answer_token_indexes'] = answer_tokens


            elif isinstance(value, np.ndarray):
                final_sample[key] = torch.from_numpy(value).float()
            else:
                final_sample[key] = value

        # Build ego_status: concatenate historical low-dimensional states
        # Order: speed_hist, theta_hist, throttle_hist, brake_hist, command_hist, waypoints_hist
        ego_status_components = []
        
        # 1. speed_hist
        speed_data = final_sample['speed']
        ego_status_components.append(speed_data.unsqueeze(-1))  # (obs_horizon, 1)
        
        # 2. theta_hist
        theta_data = final_sample['theta_hist']
        ego_status_components.append(theta_data.unsqueeze(-1))  # (obs_horizon, 1)
        
        throttle_data = final_sample['throttle_hist']
        ego_status_components.append(throttle_data.unsqueeze(-1))  # (obs_horizon, 1)

        brake_data = final_sample['brake_hist']
        ego_status_components.append(brake_data.unsqueeze(-1))  # (obs_horizon, 1)
        
        # 5. command_hist (one-hot, shape: (obs_horizon, 6))
        command_data = final_sample['command_hist']
        ego_status_components.append(command_data)  # (obs_horizon, 6)

        # 5. target point
        command_data = final_sample['target_point_hist']
        ego_status_components.append(command_data)  # (obs_horizon, 2)
        
        # 6. waypoints_hist (shape: (obs_horizon, 2))
        waypoints_data = final_sample['waypoints_hist']
        ego_status_components.append(waypoints_data)  # (obs_horizon, 2)
        
        # Concatenate all components along the feature dimension
        final_sample['ego_status'] = torch.cat(ego_status_components, dim=-1)  # (obs_horizon, feature_dim)

        return final_sample



    def load_image(self, image_paths, sample_path):
        images = []
        for img_path in image_paths:
            full_img_path = os.path.join(self.image_data_root, img_path)
            try:
                img = Image.open(full_img_path)
                img_tensor = self.image_transform(img)
                images.append(img_tensor)
            except Exception as e:
                print(f"Error loading image {full_img_path}: {e}")
                images.append(torch.zeros(3, 256, 928))
        
        if len(images) > 0:
            images_tensor = torch.stack(images)
        else:
            images_tensor = torch.zeros(2, 3, 256, 928)  # 默认obs_horizon=2

        return images_tensor

    def load_lidar_bev(self, bev_paths, sample_path):
        images = []
        for bev_path in bev_paths:
            full_bev_path = os.path.join(self.image_data_root, bev_path)
            bev_image = Image.open(full_bev_path)
            bev_tensor = self.lidar_bev_transform(bev_image)
            images.append(bev_tensor)

        images_tensor = torch.stack(images)

        return images_tensor

# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_trajectory(sample, obs_horizon, rand_idx, save_dir='/root/z_projects/code/MoT-DP-1/image'):

    agent_pos = sample.get('agent_pos')
    waypoints_hist = sample.get('waypoints_hist')
    target_point = sample.get('target_point')
    target_point_hist = sample.get('target_point_hist')
    
    if isinstance(agent_pos, torch.Tensor):
        agent_pos = agent_pos.numpy()
    if isinstance(waypoints_hist, torch.Tensor):
        waypoints_hist = waypoints_hist.numpy()
    if isinstance(target_point, torch.Tensor):
        target_point = target_point.numpy()
    if isinstance(target_point_hist, torch.Tensor):
        target_point_hist = target_point_hist.numpy()
    
    plt.figure(figsize=(12, 12))
    
    # ========== Plot Historical Waypoints ==========
    if waypoints_hist is not None and len(waypoints_hist) > 0:
        # waypoints_hist shape: (obs_horizon, 2)
        # Plot line connecting historical points
        plt.plot(waypoints_hist[:, 1], waypoints_hist[:, 0], 'b-', 
                linewidth=1.5, alpha=0.5, label='History trajectory', zorder=2)
        
        # Plot each historical waypoint as discrete point
        for i, waypoint in enumerate(waypoints_hist[:-1]):  # Exclude last (current frame)
            plt.plot(waypoint[1], waypoint[0], 'bo', markersize=8, zorder=3)
    
    # ========== Plot Historical Target Points ==========
    if target_point_hist is not None and len(target_point_hist) > 0:
        # target_point_hist shape: (obs_horizon, 2)
        # Plot each historical target point with different colors based on time
        colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(target_point_hist)))
        for i, target_pt in enumerate(target_point_hist):
            plt.plot(target_pt[1], target_pt[0], 'D', color=colors[i], 
                    markersize=6, alpha=0.7, zorder=3)
            # Add text label for frame index
            plt.text(target_pt[1], target_pt[0] + 1.5, f't-{len(target_point_hist)-1-i}', 
                    fontsize=8, ha='center', alpha=0.6)
    
    # ========== Plot Current Position (Origin) ==========
    plt.plot(0, 0, 'ko', markersize=15, label='Current position (t=0)', 
            markeredgecolor='yellow', markeredgewidth=2, zorder=5)
    
    # ========== Plot Future Waypoints (Predictions) ==========
    if len(agent_pos) > 0:
        # Plot line connecting future points
        future_waypoints = np.vstack([[[0, 0]], agent_pos])
        plt.plot(future_waypoints[:, 1], future_waypoints[:, 0], 'r--', 
                linewidth=2, alpha=0.6, label='Predicted trajectory', zorder=2)
        
        # Plot each future waypoint as discrete point
        for i, waypoint in enumerate(agent_pos, 1):
            plt.plot(waypoint[1], waypoint[0], 'r^', markersize=10, 
                    markeredgecolor='darkred', markeredgewidth=1, zorder=4)
    
    # ========== Plot Current Target Point ==========
    if target_point is not None:
        if target_point.ndim == 2:
            target_point = target_point[0]
        plt.plot(target_point[1], target_point[0], 'g*', markersize=30, 
                label='Current target point (t=0)', markeredgecolor='darkgreen', markeredgewidth=2, zorder=6)
    
    # ========== Formatting ==========
    plt.xlabel('Y (ego frame, lateral / m)', fontsize=12, fontweight='bold')
    plt.ylabel('X (ego frame, longitudinal / m)', fontsize=12, fontweight='bold')
    plt.title(f'Sample {rand_idx}: Trajectory with History Target Points', fontsize=13, fontweight='bold')
    
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Add legend with custom formatting
    plt.legend(fontsize=10, loc='best', framealpha=0.95, edgecolor='black')
    
    # Add axis line at origin
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Set axis range to be square and centered at origin
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    
    # Calculate the maximum range needed
    x_range = max(abs(xlim[0]), abs(xlim[1]))
    y_range = max(abs(ylim[0]), abs(ylim[1]))
    max_range = max(x_range, y_range, 1.0)
    
    # Set symmetric square limits around origin
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'sample_{rand_idx}_trajectory.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ 保存轨迹可视化到: {save_path}")


def visualize_observation_images(sample, obs_horizon, rand_idx, save_dir='/root/z_projects/code/MoT-DP-1/image'):

    images = sample['image']  # shape: (obs_horizon, C, H, W)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    os.makedirs(save_dir, exist_ok=True)
    
    for t, img_tensor in enumerate(images):
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor * std + mean
            img_tensor = torch.clamp(img_tensor, 0, 1)
            img_arr = img_tensor.numpy()
        else:
            img_arr = img_tensor
            
        if img_arr.shape[0] == 3:
            img_vis = np.moveaxis(img_arr, 0, -1)
            img_vis = (img_vis * 255).astype(np.uint8)
        else:
            img_vis = img_arr.astype(np.uint8)
        
        plt.figure(figsize=(20, 5))
        plt.imshow(img_vis)
        plt.title(f'Random Sample {rand_idx} - Obs Image t={t}', fontsize=12)
        plt.axis('off')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)

        save_path = os.path.join(save_dir, f'sample_{rand_idx}_obs_image_t{t}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存观测图像到: {save_path}")




def print_sample_details(sample, dataset, rand_idx, obs_horizon, 
                        save_dir='/home/wang/Project/MoT-DP/image'):
    print(f"\n样本 {rand_idx} 的详细信息:")
    
    agent_pos = sample.get('agent_pos')
    if agent_pos is not None:
        obs_agent_pos = agent_pos[:obs_horizon]
        pred_agent_pos = agent_pos[obs_horizon:]
        print(f"观测位置: {obs_agent_pos}")
        print(f"预测位置: {pred_agent_pos}")

    target_point = sample.get('target_point')
    if target_point is not None:
        if target_point.ndim == 2:
            target_point = target_point[0]
        print(f"目标点 (相对): {target_point}")
        if agent_pos is not None and len(agent_pos) > obs_horizon - 1:
            last_obs = agent_pos[obs_horizon - 1]
            distance_to_target = np.linalg.norm(target_point - last_obs)
            print(f"目标点距离最后观测点的距离: {distance_to_target:.3f}")

    target_point_hist = sample.get('target_point_hist')
    if target_point_hist is not None:
        print(f"\n历史目标点 (target_point_hist):")
        print(f"  形状: {target_point_hist.shape}")
        for i, tp in enumerate(target_point_hist):
            frame_idx = len(target_point_hist) - 1 - i
            print(f"  t-{frame_idx}: {tp}")

    print(f"\nDataset length: {len(dataset)}")
    first_sample = dataset[0]
    print("Sample keys:", first_sample.keys())
    if 'image' in first_sample:
        print("Image shape:", first_sample['image'].shape)
    if 'agent_pos' in first_sample:
        print("Agent pos shape:", first_sample['agent_pos'].shape)

    print("\nKey fields:")
    for key in ['town_name', 'speed', 'command', 'next_command', 'target_point', 
                'target_point_hist', 'ego_waypoints', 'image', 'agent_pos', 'meta_action_direction', 'meta_action_speed', 'gen_vit_tokens']:
        if key in first_sample:
            value = first_sample[key]
            print(f"  {key}: shape={getattr(value, 'shape', 'N/A')}, type={type(value)}")
            if hasattr(value, '__len__') and not isinstance(value, str):
                print(f"    Length: {len(value)}")


# ============================================================================
# Test Function
# ============================================================================

def test_pdm():
    """Test with PDM Lite dataset."""
    import random
    
    dataset_path = '/share-data/pdm_lite/tmp_data/train'
    obs_horizon = 4
    """
    Prints detailed information about a sample and the dataset.
    
    Args:
        sample (dict): Data sample
        dataset (CARLAImageDataset): Dataset object
        rand_idx (int): Sample index
        obs_horizon (int): Number of observation frames
        save_dir (str): Directory for saving visualizations
    """
    print("\n========== Testing PDM Lite Dataset ==========")
    dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        image_data_root='/share-data/pdm_lite/'
    )
    
    print(f"\n总样本数: {len(dataset)}")
    if len(dataset) == 0:
        print("数据为空，无法进行测试。")
        return

    rand_idx = random.choice(range(len(dataset)))
    rand_sample = dataset[rand_idx]
    print(f"\n随机选择的样本索引: {rand_idx}")

    print("\nSample keys:", rand_sample.keys())
    for key, value in rand_sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}")

    visualize_trajectory(rand_sample, obs_horizon, rand_idx)
    print_sample_details(rand_sample, dataset, rand_idx, obs_horizon)


    
if __name__ == "__main__":
    test_pdm()

