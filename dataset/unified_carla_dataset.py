
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


class CARLAImageDataset(torch.utils.data.Dataset):
    """
    Dataset for CARLA data aligned with Transfuser format.
    Loads:
      - rgb: current frame RGB image path (.jpg)
      - lidar: current frame LiDAR point cloud path (.laz)
      - Low-dimensional states (speed_hist, theta_hist, command_hist, etc.)
    """
    
    def __init__(self,
                 dataset_path: str,
                 data_root: str,
                 ):
        """
        Args:
            dataset_path: Path to preprocessed pkl files
            data_root: Root path where raw data (rgb, lidar) is stored
        """
        self.data_root = data_root
        self.dataset_path = dataset_path

        # Image transform aligned with Transfuser:
        # - Original: 1024x512 (W x H)
        # - Crop to: 1024x384 (crop top 384 pixels, keep full width)
        # - Convert to tensor: (C, H, W) = (3, 384, 1024)
        # - Keep values in [0, 255] range (Transfuser normalizes inside model)
        # Note: Transfuser uses cv2 to load and np.transpose, we use PIL + custom transform
        self.crop_height = 384
        self.crop_width = 1024
    
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

        # Convert sample data
        final_sample = dict()
        
        for key, value in sample.items():
            if key == 'rgb':
                # Store RGB image path (relative path)
                final_sample['rgb_path'] = value
                # Load and transform RGB image
                full_rgb_path = os.path.join(self.data_root, value)
                rgb_tensor = self.load_image(full_rgb_path)
                final_sample['rgb'] = rgb_tensor
                
            elif key == 'lidar':
                # Store LiDAR path (relative path to .laz file)
                final_sample['lidar_path'] = value
                # Note: LiDAR point cloud loading is handled separately
                # as it requires laspy and special processing
                
            elif key == 'speed_hist':
                speed_data = sample['speed_hist']
                final_sample['speed'] = torch.from_numpy(speed_data).float()
                
            elif key == 'ego_waypoints':
                # agent_pos excludes the current position (index 0)
                final_sample['agent_pos'] = torch.from_numpy(sample['ego_waypoints'][1:]).float()
                final_sample['ego_waypoints'] = torch.from_numpy(sample['ego_waypoints']).float()
                
            elif isinstance(value, np.ndarray):
                final_sample[key] = torch.from_numpy(value).float()
            else:
                final_sample[key] = value

        # Build ego_status: concatenate historical low-dimensional states
        # Order: speed_hist, theta_hist, command_hist, target_point_hist, waypoints_hist
        ego_status_components = []
        
        # 1. speed_hist
        speed_data = final_sample['speed']
        ego_status_components.append(speed_data.unsqueeze(-1))  # (obs_horizon, 1)
        
        # 2. theta_hist
        theta_data = final_sample['theta_hist']
        ego_status_components.append(theta_data.unsqueeze(-1))  # (obs_horizon, 1)
        
        # 3. command_hist (one-hot, shape: (obs_horizon, 6))
        command_data = final_sample['command_hist']
        ego_status_components.append(command_data)  # (obs_horizon, 6)

        # 4. target_point_hist
        target_point_data = final_sample['target_point_hist']
        ego_status_components.append(target_point_data)  # (obs_horizon, 2)
        
        # 5. waypoints_hist (shape: (obs_horizon, 2))
        waypoints_data = final_sample['waypoints_hist']
        ego_status_components.append(waypoints_data)  # (obs_horizon, 2)
        
        # Concatenate all components along the feature dimension
        final_sample['ego_status'] = torch.cat(ego_status_components, dim=-1)  # (obs_horizon, 12)

        return final_sample

    def load_image(self, image_path):
        """
        Load and transform a single RGB image (aligned with Transfuser data.py).
        - Original: 1024x512 (W x H)
        - Crop to: 1024x384 (top 384 pixels)
        - Output: (3, 384, 1024) tensor with values in [0, 255]
        
        Note: Transfuser normalizes images inside the model (normalize_imagenet),
        so we keep values in [0, 255] range here.
        """
        try:
            # Load image using PIL (RGB format)
            img = Image.open(image_path).convert('RGB')
            
            # Crop top 384 pixels (same as Transfuser's crop_array)
            # PIL crop: (left, upper, right, lower)
            img_cropped = img.crop((0, 0, self.crop_width, self.crop_height))
            
            # Convert to numpy array (H, W, C) with values [0, 255]
            img_np = np.array(img_cropped, dtype=np.float32)
            
            # Transpose to (C, H, W) format (same as Transfuser's np.transpose)
            img_np = np.transpose(img_np, (2, 0, 1))
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img_np)
            
            img.close()
            return img_tensor
        except Exception as e:
            print(f"Error loading image {image_path}: {e}!!!!!")
            return torch.zeros(3, self.crop_height, self.crop_width)  # (C, H, W)

# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_trajectory(sample, obs_horizon, rand_idx, save_dir='/home/wang/Project/MoT-DP/image'):

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
    
    # Skip tight_layout due to numpy/matplotlib compatibility issues
    # plt.tight_layout() is already handled by bbox_inches='tight' in savefig
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'sample_{rand_idx}_trajectory.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ 保存轨迹可视化到: {save_path}")


def visualize_observation_images(sample, rand_idx, save_dir='/home/wang/Project/MoT-DP/image'):
    """Visualize the RGB image from a sample."""
    rgb = sample.get('rgb')  # shape: (C, H, W) = (3, 384, 1024)
    
    if rgb is None:
        print("No RGB image found in sample")
        return
    
    print(f"RGB tensor shape: {rgb.shape if isinstance(rgb, torch.Tensor) else 'not a tensor'}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    if isinstance(rgb, torch.Tensor):
        img_arr = rgb.numpy()
    else:
        img_arr = rgb
    
    print(f"Image array shape: {img_arr.shape}, dtype: {img_arr.dtype}")
    print(f"Image array range: [{img_arr.min():.3f}, {img_arr.max():.3f}]")
        
    if img_arr.ndim == 3 and img_arr.shape[0] == 3:
        # Convert from (C, H, W) to (H, W, C)
        img_vis = np.moveaxis(img_arr, 0, -1)
    else:
        img_vis = img_arr

    # Ensure uint8
    if img_vis.dtype != np.uint8:
        img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)
    
    print(f"Visualization shape: {img_vis.shape}, dtype: {img_vis.dtype}")
    print(f"Visualization range: [{img_vis.min()}, {img_vis.max()}]")
    
    # Save using PIL directly
    from PIL import Image
    img_pil = Image.fromarray(img_vis)
    save_path = os.path.join(save_dir, f'sample_{rand_idx}_rgb.png')
    img_pil.save(save_path)
    print(f"✓ 保存RGB图像到: {save_path}")




def print_sample_details(sample, dataset, rand_idx, obs_horizon, 
                        save_dir='/home/wang/Project/MoT-DP/image'):
    """Print detailed information about a sample."""
    print(f"\n样本 {rand_idx} 的详细信息:")
    
    # RGB and LiDAR paths
    rgb_path = sample.get('rgb_path')
    lidar_path = sample.get('lidar_path')
    if rgb_path:
        print(f"RGB路径: {rgb_path}")
    if lidar_path:
        print(f"LiDAR路径: {lidar_path}")
    
    agent_pos = sample.get('agent_pos')
    if agent_pos is not None:
        print(f"\n未来轨迹点 (agent_pos): shape={agent_pos.shape}")
        print(f"  {agent_pos}")

    target_point = sample.get('target_point')
    if target_point is not None:
        if target_point.ndim == 2:
            target_point = target_point[0]
        print(f"\n当前目标点 (target_point): {target_point}")

    target_point_hist = sample.get('target_point_hist')
    if target_point_hist is not None:
        print(f"\n历史目标点 (target_point_hist): shape={target_point_hist.shape}")
        for i, tp in enumerate(target_point_hist):
            frame_idx = len(target_point_hist) - 1 - i
            print(f"  t-{frame_idx}: {tp}")

    # ego_status info
    ego_status = sample.get('ego_status')
    if ego_status is not None:
        print(f"\nego_status: shape={ego_status.shape}")
        print(f"  (包含: speed, theta, command_one_hot, target_point_hist, waypoints_hist)")

    print(f"\n========== Dataset Info ==========")
    print(f"Dataset length: {len(dataset)}")
    first_sample = dataset[0]
    print("Sample keys:", list(first_sample.keys()))
    
    print("\nKey fields:")
    for key in ['rgb', 'lidar_path', 'speed', 'command', 'next_command', 'target_point', 
                'target_point_hist', 'ego_waypoints', 'agent_pos', 'ego_status',
                'waypoints_hist', 'speed_hist', 'theta_hist', 'command_hist']:
        if key in first_sample:
            value = first_sample[key]
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: type={type(value).__name__}")


# ============================================================================
# Test Function
# ============================================================================

def test_pdm():
    """Test with PDM Lite dataset (aligned with Transfuser format)."""
    import random
    
    # Update these paths according to your setup
    dataset_path = '/home/wang/Project/carla_garage/data/tmp_data/val'
    data_root = '/home/wang/Project/carla_garage/data'
    obs_horizon = 4
    
    print("\n========== Testing PDM Lite Dataset (Transfuser Format) ==========")
    dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        data_root=data_root
    )
    
    print(f"\n总样本数: {len(dataset)}")
    if len(dataset) == 0:
        print("数据为空，无法进行测试。")
        return

    rand_idx = random.choice(range(len(dataset)))
    rand_sample = dataset[rand_idx]
    print(f"\n随机选择的样本索引: {rand_idx}")

    print("\n========== Sample Keys ==========")
    for key, value in rand_sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, str):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: type={type(value).__name__}")

    # Print detailed sample info
    print_sample_details(rand_sample, dataset, rand_idx, obs_horizon)
    
    # Visualize trajectory
    visualize_trajectory(rand_sample, obs_horizon, rand_idx)
    
    # Visualize RGB image
    visualize_observation_images(rand_sample, rand_idx)

    
if __name__ == "__main__":
    test_pdm()


