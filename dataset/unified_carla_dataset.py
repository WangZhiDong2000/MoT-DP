
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
import cv2
import laspy


class CARLAImageDataset(torch.utils.data.Dataset):
    """
    Dataset for CARLA data strictly aligned with Transfuser format.
    
    Supports two modes:
      - 'raw': Load raw RGB images and LiDAR point clouds, process them on-the-fly
      - 'feature': Load pre-extracted Transfuser features (paths only, not values)
    
    RGB Processing (same as Transfuser data.py):
      1. cv2.imread -> BGR format
      2. cv2.cvtColor BGR->RGB
      3. crop_array: crop to (384, 1024) from top
      4. np.transpose to (C, H, W) = (3, 384, 1024)
      5. Values in [0, 255] range (normalization happens inside model)
    
    LiDAR Processing (same as Transfuser data.py):
      1. laspy.read -> xyz point cloud (N, 3)
      2. lidar_to_histogram_features -> (2, 256, 256) BEV histogram
         - Split by height (below/above lidar_split_height)
         - Voxelize to 256x256 grid with pixels_per_meter=4.0
         - Range: x=[-32,32], y=[-32,32]
    """
    
    def __init__(self,
                 dataset_path: str,
                 data_root: str,
                 mode: str = 'raw',
                 ):
        """
        Args:
            dataset_path: Path to preprocessed pkl files
            data_root: Root path where raw data (rgb, lidar) is stored
            mode: 'raw' to load and process raw RGB/LiDAR, 
                  'feature' to load pre-extracted feature paths
        """
        assert mode in ['raw', 'feature'], f"mode must be 'raw' or 'feature', got '{mode}'"
        self.mode = mode
        self.data_root = data_root
        self.dataset_path = dataset_path

        # ========== Transfuser Config Parameters (from config.py) ==========
        # All parameters must match exactly with Transfuser to use pretrained backbone
        
        # Camera and Image cropping (lines 360-366)
        self.camera_width = 1024
        self.camera_height = 512
        self.camera_fov = 110
        self.crop_image = True
        self.cropped_height = 384  # crops off the bottom part
        self.cropped_width = 1024  # crops off both sides symmetrically
        
        # Dataloader parameters (lines 370-382)
        self.carla_fps = 20
        self.seq_len = 1
        self.img_seq_len = 1
        self.lidar_seq_len = 1
        self.lidar_resolution_width = 256
        self.lidar_resolution_height = 256
        self.crop_bev = False
        self.crop_bev_height_only_from_behind = False
        
        # LiDAR voxelization parameters (lines 392-410)
        self.pixels_per_meter = 4.0
        self.pixels_per_meter_collection = 2.0
        self.hist_max_per_pixel = 5
        self.lidar_split_height = 0.2  # Height to split LiDAR into 2 channels (relative to lidar_pos[2])
        self.realign_lidar = True
        self.use_ground_plane = False  # Whether to use ground plane channel
        
        # LiDAR range for voxelization (lines 401-410)
        self.min_x = -32
        self.max_x = 32
        self.min_y = -32
        self.max_y = 32
        self.min_z = -4
        self.max_z = 4
        self.min_z_projection = -10
        self.max_z_projection = 14
        
        # LiDAR height filter (line 763)
        self.max_height_lidar = 100.0  # Points from LiDAR higher than this are discarded
    
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
                
                if self.mode == 'raw':
                    # Load and transform RGB image (strictly following Transfuser)
                    full_rgb_path = os.path.join(self.data_root, value)
                    rgb_tensor = self.load_image(full_rgb_path)
                    final_sample['rgb'] = rgb_tensor
                # In 'feature' mode, features are loaded from transfuser_features/transfuser_fused_features keys
                
            elif key == 'lidar':
                # Store LiDAR path (relative path to .laz file)
                final_sample['lidar_path'] = value
                
                if self.mode == 'raw':
                    # Load and transform LiDAR point cloud (strictly following Transfuser)
                    full_lidar_path = os.path.join(self.data_root, value)
                    lidar_bev = self.load_lidar(full_lidar_path)
                    final_sample['lidar_bev'] = lidar_bev
                # In 'feature' mode, lidar_bev is not needed (features already contain fused info)
            
            elif key == 'transfuser_features':
                # Pre-extracted Transfuser features path (only used in 'feature' mode)
                if self.mode == 'feature':
                    # Store full path (not loading values to save memory)
                    final_sample['features'] = os.path.join(self.data_root, value)
                    
            elif key == 'transfuser_fused_features':
                # Pre-extracted Transfuser fused features path (only used in 'feature' mode)
                if self.mode == 'feature':
                    # Store full path (not loading values to save memory)
                    final_sample['fused_features'] = os.path.join(self.data_root, value)
                
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
        
        # Fallback: if in 'feature' mode but features not in pkl, derive from rgb path
        if self.mode == 'feature' and 'features' not in final_sample and 'rgb_path' in final_sample:
            rgb_path = final_sample['rgb_path']
            rgb_dir = os.path.dirname(rgb_path)  # "route_xxx/rgb"
            scene_dir = os.path.dirname(rgb_dir)  # "route_xxx"
            frame_id = os.path.splitext(os.path.basename(rgb_path))[0]  # "00000"
            
            features_rel_path = os.path.join(scene_dir, 'transfuser_features', f'{frame_id}_features.pt')
            fused_features_rel_path = os.path.join(scene_dir, 'transfuser_features', f'{frame_id}_fused_features.pt')
            
            final_sample['features'] = os.path.join(self.data_root, features_rel_path)
            final_sample['fused_features'] = os.path.join(self.data_root, fused_features_rel_path)

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

    def crop_array(self, images_i):
        """
        Crop RGB images to the desired height and width.
        Strictly follows Transfuser's transfuser_utils.crop_array().
        
        Args:
            images_i: numpy array with shape (H, W, C) or (H, W)
        Returns:
            Cropped array with shape (cropped_height, cropped_width, C) or (cropped_height, cropped_width)
        """
        if self.crop_image:
            # crops rgb/depth/semantics from the bottom to cropped_height 
            # and symmetrically from both sides to cropped_width
            assert self.cropped_height <= images_i.shape[0]
            assert self.cropped_width <= images_i.shape[1]
            side_crop_amount = (images_i.shape[1] - self.cropped_width) // 2
            if len(images_i.shape) > 2:  # for rgb, we have 3 channels
                return images_i[0:self.cropped_height, side_crop_amount:images_i.shape[1] - side_crop_amount, :]
            else:  # for depth and semantics, there is no channel dimension
                return images_i[0:self.cropped_height, side_crop_amount:images_i.shape[1] - side_crop_amount]
        else:
            return images_i

    def load_image(self, image_path):
        """
        Load and transform a single RGB image.
        Strictly follows Transfuser data.py:
          1. cv2.imread (BGR)
          2. cv2.cvtColor BGR->RGB  
          3. crop_array
          4. np.transpose to (C, H, W)
        
        Output: (3, 384, 1024) tensor with values in [0, 255] float32
        """
        try:
            # Step 1: Load image using cv2 (BGR format) - same as Transfuser
            images_i = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if images_i is None:
                raise FileNotFoundError(f"cv2.imread returned None for {image_path}")
            
            # Step 2: Convert BGR to RGB - same as Transfuser
            images_i = cv2.cvtColor(images_i, cv2.COLOR_BGR2RGB)
            
            # Step 3: Crop to (384, 1024) - same as Transfuser's crop_array
            images_i = self.crop_array(images_i)
            
            # Step 4: Transpose to (C, H, W) format - same as Transfuser
            # data['rgb'] = np.transpose(processed_image, (2, 0, 1))
            images_i = np.transpose(images_i, (2, 0, 1)).astype(np.float32)
            
            # Convert to tensor
            img_tensor = torch.from_numpy(images_i)
            
            return img_tensor
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.zeros(3, self.cropped_height, self.cropped_width, dtype=torch.float32)

    def load_lidar(self, lidar_path):
        """
        Load and transform LiDAR point cloud to BEV histogram.
        Strictly follows Transfuser data.py:
          1. laspy.read -> xyz point cloud
          2. lidar_to_histogram_features -> (C, 256, 256) BEV
        
        Output: (1 or 2, 256, 256) tensor depending on use_ground_plane
        """
        try:
            # Step 1: Load LiDAR using laspy - same as Transfuser
            las_object = laspy.read(lidar_path)
            lidar = las_object.xyz  # (N, 3) numpy array
            
            # Step 2: Convert to histogram features - same as Transfuser
            lidar_bev = self.lidar_to_histogram_features(lidar)
            
            # Convert to tensor
            lidar_tensor = torch.from_numpy(lidar_bev)
            
            return lidar_tensor
        except Exception as e:
            print(f"Error loading LiDAR {lidar_path}: {e}")
            # Return zeros with correct shape based on use_ground_plane
            num_channels = 2 if self.use_ground_plane else 1
            return torch.zeros(num_channels, self.lidar_resolution_height, self.lidar_resolution_width, 
                             dtype=torch.float32)

    def lidar_to_histogram_features(self, lidar):
        """
        Convert LiDAR point cloud into 2-bin histogram over a fixed size grid.
        Strictly follows Transfuser data.py lidar_to_histogram_features().
        
        Args:
            lidar: (N, 3) numpy array, LiDAR point cloud in ego coordinates
        Returns:
            (C, H, W) numpy array, LiDAR as sparse BEV image
            C = 2 if use_ground_plane else 1
        """
        def splat_points(point_cloud):
            """Voxelize points to 256x256 grid."""
            # 256 x 256 grid
            xbins = np.linspace(
                self.min_x, self.max_x,
                (self.max_x - self.min_x) * int(self.pixels_per_meter) + 1
            )
            ybins = np.linspace(
                self.min_y, self.max_y,
                (self.max_y - self.min_y) * int(self.pixels_per_meter) + 1
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self.hist_max_per_pixel] = self.hist_max_per_pixel
            overhead_splat = hist / self.hist_max_per_pixel
            # The transpose here is an efficient axis swap.
            # Comes from the fact that carla is x front, y right, whereas the image is y front, x right
            # (x height channel, y width channel)
            return overhead_splat.T

        # Remove points above the vehicle
        lidar = lidar[lidar[..., 2] < self.max_height_lidar]
        
        # Split by height
        below = lidar[lidar[..., 2] <= self.lidar_split_height]
        above = lidar[lidar[..., 2] > self.lidar_split_height]
        
        below_features = splat_points(below)
        above_features = splat_points(above)
        
        if self.use_ground_plane:
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        
        # Transpose to (C, H, W) format
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)
        return features

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


def visualize_lidar_bev(sample, rand_idx, save_dir='/home/wang/Project/MoT-DP/image'):
    """Visualize the LiDAR BEV histogram from a sample."""
    lidar_bev = sample.get('lidar_bev')  # shape: (C, H, W) = (1 or 2, 256, 256)
    
    if lidar_bev is None:
        print("No LiDAR BEV found in sample")
        return
    
    print(f"LiDAR BEV tensor shape: {lidar_bev.shape if isinstance(lidar_bev, torch.Tensor) else 'not a tensor'}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    if isinstance(lidar_bev, torch.Tensor):
        lidar_arr = lidar_bev.numpy()
    else:
        lidar_arr = lidar_bev
    
    print(f"LiDAR array shape: {lidar_arr.shape}, dtype: {lidar_arr.dtype}")
    print(f"LiDAR array range: [{lidar_arr.min():.3f}, {lidar_arr.max():.3f}]")
    
    # Create visualization
    # After Transfuser's transpose: shape is (C, H, W) where H=Y direction, W=X direction
    # - Y axis (height): left (-32m) to right (+32m)  
    # - X axis (width): back (-32m) to front (+32m)
    num_channels = lidar_arr.shape[0]
    fig, axes = plt.subplots(1, num_channels, figsize=(8 * num_channels, 7))
    if num_channels == 1:
        axes = [axes]
    
    channel_names = ['Above Split Height', 'Below Split Height'] if num_channels == 2 else ['Above Split Height']
    
    # Coordinate ranges after Transfuser transpose
    min_x, max_x = -32, 32  # X: longitudinal (front-back), displayed as image width
    min_y, max_y = -32, 32  # Y: lateral (left-right), displayed as image height
    
    for i, (ax, name) in enumerate(zip(axes, channel_names)):
        # Use origin='lower' and extent to show proper coordinates
        # extent=[left, right, bottom, top] for image coordinates
        # After transpose: axis 0 (rows) = Y, axis 1 (cols) = X
        im = ax.imshow(lidar_arr[i], cmap='hot', vmin=0, vmax=1, 
                      origin='lower', extent=[min_x, max_x, min_y, max_y])
        ax.set_title(f'{name}\nChannel {i}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (Longitudinal: Back <- 0 -> Front) [m]', fontsize=10)
        ax.set_ylabel('Y (Lateral: Left <- 0 -> Right) [m]', fontsize=10)
        
        # Mark vehicle position at origin
        ax.plot(0, 0, 'c*', markersize=15, markeredgewidth=2, 
               markeredgecolor='white', label='Vehicle (0,0)')
        ax.axhline(y=0, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(im, ax=ax, label='Normalized LiDAR hits [0-1]')
    
    plt.suptitle(f'Sample {rand_idx} - LiDAR BEV (256x256)\nAfter Transfuser Transpose', 
                fontsize=14, fontweight='bold')
    
    save_path = os.path.join(save_dir, f'sample_{rand_idx}_lidar_bev.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ 保存LiDAR BEV到: {save_path}")




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
    for key in ['rgb', 'lidar_bev', 'lidar_path', 'speed', 'command', 'next_command', 'target_point', 
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
    dataset_path = '/home/wang/Project/carla_garage/data_mini/tmp_data/val'
    data_root = '/home/wang/Project/carla_garage/data_mini'
    obs_horizon = 4
    
    print("\n========== Testing PDM Lite Dataset (Transfuser Format) ==========")
    dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        data_root=data_root,
        mode='feature'  # Change to 'feature' to test feature loading
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
    
    # # Visualize RGB image
    # visualize_observation_images(rand_sample, rand_idx)
    
    # # Visualize LiDAR BEV
    # visualize_lidar_bev(rand_sample, rand_idx)

    
if __name__ == "__main__":
    test_pdm()


