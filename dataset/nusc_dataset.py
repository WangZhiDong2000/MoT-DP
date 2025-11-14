"""
nuScenes Dataset Loader for preprocessed nuScenes data.

This module loads preprocessed nuScenes samples with:
- BEV features from LIDAR
- Historical waypoints
- Future trajectories
- Navigation commands
- Ego status (velocity, acceleration, etc.)
"""

import numpy as np
from PIL import Image
import os
import sys
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torchvision.transforms as transforms


class NUSCDataset(torch.utils.data.Dataset):
    """
    Dataset for preprocessed nuScenes data with visualization support.
    
    Each sample contains:
    - lidar_token: BEV features (obs_horizon, 196, 512)
    - lidar_token_global: Global BEV features (obs_horizon, 1, 512)
    - hist_waypoints: Historical waypoints in ego frame (obs_horizon, 2)
    - fut_waypoints: Future waypoints in ego frame (action_horizon, 2)
    - fut_valid_mask: Valid mask for future waypoints (action_horizon,)
    - ego_status: Ego vehicle status (10,) - velocity, acceleration, etc.
    - nav_command: Navigation command (3,) - left, straight, right
    
    Args:
        processed_data_path (str): Path to preprocessed pkl file
        dataset_root (str): Root directory of nuScenes dataset
        mode (str): 'train' or 'val'
        load_bev_images (bool): Whether to load BEV images for visualization (default: False for train, True for val)
    """
    
    def __init__(self,
                 processed_data_path: str,
                 dataset_root: str,
                 mode: str = 'train',
                 load_bev_images: bool = None):
        
        self.dataset_root = dataset_root
        self.mode = mode
        
        # Auto-determine whether to load BEV images based on mode
        if load_bev_images is None:
            load_bev_images = (mode == 'val')
        self.load_bev_images = load_bev_images
        
        # BEV image transform for visualization
        if self.load_bev_images:
            self.lidar_bev_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # Load preprocessed data
        print(f"Loading preprocessed data from {processed_data_path}...")
        with open(processed_data_path, 'rb') as f:
            # Handle numpy version compatibility
            import sys
            import numpy
            sys.modules['numpy._core'] = numpy.core
            sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
            
            data = pickle.load(f)
        
        self.samples = data['samples']
        self.metadata = data['metadata']
        self.config = data['config']
        
        self.obs_horizon = self.config['obs_horizon']
        self.action_horizon = self.config['action_horizon']
        
        print(f"Loaded {len(self.samples)} samples from {self.metadata['version']}")
        print(f"Config: obs_horizon={self.obs_horizon}, action_horizon={self.action_horizon}")
        print(f"Load BEV images: {self.load_bev_images}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load BEV features for all historical frames
        lidar_tokens = []
        lidar_token_globals = []
        
        if 'hist_bev_token_paths' in sample and 'hist_bev_token_global_paths' in sample:
            # New format: load historical BEV features
            for token_path, token_global_path in zip(
                sample['hist_bev_token_paths'], 
                sample['hist_bev_token_global_paths']
            ):
                try:
                    token = torch.load(token_path, weights_only=True)
                    token_global = torch.load(token_global_path, weights_only=True)
                    lidar_tokens.append(token)
                    lidar_token_globals.append(token_global)
                except Exception as e:
                    print(f"Error loading BEV features from {token_path}: {e}")
                    # Create dummy features
                    lidar_tokens.append(torch.zeros(196, 512))
                    lidar_token_globals.append(torch.zeros(1, 512))
            
            # Stack to (obs_horizon, 196, 512) and (obs_horizon, 1, 512)
            lidar_token = torch.stack(lidar_tokens, dim=0)
            lidar_token_global = torch.stack(lidar_token_globals, dim=0)
        else:
            # Old format: load current frame and repeat (backward compatibility)
            try:
                token = torch.load(sample['bev_token_path'], weights_only=True)
                token_global = torch.load(sample['bev_token_global_path'], weights_only=True)
            except Exception as e:
                print(f"Error loading BEV features for sample {idx}: {e}")
                # Create dummy features
                token = torch.zeros(196, 512)
                token_global = torch.zeros(1, 512)
            
            # Repeat current frame for obs_horizon
            # Shape: (196, 512) -> (obs_horizon, 196, 512)
            lidar_token = token.unsqueeze(0).repeat(self.obs_horizon, 1, 1)
            # Shape: (1, 512) -> (obs_horizon, 1, 512)
            lidar_token_global = token_global.unsqueeze(0).repeat(self.obs_horizon, 1, 1)
        
        # Convert numpy arrays to tensors (using torch.tensor for compatibility)
        hist_waypoints = torch.tensor(sample['hist_waypoints'], dtype=torch.float32)  # (obs_horizon, 2)
        fut_waypoints = torch.tensor(sample['fut_waypoints'], dtype=torch.float32)    # (action_horizon, 2)
        fut_valid_mask = torch.tensor(sample['fut_valid_mask'], dtype=torch.bool)     # (action_horizon,)
        
        # Load historical ego_status and nav_command
        if 'hist_ego_status' in sample and 'hist_nav_command' in sample:
            # New format: use historical data
            ego_status = torch.tensor(sample['hist_ego_status'], dtype=torch.float32)  # (obs_horizon, 10)
            nav_command = torch.tensor(sample['hist_nav_command'], dtype=torch.float32)  # (obs_horizon, 3)
        else:
            # Old format: repeat current frame (backward compatibility)
            ego_status_single = torch.tensor(sample['ego_status'], dtype=torch.float32)  # (10,)
            nav_command_single = torch.tensor(sample['nav_command'], dtype=torch.float32)  # (3,)
            ego_status = ego_status_single.unsqueeze(0).repeat(self.obs_horizon, 1)  # (obs_horizon, 10)
            nav_command = nav_command_single.unsqueeze(0).repeat(self.obs_horizon, 1)  # (obs_horizon, 3)
        
        # Concatenate ego_status and nav_command into single vector
        # ego_status: (obs_horizon, 10) - [accel(3), rot_rate(3), vel(3), steer(1)]
        # nav_command: (obs_horizon, 3) - [left, straight, right]
        # Combined: (obs_horizon, 13)
        ego_status_with_command = torch.cat([ego_status, nav_command], dim=-1)
        
        # Load obstacle information for future frames
        fut_obstacles = []
        if 'fut_obstacles' in sample:
            for obs_dict in sample['fut_obstacles']:
                fut_obstacles.append({
                    'gt_boxes': torch.tensor(obs_dict['gt_boxes'], dtype=torch.float32),  # (N, 7)
                    'gt_names': obs_dict['gt_names'],  # (N,) numpy array of strings
                    'gt_velocity': torch.tensor(obs_dict['gt_velocity'], dtype=torch.float32),  # (N, 2)
                })
        
        # Prepare output dictionary
        output = {
            # BEV features
            'lidar_token': lidar_token,                    # (obs_horizon, 196, 512)
            'lidar_token_global': lidar_token_global,      # (obs_horizon, 1, 512)
            
            # Waypoints
            'hist_waypoints': hist_waypoints,              # (obs_horizon, 2)
            'agent_pos': fut_waypoints,                    # (action_horizon, 2) - target trajectory
            
            # Valid mask
            'fut_valid_mask': fut_valid_mask,              # (action_horizon,)
            
            # Ego status with command concatenated (obs_horizon dimension)
            'ego_status': ego_status_with_command,         # (obs_horizon, 13) - [ego_status(10), command(3)]
            'command': nav_command,                        # (obs_horizon, 3) - kept for compatibility
            
            # Obstacle information for future frames
            'fut_obstacles': fut_obstacles,                # List of action_horizon dicts with gt_boxes, gt_names, gt_velocity
            
            # Metadata
            'scene_token': sample['scene_token'],
            'sample_token': sample['sample_token'],
            'timestamp': sample['timestamp'],
            'lidar_token_str': sample['lidar_token'],
        }
        
        # Load BEV image for visualization if requested
        if self.load_bev_images:
            lidar_token = sample['lidar_token']
            
            # Construct BEV image path
            bev_image_path = os.path.join(
                self.dataset_root,
                'samples',
                'LIDAR_TOP_BEV',
                f'{lidar_token}.png'
            )
            
            try:
                bev_image = Image.open(bev_image_path).convert('RGB')
                # Convert to numpy array then to tensor for compatibility
                bev_np = np.array(bev_image, dtype=np.float32) / 255.0
                bev_tensor = torch.tensor(bev_np, dtype=torch.float32).permute(2, 0, 1)
                # Apply normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                bev_tensor = (bev_tensor - mean) / std
                # Repeat for obs_horizon
                output['lidar_bev'] = bev_tensor.unsqueeze(0).repeat(self.obs_horizon, 1, 1, 1)
                output['lidar_bev_path'] = bev_image_path
            except Exception as e:
                print(f"Warning: Could not load BEV image {bev_image_path}: {e}")
                output['lidar_bev'] = torch.zeros(self.obs_horizon, 3, 448, 448)
                output['lidar_bev_path'] = None
        
        return output




# ============================================================================
# Utility Functions
# ============================================================================

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Handles fut_obstacles with variable number of obstacles per frame.
    """
    # Stack all tensors in the batch
    output = {}
    
    for key in batch[0].keys():
        if key == 'fut_obstacles':
            # Keep fut_obstacles as list of lists (batch, action_horizon)
            # Each element contains dict with variable-length tensors
            output[key] = [sample[key] for sample in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            output[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(batch[0][key], (str, int, float)):
            output[key] = [sample[key] for sample in batch]
        else:
            output[key] = [sample[key] for sample in batch]
    
    return output


def visualize_sample(sample, save_path=None):
    """
    Visualize a single sample with waypoints, obstacles, and BEV image.
    
    Args:
        sample: Dictionary containing sample data
        save_path: Path to save the visualization (optional)
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, axes = plt.subplots(1, 3, figsize=(27, 7))
    
    # Plot 1: Waypoints and Obstacles
    ax1 = axes[0]
    
    # NOTE: In our data, x is forward (longitudinal), y is lateral
    # For visualization, we swap them so forward (x) appears upward on the plot
    # Plot coordinates: horizontal=y (lateral), vertical=x (forward)
    
    # Historical waypoints (blue)
    hist_wp = sample['hist_waypoints'].cpu().numpy()
    ax1.plot(hist_wp[:, 1], hist_wp[:, 0], 'b.-', label='Historical', markersize=10, linewidth=2)
    
    # Future waypoints (red)
    fut_wp = sample['agent_pos'].cpu().numpy()
    valid_mask = sample['fut_valid_mask'].cpu().numpy()
    ax1.plot(fut_wp[:, 1], fut_wp[:, 0], 'r.-', label='Future', markersize=10, linewidth=2)
    
    # Mark invalid waypoints
    invalid_wp = fut_wp[~valid_mask]
    if len(invalid_wp) > 0:
        ax1.scatter(invalid_wp[:, 1], invalid_wp[:, 0], c='orange', s=100, 
                   marker='x', label='Invalid', zorder=10)
    
    # Current position (ego vehicle)
    ax1.scatter(0, 0, c='green', s=200, marker='*', label='Ego (Current)', zorder=10)
    
    # Draw ego vehicle as a rectangle
    # ego_length (x-direction) will be vertical, ego_width (y-direction) will be horizontal
    ego_length = 4.084
    ego_width = 1.730
    ego_rect = patches.Rectangle(
        (-ego_width/2, -ego_length/2), ego_width, ego_length,
        linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.5, zorder=5
    )
    ax1.add_patch(ego_rect)
    
    # Visualize obstacles from the first future frame (if available)
    if 'fut_obstacles' in sample and len(sample['fut_obstacles']) > 0:
        obstacles_frame_0 = sample['fut_obstacles'][0]  # First future frame
        gt_boxes = obstacles_frame_0['gt_boxes'].cpu().numpy()  # (N, 7) [x, y, z, w, l, h, yaw]
        gt_names = obstacles_frame_0['gt_names']  # (N,) string array
        
        # Color map for different obstacle types
        color_map = {
            'car': 'red',
            'truck': 'darkred',
            'bus': 'orange',
            'pedestrian': 'blue',
            'bicycle': 'cyan',
            'motorcycle': 'magenta',
            'traffic_cone': 'yellow',
            'barrier': 'gray',
            'construction_vehicle': 'brown',
            'trailer': 'pink',
        }
        
        # Limit visualization to nearby obstacles (within 60m)
        max_dist = 60.0
        obstacles_drawn = 0
        
        for i, (box, name) in enumerate(zip(gt_boxes, gt_names)):
            # gt_boxes format: [x, y, z, l, w, h, yaw]
            # where l=length (x-direction), w=width (y-direction)
            x, y, z, l, w, h, yaw = box
            
            # Skip if too far
            dist = np.sqrt(x**2 + y**2)
            if dist > max_dist:
                continue
            
            # Get color for this obstacle type
            color = color_map.get(name, 'purple')
            
            # Draw bounding box (2D top-down view)
            # Data frame: x=forward, y=lateral (left), yaw=rotation around z-axis
            # Plot frame: horizontal=y (lateral), vertical=x (forward)
            
            # Step 1: Compute corners in data frame
            # gt_boxes format: l=length(x-direction), w=width(y-direction)
            # Box corners in local frame (before rotation)
            corners_local = np.array([
                [l/2,  w/2],   # right-front
                [l/2, -w/2],   # left-front
                [-l/2, -w/2],  # left-rear
                [-l/2,  w/2]   # right-rear
            ])
            
            # Rotation matrix in data frame
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rotation_matrix = np.array([[cos_yaw, -sin_yaw],
                                       [sin_yaw, cos_yaw]])
            
            # Rotate and translate in data frame
            corners_data = (rotation_matrix @ corners_local.T).T + np.array([x, y])
            
            # Step 2: Convert to plot frame by swapping x and y
            corners_plot = corners_data[:, [1, 0]]  # Swap columns: [x, y] -> [y, x]
            
            # Create polygon
            polygon = patches.Polygon(
                corners_plot, closed=True,
                linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.3
            )
            ax1.add_patch(polygon)
            
            # Add center point (also swap coordinates)
            ax1.scatter(y, x, c=color, s=20, marker='o', alpha=0.7)
            
            obstacles_drawn += 1
            
            # Limit total obstacles drawn for clarity
            if obstacles_drawn >= 30:
                break
        
        # Add legend entry for obstacles
        if obstacles_drawn > 0:
            ax1.scatter([], [], c='gray', s=100, marker='s', alpha=0.3, 
                       label=f'Obstacles (t+1, {obstacles_drawn} shown)')
    
    # Axis labels: horizontal is lateral (y), vertical is forward (x)
    ax1.set_xlabel('Lateral (Y) [meters]', fontsize=12)
    ax1.set_ylabel('Forward (X) [meters]', fontsize=12)
    ax1.set_title('Trajectory and Obstacles (Ego Frame)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.set_xlim(-30, 30)   # Lateral range (y-axis in data)
    ax1.set_ylim(-20, 60)   # Forward range (x-axis in data)
    
    # Plot 2: Future Obstacle Motion Visualization
    ax2 = axes[1]
    
    # Draw ego trajectory first
    ax2.plot(hist_wp[:, 1], hist_wp[:, 0], 'b.-', label='Historical', markersize=8, linewidth=1.5, alpha=0.7)
    ax2.plot(fut_wp[:, 1], fut_wp[:, 0], 'r.-', label='Future Traj', markersize=8, linewidth=1.5, alpha=0.7)
    ax2.scatter(0, 0, c='green', s=200, marker='*', label='Ego (Current)', zorder=10)
    
    # Draw current ego vehicle
    ego_rect = patches.Rectangle(
        (-ego_width/2, -ego_length/2), ego_width, ego_length,
        linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.5, zorder=5
    )
    ax2.add_patch(ego_rect)
    
    # Visualize obstacles across all future frames with motion trails
    if 'fut_obstacles' in sample and len(sample['fut_obstacles']) > 0:
        # Color gradient for time progression
        import matplotlib.cm as cm
        num_future_frames = len(sample['fut_obstacles'])
        colors = cm.Reds(np.linspace(0.3, 0.9, num_future_frames))
        
        # Track obstacle positions across frames to draw motion trails
        obstacle_tracks = {}  # key: obstacle index in first frame, value: list of positions
        
        color_map = {
            'car': 'red',
            'truck': 'darkred',
            'bus': 'orange',
            'pedestrian': 'blue',
            'bicycle': 'cyan',
            'motorcycle': 'magenta',
            'traffic_cone': 'yellow',
            'barrier': 'gray',
            'construction_vehicle': 'brown',
            'trailer': 'pink',
        }
        
        max_dist = 60.0
        obstacles_tracked = 0
        
        # For simplicity, track obstacles that appear in the first frame
        # and show their positions in subsequent frames
        first_frame_obstacles = sample['fut_obstacles'][0]
        first_gt_boxes = first_frame_obstacles['gt_boxes'].cpu().numpy()
        first_gt_names = first_frame_obstacles['gt_names']
        
        # Iterate through each obstacle in the first frame
        for obs_idx, (box, name) in enumerate(zip(first_gt_boxes, first_gt_names)):
            x, y, z, l, w, h, yaw = box
            dist = np.sqrt(x**2 + y**2)
            if dist > max_dist:
                continue
            
            # Draw obstacle boxes at each timestep with transparency
            for frame_idx in range(num_future_frames):
                if frame_idx < len(sample['fut_obstacles']):
                    frame_obstacles = sample['fut_obstacles'][frame_idx]
                    if obs_idx < len(frame_obstacles['gt_boxes']):
                        box = frame_obstacles['gt_boxes'][obs_idx].cpu().numpy()
                        x, y, z, l, w, h, yaw = box
                        
                        # Compute corners
                        corners_local = np.array([
                            [l/2,  w/2],
                            [l/2, -w/2],
                            [-l/2, -w/2],
                            [-l/2,  w/2]
                        ])
                        
                        cos_yaw = np.cos(yaw)
                        sin_yaw = np.sin(yaw)
                        rotation_matrix = np.array([[cos_yaw, -sin_yaw],
                                                   [sin_yaw, cos_yaw]])
                        
                        corners_data = (rotation_matrix @ corners_local.T).T + np.array([x, y])
                        corners_plot = corners_data[:, [1, 0]]
                        
                        # Alpha decreases with time (earlier frames more transparent)
                        alpha = 0.15 + 0.25 * (frame_idx / max(1, num_future_frames - 1))
                        
                        polygon = patches.Polygon(
                            corners_plot, closed=True,
                            linewidth=1.0, edgecolor=color_map.get(name, 'purple'), 
                            facecolor=color_map.get(name, 'purple'), alpha=alpha
                        )
                        ax2.add_patch(polygon)
                        
                        # Add small marker at obstacle center
                        if frame_idx == 0:  # First frame: larger marker
                            ax2.scatter(y, x, c=color_map.get(name, 'purple'), s=30, marker='o', alpha=0.8, zorder=3)
                        else:  # Future frames: smaller markers
                            ax2.scatter(y, x, c=color_map.get(name, 'purple'), s=15, marker='o', alpha=0.6, zorder=3)
            
            obstacles_tracked += 1
            if obstacles_tracked >= 20:  # Limit for clarity
                break
        
        if obstacles_tracked > 0:
            # Add legend entry
            ax2.scatter([], [], c='gray', s=100, marker='s', alpha=0.3, 
                       label=f'Obstacles Motion ({obstacles_tracked} tracked)')
    
    ax2.set_xlabel('Lateral (Y) [meters]', fontsize=12)
    ax2.set_ylabel('Forward (X) [meters]', fontsize=12)
    ax2.set_title(f'Future Obstacle Motion ({num_future_frames} frames)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.set_xlim(-30, 30)
    ax2.set_ylim(-20, 60)
    
    # Plot 3: BEV image (if available)
    ax3 = axes[2]
    if 'lidar_bev' in sample and sample['lidar_bev'] is not None:
        # Take the last frame from obs_horizon
        bev_tensor = sample['lidar_bev'][-1].cpu()
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        bev_tensor = bev_tensor * std + mean
        bev_tensor = torch.clamp(bev_tensor, 0, 1)
        bev_img = bev_tensor.numpy().transpose(1, 2, 0)
        ax3.imshow(bev_img)
        ax3.set_title('BEV Image (Current Frame)', fontsize=14, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No BEV Image', ha='center', va='center', fontsize=20)
        ax3.set_title('BEV Image (Not Available)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Add info text
    nav_cmd = sample['command'][0].cpu()
    cmd_names = ['Left', 'Straight', 'Right']
    cmd_idx = torch.argmax(nav_cmd).item()
    # ego_status is now (obs_horizon, 13) - [accel(3), rot_rate(3), vel(3), steer(1), command(3)]
    # Extract velocity from first frame: vel is at indices 6:9
    ego_status = sample['ego_status'][0].cpu()  # (13,)
    vel_x, vel_y, vel_z = ego_status[6], ego_status[7], ego_status[8]
    speed = torch.sqrt(vel_x**2 + vel_y**2 + vel_z**2).item()
    
    # Count obstacles
    num_obstacles = 0
    if 'fut_obstacles' in sample and len(sample['fut_obstacles']) > 0:
        num_obstacles = len(sample['fut_obstacles'][0]['gt_boxes'])
    
    info_text = f"Command: {cmd_names[cmd_idx]} | Speed: {speed:.2f} m/s | "
    info_text += f"Obstacles: {num_obstacles} | Scene: {sample['scene_token'][:8]}..."
    
    fig.suptitle(info_text, fontsize=13, y=0.98, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_data', type=str, 
                       default='/home/wang/Dataset/v1.0-mini/processed_data/nuscenes_processed_train.pkl',
                       help='Path to preprocessed data')
    parser.add_argument('--dataset_root', type=str,
                       default='/home/wang/Dataset/v1.0-mini',
                       help='Root directory of nuScenes dataset')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--visualize', default=True, action='store_true', help='Visualize samples')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Create dataset
    dataset = NUSCDataset(
        processed_data_path=args.processed_data,
        dataset_root=args.dataset_root,
        mode=args.mode,
        load_bev_images=args.visualize  # Load BEV images if visualization is requested
    )
    
    print(f"\n{'='*70}")
    print(f"Dataset Statistics:")
    print(f"{'='*70}")
    print(f"Total samples: {len(dataset)}")
    print(f"Observation horizon: {dataset.obs_horizon}")
    print(f"Action horizon: {dataset.action_horizon}")
    
    # Test loading a few samples
    print(f"\n{'='*70}")
    print(f"Testing Sample Loading:")
    print(f"{'='*70}")
    
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key:25s}: shape={str(value.shape):25s}, dtype={value.dtype}")
            elif isinstance(value, list) and key == 'fut_obstacles':
                print(f"  {key:25s}: {len(value)} frames")
                for frame_idx, obs_dict in enumerate(value):
                    num_obs = len(obs_dict['gt_boxes'])
                    print(f"    Frame {frame_idx}: {num_obs} obstacles")
            elif isinstance(value, (str, int, float)):
                value_str = str(value) if len(str(value)) < 50 else str(value)[:50] + '...'
                print(f"  {key:25s}: {value_str}")
    
    # Visualize samples
    if args.visualize:
        print(f"\n{'='*70}")
        print(f"Generating Visualizations:")
        print(f"{'='*70}")
        
        import os
        os.makedirs('/home/wang/Project/MoT-DP/image/nusc_samples', exist_ok=True)
        
        for i in range(min(args.num_samples, len(dataset))):
            sample = dataset[i]
            save_path = f'/home/wang/Project/MoT-DP/image/nusc_samples/sample_{i:03d}.png'
            visualize_sample(sample, save_path)
        
        print(f"✓ Saved {min(args.num_samples, len(dataset))} visualizations")
