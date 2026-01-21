"""
Senna nuScenes Dataset Loader for preprocessed Senna data with hidden states.

This module loads preprocessed Senna nuScenes samples with:
- BEV features from LIDAR
- Historical waypoints
- Future trajectories
- Navigation commands
- Ego status (velocity, acceleration, etc.)
- ViT hidden states
- Action hidden states (decision features)
- Reasoning hidden states
- Anchor trajectory (predicted trajectory from Senna model)
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from PIL import Image
import pickle
import torch
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


class SennaDataset(torch.utils.data.Dataset):
    """
    Dataset for preprocessed Senna nuScenes data with visualization support.
    
    Each sample contains:
    - lidar_token: BEV features (obs_horizon, 196, 512)
    - lidar_token_global: Global BEV features (obs_horizon, 1, 512)
    - hist_waypoints: Historical waypoints in ego frame (obs_horizon, 2)
    - fut_waypoints: Future waypoints in ego frame (action_horizon, 2)
    - fut_valid_mask: Valid mask for future waypoints (action_horizon,)
    - ego_status: Ego vehicle status (obs_horizon, 9)
    - nav_command: Navigation command (obs_horizon, 3)
    - vit_hidden_states: ViT visual features (128, 2560)
    - action_hidden_states: Decision/action features (4, 2560)
    - reasoning_hidden_states: Reasoning features (20, 2560)
    - anchor_trajectory: Anchor trajectory prediction (1, 6, 2)
    
    Args:
        processed_data_path (str): Path to preprocessed pkl file with features
        dataset_root (str): Root directory of nuScenes dataset
        mode (str): 'train' or 'val'
        load_bev_images (bool): Whether to load BEV images for visualization
        load_hidden_states (bool): Whether to load hidden state features (default: True)
    """
    
    def __init__(self,
                 processed_data_path: str,
                 dataset_root: str,
                 mode: str = 'train',
                 load_bev_images: bool = None,
                 load_hidden_states: bool = True):
        
        self.dataset_root = dataset_root
        self.mode = mode
        self.load_hidden_states = load_hidden_states
        
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
        print(f"Loading preprocessed Senna data from {processed_data_path}...")
        with open(processed_data_path, 'rb') as f:
            sys.modules['numpy._core'] = np.core
            sys.modules['numpy._core.multiarray'] = np.core.multiarray
            data = pickle.load(f)
        
        self.samples = data['samples']
        self.config = data.get('config', {})
        
        # Determine obs_horizon and action_horizon from data
        if len(self.samples) > 0:
            sample = self.samples[0]
            self.obs_horizon = len(sample.get('hist_bev_token_paths', [])) or 5
            self.action_horizon = len(sample.get('fut_waypoints', [])) or 6
        else:
            self.obs_horizon = 5
            self.action_horizon = 6
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"Config: obs_horizon={self.obs_horizon}, action_horizon={self.action_horizon}")
        print(f"Load BEV images: {self.load_bev_images}")
        print(f"Load hidden states: {self.load_hidden_states}")
        
        # Check if hidden states are available
        if self.load_hidden_states and len(self.samples) > 0:
            has_vit = 'vit_hidden_states' in self.samples[0]
            has_action = 'action_hidden_states' in self.samples[0]
            has_reasoning = 'reasoning_hidden_states' in self.samples[0]
            has_anchor = 'anchor_trajectory' in self.samples[0]
            print(f"Hidden states available: vit={has_vit}, action={has_action}, "
                  f"reasoning={has_reasoning}, anchor={has_anchor}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load BEV features for all historical frames
        lidar_tokens = []
        lidar_token_globals = []
        
        for token_path, token_global_path in zip(
                sample['hist_bev_token_paths'], 
                sample['hist_bev_token_global_paths']
            ):
            token = torch.load(token_path, weights_only=True)
            token_global = torch.load(token_global_path, weights_only=True)
            lidar_tokens.append(token)
            lidar_token_globals.append(token_global)
        
        lidar_token = torch.stack(lidar_tokens, dim=0)
        lidar_token_global = torch.stack(lidar_token_globals, dim=0)
        
        # Convert numpy arrays to tensors 
        hist_waypoints_offset = torch.tensor(sample['hist_waypoints'], dtype=torch.float32)  # (obs_horizon-1, 2) - offset format
        # Convert offset to absolute coordinates using cumsum (negative direction for history)
        hist_waypoints = -torch.cumsum(hist_waypoints_offset, dim=0)  # (obs_horizon-1, 2) - absolute format, negative direction
        
        fut_waypoints_offset = torch.tensor(sample['fut_waypoints'], dtype=torch.float32)  # (action_horizon, 2) - offset format
        # Convert offset to absolute coordinates using cumsum
        fut_waypoints = torch.cumsum(fut_waypoints_offset, dim=0)  # (action_horizon, 2) - absolute format
        fut_valid_mask = torch.tensor(sample['fut_valid_mask'], dtype=torch.bool)     # (action_horizon,)
        ego_status = torch.tensor(sample['hist_ego_status'], dtype=torch.float32)     # (obs_horizon, 9)
        nav_command = torch.tensor(sample['hist_nav_command'], dtype=torch.float32)   # (obs_horizon, 3)
        ego_status_with_command = torch.cat([ego_status, nav_command], dim=-1)        # (obs_horizon, 12)
        
        # Concatenate waypoints with ego status
        # Note: hist_waypoints may have different length than obs_horizon
        # Pad hist_waypoints if necessary
        if hist_waypoints.shape[0] < ego_status.shape[0]:
            # Pad with zeros at the beginning
            pad_size = ego_status.shape[0] - hist_waypoints.shape[0]
            hist_waypoints_padded = torch.cat([
                torch.zeros(pad_size, 2, dtype=torch.float32),
                hist_waypoints
            ], dim=0)
        else:
            hist_waypoints_padded = hist_waypoints[-ego_status.shape[0]:]
        
        ego_status_with_waypoints = torch.cat([ego_status_with_command, hist_waypoints_padded], dim=-1)  # (obs_horizon, 14)
        
        # Note: fut_obstacles are not included in output to avoid collate issues with variable-length tensors
        
        output = {
            # BEV features
            'lidar_token': lidar_token,                    # (obs_horizon, 196, 512)
            'lidar_token_global': lidar_token_global,      # (obs_horizon, 1, 512)
            
            # Waypoints
            'hist_waypoints': hist_waypoints,              # (obs_horizon-1, 2) or (4, 2)
            'agent_pos': fut_waypoints,                    # (action_horizon, 2) - target trajectory
            
            # Valid mask
            'fut_valid_mask': fut_valid_mask,              # (action_horizon,)
            
            # Ego status with command concatenated (obs_horizon dimension)
            'ego_status': ego_status_with_waypoints,       # (obs_horizon, 14) - [accel(3), rot_rate(3), vel(3), command(3), waypoint(2)]
            'command': nav_command,                        # (obs_horizon, 3) - kept for compatibility
            
            # Metadata
            'scene_token': sample['scene_token'],
            'sample_token': sample['sample_token'],
            'timestamp': sample['timestamp'],
            'lidar_token_str': sample['lidar_token'],
        }
        
        # Load hidden state features
        if self.load_hidden_states:
            # ViT hidden states (128, 2560) -> convert float16 to float32
            if 'vit_hidden_states' in sample and sample['vit_hidden_states'] is not None:
                vit_hs = sample['vit_hidden_states']
                if isinstance(vit_hs, np.ndarray):
                    output['vit_hidden_states'] = torch.tensor(vit_hs, dtype=torch.float32)
                else:
                    output['vit_hidden_states'] = vit_hs.float() if isinstance(vit_hs, torch.Tensor) else torch.tensor(vit_hs, dtype=torch.float32)
            else:
                output['vit_hidden_states'] = None
            
            # Action hidden states (4, 2560)
            if 'action_hidden_states' in sample and sample['action_hidden_states'] is not None:
                action_hs = sample['action_hidden_states']
                if isinstance(action_hs, np.ndarray):
                    output['action_hidden_states'] = torch.tensor(action_hs, dtype=torch.float32)
                else:
                    output['action_hidden_states'] = action_hs.float() if isinstance(action_hs, torch.Tensor) else torch.tensor(action_hs, dtype=torch.float32)
            else:
                output['action_hidden_states'] = None
            
            # Reasoning hidden states (20, 2560)
            if 'reasoning_hidden_states' in sample and sample['reasoning_hidden_states'] is not None:
                reasoning_hs = sample['reasoning_hidden_states']
                if isinstance(reasoning_hs, np.ndarray):
                    output['reasoning_hidden_states'] = torch.tensor(reasoning_hs, dtype=torch.float32)
                else:
                    output['reasoning_hidden_states'] = reasoning_hs.float() if isinstance(reasoning_hs, torch.Tensor) else torch.tensor(reasoning_hs, dtype=torch.float32)
            else:
                output['reasoning_hidden_states'] = None
            
            # Anchor trajectory (1, 6, 2) -> squeeze to (6, 2)
            if 'anchor_trajectory' in sample and sample['anchor_trajectory'] is not None:
                anchor_traj = sample['anchor_trajectory']
                if isinstance(anchor_traj, np.ndarray):
                    anchor_tensor = torch.tensor(anchor_traj, dtype=torch.float32)
                else:
                    anchor_tensor = anchor_traj.float() if isinstance(anchor_traj, torch.Tensor) else torch.tensor(anchor_traj, dtype=torch.float32)
                # Squeeze first dimension if shape is (1, 6, 2)
                if anchor_tensor.dim() == 3 and anchor_tensor.shape[0] == 1:
                    anchor_tensor = anchor_tensor.squeeze(0)  # (6, 2)
                output['anchor_trajectory'] = anchor_tensor
            else:
                output['anchor_trajectory'] = None
        
        # Load BEV images for visualization
        if self.load_bev_images:
            lidar_token_str = sample['lidar_token']
            bev_image_path = os.path.join(
                self.dataset_root,
                'samples',
                'LIDAR_TOP_BEV',
                f'{lidar_token_str}.png'
            )
            
            if os.path.exists(bev_image_path):
                bev_image = Image.open(bev_image_path).convert('RGB')
                bev_np = np.array(bev_image, dtype=np.float32) / 255.0
                bev_tensor = torch.tensor(bev_np, dtype=torch.float32).permute(2, 0, 1)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                bev_tensor = (bev_tensor - mean) / std
                output['lidar_bev'] = bev_tensor.unsqueeze(0).repeat(self.obs_horizon, 1, 1, 1)
                output['lidar_bev_path'] = bev_image_path
            else:
                output['lidar_bev'] = None
                output['lidar_bev_path'] = None
        
        return output


# ============================================================================
# Utility Functions
# ============================================================================

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Handles fut_obstacles with variable number of obstacles per frame.
    Also handles hidden states that may be None.
    """
    output = {}
    
    for key in batch[0].keys():
        if key == 'fut_obstacles':
            output[key] = [sample[key] for sample in batch]
        elif key in ['vit_hidden_states', 'action_hidden_states', 'reasoning_hidden_states', 
                     'anchor_trajectory', 'lidar_bev', 'lidar_bev_path']:
            # Handle optional features that may be None
            values = [sample[key] for sample in batch]
            if all(v is not None for v in values):
                if isinstance(values[0], torch.Tensor):
                    output[key] = torch.stack(values)
                else:
                    output[key] = values
            else:
                output[key] = values  # Keep as list with Nones
        elif isinstance(batch[0][key], torch.Tensor):
            output[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(batch[0][key], (str, int, float)):
            output[key] = [sample[key] for sample in batch]
        else:
            output[key] = [sample[key] for sample in batch]

    return output


def visualize_sample(sample, save_path=None):
    """
    Visualize a single sample with waypoints and anchor trajectory.
    
    Visualizes:
    - Historical ego trajectory (blue)
    - Future GT trajectory (red)
    - Anchor trajectory from Senna model (green dashed)
    
    Does NOT visualize:
    - GT boxes / obstacles
    
    Args:
        sample: Dictionary containing sample data
        save_path: Path to save the visualization (optional)
    """
    matplotlib.use('Agg')  
    fig, axes = plt.subplots(1, 3, figsize=(27, 7))
    
    # Plot 1: Waypoints and Anchor Trajectory
    ax1 = axes[0]
    
    # NOTE: In our data, x is forward (longitudinal), y is lateral
    # For visualization, we swap them so forward (x) appears upward on the plot
    # Plot coordinates: horizontal=y (lateral), vertical=x (forward)
    
    # Historical waypoints (blue)
    hist_wp = sample['hist_waypoints'].cpu().numpy()
    ax1.plot(hist_wp[:, 1], hist_wp[:, 0], 'b.-', label='Historical', markersize=10, linewidth=2)
    
    # Future waypoints / GT trajectory (red)
    fut_wp = sample['agent_pos'].cpu().numpy()
    valid_mask = sample['fut_valid_mask'].cpu().numpy()
    ax1.plot(fut_wp[:, 1], fut_wp[:, 0], 'r.-', label='Future GT', markersize=10, linewidth=2)
    
    # Mark invalid waypoints
    invalid_wp = fut_wp[~valid_mask]
    if len(invalid_wp) > 0:
        ax1.scatter(invalid_wp[:, 1], invalid_wp[:, 0], c='orange', s=100, 
                   marker='x', label='Invalid', zorder=10)
    
    # Anchor trajectory (green dashed)
    if 'anchor_trajectory' in sample and sample['anchor_trajectory'] is not None:
        anchor_traj = sample['anchor_trajectory'].cpu().numpy()
        # anchor_trajectory shape: (1, 6, 2) or (6, 2)
        if anchor_traj.ndim == 3:
            anchor_traj = anchor_traj[0]  # Remove batch dimension
        ax1.plot(anchor_traj[:, 1], anchor_traj[:, 0], 'g--', 
                label='Anchor Traj', markersize=8, linewidth=2, marker='s')
    
    # Current position (ego vehicle)
    ax1.scatter(0, 0, c='green', s=200, marker='*', label='Ego (Current)', zorder=10)
    
    # Draw ego vehicle as a rectangle
    ego_length = 4.084
    ego_width = 1.730
    ego_rect = patches.Rectangle(
        (-ego_width/2, -ego_length/2), ego_width, ego_length,
        linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.5, zorder=5
    )
    ax1.add_patch(ego_rect)
    
    # Axis labels
    ax1.set_xlabel('Lateral (Y) [meters]', fontsize=12)
    ax1.set_ylabel('Forward (X) [meters]', fontsize=12)
    ax1.set_title('Trajectory Comparison (Ego Frame)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.set_xlim(-30, 30)   
    ax1.set_ylim(-20, 60)   
    
    # Plot 2: Trajectory Comparison (zoomed in)
    ax2 = axes[1]
    
    # Historical trajectory
    ax2.plot(hist_wp[:, 1], hist_wp[:, 0], 'b.-', label='Historical', markersize=12, linewidth=2.5)
    
    # Future GT trajectory
    ax2.plot(fut_wp[:, 1], fut_wp[:, 0], 'r.-', label='Future GT', markersize=12, linewidth=2.5)
    
    # Anchor trajectory
    if 'anchor_trajectory' in sample and sample['anchor_trajectory'] is not None:
        anchor_traj = sample['anchor_trajectory'].cpu().numpy()
        if anchor_traj.ndim == 3:
            anchor_traj = anchor_traj[0]
        ax2.plot(anchor_traj[:, 1], anchor_traj[:, 0], 'g--s', 
                label='Anchor Traj', markersize=10, linewidth=2.5)
        
        # Calculate error between GT and anchor
        if fut_wp.shape == anchor_traj.shape:
            error = np.sqrt(np.sum((fut_wp - anchor_traj)**2, axis=1))
            avg_error = np.mean(error)
            ax2.set_title(f'Trajectory Comparison (Zoomed)\nAvg Error: {avg_error:.3f} m', 
                         fontsize=14, fontweight='bold')
        else:
            ax2.set_title('Trajectory Comparison (Zoomed)', fontsize=14, fontweight='bold')
    else:
        ax2.set_title('Trajectory Comparison (Zoomed)', fontsize=14, fontweight='bold')
    
    # Ego vehicle
    ax2.scatter(0, 0, c='green', s=300, marker='*', label='Ego', zorder=10)
    ego_rect2 = patches.Rectangle(
        (-ego_width/2, -ego_length/2), ego_width, ego_length,
        linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.5, zorder=5
    )
    ax2.add_patch(ego_rect2)
    
    ax2.set_xlabel('Lateral (Y) [meters]', fontsize=12)
    ax2.set_ylabel('Forward (X) [meters]', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Dynamic zoom based on trajectory extent
    all_y = np.concatenate([hist_wp[:, 1], fut_wp[:, 1]])
    all_x = np.concatenate([hist_wp[:, 0], fut_wp[:, 0]])
    if 'anchor_trajectory' in sample and sample['anchor_trajectory'] is not None:
        anchor_traj = sample['anchor_trajectory'].cpu().numpy()
        if anchor_traj.ndim == 3:
            anchor_traj = anchor_traj[0]
        all_y = np.concatenate([all_y, anchor_traj[:, 1]])
        all_x = np.concatenate([all_x, anchor_traj[:, 0]])
    
    x_margin = max(5, (all_x.max() - all_x.min()) * 0.2)
    y_margin = max(5, (all_y.max() - all_y.min()) * 0.2)
    ax2.set_xlim(all_y.min() - y_margin, all_y.max() + y_margin)
    ax2.set_ylim(all_x.min() - x_margin, all_x.max() + x_margin)
    
    # Plot 3: BEV image or Hidden States Info
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
        # Show hidden states info if available
        info_lines = ['Hidden States Information:']
        info_lines.append('')
        
        if 'vit_hidden_states' in sample and sample['vit_hidden_states'] is not None:
            vit_hs = sample['vit_hidden_states']
            if isinstance(vit_hs, torch.Tensor):
                info_lines.append(f'ViT Hidden States: {tuple(vit_hs.shape)}')
            else:
                info_lines.append(f'ViT Hidden States: {vit_hs.shape}')
        
        if 'action_hidden_states' in sample and sample['action_hidden_states'] is not None:
            action_hs = sample['action_hidden_states']
            if isinstance(action_hs, torch.Tensor):
                info_lines.append(f'Action Hidden States: {tuple(action_hs.shape)}')
            else:
                info_lines.append(f'Action Hidden States: {action_hs.shape}')
        
        if 'reasoning_hidden_states' in sample and sample['reasoning_hidden_states'] is not None:
            reasoning_hs = sample['reasoning_hidden_states']
            if isinstance(reasoning_hs, torch.Tensor):
                info_lines.append(f'Reasoning Hidden States: {tuple(reasoning_hs.shape)}')
            else:
                info_lines.append(f'Reasoning Hidden States: {reasoning_hs.shape}')
        
        if 'anchor_trajectory' in sample and sample['anchor_trajectory'] is not None:
            anchor_hs = sample['anchor_trajectory']
            if isinstance(anchor_hs, torch.Tensor):
                info_lines.append(f'Anchor Trajectory: {tuple(anchor_hs.shape)}')
            else:
                info_lines.append(f'Anchor Trajectory: {anchor_hs.shape}')
        
        ax3.text(0.5, 0.5, '\n'.join(info_lines), ha='center', va='center', 
                fontsize=14, family='monospace', transform=ax3.transAxes)
        ax3.set_title('Hidden States Info', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Add info text
    nav_cmd = sample['command'][-1].cpu()  # Use last frame's command
    cmd_names = ['Right', 'Left', 'Straight']
    cmd_idx = torch.argmax(nav_cmd).item()
    ego_status = sample['ego_status'][-1].cpu()  # (14,)
    # Ego status: [accel(3), rot_rate(3), vel(3), command(3), waypoint(2)]
    vel_x, vel_y, vel_z = ego_status[6], ego_status[7], ego_status[8]
    speed = torch.sqrt(vel_x**2 + vel_y**2 + vel_z**2).item()
    
    info_text = f"Command: {cmd_names[cmd_idx]} | Speed: {speed:.2f} m/s | "
    info_text += f"Scene: {sample['scene_token'][:8]}..."
    
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
    from dataset.config_loader import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_data', type=str, default=None,
                        help='Path to preprocessed Senna data with features')
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='Root directory of nuScenes dataset')
    parser.add_argument('--mode', type=str, default=None, choices=['train', 'val'])
    parser.add_argument('--visualize', action='store_true', default=None, help='Visualize samples')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualization images')
    parser.add_argument('--save_prefix', type=str, default=None,
                        help='Prefix for saved visualization filenames')

    args = parser.parse_args()

    defaults = {
        'processed_data': '/mnt/data2/nuscenes/processed_data/senna_val_4obs_with_features.pkl',
        'dataset_root': '/mnt/data2/nuscenes',
        'mode': 'val',
        'visualize': True,
        'num_samples': 5,
        'output_dir': os.path.join(project_root, 'image', 'senna_samples'),
        'save_prefix': 'senna_sample',
    }

    cfg = load_config('senna_dataset', vars(args), defaults=defaults)

    # Create dataset
    dataset = SennaDataset(
        processed_data_path=cfg['processed_data'],
        dataset_root=cfg['dataset_root'],
        mode=cfg['mode'],
        load_bev_images=cfg.get('visualize', False),
        load_hidden_states=True
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
                print(f"  {key:30s}: shape={str(value.shape):25s}, dtype={value.dtype}")
            elif isinstance(value, list) and key == 'fut_obstacles':
                print(f"  {key:30s}: {len(value)} frames")
                for frame_idx, obs_dict in enumerate(value):
                    num_obs = len(obs_dict['gt_boxes'])
                    print(f"    Frame {frame_idx}: {num_obs} obstacles")
            elif value is None:
                print(f"  {key:30s}: None")
            elif isinstance(value, (str, int, float)):
                value_str = str(value) if len(str(value)) < 40 else str(value)[:40] + '...'
                print(f"  {key:30s}: {value_str}")
    
    # Visualize samples
    if cfg.get('visualize', False):
        print(f"\n{'='*70}")
        print(f"Generating Visualizations:")
        print(f"{'='*70}")
        output_dir = cfg.get('output_dir')
        os.makedirs(output_dir, exist_ok=True)
        num_samples_to_visualize = min(cfg.get('num_samples', 5), len(dataset))
        random_indices = random.sample(range(len(dataset)), num_samples_to_visualize)
        print(f"Randomly selected sample indices: {random_indices}")
        
        for idx, sample_idx in enumerate(random_indices):
            sample = dataset[sample_idx]
            save_path = os.path.join(output_dir, f'{cfg.get("save_prefix", "senna_sample")}_{sample_idx:05d}.png')
            visualize_sample(sample, save_path)
        
        print(f"✓ Saved {num_samples_to_visualize} visualizations to {output_dir}")
