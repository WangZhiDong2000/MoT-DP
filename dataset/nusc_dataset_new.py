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
    Dataset for preprocessed nuScenes data.
    
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
    """
    
    def __init__(self,
                 processed_data_path: str,
                 dataset_root: str,
                 mode: str = 'train'):
        
        self.dataset_root = dataset_root
        self.mode = mode
        
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load BEV features
        try:
            lidar_token = torch.load(sample['bev_token_path'], weights_only=True)
            lidar_token_global = torch.load(sample['bev_token_global_path'], weights_only=True)
        except Exception as e:
            print(f"Error loading BEV features for sample {idx}: {e}")
            # Create dummy features
            lidar_token = torch.zeros(196, 512)
            lidar_token_global = torch.zeros(1, 512)
        
        # Expand to obs_horizon by repeating the current frame
        # Shape: (196, 512) -> (obs_horizon, 196, 512)
        lidar_token = lidar_token.unsqueeze(0).repeat(self.obs_horizon, 1, 1)
        # Shape: (1, 512) -> (obs_horizon, 1, 512)
        lidar_token_global = lidar_token_global.unsqueeze(0).repeat(self.obs_horizon, 1, 1)
        
        # Convert numpy arrays to tensors (using torch.tensor for compatibility)
        hist_waypoints = torch.tensor(sample['hist_waypoints'], dtype=torch.float32)  # (obs_horizon, 2)
        fut_waypoints = torch.tensor(sample['fut_waypoints'], dtype=torch.float32)    # (action_horizon, 2)
        fut_valid_mask = torch.tensor(sample['fut_valid_mask'], dtype=torch.bool)     # (action_horizon,)
        ego_status = torch.tensor(sample['ego_status'], dtype=torch.float32)          # (10,)
        nav_command = torch.tensor(sample['nav_command'], dtype=torch.float32)        # (3,)
        
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
            
            # Ego status and command
            'speed': ego_status.unsqueeze(0).repeat(self.obs_horizon, 1),  # (obs_horizon, 10)
            'command': nav_command.unsqueeze(0).repeat(self.obs_horizon, 1),  # (obs_horizon, 3)
            
            # Metadata
            'scene_token': sample['scene_token'],
            'sample_token': sample['sample_token'],
            'timestamp': sample['timestamp'],
            'lidar_token_str': sample['lidar_token'],
        }
        
        return output


class NUSCDatasetWithVisualization(NUSCDataset):
    """
    Extended nuScenes dataset with visualization support.
    Loads BEV images for visualization in validation mode.
    """
    
    def __init__(self,
                 processed_data_path: str,
                 dataset_root: str,
                 mode: str = 'train'):
        super().__init__(processed_data_path, dataset_root, mode)
        
        # BEV image transform for visualization
        self.lidar_bev_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __getitem__(self, idx):
        output = super().__getitem__(idx)
        
        # Load BEV image for visualization in val mode
        if self.mode == 'val':
            sample = self.samples[idx]
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
    """
    # Stack all tensors in the batch
    output = {}
    
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            output[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(batch[0][key], (str, int, float)):
            output[key] = [sample[key] for sample in batch]
        else:
            output[key] = [sample[key] for sample in batch]
    
    return output


def visualize_sample(sample, save_path=None):
    """
    Visualize a single sample with waypoints and BEV image.
    
    Args:
        sample: Dictionary containing sample data
        save_path: Path to save the visualization (optional)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Waypoints
    ax1 = axes[0]
    
    # Historical waypoints (blue)
    hist_wp = sample['hist_waypoints'].cpu().numpy()
    ax1.plot(hist_wp[:, 0], hist_wp[:, 1], 'b.-', label='Historical', markersize=10, linewidth=2)
    
    # Future waypoints (red)
    fut_wp = sample['agent_pos'].cpu().numpy()
    valid_mask = sample['fut_valid_mask'].cpu().numpy()
    ax1.plot(fut_wp[:, 0], fut_wp[:, 1], 'r.-', label='Future', markersize=10, linewidth=2)
    
    # Mark invalid waypoints
    invalid_wp = fut_wp[~valid_mask]
    if len(invalid_wp) > 0:
        ax1.scatter(invalid_wp[:, 0], invalid_wp[:, 1], c='orange', s=100, 
                   marker='x', label='Invalid', zorder=10)
    
    # Current position
    ax1.scatter(0, 0, c='green', s=200, marker='*', label='Current', zorder=10)
    
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_title('Trajectory (Ego Frame)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: BEV image (if available)
    ax2 = axes[1]
    if 'lidar_bev' in sample and sample['lidar_bev'] is not None:
        # Take the last frame from obs_horizon
        bev_tensor = sample['lidar_bev'][-1].cpu()
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        bev_tensor = bev_tensor * std + mean
        bev_tensor = torch.clamp(bev_tensor, 0, 1)
        bev_img = bev_tensor.numpy().transpose(1, 2, 0)
        ax2.imshow(bev_img)
        ax2.set_title('BEV Image')
    else:
        ax2.text(0.5, 0.5, 'No BEV Image', ha='center', va='center', fontsize=20)
        ax2.set_title('BEV Image (Not Available)')
    ax2.axis('off')
    
    # Add info text
    nav_cmd = sample['command'][0].cpu()
    cmd_names = ['Left', 'Straight', 'Right']
    cmd_idx = torch.argmax(nav_cmd).item()
    speed = sample['speed'][0, 0].item() if sample['speed'].numel() > 0 else 0.0
    
    info_text = f"Command: {cmd_names[cmd_idx]}\n"
    info_text += f"Speed: {speed:.2f} m/s\n"
    info_text += f"Scene: {sample['scene_token'][:8]}...\n"
    
    fig.suptitle(info_text, fontsize=12, y=0.98)
    
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
    if args.visualize and args.mode == 'val':
        dataset = NUSCDatasetWithVisualization(
            processed_data_path=args.processed_data,
            dataset_root=args.dataset_root,
            mode=args.mode
        )
    else:
        dataset = NUSCDataset(
            processed_data_path=args.processed_data,
            dataset_root=args.dataset_root,
            mode=args.mode
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
