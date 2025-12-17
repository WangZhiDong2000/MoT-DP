#!/usr/bin/env python3
"""
Transfuser Feature Extraction Script

This script extracts features from RGB images and LiDAR point clouds using the 
pretrained Transfuser backbone. The processing follows exactly the same pipeline 
as in preprocess_pdm_lite.py to ensure consistency.

Input:
    - RGB images: <scene_path>/rgb/*.jpg
    - LiDAR point clouds: <scene_path>/lidar/*.laz

Output:
    - features: <scene_path>/transfuser_features/<frame_id>_features.pt
    - fused_features: <scene_path>/transfuser_features/<frame_id>_fused_features.pt

Note: The forward pass returns (features, fused_features, image_feature_grid),
      but image_feature_grid is NOT saved as per requirements.

Usage:
    python preprocess_transfuser_features.py --data_root /path/to/data --checkpoint_dir /path/to/checkpoint
"""

import numpy as np
import torch
import cv2
import laspy
import os
import sys
import glob
import argparse
import json
from pathlib import Path
from tqdm import tqdm

# Add project root and Transfuser to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

transfuser_path = project_root / 'Transfuser' / 'team_code'
if str(transfuser_path) not in sys.path:
    sys.path.insert(0, str(transfuser_path))


# ============================================================================
# Config class - matches Transfuser config.py exactly
# ============================================================================
class TransfuserConfig:
    """
    Minimal config class containing only parameters needed for feature extraction.
    All values must match Transfuser config.py exactly.
    """
    def __init__(self):
        # Image parameters (config.py lines 360-366)
        self.camera_width = 1024
        self.camera_height = 512
        self.camera_fov = 110
        self.crop_image = True
        self.cropped_height = 384
        self.cropped_width = 1024
        
        # LiDAR parameters (config.py lines 370-410)
        self.lidar_resolution_width = 256
        self.lidar_resolution_height = 256
        self.pixels_per_meter = 4.0
        self.hist_max_per_pixel = 5
        self.lidar_split_height = 0.2
        self.use_ground_plane = False
        self.min_x = -32
        self.max_x = 32
        self.min_y = -32
        self.max_y = 32
        self.max_height_lidar = 100.0
        
        # Model architecture parameters - will be overwritten by checkpoint config
        self.image_architecture = 'regnety_032'
        self.lidar_architecture = 'regnety_032'
        self.use_velocity = True
        self.seq_len = 1
        self.img_seq_len = 1
        self.lidar_seq_len = 1
        
        # Transformer parameters
        self.n_layer = 4
        self.n_head = 4
        self.block_exp = 4
        self.embd_pdrop = 0.1
        self.attn_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.gpt_linear_layer_init_mean = 0.0
        self.gpt_linear_layer_init_std = 0.02
        self.gpt_layer_norm_init_weight = 1.0
        
        # Feature fusion parameters
        self.img_vert_anchors = 6
        self.img_horz_anchors = 16
        self.lidar_vert_anchors = 8
        self.lidar_horz_anchors = 8
        
        # Model output configuration - will be overwritten by checkpoint config
        self.transformer_decoder_join = True
        self.add_features = True
        self.detect_boxes = True
        self.use_bev_semantic = True
        self.use_semantic = False
        self.use_depth = False
        self.bev_features_chanels = 64
        self.bev_upsample_factor = 2
        self.bev_down_sample_factor = 4
        
        # Normalization
        self.normalize_imagenet = True
        
        # Perspective parameters
        self.perspective_downsample_factor = 1


# ============================================================================
# Image Processing - EXACTLY matches Transfuser data.py
# ============================================================================
def crop_array(config, images_i):
    """
    Crop RGB images to the desired height and width.
    EXACTLY matches Transfuser transfuser_utils.py crop_array() (lines 809-823)
    
    Args:
        config: Config object with cropping parameters
        images_i: numpy array with shape (H, W, C) or (H, W)
    Returns:
        Cropped array
    """
    if config.crop_image:
        assert config.cropped_height <= images_i.shape[0]
        assert config.cropped_width <= images_i.shape[1]
        side_crop_amount = (images_i.shape[1] - config.cropped_width) // 2
        if len(images_i.shape) > 2:
            return images_i[0:config.cropped_height, side_crop_amount:images_i.shape[1] - side_crop_amount, :]
        else:
            return images_i[0:config.cropped_height, side_crop_amount:images_i.shape[1] - side_crop_amount]
    else:
        return images_i


def load_image(image_path, config):
    """
    Load and preprocess RGB image.
    EXACTLY matches Transfuser data.py (lines 404-406):
        images_i = cv2.imread(str(images[i], encoding='utf-8'), cv2.IMREAD_COLOR)
        images_i = cv2.cvtColor(images_i, cv2.COLOR_BGR2RGB)
        images_i = t_u.crop_array(self.config, images_i)
    
    And data.py line 580:
        data['rgb'] = np.transpose(processed_image, (2, 0, 1))
    
    Args:
        image_path: Path to RGB image
        config: Config object
    Returns:
        image: numpy array (C, H, W) = (3, 384, 1024), values in [0, 255]
    """
    # Step 1: Load with cv2 (BGR format)
    images_i = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if images_i is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    
    # Step 2: Convert BGR to RGB
    images_i = cv2.cvtColor(images_i, cv2.COLOR_BGR2RGB)
    
    # Step 3: Crop to (384, 1024)
    images_i = crop_array(config, images_i)
    
    # Step 4: Transpose to (C, H, W)
    images_i = np.transpose(images_i, (2, 0, 1))
    
    return images_i.astype(np.float32)


# ============================================================================
# LiDAR Processing - EXACTLY matches Transfuser data.py
# ============================================================================
def lidar_to_histogram_features(lidar, config):
    """
    Convert LiDAR point cloud into 2-bin histogram over a fixed size grid.
    EXACTLY matches Transfuser data.py lidar_to_histogram_features() (lines 916-946)
    
    Args:
        lidar: (N, 3) numpy array, LiDAR point cloud
        config: Config object with LiDAR parameters
    Returns:
        features: (C, H, W) numpy array, LiDAR as sparse BEV image
    """
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
        return overhead_splat.T

    # Remove points above the vehicle
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


def load_lidar(lidar_path, config):
    """
    Load and process LiDAR point cloud.
    EXACTLY matches Transfuser data.py (lines 401-402):
        las_object = laspy.read(str(lidars[i], encoding='utf-8'))
        lidars_i = las_object.xyz
    
    Args:
        lidar_path: Path to .laz file
        config: Config object
    Returns:
        lidar_bev: (C, H, W) numpy array
    """
    # Step 1: Load with laspy
    las_object = laspy.read(lidar_path)
    lidar = las_object.xyz  # (N, 3) numpy array
    
    # Step 2: Convert to histogram features
    lidar_bev = lidar_to_histogram_features(lidar, config)
    
    return lidar_bev


# ============================================================================
# Model Loading
# ============================================================================
def load_transfuser_model(checkpoint_dir, config, device='cuda'):
    """
    Load pretrained Transfuser model.
    
    Args:
        checkpoint_dir: Directory containing model checkpoint and config
        config: Config object
        device: Device to load model on
    Returns:
        model: Loaded TransfuserBackbone in eval mode
    """
    # Import TransfuserBackbone directly to avoid path conflicts
    import importlib.util
    transfuser_module_path = os.path.join(project_root, 'model', 'transfuser.py')
    spec = importlib.util.spec_from_file_location("transfuser_backbone", transfuser_module_path)
    transfuser_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transfuser_module)
    TransfuserBackbone = transfuser_module.TransfuserBackbone
    
    print(f"Loading Transfuser model from: {checkpoint_dir}")
    
    # Load config from checkpoint if available
    config_path = os.path.join(checkpoint_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        # Update config with saved values
        for key, value in saved_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        print("✓ Loaded config from checkpoint")
    
    # Create model
    model = TransfuserBackbone(config)
    
    # Find and load checkpoint
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'model_*.pth'))
    if checkpoint_files:
        # Use the first checkpoint (they should all be the same for ensemble)
        checkpoint_path = sorted(checkpoint_files)[0]
        print(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Filter out non-backbone keys and remove prefix if needed
        backbone_state_dict = {}
        for k, v in state_dict.items():
            # Remove various prefixes that might exist
            new_key = k
            for prefix in ['module.', '_model.', 'backbone.']:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            backbone_state_dict[new_key] = v
        
        # Load with strict=False to handle potential mismatches
        missing, unexpected = model.load_state_dict(backbone_state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print("✓ Checkpoint loaded successfully")
    else:
        print("⚠ Warning: No checkpoint found, using random initialization")
    
    model = model.to(device)
    model.eval()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    return model


# ============================================================================
# Feature Extraction
# ============================================================================
def extract_features(model, image, lidar_bev, config, device='cuda', verbose=False):
    """
    Extract features from image and LiDAR using Transfuser backbone.
    
    The forward pass in transfuser.py returns:
        features, fused_features, image_feature_grid
    
    We only save features and fused_features (not image_feature_grid).
    
    Args:
        model: TransfuserBackbone model
        image: (3, H, W) numpy array, RGB image
        lidar_bev: (C, H, W) numpy array, LiDAR BEV
        config: Config object
        device: Device
        verbose: Print timing information
    Returns:
        features: numpy array, BEV features
        fused_features: numpy array, fused global features
        forward_time: Time taken for forward pass (in milliseconds)
    """
    import time
    
    # Convert to tensors and add batch dimension
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)  # (1, 3, 384, 1024)
    lidar_tensor = torch.from_numpy(lidar_bev).unsqueeze(0).to(device)  # (1, C, 256, 256)
    
    with torch.no_grad():
        # Warm up (optional)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Record time
        start_time = time.time()
        
        # Forward pass
        features, fused_features, image_feature_grid = model(image_tensor, lidar_tensor)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
    
    forward_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    if verbose:
        print(f"  Forward pass: {forward_time:.2f} ms")
    
    # Convert to numpy
    if features is not None:
        features = features.cpu().numpy()
    fused_features = fused_features.cpu().numpy()
    
    return features, fused_features, forward_time


# ============================================================================
# Main Processing
# ============================================================================
def process_scene(scene_path, model, config, device='cuda', overwrite=False, verbose=False):
    """
    Process all frames in a scene and save features.
    
    Args:
        scene_path: Path to scene directory
        model: TransfuserBackbone model
        config: Config object
        device: Device
        overwrite: Whether to overwrite existing features
        verbose: Print timing information for each frame
    Returns:
        num_processed: Number of frames processed
        avg_forward_time: Average forward pass time (in milliseconds)
    """
    import time as time_module
    
    rgb_dir = os.path.join(scene_path, 'rgb')
    lidar_dir = os.path.join(scene_path, 'lidar')
    output_dir = os.path.join(scene_path, 'transfuser_features')
    
    if not os.path.exists(rgb_dir) or not os.path.exists(lidar_dir):
        return 0, 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all RGB files
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.jpg')))
    if not rgb_files:
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
    
    num_processed = 0
    forward_times = []
    
    for rgb_path in rgb_files:
        frame_id = os.path.splitext(os.path.basename(rgb_path))[0]
        
        # Check if features already exist (use .pt format)
        features_path = os.path.join(output_dir, f'{frame_id}_features.pt')
        fused_features_path = os.path.join(output_dir, f'{frame_id}_fused_features.pt')
        
        if not overwrite and os.path.exists(features_path) and os.path.exists(fused_features_path):
            continue
        
        # Find corresponding LiDAR file
        lidar_path = os.path.join(lidar_dir, f'{frame_id}.laz')
        if not os.path.exists(lidar_path):
            continue
        
        try:
            # Load data
            image = load_image(rgb_path, config)
            lidar_bev = load_lidar(lidar_path, config)
            
            # Extract features (with timing)
            features, fused_features, forward_time = extract_features(
                model, image, lidar_bev, config, device, verbose=verbose
            )
            forward_times.append(forward_time)
            
            # Save features as .pt files (PyTorch format)
            if features is not None:
                torch.save(torch.from_numpy(features), features_path)
            torch.save(torch.from_numpy(fused_features), fused_features_path)
            
            num_processed += 1
            
        except Exception as e:
            print(f"Error processing {rgb_path}: {e}")
            continue
    
    # Calculate average forward time
    avg_forward_time = sum(forward_times) / len(forward_times) if forward_times else 0
    
    return num_processed, avg_forward_time


def find_all_scenes(data_root):
    """
    Find all scene directories containing rgb and lidar subdirectories.
    
    Args:
        data_root: Root data directory
    Returns:
        scene_paths: List of scene directory paths
    """
    scene_paths = []
    
    for root, dirs, files in os.walk(data_root):
        # Check if this directory has both 'rgb' and 'lidar' subdirectories
        if 'rgb' in dirs and 'lidar' in dirs:
            scene_paths.append(root)
    
    return sorted(scene_paths)


def main():
    parser = argparse.ArgumentParser(description='Extract Transfuser features from RGB and LiDAR data')
    parser.add_argument('--data_root', type=str, 
                        default='/home/wang/Project/carla_garage/data_mini',
                        help='Root directory of data')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/home/wang/Project/carla_garage/leaderboard/leaderboard/pretrained_models/all_towns',
                        help='Directory containing pretrained Transfuser checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing features')
    parser.add_argument('--scene', type=str, default=None,
                        help='Process only a specific scene (relative path from data_root)')
    args = parser.parse_args()
    
    # Create config
    config = TransfuserConfig()
    
    # Load model first
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = load_transfuser_model(args.checkpoint_dir, config, device)
    
    # Find scenes to process
    if args.scene:
        scene_paths = [os.path.join(args.data_root, args.scene)]
    else:
        print("Scanning for scenes...")
        scene_paths = find_all_scenes(args.data_root)
    
    print(f"Found {len(scene_paths)} scenes to process")
    
    # Process each scene
    total_processed = 0
    all_forward_times = []
    
    for scene_path in tqdm(scene_paths, desc='Processing scenes'):
        num_processed, avg_forward_time = process_scene(
            scene_path, model, config, device, args.overwrite, verbose=False
        )
        total_processed += num_processed
        all_forward_times.append(avg_forward_time)
    
    # Calculate overall statistics
    overall_avg_time = sum(all_forward_times) / len([t for t in all_forward_times if t > 0]) if any(all_forward_times) else 0
    
    print(f"\n{'='*60}")
    print(f"✓ Processing Complete")
    print(f"{'='*60}")
    print(f"Total frames processed: {total_processed}")
    print(f"Scenes processed: {len(scene_paths)}")
    print(f"\nTiming Statistics:")
    print(f"  Average forward pass time: {overall_avg_time:.2f} ms/frame")
    if overall_avg_time > 0:
        fps = 1000 / overall_avg_time
        print(f"  Equivalent throughput: {fps:.1f} fps")
    print(f"\nFeatures saved to: <scene_path>/transfuser_features/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
