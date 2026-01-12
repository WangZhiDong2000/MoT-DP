#!/usr/bin/env python3
import os
import sys
import time
import torch
import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dataset.unified_carla_dataset import CARLAImageDataset
from policy.diffusion_dit_carla_policy import DiffusionDiTCarlaPolicy


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config, checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint['config']

    policy = DiffusionDiTCarlaPolicy(config).to(device)
    
    if 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model weights from checkpoint")
        if 'epoch' in checkpoint:
            print(f"  Checkpoint epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    else:
        print("⚠ No model_state_dict found in checkpoint")
    
    policy.eval()
    return policy




def main():
    # Configuration
    config_path = os.path.join(project_root, "config", "pdm_local.yaml")
    checkpoint_path = os.path.join(project_root, "checkpoints", "carla_dit_best", "carla_policy_best.pt")
    config = load_config(config_path)
    
    # Dataset paths
    dataset_path = os.path.join(
        config.get('training', {}).get('dataset_path', ''), 'val'
    )
    image_data_root = config.get('training', {}).get('image_data_root', '')
    
    print(f"Dataset path: {dataset_path}")
    print(f"Image data root: {image_data_root}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        image_data_root=image_data_root,
        mode='val'
    )
    print(f"✓ Loaded {len(dataset)} validation samples")
    
    policy = load_model(config, checkpoint_path, device)
    
    random.seed(62)  # For reproducibility
    sample_indices = random.sample(range(len(dataset)), 4)
    print(f"Selected sample indices: {sample_indices}")
    
    samples = []
    predictions = []
    rgb_images = []
    rgb_last_frames = []  
    
    # Process each sample
    with torch.no_grad():
        for idx in sample_indices:
            sample = dataset[idx]
            rgb_hist_paths = sample['rgb_hist_jpg']
            loaded_frames = []
            for path in rgb_hist_paths:
                full_path = os.path.join(image_data_root, path)
                img = Image.open(full_path)
                img_array = np.array(img)
                loaded_frames.append(img_array)
  
            rgb_hist = np.array(loaded_frames)
            print(f"  Sample {idx}: Loaded {len(loaded_frames)} frames from paths, RGB hist shape: {rgb_hist.shape}")
            rgb_images.append(rgb_hist)
            last_frame = rgb_hist[-1]
            rgb_last_frames.append(last_frame)

            
            # Prepare observation dict for dp model
            mot_dict ={
                'rgb_hist_jpg': rgb_hist,  # List of file paths
            }
            # gen_vit_tokens, reasoning_query_tokens= mot_model.forward(mot_dict)
            obs_dict = {
                'lidar_token': sample['lidar_token'].unsqueeze(0).to(device),  # Keep last 4 frames (indices 1,2,3,4) from 5-frame horizon
                'lidar_token_global': sample['lidar_token_global'].unsqueeze(0).to(device),  # Keep last 4 frames in time dimension
                'ego_status': sample['ego_status'].unsqueeze(0).to(device),  # Keep last 4 frames from 5-frame horizon
                'gen_vit_tokens': sample['gen_vit_tokens'].unsqueeze(0).to(device),  # (1, ...)
                'reasoning_query_tokens': sample['reasoning_query_tokens'].unsqueeze(0).to(device),  # (1, ...)
                'anchor': sample['anchor']
            }

            
            # Predict
            time1=time.time()
            result = policy.predict_action(obs_dict)
            pred = result['action'][0]  # (pred_horizon, 2)
            time2=time.time()
            print(f"Prediction time for sample {idx}: {time2 - time1:.4f} seconds")
            
            samples.append(sample)
            predictions.append(pred)
            
            print(f"  Sample {idx}: Prediction shape {pred.shape}")
    
    save_dir = os.path.join(project_root, "image")
    os.makedirs(save_dir, exist_ok=True)
    data_save_path = os.path.join(save_dir, "trajectory_data.npz")
    
    # Save trajectory data
    np.savez(data_save_path,
             sample_indices=sample_indices,
             predictions=[p for p in predictions],
             waypoints_hist=[s['waypoints_hist'].cpu().numpy() if isinstance(s['waypoints_hist'], torch.Tensor) else s['waypoints_hist'] for s in samples],
             agent_pos=[s['agent_pos'].cpu().numpy() if isinstance(s['agent_pos'], torch.Tensor) else s['agent_pos'] for s in samples])
    print(f"✓ Saved trajectory data to: {data_save_path}")
    
    # Save RGB images (individual frames)
    for i, (sample_idx, rgb_image) in enumerate(zip(sample_indices, rgb_images)):
        # Convert different types to numpy
        if isinstance(rgb_image, torch.Tensor):
            rgb_array = rgb_image.cpu().numpy()
        elif isinstance(rgb_image, list):
            rgb_array = np.array(rgb_image)
        else:
            rgb_array = rgb_image
        
        # If it's a sequence of frames, save the last one
        if rgb_array.ndim == 4:  # (T, H, W, C) or similar
            rgb_array = rgb_array[-1]
        
        # Handle different possible shapes (C, H, W) or (H, W, C)
        if rgb_array.ndim == 3:
            if rgb_array.shape[0] in [3, 4]:  # Likely (C, H, W) format
                rgb_array = np.transpose(rgb_array, (1, 2, 0))
            
            # Normalize to 0-255 if values are in 0-1 range
            if rgb_array.max() <= 1.0:
                rgb_array = (rgb_array * 255).astype(np.uint8)
            else:
                rgb_array = rgb_array.astype(np.uint8)
            
            # Save image
            if rgb_array.shape[2] >= 3:
                img = Image.fromarray(rgb_array[:, :, :3])
            else:
                img = Image.fromarray(rgb_array)
            image_save_path = os.path.join(save_dir, f"rgb_sample_{sample_idx}.png")
            img.save(image_save_path)
            print(f"✓ Saved RGB image to: {image_save_path}")
    
    # # Create figure with 4 subplots for last frames
    # fig = plt.figure(figsize=(16, 12))
    # for i, (sample_idx, last_frame) in enumerate(zip(sample_indices, rgb_last_frames)):
    #     ax = plt.subplot(2, 2, i + 1)
        
    #     # Convert different types to numpy
    #     if isinstance(last_frame, torch.Tensor):
    #         frame_array = last_frame.cpu().numpy()
    #     elif isinstance(last_frame, list):
    #         frame_array = np.array(last_frame)
    #     else:
    #         frame_array = last_frame
        
    #     print(f"  Frame {i} - Before processing: shape={frame_array.shape}, ndim={frame_array.ndim}, dtype={frame_array.dtype}")
        
    #     # Handle different possible shapes (C, H, W) or (H, W, C)
    #     if frame_array.ndim == 3:
    #         if frame_array.shape[0] in [3, 4]:  # Likely (C, H, W) format
    #             frame_array = np.transpose(frame_array, (1, 2, 0))
            
    #         # Normalize to 0-255 if values are in 0-1 range
    #         if frame_array.max() <= 1.0:
    #             frame_array = (frame_array * 255).astype(np.uint8)
    #         else:
    #             frame_array = frame_array.astype(np.uint8)
            
    #         print(f"  Frame {i} - After processing: shape={frame_array.shape}")
            
    #         # Display image, handle both RGB and RGBA
    #         if frame_array.shape[2] >= 3:
    #             ax.imshow(frame_array[:, :, :3])
    #         else:
    #             ax.imshow(frame_array)
    #     else:
    #         print(f"  ⚠ Frame {i} - Unexpected shape: {frame_array.shape}, skipping...")
    #         ax.text(0.5, 0.5, f'Invalid frame shape: {frame_array.shape}', 
    #                ha='center', va='center', transform=ax.transAxes)
        
    #     ax.set_title(f'Sample {sample_idx} - Last Frame', fontsize=12, fontweight='bold')
    #     ax.axis('off')
    
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.2, wspace=0.2)
    # last_frame_path = os.path.join(save_dir, "rgb_samples_last_frames.png")
    # plt.savefig(last_frame_path, dpi=150, format='png', bbox_inches='tight')
    # plt.close()
    # print(f"✓ Saved last frames figure to: {last_frame_path}")
    

    



if __name__ == "__main__":
    main()
