#!/usr/bin/env python
"""
Extract SparseDrive sparse features from nuScenes dataset.

This script loads the SparseDrive Stage1 model and extracts sparse perception features
for each sample in the nuScenes dataset. The features are saved to disk for later use
in trajectory prediction models.

Features saved per sample:
- det_instance_feature: (50, 256) - Top-50 detection instance features
- det_anchor_embed: (50, 256) - Detection anchor embeddings
- det_classification: (50, 10) - Detection class scores
- det_prediction: (50, 11) - Detection box predictions (xyz, wlh, sin/cos yaw, vx, vy)
- map_instance_feature: (10, 256) - Top-10 map instance features
- map_anchor_embed: (10, 256) - Map anchor embeddings
- ego_feature: (1, 256) - Ego vehicle feature from BEV

Usage:
    cd /root/z_projects/code/MoT-DP-1/SparseDrive
    python ../dataset/extract_sparsedrive_features.py --split train
    python ../dataset/extract_sparsedrive_features.py --split val
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add SparseDrive to path - must be run from SparseDrive directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SPARSEDRIVE_ROOT = os.path.join(PROJECT_ROOT, 'SparseDrive')
sys.path.insert(0, SPARSEDRIVE_ROOT)


def topk(confidence, k, *inputs):
    """Select top-k elements based on confidence scores."""
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (
        indices + torch.arange(bs, device=indices.device)[:, None] * N
    ).reshape(-1)
    outputs = []
    for input_tensor in inputs:
        outputs.append(input_tensor.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


def extract_sparse_features(det_output, map_output, ego_feature, num_det=50, num_map=10):
    """Extract sparse features from detection and map outputs."""
    # Detection features - TopK selection
    instance_feature = det_output["instance_feature"]
    anchor_embed = det_output["anchor_embed"]
    det_classification = det_output["classification"][-1].sigmoid()
    det_prediction = det_output["prediction"][-1]
    det_confidence = det_classification.max(dim=-1).values
    
    _, (det_feat, det_anchor, det_cls, det_pred) = topk(
        det_confidence, num_det,
        instance_feature, anchor_embed, det_classification, det_prediction
    )
    
    # Map features - TopK selection
    map_instance_feature = map_output["instance_feature"]
    map_anchor_embed = map_output["anchor_embed"]
    map_classification = map_output["classification"][-1].sigmoid()
    map_confidence = map_classification.max(dim=-1).values
    
    _, (map_feat, map_anchor) = topk(
        map_confidence, num_map,
        map_instance_feature, map_anchor_embed
    )
    
    return {
        'det_instance_feature': det_feat.half().cpu(),      # (B, 50, 256)
        'det_anchor_embed': det_anchor.half().cpu(),        # (B, 50, 256)
        'det_classification': det_cls.half().cpu(),         # (B, 50, 10)
        'det_prediction': det_pred.half().cpu(),            # (B, 50, 11)
        'map_instance_feature': map_feat.half().cpu(),      # (B, 10, 256)
        'map_anchor_embed': map_anchor.half().cpu(),        # (B, 10, 256)
        'ego_feature': ego_feature.half().cpu(),            # (B, 1, 256)
    }


def main():
    parser = argparse.ArgumentParser(description='Extract SparseDrive features')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                        help='Dataset split to process')
    parser.add_argument('--data-root', type=str, default='/mnt/data2/nuscenes',
                        help='Path to nuScenes data')
    parser.add_argument('--infos-root', type=str, default='/root/z_projects/code/MoT-DP-1/data/infos',
                        help='Path to info pkl files')
    parser.add_argument('--checkpoint', type=str, 
                        default='/root/z_projects/code/MoT-DP-1/checkpoints/nusc_sparsedrive/sparsedrive_stage1.pth',
                        help='Path to SparseDrive checkpoint')
    parser.add_argument('--output-dir', type=str, default='/mnt/data2/nuscenes/samples/SPARSEDRIVE_FEATURE',
                        help='Output directory for features')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--num-det', type=int, default=50,
                        help='Number of detection instances to keep')
    parser.add_argument('--num-map', type=int, default=10,
                        help='Number of map instances to keep')
    args = parser.parse_args()
    
    # Import mmcv and related modules
    import mmcv
    from mmcv import Config
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmcv.parallel import MMDataParallel
    from mmdet.datasets import build_dataset
    from mmdet.datasets import build_dataloader as build_dataloader_origin
    from mmdet.models import build_detector
    
    # Set CUDA device
    torch.cuda.set_device(args.gpu_id)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading config from {SPARSEDRIVE_ROOT}...")
    config_path = os.path.join(SPARSEDRIVE_ROOT, 'projects/configs/sparsedrive_small_stage1.py')
    cfg = Config.fromfile(config_path)
    
    # Import plugin modules
    import importlib
    plugin_dir = cfg.plugin_dir
    _module_dir = os.path.dirname(plugin_dir)
    _module_dir = _module_dir.split("/")
    _module_path = _module_dir[0]
    for m in _module_dir[1:]:
        _module_path = _module_path + "." + m
    print(f"Importing plugin: {_module_path}")
    plg_lib = importlib.import_module(_module_path)
    
    # Update data paths
    cfg.data_root = args.data_root
    
    # Update anno paths based on split
    if args.split == 'train':
        ann_file = os.path.join(args.infos_root, 'nuscenes_infos_train.pkl')
    else:
        ann_file = os.path.join(args.infos_root, 'nuscenes_infos_val.pkl')
    
    # Update kmeans paths to use project's data
    kmeans_dir = os.path.join(PROJECT_ROOT, 'data/kmeans')
    cfg.model.head.det_head.instance_bank.anchor = os.path.join(kmeans_dir, 'kmeans_det_900.npy')
    
    # Create map anchor if needed
    map_anchor_path = os.path.join(kmeans_dir, 'kmeans_map_100.npy')
    if not os.path.exists(map_anchor_path):
        print(f"Creating default map anchor at {map_anchor_path}...")
        # Create default map anchor with shape (100, 40) - 100 anchors, 20 points * 2 coords
        default_map_anchor = np.random.randn(100, 40).astype(np.float32) * 0.01
        np.save(map_anchor_path, default_map_anchor)
    cfg.model.head.map_head.instance_bank.anchor = map_anchor_path
    
    # Configure data augmentation for test mode
    data_aug_conf = {
        "resize_lim": (0.40, 0.47),
        "final_dim": (256, 704),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
        "rot3d_range": [0, 0],
    }
    
    # Configure test dataset
    cfg.data.test.ann_file = ann_file
    cfg.data.test.data_root = args.data_root
    cfg.data.test.data_aug_conf = data_aug_conf
    cfg.data.test.test_mode = True
    
    # Disable training configs
    cfg.model.train_cfg = None
    
    print(f"Building dataset for {args.split}...")
    dataset = build_dataset(cfg.data.test)
    print(f"Dataset size: {len(dataset)}")
    
    data_loader = build_dataloader_origin(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=args.num_workers,
        dist=False,
        shuffle=False,
    )
    
    print(f"Building model...")
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    
    # Enable FP16
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    
    # Set model classes
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES
    
    # Wrap model
    model = MMDataParallel(model, device_ids=[args.gpu_id])
    model.eval()
    
    # Reset instance banks to clear temporal information
    model.module.head.det_head.instance_bank.reset()
    model.module.head.map_head.instance_bank.reset()
    
    print(f"Extracting features for {len(dataset)} samples...")
    print(f"Output directory: {output_dir}")
    
    # Process each batch
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, data in enumerate(tqdm(data_loader, desc=f"Processing {args.split}")):
        try:
            # Get sample tokens from img_metas
            if 'img_metas' in data:
                img_metas = data['img_metas'].data[0]
                # Try to get token from the first meta
                tokens = []
                for meta in img_metas:
                    token = meta.get('token', None)
                    if token is None:
                        # Try to extract from filename
                        token = f"sample_{i}"
                    tokens.append(token)
            else:
                tokens = [f"sample_{i}"]
            
            # Check if all samples in batch already processed
            all_exist = all(
                (output_dir / f"{token}.npz").exists() 
                for token in tokens
            )
            if all_exist:
                skipped_count += len(tokens)
                continue
            
            # Move data to GPU
            img = data['img'].data[0].cuda()
            batch_size = img.shape[0]
            
            # Prepare data dict for model
            data_dict = {}
            if 'img_metas' in data:
                data_dict['img_metas'] = data['img_metas'].data[0]
            if 'timestamp' in data:
                if hasattr(data['timestamp'], 'data'):
                    data_dict['timestamp'] = data['timestamp'].data[0].cuda()
                else:
                    data_dict['timestamp'] = data['timestamp'].cuda()
            if 'projection_mat' in data:
                if hasattr(data['projection_mat'], 'data'):
                    proj_mat = data['projection_mat'].data[0].cuda()
                else:
                    proj_mat = data['projection_mat'].cuda()
                # Add batch dimension if needed: (num_cams, 4, 4) -> (bs, num_cams, 4, 4)
                if proj_mat.dim() == 3:
                    proj_mat = proj_mat.unsqueeze(0).expand(batch_size, -1, -1, -1)
                data_dict['projection_mat'] = proj_mat
            if 'image_wh' in data:
                if hasattr(data['image_wh'], 'data'):
                    img_wh = data['image_wh'].data[0].cuda()
                else:
                    img_wh = data['image_wh'].cuda()
                # Add batch dimension if needed: (num_cams, 2) -> (bs, num_cams, 2)
                if img_wh.dim() == 2:
                    img_wh = img_wh.unsqueeze(0).expand(batch_size, -1, -1)
                data_dict['image_wh'] = img_wh
            
            # Extract features
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=fp16_cfg is not None):
                    # Get feature maps from backbone
                    feature_maps = model.module.extract_feat(img)
                    
                    # Reset instance bank for each sample to avoid temporal contamination
                    model.module.head.det_head.instance_bank.reset()
                    model.module.head.map_head.instance_bank.reset()
                    
                    # Run detection head
                    det_output = model.module.head.det_head(feature_maps, data_dict)
                    
                    # Run map head
                    map_output = model.module.head.map_head(feature_maps, data_dict)
                    
                    # Get ego feature from feature map pooling
                    # Use the highest resolution feature map from the last level
                    try:
                        from projects.mmdet3d_plugin.ops import feature_maps_format
                        feature_maps_inv = feature_maps_format(feature_maps, inverse=True)
                        feature_map = feature_maps_inv[0][-1][:, 0]  # (B, C, H, W)
                    except:
                        # Fallback: use the last feature map directly
                        if isinstance(feature_maps, (list, tuple)):
                            feature_map = feature_maps[-1][:, 0]  # (B, C, H, W)
                        else:
                            feature_map = feature_maps[:, 0]
                    
                    # Global average pooling for ego feature
                    ego_feature = feature_map.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
                    ego_feature = ego_feature.squeeze(-1).squeeze(-1).unsqueeze(1)  # (B, 1, C)
                    
                    # Extract sparse features
                    features = extract_sparse_features(
                        det_output, map_output, ego_feature, 
                        num_det=args.num_det, num_map=args.num_map
                    )
            
            # Save features for each sample in batch
            for b, token in enumerate(tokens):
                output_path = output_dir / f"{token}.npz"
                
                # Skip if already exists
                if output_path.exists():
                    skipped_count += 1
                    continue
                
                # Extract single sample features
                sample_features = {
                    k: v[b].numpy() for k, v in features.items()
                }
                
                # Save as compressed npz
                np.savez_compressed(output_path, **sample_features)
                processed_count += 1
                
        except Exception as e:
            error_count += 1
            if error_count <= 10:  # Only print first 10 errors
                print(f"\nError processing batch {i}: {e}")
                import traceback
                traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Done! Results:")
    print(f"  Processed: {processed_count} samples")
    print(f"  Skipped (existing): {skipped_count} samples")
    print(f"  Errors: {error_count} samples")
    print(f"  Features saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
