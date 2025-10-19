#!/usr/bin/env python3
"""
BEV Feature Loader

Helper module for efficiently loading preprocessed BEV features during training.
This module provides utilities to load the precomputed lidar_token and 
lidar_token_global features, avoiding repeated feature extraction.

Usage Example:
    from dataset.bev_feature_loader import BEVFeatureLoader
    
    # Initialize loader
    loader = BEVFeatureLoader(dataset_root='/path/to/dataset')
    
    # Load features for a specific frame
    features = loader.load_features(scene_path, frame_id)
    lidar_token = features['lidar_token']  # (seq_len, 512)
    lidar_token_global = features['lidar_token_global']  # (1, 512)
    
    # Or use batch loading
    batch_features = loader.load_batch(scene_paths, frame_ids)
"""

import numpy as np
import torch
import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings


class BEVFeatureLoader:
    """
    Loader for preprocessed BEV features
    
    This class provides efficient loading of precomputed lidar_token and 
    lidar_token_global features saved by preprocess_bev_features.py
    """
    
    def __init__(
        self, 
        dataset_root: Optional[str] = None,
        cache_size: int = 1000,
        return_torch: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize the BEV feature loader
        
        Args:
            dataset_root: Root directory of the dataset (optional)
            cache_size: Maximum number of features to cache in memory
            return_torch: If True, return torch tensors; if False, return numpy arrays
            device: Device to load torch tensors on
        """
        self.dataset_root = dataset_root
        self.cache_size = cache_size
        self.return_torch = return_torch
        self.device = device
        
        # Simple LRU-style cache
        self._cache = {}
        self._cache_order = []
        
    def _get_feature_paths(self, scene_path: str, frame_id: str) -> Tuple[str, str]:
        """
        Get the file paths for a frame's features
        
        Args:
            scene_path: Path to the scene directory
            frame_id: Frame identifier (without extension)
            
        Returns:
            token_path: Path to lidar_token file
            token_global_path: Path to lidar_token_global file
        """
        feature_dir = os.path.join(scene_path, 'lidar_bev_features')
        token_path = os.path.join(feature_dir, f'{frame_id}_token.pt')
        token_global_path = os.path.join(feature_dir, f'{frame_id}_token_global.pt')
        
        return token_path, token_global_path
    
    def _load_from_disk(self, token_path: str, token_global_path: str) -> Dict[str, torch.Tensor]:
        """
        Load features from disk
        
        Args:
            token_path: Path to lidar_token file
            token_global_path: Path to lidar_token_global file
            
        Returns:
            features: Dictionary containing 'lidar_token' and 'lidar_token_global' as tensors
        """
        if not os.path.exists(token_path):
            raise FileNotFoundError(f"Token file not found: {token_path}")
        if not os.path.exists(token_global_path):
            raise FileNotFoundError(f"Token global file not found: {token_global_path}")
        
        lidar_token = torch.load(token_path)
        lidar_token_global = torch.load(token_global_path)
        
        return {
            'lidar_token': lidar_token,
            'lidar_token_global': lidar_token_global
        }
    
    def _add_to_cache(self, cache_key: str, features: Dict[str, np.ndarray]):
        """Add features to cache with LRU eviction"""
        if cache_key in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
        else:
            # Add new entry
            self._cache[cache_key] = features
            self._cache_order.append(cache_key)
            
            # Evict oldest if cache is full
            if len(self._cache) > self.cache_size:
                oldest_key = self._cache_order.pop(0)
                del self._cache[oldest_key]
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, np.ndarray]]:
        """Get features from cache if available"""
        if cache_key in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return self._cache[cache_key]
        return None
    
    def load_features(
        self, 
        scene_path: str, 
        frame_id: str,
        use_cache: bool = True
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Load features for a single frame
        
        Args:
            scene_path: Path to the scene directory
            frame_id: Frame identifier (without extension)
            use_cache: Whether to use caching
            
        Returns:
            features: Dictionary containing 'lidar_token' and 'lidar_token_global'
                     Returns torch tensors if return_torch=True, numpy arrays otherwise
        """
        cache_key = f"{scene_path}:{frame_id}"
        
        # Try cache first
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                features = cached
            else:
                # Load from disk and cache
                token_path, token_global_path = self._get_feature_paths(scene_path, frame_id)
                features = self._load_from_disk(token_path, token_global_path)
                self._add_to_cache(cache_key, features)
        else:
            # Load directly from disk
            token_path, token_global_path = self._get_feature_paths(scene_path, frame_id)
            features = self._load_from_disk(token_path, token_global_path)
        
        # Features are already torch tensors, just move to device if needed
        if self.return_torch:
            features = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else torch.from_numpy(value).to(self.device)
                for key, value in features.items()
            }
        else:
            # Convert to numpy if requested
            features = {
                key: value.numpy() if isinstance(value, torch.Tensor) else value
                for key, value in features.items()
            }
        
        return features
    
    def load_batch(
        self,
        scene_paths: List[str],
        frame_ids: List[str],
        stack: bool = True
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Load features for a batch of frames
        
        Args:
            scene_paths: List of scene paths
            frame_ids: List of frame identifiers
            stack: If True, stack features into a single array/tensor with batch dimension
            
        Returns:
            features: Dictionary containing batched 'lidar_token' and 'lidar_token_global'
                     If stack=True, shapes are (batch, seq_len, 512) and (batch, 1, 512)
                     If stack=False, returns lists of features
        """
        if len(scene_paths) != len(frame_ids):
            raise ValueError("scene_paths and frame_ids must have the same length")
        
        batch_tokens = []
        batch_tokens_global = []
        
        for scene_path, frame_id in zip(scene_paths, frame_ids):
            features = self.load_features(scene_path, frame_id, use_cache=True)
            batch_tokens.append(features['lidar_token'])
            batch_tokens_global.append(features['lidar_token_global'])
        
        if stack:
            if self.return_torch:
                batch_tokens = torch.stack(batch_tokens, dim=0)
                batch_tokens_global = torch.stack(batch_tokens_global, dim=0)
            else:
                batch_tokens = np.stack(batch_tokens, axis=0)
                batch_tokens_global = np.stack(batch_tokens_global, axis=0)
        
        return {
            'lidar_token': batch_tokens,
            'lidar_token_global': batch_tokens_global
        }
    
    def load_sequence(
        self,
        scene_path: str,
        frame_ids: List[str]
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Load features for a sequence of frames from the same scene
        
        Args:
            scene_path: Path to the scene directory
            frame_ids: List of frame identifiers in temporal order
            
        Returns:
            features: Dictionary containing stacked features with shape:
                     - lidar_token: (T, seq_len, 512)
                     - lidar_token_global: (T, 1, 512)
        """
        scene_paths = [scene_path] * len(frame_ids)
        return self.load_batch(scene_paths, frame_ids, stack=True)
    
    def check_availability(self, scene_path: str, frame_id: str) -> bool:
        """
        Check if features are available for a given frame
        
        Args:
            scene_path: Path to the scene directory
            frame_id: Frame identifier
            
        Returns:
            available: True if both feature files exist
        """
        token_path, token_global_path = self._get_feature_paths(scene_path, frame_id)
        return os.path.exists(token_path) and os.path.exists(token_global_path)
    
    def get_feature_stats(self, scene_path: str, frame_id: str) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for features of a frame
        
        Args:
            scene_path: Path to the scene directory
            frame_id: Frame identifier
            
        Returns:
            stats: Dictionary containing statistics for each feature type
        """
        features = self.load_features(scene_path, frame_id, use_cache=False)
        
        stats = {}
        for key, value in features.items():
            if self.return_torch:
                value = value.cpu().numpy()
            
            stats[key] = {
                'shape': tuple(value.shape),
                'mean': float(value.mean()),
                'std': float(value.std()),
                'min': float(value.min()),
                'max': float(value.max())
            }
        
        return stats
    
    def clear_cache(self):
        """Clear the feature cache"""
        self._cache.clear()
        self._cache_order.clear()
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get information about the cache"""
        return {
            'size': len(self._cache),
            'max_size': self.cache_size,
            'usage_percent': (len(self._cache) / self.cache_size) * 100 if self.cache_size > 0 else 0
        }


def verify_features(scene_path: str, frame_id: str, verbose: bool = True) -> bool:
    """
    Standalone function to verify feature files
    
    Args:
        scene_path: Path to the scene directory
        frame_id: Frame identifier
        verbose: If True, print detailed information
        
    Returns:
        valid: True if features are valid
    """
    loader = BEVFeatureLoader(return_torch=False)
    
    try:
        features = loader.load_features(scene_path, frame_id, use_cache=False)
        
        if verbose:
            print(f"✓ Features verified for {scene_path}/{frame_id}")
            stats = loader.get_feature_stats(scene_path, frame_id)
            for key, stat in stats.items():
                print(f"  {key}:")
                print(f"    Shape: {stat['shape']}")
                print(f"    Mean: {stat['mean']:.4f}, Std: {stat['std']:.4f}")
                print(f"    Range: [{stat['min']:.4f}, {stat['max']:.4f}]")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"✗ Feature verification failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test BEV feature loader')
    parser.add_argument('--scene_path', type=str, required=True,
                       help='Path to scene directory')
    parser.add_argument('--frame_id', type=str, required=True,
                       help='Frame identifier')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cuda', 'cpu'],
                       help='Device for torch tensors')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BEV Feature Loader Test")
    print("=" * 70)
    
    # Test numpy loading
    print("\n1. Testing numpy loading...")
    loader_np = BEVFeatureLoader(return_torch=False)
    if verify_features(args.scene_path, args.frame_id, verbose=True):
        print("✓ Numpy loading successful")
    else:
        print("✗ Numpy loading failed")
    
    # Test torch loading
    print("\n2. Testing torch loading...")
    loader_torch = BEVFeatureLoader(return_torch=True, device=args.device)
    try:
        features = loader_torch.load_features(args.scene_path, args.frame_id)
        print(f"✓ Torch loading successful")
        print(f"  lidar_token: {features['lidar_token'].shape}, device={features['lidar_token'].device}")
        print(f"  lidar_token_global: {features['lidar_token_global'].shape}, device={features['lidar_token_global'].device}")
    except Exception as e:
        print(f"✗ Torch loading failed: {e}")
    
    # Test caching
    print("\n3. Testing cache...")
    loader_cached = BEVFeatureLoader(return_torch=False, cache_size=10)
    
    import time
    
    # First load (cold)
    start = time.time()
    _ = loader_cached.load_features(args.scene_path, args.frame_id, use_cache=True)
    cold_time = time.time() - start
    
    # Second load (warm)
    start = time.time()
    _ = loader_cached.load_features(args.scene_path, args.frame_id, use_cache=True)
    warm_time = time.time() - start
    
    print(f"  Cold load time: {cold_time*1000:.2f}ms")
    print(f"  Warm load time: {warm_time*1000:.2f}ms")
    print(f"  Speedup: {cold_time/warm_time:.2f}x")
    
    cache_info = loader_cached.get_cache_info()
    print(f"  Cache: {cache_info['size']}/{cache_info['max_size']} ({cache_info['usage_percent']:.1f}%)")
    
    print("\n" + "=" * 70)
    print("✓ All tests completed")
    print("=" * 70)
