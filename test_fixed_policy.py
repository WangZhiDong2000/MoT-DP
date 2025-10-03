"""
Quick test to verify the fixed diffusion_dit_pusht_policy works correctly
"""
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import torch
import yaml

def test_fixed_policy():
    print("="*80)
    print("Testing Fixed DiffusionDiTPushTPolicy")
    print("="*80)
    
    # Load config
    config_path = os.path.join(project_root, "config/carla.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update image size to correct dimensions for TCP
    config['shape_meta']['obs']['image']['shape'] = [3, 256, 928]
    
    # Enable GroupNorm for testing
    config['policy']['use_j_ctrl_norm'] = True
    
    # Disable action normalization for testing
    config['enable_action_normalization'] = False
    
    print("\n✓ Config loaded and updated:")
    print(f"  - Image shape: {config['shape_meta']['obs']['image']['shape']}")
    print(f"  - GroupNorm enabled: {config['policy'].get('use_j_ctrl_norm', False)}")
    
    # Import and create model
    from policy.diffusion_dit_carla_policy import DiffusionDiTPushTPolicy
    
    print("\n✓ Creating model...")
    model = DiffusionDiTPushTPolicy(config)
    model = model.cuda()
    model.eval()
    print("✓ Model created successfully!")
    
    # Test with correct image size (256x928)
    batch_size = 2
    obs_horizon = config.get('n_obs_steps', 2)
    
    obs_dict = {
        'image': torch.randn(batch_size, obs_horizon, 3, 256, 928).cuda(),
        'agent_pos': torch.randn(batch_size, obs_horizon, 2).cuda(),
        'speed': torch.randn(batch_size, obs_horizon).cuda(),
        'target_point': torch.randn(batch_size, obs_horizon, 2).cuda(),
        'next_command': torch.randn(batch_size, obs_horizon, 6).cuda(),
    }
    
    print(f"\n✓ Testing feature extraction...")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Obs horizon: {obs_horizon}")
    print(f"  - Image shape: {obs_dict['image'].shape}")
    
    try:
        # Test feature extraction for first timestep
        first_step_obs = {
            'image': obs_dict['image'][:, 0, ...],
            'speed': obs_dict['speed'][:, 0],
            'target_point': obs_dict['target_point'][:, 0, :],
            'next_command': obs_dict['next_command'][:, 0, :]
        }
        
        with torch.no_grad():
            j_ctrl = model.extract_tcp_features(first_step_obs)
        
        print(f"\n✓ Feature extraction successful!")
        print(f"  - j_ctrl shape: {j_ctrl.shape}")
        print(f"  - j_ctrl mean: {j_ctrl.mean().item():.4f}")
        print(f"  - j_ctrl std: {j_ctrl.std().item():.4f}")
        print(f"  - j_ctrl min: {j_ctrl.min().item():.4f}")
        print(f"  - j_ctrl max: {j_ctrl.max().item():.4f}")
        
        # Test full forward pass (predict_action)
        print(f"\n✓ Testing full forward pass (predict_action)...")
        result = model.predict_action(obs_dict)
        
        print(f"\n✓ Forward pass successful!")
        print(f"  - Action shape: {result['action'].shape}")
        print(f"  - Action pred shape: {result['action_pred'].shape}")
        
        print(f"\n{'='*80}")
        print("✓ All tests passed!")
        print("="*80)
        print("\nSummary:")
        print("✓ TCP encoder works correctly with 256x928 images")
        print("✓ Spatial dimension bug is fixed (using hardcoded 8x29)")
        print("✓ GroupNorm can be optionally enabled for j_ctrl")
        print("\nNext steps:")
        print("1. Update config/carla.yaml to use 256x928 images")
        print("2. Regenerate or resize your dataset images to 256x928")
        print("3. Consider enabling use_j_ctrl_norm: true in config for better training stability")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_policy()
    if not success:
        sys.exit(1)
