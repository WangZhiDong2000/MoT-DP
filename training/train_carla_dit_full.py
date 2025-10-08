#!/usr/bin/env python3
import os
import sys
import torch
import yaml
import wandb
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import time
import psutil
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from dataset.generate_pdm_dataset import CARLAImageDataset
from policy.diffusion_dit_tcp_policy import DiffusionDiTCarlaPolicy
import yaml

def create_carla_config():
    config_path = "/home/wang/projects/diffusion_policy_z/config/carla.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def compute_action_stats(train_dataset):
    print("Computing action statistics from training dataset...")
    all_actions = []
    
    if len(train_dataset) == 0:
        print("Warning: Dataset is empty! Using default statistics.")
        return {
            'min': np.array([-10.0, -10.0]),
            'max': np.array([10.0, 10.0]),
            'mean': np.array([0.0, 0.0]),
            'std': np.array([1.0, 1.0])
        }
    
    sample_size = min(30, len(train_dataset))
    indices = np.random.choice(len(train_dataset), sample_size, replace=False)
    
    print(f"Sampling {sample_size} trajectories from {len(train_dataset)} total samples...")
    
    for i in tqdm(indices, desc="Collecting action samples"):
        try:
            sample = train_dataset[i]
            agent_pos = sample['agent_pos']  # shape: (sequence_length, 2)
            
            if isinstance(agent_pos, torch.Tensor):
                agent_pos = agent_pos.numpy()
            
            obs_horizon = train_dataset.obs_horizon
            actions = agent_pos[obs_horizon:]  # shape: (action_horizon, 2)
            all_actions.append(actions)
        except Exception as e:
            print(f"Warning: Failed to load sample {i}: {e}")
            continue
    
    if len(all_actions) == 0:
        print("Warning: No valid actions collected! Using default statistics.")
        return {
            'min': np.array([-10.0, -10.0]),
            'max': np.array([10.0, 10.0]),
            'mean': np.array([0.0, 0.0]),
            'std': np.array([1.0, 1.0])
        }
    
    all_actions = np.concatenate(all_actions, axis=0)  # shape: (N * action_horizon, 2)
    
    print(f"Raw action statistics from {len(all_actions)} samples:")
    raw_min = np.min(all_actions, axis=0)
    raw_max = np.max(all_actions, axis=0)
    print(f"  Raw range: X=[{raw_min[0]:.4f}, {raw_max[0]:.4f}], Y=[{raw_min[1]:.4f}, {raw_max[1]:.4f}]")
    
    percentile_low = 1  
    percentile_high = 99  
    
    action_min = np.percentile(all_actions, percentile_low, axis=0)
    action_max = np.percentile(all_actions, percentile_high, axis=0)
    action_mean = np.mean(all_actions, axis=0)
    action_std = np.std(all_actions, axis=0)
    
    print(f"Filtered action statistics (using {percentile_low}-{percentile_high} percentile):")
    print(f"  Min: {action_min}")
    print(f"  Max: {action_max}")
    print(f"  Mean: {action_mean}")
    print(f"  Std: {action_std}")
    print(f"  Range: {action_max - action_min}")
    
    reasonable_max_range = 10.0
    if np.any(action_max - action_min > reasonable_max_range):
        print(f"  ⚠️ Range still seems large, applying conservative clipping...")
        conservative_min = np.maximum(action_min, np.array([-3.0, -3.0]))
        conservative_max = np.minimum(action_max, np.array([8.0, 4.0]))
        action_min = conservative_min
        action_max = conservative_max
        print(f"  Conservative range: X=[{action_min[0]:.4f}, {action_max[0]:.4f}], Y=[{action_min[1]:.4f}, {action_max[1]:.4f}]")
    
    action_stats = {
        'min': torch.from_numpy(action_min).float(),
        'max': torch.from_numpy(action_max).float(),
        'mean': torch.from_numpy(action_mean).float(),
        'std': torch.from_numpy(action_std).float()
    }
    
    return action_stats

def compute_driving_metrics(predicted_trajectories, target_trajectories):
    predicted_trajectories = predicted_trajectories.detach().cpu().numpy()
    target_trajectories = target_trajectories.detach().cpu().numpy()

    # L2距离误差
    l2_errors = np.linalg.norm(predicted_trajectories - target_trajectories, axis=-1)  # (B, T)

    # 不同时间步的L2误差（仿照TCP的0.5s, 1s, 1.5s, 2s评估）
    metrics = {}
    time_steps = [0, 1, 2, 3]  
    time_labels = ['step_1', 'step_2', 'step_3', 'step_4']
    
    for i, (step, label) in enumerate(zip(time_steps, time_labels)):
        if step < l2_errors.shape[1]:
            metrics[f'trajectory_error_{label}'] = np.mean(l2_errors[:, step])
    
    # 平均轨迹误差 
    metrics['trajectory_error_mean'] = np.mean(l2_errors)
    
    # x和y坐标的误差
    x_error = np.mean(np.abs(predicted_trajectories[:, :, 0] - target_trajectories[:, :, 0]))
    y_error = np.mean(np.abs(predicted_trajectories[:, :, 1] - target_trajectories[:, :, 1]))
    
    metrics['x_coordinate_mae'] = x_error
    metrics['y_coordinate_mae'] = y_error
    
    # 轨迹变化平滑性评估
    pred_diff = np.diff(predicted_trajectories, axis=1)
    target_diff = np.diff(target_trajectories, axis=1)
    smoothness_error = np.mean(np.linalg.norm(pred_diff - target_diff, axis=-1))
    metrics['trajectory_smoothness_error'] = smoothness_error
    
    return metrics

def validate_model(policy, val_loader, device):
    policy.eval()
    val_metrics = defaultdict(list)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            loss = policy.compute_loss(batch)
            val_metrics['loss'].append(loss.item())
            
            obs_dict = {
                'image': batch['image'][:, :policy.n_obs_steps],  # (B, obs_horizon, C, H, W)
                'agent_pos': batch['agent_pos'][:, :policy.n_obs_steps],  # (B, obs_horizon, 2) - 观测步的agent_pos
                'speed': batch['speed'][:, :policy.n_obs_steps],
                'target_point': batch['target_point'][:, :policy.n_obs_steps],
                'next_command': batch['next_command'][:, :policy.n_obs_steps],  
            }
            
            try:
                result = policy.predict_action(obs_dict)
                predicted_actions = torch.from_numpy(result['action']).to(device)
                target_actions = batch['agent_pos'][:, policy.n_obs_steps:policy.n_obs_steps + predicted_actions.shape[1]]
                
                if batch_idx == 0:
                    print(f"\n=== Debug Info ===")
                    print(f"Predicted actions shape: {predicted_actions.shape}")
                    print(f"Predicted actions range: [{predicted_actions.min():.4f}, {predicted_actions.max():.4f}]")
                    print(f"Predicted actions mean: {predicted_actions.mean():.4f}, std: {predicted_actions.std():.4f}")
                    print(f"Target actions shape: {target_actions.shape}")
                    print(f"Target actions range: [{target_actions.min():.4f}, {target_actions.max():.4f}]")
                    print(f"Target actions mean: {target_actions.mean():.4f}, std: {target_actions.std():.4f}")
                    print(f"Sample predicted: {predicted_actions[0, :2]}")
                    print(f"Sample target: {target_actions[0, :2]}")
                    print(f"==================\n")
                
                driving_metrics = compute_driving_metrics(predicted_actions, target_actions)
                for key, value in driving_metrics.items():
                    val_metrics[key].append(value)
            except Exception as e:
                print(f"Warning: Error in action prediction during validation: {e}")
                continue
    
    averaged_metrics = {}
    for key, values in val_metrics.items():
        if values:  
            averaged_metrics[f'val_{key}'] = np.mean(values)
        
    return averaged_metrics

def train_carla_policy():
    print("="*80)
    print("Initializing CARLA driving policy training...")
    print("="*80)
    
    # 获取初始内存状态
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
    
    # 检查系统总内存
    system_memory = psutil.virtual_memory()
    total_memory_gb = system_memory.total / 1024 / 1024 / 1024
    available_memory_gb = system_memory.available / 1024 / 1024 / 1024
    
    print(f"System total memory: {total_memory_gb:.2f} GB")
    print(f"Available memory: {available_memory_gb:.2f} GB")
    print(f"Initial process memory: {initial_memory:.2f} GB")
    
    if available_memory_gb < 20:
        print(f"\n⚠️⚠️⚠️  CRITICAL MEMORY WARNING  ⚠️⚠️⚠️")
        print(f"Available memory is low ({available_memory_gb:.1f} GB)")
        print(f"Training with large datasets may fail!")
        print(f"Please free up memory or reduce dataset size.")
    
    config = create_carla_config()
    wandb.init(
        project=config.get('logging', {}).get('wandb_project', "carla-diffusion-policy"),
        name=config.get('logging', {}).get('run_name', "carla_dit_full_validation"),
        config={
            "learning_rate": config.get('optimizer', {}).get('lr', 5e-5),
            "epochs": config.get('training', {}).get('num_epochs', 50),
            "batch_size": config.get('dataloader', {}).get('batch_size', 16),
            "obs_horizon": config.get('obs_horizon', 2),
            "action_horizon": config.get('action_horizon', 4),
            "pred_horizon": config.get('pred_horizon', 8),
            "dataset_path": config.get('training', {}).get('dataset_path', ""),
            "max_files": 10,
            "train_split": 0.8,
            "weight_decay": config.get('optimizer', {}).get('weight_decay', 1e-5),
            "num_workers": config.get('dataloader', {}).get('num_workers', 4)
        }
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_path = config.get('training', {}).get('dataset_path', "/home/wang/dataset/pkl/tmp_data")
    pred_horizon = config.get('pred_horizon', 6)
    obs_horizon = config.get('obs_horizon', 2)
    action_horizon = config.get('action_horizon', 4)
    sample_interval = config.get('training', {}).get('sample_interval', 5)  # 新增：从配置读取采样间隔
    max_files = None # 设置为 None 以加载所有数据文件
    
    print(f"\n{'='*80}")
    print(f"DATASET CONFIGURATION:")
    print(f"  Dataset path: {dataset_path}")
    print(f"  Sample interval: {sample_interval}")
    print(f"  Max files: {max_files if max_files else 'ALL (no limit)'}")
    print(f"  Pred horizon: {pred_horizon}")
    print(f"  Obs horizon: {obs_horizon}")
    print(f"  Action horizon: {action_horizon}")
    print(f"{'='*80}\n")
    
    # 开始加载训练数据集
    print(f"[{time.strftime('%H:%M:%S')}] Loading training dataset...")
    dataset_load_start = time.time()
    
    train_dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon, 
        action_horizon=action_horizon,
        max_files=max_files,
        train_split=0.8,
        mode='train',
        device=device,
        use_gpu_processing=True,
        sample_interval=sample_interval  # 新增：传入采样间隔
    )
    
    dataset_load_time = time.time() - dataset_load_start
    current_memory = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"\n[{time.strftime('%H:%M:%S')}] Training dataset loaded in {dataset_load_time:.1f} seconds")
    print(f"Memory usage after loading train dataset: {current_memory:.2f} GB (Δ {current_memory - initial_memory:.2f} GB)")
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading validation dataset...")
    val_load_start = time.time()
    
    val_dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        max_files=max_files,
        train_split=0.8,
        mode='val',
        device=device,
        use_gpu_processing=True,
        sample_interval=sample_interval  # 新增：传入采样间隔
    )
    
    val_load_time = time.time() - val_load_start
    current_memory = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"[{time.strftime('%H:%M:%S')}] Validation dataset loaded in {val_load_time:.1f} seconds")
    print(f"Memory usage after loading val dataset: {current_memory:.2f} GB (Δ {current_memory - initial_memory:.2f} GB)")
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    
    
    action_stats = None
    if config.get('enable_action_normalization', False):
        action_stats = compute_action_stats(train_dataset)
        wandb.config.update({
            "action_stats": {
                "min": action_stats['min'].tolist(),
                "max": action_stats['max'].tolist(),
                "mean": action_stats['mean'].tolist(),
                "std": action_stats['std'].tolist()
            }
        })
    else:
        print("Action normalization disabled in config")
    
    batch_size = config.get('dataloader', {}).get('batch_size', 32)
    num_workers = config.get('dataloader', {}).get('num_workers', 4)
    
    print(f"\n{'='*80}")
    print(f"DATALOADER CONFIGURATION:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    
    # 计算预估内存占用
    estimated_worker_memory = current_memory * num_workers  # 每个worker会fork主进程
    print(f"\n  ⚠️  IMPORTANT: DataLoader Memory Warning")
    print(f"  Current process memory: {current_memory:.2f} GB")
    print(f"  Each worker will fork the main process")
    print(f"  Estimated peak memory (worst case): ~{current_memory + estimated_worker_memory:.1f} GB")
    
    if num_workers > 2 and current_memory > 10:
        print(f"\n  ⚠️⚠️⚠️  CRITICAL WARNING  ⚠️⚠️⚠️")
        print(f"  High memory usage ({current_memory:.1f} GB) + multiple workers ({num_workers})")
        print(f"  This may cause:")
        print(f"    1. System running out of memory")
        print(f"    2. Workers killed by OOM (Out Of Memory) killer")
        print(f"    3. RuntimeError: DataLoader worker is killed by signal")
        print(f"    4. Very slow DataLoader initialization (5-30 minutes)")
        print(f"    5. Training startup hanging/freezing")
        print(f"  ")
        print(f"  RECOMMENDATIONS:")
        print(f"    - Set num_workers=0 in config (MOST IMPORTANT!)")
        print(f"    - Reduce batch_size (current: {batch_size})")
        print(f"    - Use smaller dataset (increase sample_interval or set max_files)")
        print(f"  ")
        print(f"  🔴 High probability of worker being killed!")
        print(f"  🔴 Change num_workers to 0 and restart training!")
    elif num_workers > 0 and current_memory > 5:
        print(f"\n  ⚠️  WARNING: Memory usage is moderate ({current_memory:.1f} GB)")
        print(f"  With {num_workers} workers, peak memory may reach ~{current_memory * (1 + num_workers):.1f} GB")
        print(f"  If you see 'DataLoader worker killed' errors:")
        print(f"    - Set num_workers=0 in config")
        print(f"    - This will solve the OOM issue")
    elif num_workers > 0:
        print(f"\n  Note: Worker initialization may take 1-5 minutes with large datasets")
        print(f"        Please be patient during first batch loading...")
    
    print(f"{'='*80}\n")
    
    print(f"[{time.strftime('%H:%M:%S')}] Creating DataLoader for training...")
    dataloader_create_start = time.time()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    dataloader_create_time = time.time() - dataloader_create_start
    print(f"[{time.strftime('%H:%M:%S')}] DataLoader created in {dataloader_create_time:.1f} seconds")
    
    val_loader = None
    if len(val_dataset) > 0:
        print(f"[{time.strftime('%H:%M:%S')}] Creating DataLoader for validation...")
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        print(f"[{time.strftime('%H:%M:%S')}] Validation DataLoader created")
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Initializing policy model...")
    model_init_start = time.time()
    policy = DiffusionDiTCarlaPolicy(config, action_stats=action_stats).to(device)
    
    model_init_time = time.time() - model_init_start
    current_memory = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"[{time.strftime('%H:%M:%S')}] Policy model initialized in {model_init_time:.1f} seconds")
    print(f"Memory usage after model init: {current_memory:.2f} GB")
    print(f"Policy action steps (n_action_steps): {policy.n_action_steps}")

    
    # print("\n=== Checking data statistics ===")
    # sample_batch = next(iter(train_loader))
    # print(f"Sample agent_pos shape: {sample_batch['agent_pos'].shape}")
    # print(f"Sample agent_pos range: [{sample_batch['agent_pos'].min():.4f}, {sample_batch['agent_pos'].max():.4f}]")
    # print(f"Sample agent_pos mean: {sample_batch['agent_pos'].mean():.4f}, std: {sample_batch['agent_pos'].std():.4f}")
    # print(f"First sample agent_pos:\n{sample_batch['agent_pos'][0]}")
    # print("===================================\n")
    
    lr = config.get('optimizer', {}).get('lr', 5e-5)
    weight_decay = config.get('optimizer', {}).get('weight_decay', 1e-5)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    
    print(f"\n{'='*80}")
    print(f"[{time.strftime('%H:%M:%S')}] STARTING TRAINING LOOP")
    print(f"{'='*80}")
    print(f"\n⏳ Loading first batch from DataLoader...")
    print(f"   This is where the training often gets stuck with large datasets!")
    print(f"   Worker processes are being initialized and forked...")
    print(f"   Expected time: {num_workers * 30 if num_workers > 0 else 5}-{num_workers * 120 if num_workers > 0 else 30} seconds")
    print(f"   Please be patient...\n")
    

    num_epochs = config.get('training', {}).get('num_epochs', 50)
    best_val_loss = float('inf')
    
    first_batch_loaded = False
    first_batch_time = None
    
    for epoch in range(num_epochs):
        policy.train()
        train_losses = []

        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*80}")
        
        if not first_batch_loaded:
            print(f"[{time.strftime('%H:%M:%S')}] ⏳ Initializing DataLoader iterator...")
            print(f"   (This will spawn {num_workers} worker processes)")
            iter_start_time = time.time()
        
        try:
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
                if not first_batch_loaded:
                    first_batch_time = time.time() - iter_start_time
                    current_memory = process.memory_info().rss / 1024 / 1024 / 1024
                    print(f"\n✓ First batch loaded successfully!")
                    print(f"   Time taken: {first_batch_time:.1f} seconds")
                    print(f"   Memory usage: {current_memory:.2f} GB")
                    print(f"   Training will now proceed normally...\n")
                    first_batch_loaded = True
                
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                optimizer.zero_grad()
                loss = policy.compute_loss(batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                
                if batch_idx % 10 == 0:
                    step = epoch * len(train_loader) + batch_idx
                    wandb.log({
                        "train/loss_step": loss.item(),
                        "train/epoch": epoch,
                        "train/step": step,
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                        "train/batch_idx": batch_idx
                    })
                    
                    print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        except RuntimeError as e:
            if "DataLoader worker" in str(e) and "killed" in str(e):
                print(f"\n{'='*80}")
                print(f"❌ ERROR: DataLoader worker was killed!")
                print(f"{'='*80}")
                print(f"\n🔴 This is an Out-Of-Memory (OOM) error!")
                print(f"The system killed the worker process because it ran out of memory.\n")
                print(f"SOLUTION:")
                print(f"1. Open config/carla.yaml")
                print(f"2. Find the 'dataloader' section")
                print(f"3. Change 'num_workers' to 0:")
                print(f"   dataloader:")
                print(f"     batch_size: {batch_size}")
                print(f"     num_workers: 0  # <- Change this!\n")
                print(f"4. Restart training\n")
                print(f"Alternative solutions:")
                print(f"  - Increase sample_interval to reduce dataset size")
                print(f"  - Set max_files to limit number of files loaded")
                print(f"  - Reduce batch_size")
                print(f"{'='*80}\n")
                raise
            else:
                raise
        

        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")
        
        wandb.log({
            "train/loss_epoch": avg_train_loss,
            "train/epoch": epoch,
            "train/samples_processed": (epoch + 1) * len(train_dataset)
        })

        if (epoch + 1) % config.get('validation_freq', 1) == 0:
            print("Validating...")
            val_metrics = validate_model(policy, val_loader, device)
        
            log_dict = {
                "epoch": epoch,
                "train/loss": avg_train_loss,
            }
        
            if 'val_loss' in val_metrics:
                log_dict["val/loss"] = val_metrics['val_loss']
        
            trajectory_metrics = {}
            for key, value in val_metrics.items():
                if 'trajectory_error' in key:
                    new_key = f"val/trajectory/{key.replace('val_', '')}"
                    trajectory_metrics[new_key] = value
                elif 'coordinate' in key:
                    new_key = f"val/coordinate/{key.replace('val_', '')}"
                    trajectory_metrics[new_key] = value
                elif 'smoothness' in key:
                    new_key = f"val/smoothness/{key.replace('val_', '')}"
                    trajectory_metrics[new_key] = value
                else:
        
                    new_key = f"val/{key.replace('val_', '')}"
                    trajectory_metrics[new_key] = value
        
            log_dict.update(trajectory_metrics)
            wandb.log(log_dict)
        
            print(f"Validation metrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
        
            val_loss = val_metrics.get('val_loss', float('inf'))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_dir = "/home/wang/projects/diffusion_policy_z/checkpoints/carla_dit"
                os.makedirs(checkpoint_dir, exist_ok=True)
            
                torch.save({
                    'model_state_dict': policy.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'train_loss': avg_train_loss,
                    'val_metrics': val_metrics
                    }, os.path.join(checkpoint_dir, "carla_policy_best.pt"))
            
                print(f"✓ New best model saved with val_loss: {val_loss:.4f}")
            
                wandb.log({
                    "best_model/epoch": epoch,
                    "best_model/val_loss": val_loss,
                    "best_model/train_loss": avg_train_loss
                })
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    wandb.log({
        "training/completed": True,
        "training/total_epochs": num_epochs,
        "training/best_val_loss": best_val_loss,
        "training/final_train_loss": avg_train_loss
    })
    wandb.finish()

if __name__ == "__main__":
    train_carla_policy()