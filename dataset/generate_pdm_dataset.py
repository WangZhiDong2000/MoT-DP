import numpy as np
from PIL import Image
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


class CARLAImageDataset(torch.utils.data.Dataset):
    """
    Dataset for preprocessed PDM Lite data in 'frame' mode.
    Each pickle file is a complete training sample with image paths.
    Images are loaded dynamically in __getitem__.
    """
    def __init__(self,
                 dataset_path: str,  
                 image_data_root:str,
                 ): 

        self.image_data_root = image_data_root
        self.dataset_path = dataset_path
        
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 928)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Processed data directory not found: {dataset_path}")

        # Load pkl files - support both direct directory and train/val subdirectories
        train_files = glob.glob(os.path.join(dataset_path, "train", "*.pkl"))
        val_files = glob.glob(os.path.join(dataset_path, "val", "*.pkl"))
        direct_files = glob.glob(os.path.join(dataset_path, "*.pkl"))
        
        # If subdirectories exist, use them; otherwise use direct files
        if train_files or val_files:
            self.sample_files = sorted(train_files + val_files)
            print(f"Found {len(self.sample_files)} preprocessed samples in '{dataset_path}' ({len(train_files)} train, {len(val_files)} val).")
        elif direct_files:
            self.sample_files = sorted(direct_files)
            print(f"Found {len(self.sample_files)} preprocessed samples in '{dataset_path}'.")
        else:
            raise FileNotFoundError(f"No pkl files found in {dataset_path} or its train/val subdirectories.")

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        sample_path = self.sample_files[idx]
        with open(sample_path, 'rb') as f:
            sample = pickle.load(f)

        # # 2. 动态加载图像
        # image_paths = sample.get('rgb_hist_jpg', [])
        # images = []
        
        # for img_path in image_paths:
        #     if img_path is None:
        #         # 如果路径为None，创建一个黑色图像
        #         images.append(torch.zeros(3, 256, 928))
        #         print(f"Warning: Image path is None in sample {sample_path}. Using black image.")
        #     else:
        #         full_img_path = os.path.join(self.image_data_root, img_path)
        #         try:
        #             img = Image.open(full_img_path)
        #             img_tensor = self.image_transform(img)
        #             images.append(img_tensor)
        #         except Exception as e:
        #             print(f"Error loading image {full_img_path}: {e}")
        #             images.append(torch.zeros(3, 256, 928))
        
        # # 堆叠图像
        # if len(images) > 0:
        #     images_tensor = torch.stack(images)
        # else:
        #     images_tensor = torch.zeros(2, 3, 256, 928)  # 默认obs_horizon=2
        
        # 2 动态加载预处理好的LiDAR BEV特征
        lidar_bev_paths = sample.get('lidar_bev_hist', [])
        lidar_tokens = []
        lidar_tokens_global = []
        
        for bev_path in lidar_bev_paths:
            # 从BEV图像路径推断特征路径
            # 例如: Town01/data/Route_0/lidar_bev/0000.png -> Town01/data/Route_0/lidar_bev_features/0000_token.pt
            bev_dir = os.path.dirname(bev_path)  # e.g., Town01/data/Route_0/lidar_bev
            frame_id = os.path.splitext(os.path.basename(bev_path))[0]  # e.g., 0000
            feature_dir = bev_dir.replace('lidar_bev', 'lidar_bev_features')
                
            token_path = os.path.join(self.image_data_root, feature_dir, f'{frame_id}_token.pt')
            token_global_path = os.path.join(self.image_data_root, feature_dir, f'{frame_id}_token_global.pt')
                
            lidar_token = torch.load(token_path)
            lidar_token_global = torch.load(token_global_path)
            lidar_tokens.append(lidar_token)
            lidar_tokens_global.append(lidar_token_global)
        
        lidar_token_tensor = torch.stack(lidar_tokens)  # (obs_horizon, seq_len, 512)
        lidar_token_global_tensor = torch.stack(lidar_tokens_global)  # (obs_horizon, 1, 512)
        
        # 3. 转换其他数据
        final_sample = dict()
        for key, value in sample.items():
            # if key == 'rgb_hist_jpg':
            #     final_sample['rgb_hist_jpg'] = image_paths  
            #     final_sample['image'] = images_tensor
            if key == 'lidar_bev_hist':
                final_sample['lidar_bev_hist'] = lidar_bev_paths  
                final_sample['lidar_token'] = lidar_token_tensor  # (obs_horizon, seq_len, 512)
                final_sample['lidar_token_global'] = lidar_token_global_tensor  # (obs_horizon, 1, 512)
            elif key == 'speed_hist':
                speed_data = sample['speed_hist']
                if isinstance(speed_data, np.ndarray):
                    final_sample['speed'] = torch.from_numpy(speed_data).float()
                elif isinstance(speed_data, (list, tuple)):
                    final_sample['speed'] = torch.tensor(speed_data, dtype=torch.float32)
                else:
                    final_sample['speed'] = speed_data
            elif key == 'ego_waypoints':
                final_sample['agent_pos'] = torch.from_numpy(sample['ego_waypoints'][1:])  # delete the first waypoint (current position always be 0,0)
            elif isinstance(value, np.ndarray):
                final_sample[key] = torch.from_numpy(value).float()
            else:
                final_sample[key] = value
        
        final_sample = self.hard_process(final_sample, obs_horizon=lidar_token_tensor.shape[0])

        return final_sample

    def hard_process(self, final_sample, obs_horizon):
        '''
        repeat command & target_point to match obs_horizon
        '''
        for key in ['command', 'next_command', 'target_point']:
            if key in final_sample:
                data = final_sample[key]
                if isinstance(data, torch.Tensor):
                    if data.ndim == 1:
                        data = data.unsqueeze(0)  # (D,) -> (1, D)
                    repeated_data = data.repeat(obs_horizon, 1)  # (obs_horizon, D)
                    final_sample[key] = repeated_data
                elif isinstance(data, np.ndarray):
                    if data.ndim == 1:
                        data = data[np.newaxis, :]  # (D,) -> (1, D)
                    repeated_data = np.tile(data, (obs_horizon, 1))  # (obs_horizon, D)
                    final_sample[key] = torch.from_numpy(repeated_data).float()
                else:
                    print(f"Warning: Unsupported data type for key '{key}' during hard_process.")
        return final_sample

def visualize_trajectory(sample, obs_horizon, rand_idx):
    """
    Visualizes and saves the agent's trajectory and target point from a sample.
    """
    agent_pos = sample.get('agent_pos')
    target_point = sample.get('target_point')
    
    if agent_pos is None:
        print("'agent_pos' not in sample. Skipping trajectory visualization.")
        return

    if agent_pos.ndim == 3 and agent_pos.shape[1] == 1:
        agent_pos = agent_pos.squeeze(1)
    
    pred_agent_pos = agent_pos
    
    plt.figure(figsize=(10, 8))
    
    # Plot observed and predicted points
    if len(pred_agent_pos) > 0:
        plt.plot(pred_agent_pos[:, 1], pred_agent_pos[:, 0], 'ro--', label='Predicted agent_pos', markersize=8, linewidth=2)
    
    # Plot target point
    if target_point is not None:
        if target_point.ndim == 2:
            target_point = target_point[0]
        plt.plot(target_point[1], target_point[0], 'g*', markersize=15, label='Target point (relative)', markeredgecolor='black', markeredgewidth=1)
    
    plt.xlabel('Y (relative to last obs)')
    plt.ylabel('X (relative to last obs)')
    plt.title(f'Sample {rand_idx}: Trajectory & Target Point\n(All positions relative to last observation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    save_path = f'/home/wang/Project/MoT-DP/image/sample_{rand_idx}_agent_pos_with_target.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"保存agent_pos和target_point图像到: {save_path}")

def visualize_observation_images(sample, obs_horizon, rand_idx):
    """
    Visualizes and saves the observation images from a sample.
    """
    if 'image' not in sample:
        print("'image' not in sample. Skipping image visualization.")
        return
        
    images = sample['image']  # shape: (obs_horizon, C, H, W)
    
    # Define the inverse normalization transform
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for t, img_tensor in enumerate(images):
        # Convert torch tensor to numpy and denormalize
        if isinstance(img_tensor, torch.Tensor):
            # Denormalize: img = img * std + mean
            img_tensor = img_tensor * std + mean
            img_tensor = torch.clamp(img_tensor, 0, 1)  # Clamp to [0, 1]
            img_arr = img_tensor.numpy()
        else:
            img_arr = img_tensor
            
        if img_arr.shape[0] == 3:  # (C, H, W)
            img_vis = np.moveaxis(img_arr, 0, -1)  # Convert to (H, W, C)
            img_vis = (img_vis * 255).astype(np.uint8)  # Convert to [0, 255]
        else:
            img_vis = img_arr.astype(np.uint8)
        
        plt.figure(figsize=(8, 2))
        plt.imshow(img_vis)
        plt.title(f'Random Sample {rand_idx} - Obs Image t={t}')
        plt.axis('off')
        plt.tight_layout()

        save_path = f'/home/wang/Project/MoT-DP/image/sample_{rand_idx}_obs_image_t{t}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"保存观测图像到: {save_path}")

def print_sample_details(sample, dataset, rand_idx, obs_horizon):
    """
    Prints detailed information about a sample and the dataset.
    """
    print(f"\n样本 {rand_idx} 的详细信息:")
    agent_pos = sample.get('agent_pos')
    if agent_pos is not None:
        obs_agent_pos = agent_pos[:obs_horizon]
        pred_agent_pos = agent_pos[obs_horizon:]
        print(f"观测位置: {obs_agent_pos}")
        print(f"预测位置: {pred_agent_pos}")

    target_point = sample.get('target_point')
    if target_point is not None:
        if target_point.ndim == 2:
            target_point = target_point[0]
        print(f"目标点 (相对): {target_point}")
        if agent_pos is not None and len(agent_pos) > obs_horizon -1:
            last_obs = agent_pos[obs_horizon-1]
            distance_to_target = np.linalg.norm(target_point - last_obs)
            print(f"目标点距离最后观测点的距离: {distance_to_target:.3f}")

    print(f"\nDataset length: {len(dataset)}")
    first_sample = dataset[0]
    print("Sample keys:", first_sample.keys())
    print("Image shape:", first_sample['image'].shape)
    print("Agent pos shape:", first_sample['agent_pos'].shape)

    print("\nNew CARLA fields:")
    for key in ['town_name', 'speed', 'command', 'next_command', 'target_point', 'ego_waypoints', 'image', 'agent_pos']:
        if key in first_sample:
            value = first_sample[key]
            print(f"  {key}: shape={getattr(value, 'shape', 'N/A')}, type={type(value)}")
            if hasattr(value, '__len__') and not isinstance(value, str):
                print(f"    Length: {len(value)}")

def test():
    import random
    # 重要：现在路径应该指向预处理好的数据集的根目录
    dataset_path = '/home/wang/Project/carla_garage/tmp_data'  # 预处理数据集路径
    obs_horizon = 2
    
    # 使用新的Dataset类
    dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        image_data_root='/home/wang/Project/carla_garage/data'  # 图像的根目录
    )
    
    print(f"\n总样本数: {len(dataset)}")
    if len(dataset) == 0:
        print("数据为空，无法进行测试。")
        return

    rand_idx = random.choice(range(len(dataset)))
    rand_sample = dataset[rand_idx]
    print(f"\n随机选择的样本索引: {rand_idx}")

    # 检查样本内容
    print("\nSample keys:", rand_sample.keys())
    for key, value in rand_sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
        else:
            print(f"  {key}: value={value}")

    visualize_trajectory(rand_sample, obs_horizon, rand_idx)
    # visualize_observation_images(rand_sample, obs_horizon, rand_idx)
    print_sample_details(rand_sample, dataset, rand_idx, obs_horizon)

    
if __name__ == "__main__":
    test()