import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
import textwrap


class CARLAImageDataset(torch.utils.data.Dataset):
    """
    Dataset for preprocessed PDM Lite data in 'frame' mode.
    Each pickle file is a complete training sample with image paths.
    Images are loaded dynamically in __getitem__.
    """
    def __init__(self,
                 dataset_path: str,  # Directory containing individual frame pkl files
                 image_data_root:str,
                 ):  # Base path for images (if None, use absolute paths from pkl)

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
        
        # 定义LiDAR BEV图像变换（不裁剪，只转换为tensor和归一化）
        self.lidar_bev_transform = transforms.Compose([
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
        # 1. 获取文件路径并加载单个样本
        sample_path = self.sample_files[idx]
        with open(sample_path, 'rb') as f:
            sample = pickle.load(f)

        # 2. 动态加载图像
        image_paths = sample.get('rgb_hist_jpg', [])
        images = []
        
        for img_path in image_paths:
            if img_path is None:
                # 如果路径为None，创建一个黑色图像
                images.append(torch.zeros(3, 256, 928))
                print(f"Warning: Image path is None in sample {sample_path}. Using black image.")
            else:
                full_img_path = os.path.join(self.image_data_root, img_path)
                try:
                    img = Image.open(full_img_path)
                    img_tensor = self.image_transform(img)
                    images.append(img_tensor)
                except Exception as e:
                    print(f"Error loading image {full_img_path}: {e}")
                    images.append(torch.zeros(3, 256, 928))
        
        # 堆叠图像
        if len(images) > 0:
            images_tensor = torch.stack(images)
        else:
            images_tensor = torch.zeros(2, 3, 256, 928)  # 默认obs_horizon=2
        
        # 2.5 动态加载LiDAR BEV图像（处理方式和image相同，但不裁剪）
        lidar_bev_paths = sample.get('lidar_bev_hist', [])
        lidar_bev_images = []
        for bev_path in lidar_bev_paths:
            if bev_path is None:
                # 如果路径为None，创建一个黑色BEV图像（336x336）
                lidar_bev_images.append(torch.zeros(3, 336, 336))
                print(f"Warning: LiDAR BEV path is None in sample {sample_path}. Using black BEV image.")
            else:
                full_bev_path = os.path.join(self.image_data_root, bev_path)
                try:
                    bev_img = Image.open(full_bev_path)
                    bev_tensor = self.lidar_bev_transform(bev_img)
                    lidar_bev_images.append(bev_tensor)
                except Exception as e:
                    print(f"Warning: Error loading BEV image {full_bev_path}: {e}")
                    lidar_bev_images.append(torch.zeros(3, 336, 336))
        
        # 堆叠LiDAR BEV图像
        if len(lidar_bev_images) > 0:
            lidar_bev_tensor = torch.stack(lidar_bev_images)
        else:
            lidar_bev_tensor = torch.zeros(2, 3, 336, 336)  # 默认obs_horizon=2, BEV尺寸336x336
        
        # 3. 转换其他数据
        final_sample = dict()
        for key, value in sample.items():
            if key == 'rgb_hist_jpg':
                final_sample['rgb_hist_jpg'] = image_paths  
                final_sample['image'] = images_tensor
            elif key == 'lidar_bev_hist':
                final_sample['lidar_bev_hist'] = lidar_bev_paths  
                final_sample['lidar_bev'] = lidar_bev_tensor
            elif key == 'speed_hist':
                # Convert speed_hist to tensor
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
        
        final_sample = self.hard_process(final_sample, obs_horizon=images_tensor.shape[0])

        return final_sample

    def hard_process(self, final_sample, obs_horizon):
        '''
        repeat command, target_point to match obs_horizon
        meta_actions are now strings, so they don't need repeating
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
    
    # Plot predicted agent positions (future waypoints)
    if len(pred_agent_pos) > 0:
        plt.plot(pred_agent_pos[:, 1], pred_agent_pos[:, 0], 'ro--', label='Predicted agent_pos', markersize=8, linewidth=2)
    
    # Target point visualization is removed as requested
    # if target_point is not None:
    #     ...
        


    
    plt.xlabel('Y (ego frame, lateral)', fontsize=12)
    plt.ylabel('X (ego frame, longitudinal)', fontsize=12)
    plt.title(f'Sample {rand_idx}: Predicted Trajectory\n(Ego coordinate frame, origin at current position)', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add origin marker
    plt.plot(0, 0, 'ko', markersize=10, label='Current position (origin)', zorder=5)
    plt.legend(fontsize=10, loc='best')  # Update legend after adding origin
    plt.tight_layout()
    
    save_path = f'/home/wang/Project/MoT-DP/image/sample_{rand_idx}_agent_pos.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"保存agent_pos图像到: {save_path}")

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

def visualize_vqa_on_image(sample, obs_horizon, rand_idx, max_qa_pairs=5):
    """
    Visualizes VQA content as conversation overlay on the observation image.
    
    Args:
        sample: The data sample containing 'image' and 'vqa' fields
        obs_horizon: Number of observation frames
        rand_idx: Sample index for saving
        max_qa_pairs: Maximum number of QA pairs to display per category
    """
    if 'image' not in sample:
        print("'image' not in sample. Skipping VQA visualization.")
        return
    
    if 'vqa' not in sample or sample['vqa'] is None:
        print("'vqa' not in sample or is None. Skipping VQA visualization.")
        return
    
    vqa_data = sample['vqa']
    images = sample['image']
    
    # Define the inverse normalization transform
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Use the last observation image (most recent frame)
    img_tensor = images[-1]
    
    # Convert torch tensor to numpy and denormalize
    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)
        img_arr = img_tensor.numpy()
    else:
        img_arr = img_tensor
        
    if img_arr.shape[0] == 3:  # (C, H, W)
        img_vis = np.moveaxis(img_arr, 0, -1)  # Convert to (H, W, C)
        img_vis = (img_vis * 255).astype(np.uint8)
    else:
        img_vis = img_arr.astype(np.uint8)
    
    # Convert to PIL Image for drawing
    pil_img = Image.fromarray(img_vis)
    img_width, img_height = pil_img.size
    
    # Create a larger canvas to add Meta Actions text on the right side
    canvas_width = img_width + 400  # Reduced from 600 since we only show meta actions
    canvas_height = img_height  # No need for extra height since we removed VQA display
    canvas = Image.new('RGB', (canvas_width, canvas_height), color=(255, 255, 255))
    canvas.paste(pil_img, (0, 0))
    
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a font, fallback to default if not available
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # VQA text area starts after the image
    text_x = img_width + 10
    text_y = 10
    text_width = 580
    line_spacing = 5
    
    # Draw title - Only show Meta Actions
    draw.text((text_x, text_y), "Meta Actions", fill=(0, 0, 0), font=font_title)
    text_y += 30
    
    # Skip VQA conversation display - only show meta actions
    # Commented out VQA display to focus on meta actions only
    # qa_categories = vqa_data.get('QA', {})
    # for category_name, qa_list in qa_categories.items():
    #     ... (VQA display code commented out)
    
    # No need to check space since we skip VQA display
    
    # Add Meta Actions section at the bottom
    text_y += 20
    draw.text((text_x, text_y), "═══ META ACTIONS ═══", fill=(150, 50, 50), font=font_title)
    text_y += 25
    
    # Display meta_action_direction (now stored as string)
    if 'meta_action_direction' in sample:
        meta_dir = sample['meta_action_direction']
        
        # Handle different data types - extract string value
        if isinstance(meta_dir, str):
            # Already a string, use directly
            direction_value = meta_dir
        elif isinstance(meta_dir, (list, np.ndarray)):
            # If it's an array, take the first element
            direction_value = meta_dir[0]
        elif isinstance(meta_dir, torch.Tensor):
            # If it's a tensor, extract the value
            if meta_dir.numel() == 1:
                direction_value = meta_dir.item()
            else:
                direction_value = meta_dir[0].item() if meta_dir[0].numel() == 1 else str(meta_dir[0])
        else:
            direction_value = str(meta_dir)
        
        # Convert string to readable format
        direction_map = {
            'FOLLOW_LANE': 'Follow Lane',
            'CHANGE_LANE_LEFT': 'Change Lane Left',
            'CHANGE_LANE_RIGHT': 'Change Lane Right',
            'GO_STRAIGHT': 'Go Straight',
            'TURN_LEFT': 'Turn Left',
            'TURN_RIGHT': 'Turn Right'
        }
        direction_str = direction_map.get(direction_value, str(direction_value))
        
        draw.text((text_x, text_y), f"Direction: {direction_str}", 
                 fill=(0, 0, 150), font=font_text)
        text_y += 20
    
    # Display meta_action_speed (now stored as string)
    if 'meta_action_speed' in sample:
        meta_speed = sample['meta_action_speed']
        
        # Handle different data types - extract string value
        if isinstance(meta_speed, str):
            # Already a string, use directly
            speed_value = meta_speed
        elif isinstance(meta_speed, (list, np.ndarray)):
            # If it's an array, take the first element
            speed_value = meta_speed[0]
        elif isinstance(meta_speed, torch.Tensor):
            # If it's a tensor, extract the value
            if meta_speed.numel() == 1:
                speed_value = meta_speed.item()
            else:
                speed_value = meta_speed[0].item() if meta_speed[0].numel() == 1 else str(meta_speed[0])
        else:
            speed_value = str(meta_speed)
        
        # Convert string to readable format
        speed_map = {
            'KEEP': 'Keep Speed',
            'ACCELERATE': 'Accelerate',
            'DECELERATE': 'Decelerate',
            'STOP': 'Stop'
        }
        speed_str = speed_map.get(speed_value, str(speed_value))
        
        draw.text((text_x, text_y), f"Speed: {speed_str}", 
                 fill=(0, 150, 0), font=font_text)
        text_y += 20
    
    # Convert back to numpy for matplotlib
    canvas_np = np.array(canvas)
    
    # Display and save
    plt.figure(figsize=(16, 8))
    plt.imshow(canvas_np)
    plt.title(f'Sample {rand_idx} - Image with Meta Actions')
    plt.axis('off')
    plt.tight_layout()
    
    save_path = f'/home/wang/Project/MoT-DP/image/sample_{rand_idx}_meta_actions.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"保存Meta Actions图像到: {save_path}")

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
    for key in ['town_name', 'speed', 'command', 'next_command', 'target_point', 'ego_waypoints', 'image', 'agent_pos', 'meta_action_direction', 'meta_action_speed']:
        if key in first_sample:
            value = first_sample[key]
            print(f"  {key}: shape={getattr(value, 'shape', 'N/A')}, type={type(value)}")
            if hasattr(value, '__len__') and not isinstance(value, str):
                print(f"    Length: {len(value)}")
            # Show meta action values
            if key == 'meta_action_direction':
                direction_map = ['FOLLOW_LANE', 'CHANGE_LANE_LEFT', 'CHANGE_LANE_RIGHT', 'GO_STRAIGHT', 'TURN_LEFT', 'TURN_RIGHT']
                if hasattr(value, 'shape') and value.shape[0] == obs_horizon:
                    direction_idx = value[0].argmax()
                else:
                    direction_idx = value.argmax() if hasattr(value, 'argmax') else 0
                print(f"    Value: {direction_map[direction_idx]}")
            if key == 'meta_action_speed':
                speed_map = ['KEEP', 'ACCELERATE', 'DECELERATE', 'STOP']
                if hasattr(value, 'shape') and value.shape[0] == obs_horizon:
                    speed_idx = value[0].argmax()
                else:
                    speed_idx = value.argmax() if hasattr(value, 'argmax') else 0
                print(f"    Value: {speed_map[speed_idx]}")

def test():
    import random
    # 重要：现在路径应该指向预处理好的数据集的根目录
    dataset_path = '/home/wang/Dataset/b2d_10scene/tmp_data_vqa_with_meta'  # 预处理数据集路径（带VQA和meta action）
    obs_horizon = 2
    
    # 使用新的Dataset类
    dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        image_data_root='/home/wang/Dataset/b2d_10scene'  # 图像的根目录
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
    visualize_observation_images(rand_sample, obs_horizon, rand_idx)
    visualize_vqa_on_image(rand_sample, obs_horizon, rand_idx, max_qa_pairs=5)
    print_sample_details(rand_sample, dataset, rand_idx, obs_horizon)

    
if __name__ == "__main__":
    test()