
import numpy as np
from PIL import Image
import os
import zipfile
import io
import sys
import pickle
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import matplotlib.pyplot as plt



def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def create_sample_indices_no_padding(
        episode_ends: np.ndarray, sequence_length: int):
    """创建样本索引，不使用填充，只保留完整的序列"""
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        # 只生成完整长度的序列，不进行填充
        if episode_length >= sequence_length:
            for idx in range(episode_length - sequence_length + 1):
                buffer_start_idx = start_idx + idx
                buffer_end_idx = start_idx + idx + sequence_length
                # 不需要填充，所以sample范围就是整个sequence_length
                sample_start_idx = 0
                sample_end_idx = sequence_length
                indices.append([
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        if key == 'agent_pos':
            # agent_pos已经是完整的序列，直接返回对应的样本
            sample_idx = buffer_start_idx  # agent_pos索引对应样本索引
            result[key] = input_arr[sample_idx]  # 直接取(8, 2)的序列
        else:
            sample = input_arr[buffer_start_idx:buffer_end_idx]
            data = sample
            
            if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
                if isinstance(input_arr, list) and key == 'town_name':
                    data = [None] * sequence_length
                    if sample_start_idx > 0:
                        # Pad beginning with first element
                        for i in range(sample_start_idx):
                            data[i] = sample[0] if len(sample) > 0 else None
                    if sample_end_idx < sequence_length:
                        # Pad end with last element
                        for i in range(sample_end_idx, sequence_length):
                            data[i] = sample[-1] if len(sample) > 0 else None
                    # Fill middle with actual data
                    for i, item in enumerate(sample):
                        if sample_start_idx + i < sample_end_idx:
                            data[sample_start_idx + i] = item
                elif isinstance(input_arr, list) and key == 'rgb_hist_jpg':
                    data = [None] * sequence_length
                    if sample_start_idx > 0:
                        # Pad beginning with first element
                        for i in range(sample_start_idx):
                            data[i] = sample[0] if len(sample) > 0 else None
                    if sample_end_idx < sequence_length:
                        # Pad end with last element
                        for i in range(sample_end_idx, sequence_length):
                            data[i] = sample[-1] if len(sample) > 0 else None
                    # Fill middle with actual data
                    for i, item in enumerate(sample):
                        if sample_start_idx + i < sample_end_idx:
                            data[sample_start_idx + i] = item
                else:
                    data = np.zeros(
                        shape=(sequence_length,) + input_arr.shape[1:],
                        dtype=input_arr.dtype)
                    if sample_start_idx > 0:
                        data[:sample_start_idx] = sample[0]
                    if sample_end_idx < sequence_length:
                        data[sample_end_idx:] = sample[-1]
                    data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
    
    return result



class CARLAImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 max_files: int = None,  
                 train_split: float = 0.8,  
                 mode: str = 'train'):  

        pkl_files = glob.glob(os.path.join(dataset_path, "*.pkl"))
        print(f"Found {len(pkl_files)} PKL files")
        pkl_files = sorted(pkl_files)
        
        # Split files for train/validation 
        num_train_files = int(len(pkl_files) * train_split)
        if mode == 'train':
            pkl_files = pkl_files[:num_train_files]
            print(f"Using {len(pkl_files)} files for training")
        elif mode == 'val':
            pkl_files = pkl_files[num_train_files:]
            print(f"Using {len(pkl_files)} files for validation")
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'val'")
        
        if max_files is not None and len(pkl_files) > max_files:
            print(f"Limiting to first {max_files} files...")
            pkl_files = pkl_files[:max_files]
        
        all_data = []
        all_episode_data = []  # 保存每个episode的原始数据，用于计算episode_ends
        
        for pkl_file in sorted(pkl_files):
            print(f"Loading {os.path.basename(pkl_file)}")
            with open(pkl_file, 'rb') as f:
                episode_data = pickle.load(f)
            
            all_data.extend(episode_data)
            all_episode_data.append(episode_data)
        
        print(f"Total samples loaded: {len(all_data)}")
        print(f"Episodes: {len(all_episode_data)}")
        
        processed_data = {
            'town_name': [],
            'speed': [],
            'command': [],
            'next_command': [],
            'target_point': [],
            'ego_waypoints': [],
            'rgb_hist_jpg': [],
        }
        
        # 重新建立episode_ends，基于过滤后的数据
        episode_ends = []
        current_length = 0
        
        for episode_data in all_episode_data:
            # 计算当前episode中有效的数据点数量
            valid_count = 0
            for item in episode_data:
                if len(item['ego_waypoints']) >= 1:
                    valid_count += 1
            
            if valid_count >= pred_horizon:  # 只有当episode长度>=pred_horizon时才处理
                # 处理当前episode的数据点
                for item in episode_data:
                    if len(item['ego_waypoints']) >= 1:
                        processed_data['town_name'].append(item['town_name'])
                        processed_data['speed'].append(item['speed'])
                        processed_data['command'].append(item['command'])
                        processed_data['next_command'].append(item['next_command'])
                        processed_data['target_point'].append(item['target_point'])
                        processed_data['ego_waypoints'].append(item['ego_waypoints'])
                        
                        # Only keep the most recent frame
                        processed_data['rgb_hist_jpg'].append(item['rgb_hist_jpg'][-1])
                
                current_length += valid_count
                episode_ends.append(current_length)
        
        print(f"Valid episodes: {len(episode_ends)}, total samples: {current_length}")
        
        # 重新计算agent_pos - 使用ego_waypoints的真实位置数据
        if current_length >= pred_horizon:
            print("Analyzing ego_waypoints structure for agent position data...")
            
            # 检查第一个episode的数据来理解ego_waypoints结构
            if all_episode_data:
                first_episode = all_episode_data[0]
                print(f"First episode has {len(first_episode)} samples")
                
                # 详细分析ego_waypoints的结构
                for i in range(min(3, len(first_episode))):
                    sample = first_episode[i]
                    if 'ego_waypoints' in sample:
                        ego_wp = np.array(sample['ego_waypoints'])
                        print(f"  Sample {i}: ego_waypoints shape = {ego_wp.shape}")
                        print(f"    ego_waypoints[0]: {ego_wp[0]} (current position)")
                        if len(ego_wp) > 1:
                            print(f"    ego_waypoints[1]: {ego_wp[1]} (next waypoint)")
                        if len(ego_wp) > 2:
                            print(f"    ego_waypoints[2]: {ego_wp[2]} (waypoint 2)")
                        print()
                
                # 检查不同时间步的ego_waypoints变化
                print("Checking ego_waypoints changes across time steps:")
                for i in range(min(5, len(first_episode))):
                    sample = first_episode[i]
                    if 'ego_waypoints' in sample:
                        ego_wp = np.array(sample['ego_waypoints'])
                        # 尝试使用不同waypoint作为真实位置
                        print(f"  Step {i}:")
                        for wp_idx in range(min(3, len(ego_wp))):
                            print(f"    waypoint[{wp_idx}]: {ego_wp[wp_idx]}")
                
                # 简化方案：正确构建agent_pos轨迹
                print("\nBuilding correct agent_pos trajectory...")
                
                # 正确的agent_pos格式：
                # - 形状: (pred_horizon, 2) = (8, 2)
                # - 前obs_horizon个(3个)：观测位置
                # - 后action_horizon个(5个)：预测位置  
                # - 以最后一个观测位置为原点，其他位置都是相对距离
                
                # 重新处理数据，为每个样本创建正确的agent_pos序列
                processed_data['agent_pos'] = []
                
                for ep_idx, ep_data in enumerate(all_episode_data):
                    valid_items = [item for item in ep_data if len(item['ego_waypoints']) >= 1]
                    if len(valid_items) >= pred_horizon:
                        
                        # 为episode中的每个可能的序列起点创建agent_pos
                        for start_idx in range(len(valid_items) - pred_horizon + 1):
                            # 构建长度为pred_horizon的序列
                            sequence_positions = []
                            
                            # 收集序列中每个时间步的ego_waypoints[0]（当前位置）
                            for seq_step in range(pred_horizon):
                                step_idx = start_idx + seq_step
                                if step_idx < len(valid_items):
                                    item = valid_items[step_idx]
                                    ego_wp = np.array(item['ego_waypoints'])
                                    # 使用ego_waypoints[1]作为这个时间步的"实际位置"
                                    # 因为ego_waypoints[0]总是[0,0]
                                    if len(ego_wp) > 1:
                                        current_pos = ego_wp[1].copy()
                                    else:
                                        current_pos = np.array([0.0, 0.0])
                                    sequence_positions.append(current_pos)
                                else:
                                    # 超出范围，使用最后已知位置
                                    sequence_positions.append(sequence_positions[-1].copy())
                            
                            # 以最后一个观测位置(index=obs_horizon-1)为原点，计算相对位置
                            reference_pos = sequence_positions[obs_horizon - 1]  # 最后一个观测位置
                            relative_positions = []
                            
                            for pos in sequence_positions:
                                relative_pos = pos - reference_pos
                                relative_positions.append(relative_pos)
                            
                            processed_data['agent_pos'].append(np.array(relative_positions))
                            
                            # 显示第一个序列的调试信息
                            if ep_idx == 0 and start_idx == 0:
                                print(f"First sequence agent_pos:")
                                print(f"  Reference position (obs_horizon-1): {reference_pos}")
                                for step, rel_pos in enumerate(relative_positions):
                                    label = "obs" if step < obs_horizon else "pred"
                                    print(f"  Step {step} ({label}): {rel_pos}")
                
                print(f"Generated {len(processed_data['agent_pos'])} agent_pos sequences")
                
                # 验证形状
                if len(processed_data['agent_pos']) > 0:
                    sample_agent_pos = processed_data['agent_pos'][0]
                    print(f"Agent_pos shape per sample: {sample_agent_pos.shape}")
                    print(f"Expected shape: ({pred_horizon}, 2)")
                    
                    if sample_agent_pos.shape == (pred_horizon, 2):
                        print("✅ Agent_pos shape is correct!")
                    else:
                        print("❌ Agent_pos shape is incorrect!")
                
                # 验证最终结果
                final_positions = np.array(processed_data['agent_pos'])
                print(f"\nFinal agent_pos summary:")
                print(f"  Shape: {final_positions.shape}")
                print(f"  Non-zero positions: {np.sum(np.any(np.abs(final_positions) > 0.01, axis=1))}")
                print(f"  Max displacement: {np.max(np.linalg.norm(final_positions, axis=1)):.3f}")
            
        else:
            print("Warning: Not enough samples to create relative positions")
        
        for key in ['speed', 'command', 'next_command', 'target_point', 'ego_waypoints']:
            processed_data[key] = np.array(processed_data[key])
        
        # agent_pos 已经是列表格式，转换为numpy数组
        processed_data['agent_pos'] = np.array(processed_data['agent_pos'])  # shape: (N, pred_horizon, 2)
        
        print("Processing images...")
        image_data = []
        image_orig_shape = []
        for i, img_bytes in enumerate(processed_data['rgb_hist_jpg']):
            if i % 100 == 0:
                print(f"  Processing image {i}/{len(processed_data['rgb_hist_jpg'])}")
            img = Image.open(io.BytesIO(img_bytes))
            image_orig_shape.append(img.size)  
            img_resized = img.resize((900, 256))
            img_array = np.array(img_resized) / 1.0 
            img_array = np.moveaxis(img_array, -1, 0)  
            image_data.append(img_array)
        processed_data['image'] = np.array(image_data)
        processed_data['image_orig_shape'] = np.array(image_orig_shape)  
        

        episode_ends = np.array(episode_ends)

        # 创建样本索引，不使用填充
        indices = create_sample_indices_no_padding(
            episode_ends=episode_ends,
            sequence_length=pred_horizon)

        # 只对速度做归一化（除以12），其余特征全部保留原始值
        processed_data['speed'] = np.array(processed_data['speed']) / 12.0

        self.indices = indices
        self.data = processed_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

      

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]


        # 直接用原始数据（只对速度做了归一化）
        sample = sample_sequence(
            train_data=self.data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        if 'image_orig_shape' in sample:
            sample['image_orig_shape'] = torch.tensor(sample['image_orig_shape'], dtype=torch.int32)
        else:
            sample['image_orig_shape'] = torch.zeros((self.obs_horizon, 2), dtype=torch.int32)

        # 对于条件数据，只保留观测部分
        sample['image'] = sample['image'][:self.obs_horizon,:]
        sample['image_orig_shape'] = sample['image_orig_shape'][:self.obs_horizon,:]
        sample['speed'] = sample['speed'][:self.obs_horizon]
        sample['command'] = sample['command'][:self.obs_horizon,:]
        sample['next_command'] = sample['next_command'][:self.obs_horizon,:]
        sample['target_point'] = sample['target_point'][:self.obs_horizon,:]


        return sample

  


       
def test():
    import random
    dataset_path = '/home/wang/projects/diffusion_policy_z/data/tmp_data'
    pred_horizon = 8  # 总的序列长度（包含观测+预测）
    obs_horizon = 3   # 观测的时间步数
    action_horizon = 5  # 动作预测的时间步数
    dataset = CARLAImageDataset(
            dataset_path=dataset_path,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            max_files=20
        )
        # 只在无填充样本中随机选择
    if True:
        # 由于现在所有样本都是无填充的，直接使用所有样本
        print(f"\n总样本数: {len(dataset)}")
        all_samples = list(range(len(dataset)))

        if len(all_samples) > 0:
            rand_idx = random.choice(all_samples)
            rand_sample = dataset[rand_idx]
            print(f"\n随机选择的样本索引: {rand_idx}")
            # agent_pos 可视化
            agent_pos = rand_sample['agent_pos'] if 'agent_pos' in rand_sample else None
            if agent_pos is not None:
                if agent_pos.ndim == 3 and agent_pos.shape[1] == 1:
                    agent_pos = agent_pos.squeeze(1)
                obs_agent_pos = agent_pos[:obs_horizon]
                pred_agent_pos = agent_pos[obs_horizon:]
                plt.figure(figsize=(8, 6))
                # 绘制观测点
                if len(obs_agent_pos) > 0:
                    plt.plot(obs_agent_pos[:, 1], obs_agent_pos[:, 0], 'bo-', label='Observed agent_pos', markersize=6)
                # 绘制预测点
                if len(pred_agent_pos) > 0:
                    plt.plot(pred_agent_pos[:, 1], pred_agent_pos[:, 0], 'ro--', label='Predicted agent_pos', markersize=6)
                plt.xlabel('Y')
                plt.ylabel('X')
                plt.title(f'Random Sample {rand_idx}: agent_pos (blue=obs, red=pred)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(f'/home/wang/projects/diffusion_policy_z/sample_{rand_idx}_agent_pos.png', dpi=150, bbox_inches='tight')
                plt.show()
                print(f"保存agent_pos图像到: sample_{rand_idx}_agent_pos.png")

            # image 可视化（只显示观测部分）
            if 'image' in rand_sample:
                images = rand_sample['image']  # shape: (obs_horizon, C, H, W)
                for t, img_arr in enumerate(images):
                    if img_arr.shape[0] == 3:  # (C, H, W)
                        img_vis = np.moveaxis(img_arr, 0, -1).astype(np.uint8)
                    else:
                        img_vis = img_arr.astype(np.uint8)
                    plt.figure(figsize=(8, 2))
                    plt.imshow(img_vis)
                    plt.title(f'Random Sample {rand_idx} - Obs Image t={t}')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(f'/home/wang/projects/diffusion_policy_z/sample_{rand_idx}_obs_image_t{t}.png', dpi=150, bbox_inches='tight')
                    plt.show()
                    print(f"保存观测图像到: sample_{rand_idx}_obs_image_t{t}.png")


            

            print(f"Dataset length: {len(dataset)}")
            sample = dataset[0]
            print("Sample keys:", sample.keys())
            print("Image shape:", sample['image'].shape)
            print("Agent pos shape:", sample['agent_pos'].shape)

            print("\nNew CARLA fields:")
            for key in ['town_name', 'speed', 'command', 'next_command', 'target_point', 'ego_waypoints', 'image', 'agent_pos']:
                if key in sample:
                    value = sample[key]
                    print(f"  {key}: shape={getattr(value, 'shape', 'N/A')}, type={type(value)}")
                    if hasattr(value, '__len__') and not isinstance(value, str):
                        print(f"    Length: {len(value)}")


    
if __name__ == "__main__":
    test()
