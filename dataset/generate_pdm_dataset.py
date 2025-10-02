import numpy as np
from PIL import Image
import os
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
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        # no padding, only full sequences
        if episode_length >= sequence_length:
            for idx in range(episode_length - sequence_length + 1):
                buffer_start_idx = start_idx + idx
                buffer_end_idx = start_idx + idx + sequence_length
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
            sample_idx = buffer_start_idx  
            result[key] = input_arr[sample_idx]  
        else:
            sample = input_arr[buffer_start_idx:buffer_end_idx]
            data = sample
            
            if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
                if isinstance(input_arr, list) and key == 'town_name':
                    data = [None] * sequence_length
                    if sample_start_idx > 0:
                        for i in range(sample_start_idx):
                            data[i] = sample[0] if len(sample) > 0 else None
                    if sample_end_idx < sequence_length:
                        for i in range(sample_end_idx, sequence_length):
                            data[i] = sample[-1] if len(sample) > 0 else None
                    for i, item in enumerate(sample):
                        if sample_start_idx + i < sample_end_idx:
                            data[sample_start_idx + i] = item
                elif isinstance(input_arr, list) and key == 'rgb_hist_jpg':
                    data = [None] * sequence_length
                    if sample_start_idx > 0:
                        for i in range(sample_start_idx):
                            data[i] = sample[0] if len(sample) > 0 else None
                    if sample_end_idx < sequence_length:
                        for i in range(sample_end_idx, sequence_length):
                            data[i] = sample[-1] if len(sample) > 0 else None
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
        all_episode_data = []  
        
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
        
        episode_ends = []
        current_length = 0
        
        for episode_data in all_episode_data:
            valid_count = 0
            for item in episode_data:
                if len(item['ego_waypoints']) >= 1:
                    valid_count += 1
            
            if valid_count >= pred_horizon:  # only keep episodes with enough length
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
        
        # 首先生成agent_pos序列
        '''
        训练轨迹生成策略：
        观测部分 (历史轨迹):
            Time 0: ego_waypoints[0] = [0,0] (当前位置基准)
            Time 1: 使用Time 0的ego_waypoints[1] (从0到1的实际移动)
        预测部分 (使用Time 1的ego_waypoints预测未来):
            Time 2: ego_waypoints[2] 
            Time 3: ego_waypoints[3] 
            Time 4: ego_waypoints[4] 
            Time 5: ego_waypoints[5] 
            Time 6: ego_waypoints[6] 
        '''
        if current_length >= pred_horizon:
            agent_pos_sequences = []
            sequence_data = {
                'town_name': [],
                'speed': [],
                'command': [],
                'next_command': [],
                'target_point': [],
                'ego_waypoints': [],
                'rgb_hist_jpg': []
            }
            episode_ends_for_sequences = []
            sequence_count = 0
            
            for ep_idx, ep_data in enumerate(all_episode_data):
                valid_items = [item for item in ep_data if len(item['ego_waypoints']) >= 1]
                episode_sequence_count = 0
                
                if len(valid_items) >= pred_horizon:
                    for start_idx in range(len(valid_items) - pred_horizon + 1):
                        sequence_positions = []
                        
                        # 观测部分
                        for obs_step in range(obs_horizon):
                            if obs_step == 0:
                                current_pos = np.array([0.0, 0.0])
                            else:
                                prev_step_idx = start_idx + obs_step - 1
                                if prev_step_idx < len(valid_items):
                                    prev_item = valid_items[prev_step_idx]
                                    prev_ego_wp = np.array(prev_item['ego_waypoints'])
                                    if len(prev_ego_wp) > 1:
                                        current_pos = prev_ego_wp[1].copy()
                                    else:
                                        current_pos = np.array([0.0, 0.0])
                                else:
                                    current_pos = np.array([0.0, 0.0])
                            sequence_positions.append(current_pos)
                        
                        # 预测部分
                        last_obs_idx = start_idx + obs_horizon - 1
                        if last_obs_idx < len(valid_items):
                            last_obs_item = valid_items[last_obs_idx]
                            last_ego_wp = np.array(last_obs_item['ego_waypoints'])
                            
                            for pred_step in range(action_horizon):
                                waypoint_idx = pred_step + 2
                                if len(last_ego_wp) > waypoint_idx:
                                    pred_pos = last_ego_wp[waypoint_idx].copy()
                                elif len(last_ego_wp) > 1:
                                    pred_pos = last_ego_wp[-1].copy()
                                else:
                                    pred_pos = np.array([0.0, 0.0])
                                sequence_positions.append(pred_pos)
                        else:
                            for pred_step in range(action_horizon):
                                sequence_positions.append(np.array([0.0, 0.0]))
                        
                        # 计算相对位置
                        reference_pos = sequence_positions[obs_horizon - 1]
                        relative_positions = []
                        for pos in sequence_positions:
                            relative_pos = pos - reference_pos
                            relative_positions.append(relative_pos)
                        
                        agent_pos_sequences.append(np.array(relative_positions))
                        
                        # 转换target_point为相对位置
                        seq_start_item = valid_items[start_idx]
                        original_target_point = np.array(seq_start_item['target_point'])
                        relative_target_point = original_target_point - reference_pos
                        
                        # 为这个序列收集对应的数据（使用序列的起始时间步数据）
                        sequence_data['town_name'].append(seq_start_item['town_name'])
                        sequence_data['speed'].append(seq_start_item['speed'])
                        sequence_data['command'].append(seq_start_item['command'])
                        sequence_data['next_command'].append(seq_start_item['next_command'])
                        sequence_data['target_point'].append(relative_target_point)  # 使用相对位置
                        sequence_data['ego_waypoints'].append(seq_start_item['ego_waypoints'])
                        sequence_data['rgb_hist_jpg'].append(seq_start_item['rgb_hist_jpg'][-1])
                        
                        episode_sequence_count += 1
                        sequence_count += 1
                
                if episode_sequence_count > 0:
                    episode_ends_for_sequences.append(sequence_count)
            
            print(f"Generated {len(agent_pos_sequences)} agent_pos sequences")
            processed_data = sequence_data
            processed_data['agent_pos'] = agent_pos_sequences
            episode_ends = episode_ends_for_sequences
                    
        else:
            print("Warning: Not enough samples to create agent_pos sequences")
        
        for key in ['speed', 'command', 'next_command', 'target_point', 'ego_waypoints']:
            processed_data[key] = np.array(processed_data[key])
        
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
        indices = create_sample_indices_no_padding(
            episode_ends=episode_ends,
            sequence_length=pred_horizon)

        # 只对速度做归一化（除以12），其余特征全部使用原始值
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
    pred_horizon = 7  # 总的序列长度（包含观测+预测）
    obs_horizon = 2   # 观测的时间步数
    action_horizon = 5  # 动作预测的时间步数
    dataset = CARLAImageDataset(
            dataset_path=dataset_path,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            max_files=20
        )
    if True:
        print(f"\n总样本数: {len(dataset)}")
        all_samples = list(range(len(dataset)))

        if len(all_samples) > 0:
            rand_idx = random.choice(all_samples)
            rand_sample = dataset[rand_idx]
            print(f"\n随机选择的样本索引: {rand_idx}")
            # agent_pos 可视化
            agent_pos = rand_sample['agent_pos'] if 'agent_pos' in rand_sample else None
            target_point = rand_sample['target_point'] if 'target_point' in rand_sample else None
            
            if agent_pos is not None:
                if agent_pos.ndim == 3 and agent_pos.shape[1] == 1:
                    agent_pos = agent_pos.squeeze(1)
                obs_agent_pos = agent_pos[:obs_horizon]
                pred_agent_pos = agent_pos[obs_horizon:]
                
                plt.figure(figsize=(10, 8))
                
                # 绘制观测点
                if len(obs_agent_pos) > 0:
                    plt.plot(obs_agent_pos[:, 1], obs_agent_pos[:, 0], 'bo-', label='Observed agent_pos', markersize=8, linewidth=2)
                
                # 绘制预测点
                if len(pred_agent_pos) > 0:
                    plt.plot(pred_agent_pos[:, 1], pred_agent_pos[:, 0], 'ro--', label='Predicted agent_pos', markersize=8, linewidth=2)
                
                # 绘制target_point（相对于最后观测位置）
                if target_point is not None:
                    if target_point.ndim == 2:  # 如果是序列格式，取第一个
                        target_point = target_point[0]
                    plt.plot(target_point[1], target_point[0], 'g*', markersize=15, label='Target point (relative)', markeredgecolor='black', markeredgewidth=1)
                    
                    # 从最后观测点到目标点画一条虚线
                    if len(obs_agent_pos) > 0:
                        last_obs = obs_agent_pos[-1]
                        plt.plot([last_obs[1], target_point[1]], [last_obs[0], target_point[0]], 'g--', alpha=0.5, linewidth=1, label='Direction to target')
                
                # 标记最后一个观测点（参考原点）
                if len(obs_agent_pos) > 0:
                    last_obs = obs_agent_pos[-1]
                    plt.plot(last_obs[1], last_obs[0], 'ks', markersize=10, label='Reference point (last obs)', markerfacecolor='yellow', markeredgecolor='black')
                
                plt.xlabel('Y (relative to last obs)')
                plt.ylabel('X (relative to last obs)')
                plt.title(f'Sample {rand_idx}: Trajectory & Target Point\n(All positions relative to last observation)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(f'/home/wang/projects/diffusion_policy_z/sample_{rand_idx}_agent_pos_with_target.png', dpi=150, bbox_inches='tight')
                plt.show()
                print(f"保存agent_pos和target_point图像到: sample_{rand_idx}_agent_pos_with_target.png")
                print(f"\n样本 {rand_idx} 的详细信息:")
                print(f"观测位置: {obs_agent_pos}")
                print(f"预测位置: {pred_agent_pos}")
                if target_point is not None:
                    print(f"目标点 (相对): {target_point}")
                    if len(obs_agent_pos) > 0:
                        distance_to_target = np.linalg.norm(target_point - obs_agent_pos[-1])
                        print(f"目标点距离最后观测点的距离: {distance_to_target:.3f}")
            # image 可视化
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
