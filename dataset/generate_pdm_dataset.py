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
                 mode: str = 'train',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 use_gpu_processing: bool = True):  

        self.device = device
        self.use_gpu_processing = use_gpu_processing and torch.cuda.is_available()
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        
        # 设置图像变换 - 使用ImageNet标准归一化
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 928)),  # 高度x宽度
            transforms.ToTensor(),  # 转换为[0,1]并变为CHW格式
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet均值
                std=[0.229, 0.224, 0.225]    # ImageNet标准差
            )
        ])
        
        print(f"Using device: {self.device}")
        print(f"GPU image processing: {self.use_gpu_processing}")

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
        

        self.pkl_files = pkl_files
        self.file_to_sequences = {}  
        self.sequence_to_file = []   
        self.sequence_metadata = []  
        
        total_sequences = 0
        print("Scanning files for sequence indexing...")
        
        for file_idx, pkl_file in enumerate(tqdm(pkl_files, desc='Indexing files')):
            with open(pkl_file, 'rb') as f:
                episode_data = pickle.load(f)
            
            valid_items = [item for item in episode_data if len(item['ego_waypoints']) >= 1]
            
            if len(valid_items) >= pred_horizon:
                file_sequence_count = len(valid_items) - pred_horizon + 1
                self.file_to_sequences[file_idx] = (total_sequences, total_sequences + file_sequence_count)
                
                for seq_idx in range(file_sequence_count):
                    self.sequence_to_file.append(file_idx)
                    self.sequence_metadata.append({
                        'file_idx': file_idx,
                        'start_idx': seq_idx,
                        'global_seq_idx': total_sequences + seq_idx
                    })
                
                total_sequences += file_sequence_count
                print(f"File {os.path.basename(pkl_file)}: {file_sequence_count} sequences")
        
        print(f"Total sequences available: {total_sequences}")
        

        self.total_sequences = total_sequences
        episode_ends = np.array([total_sequences])
        
        self.indices = create_sample_indices_no_padding(
            episode_ends=episode_ends,
            sequence_length=1)  

        self._file_cache = {}
        self._cache_size = 3 

    def _load_file_data(self, file_idx):

        if file_idx in self._file_cache:
            return self._file_cache[file_idx]
        
        if len(self._file_cache) >= self._cache_size:
            oldest_key = next(iter(self._file_cache))
            del self._file_cache[oldest_key]
        
        pkl_file = self.pkl_files[file_idx]
        with open(pkl_file, 'rb') as f:
            episode_data = pickle.load(f)
        
        valid_items = [item for item in episode_data if len(item['ego_waypoints']) >= 1]
        
        sequences_data = []
        if len(valid_items) >= self.pred_horizon:
            for start_idx in range(len(valid_items) - self.pred_horizon + 1):
                sequence_positions = []
                
                # 观测部分
                for obs_step in range(self.obs_horizon):
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
                last_obs_idx = start_idx + self.obs_horizon - 1
                if last_obs_idx < len(valid_items):
                    last_obs_item = valid_items[last_obs_idx]
                    last_ego_wp = np.array(last_obs_item['ego_waypoints'])
                    
                    for pred_step in range(self.action_horizon):
                        waypoint_idx = pred_step + 2
                        if len(last_ego_wp) > waypoint_idx:
                            pred_pos = last_ego_wp[waypoint_idx].copy()
                        elif len(last_ego_wp) > 1:
                            pred_pos = last_ego_wp[-1].copy()
                        else:
                            pred_pos = np.array([0.0, 0.0])
                        sequence_positions.append(pred_pos)
                else:
                    for pred_step in range(self.action_horizon):
                        sequence_positions.append(np.array([0.0, 0.0]))
                
                # 计算相对位置
                reference_pos = sequence_positions[self.obs_horizon - 1]
                relative_positions = []
                for pos in sequence_positions:
                    relative_pos = pos - reference_pos
                    relative_positions.append(relative_pos)
                
                agent_pos = np.array(relative_positions)
                
                # 转换target_point为相对位置
                seq_start_item = valid_items[start_idx]
                original_target_point = np.array(seq_start_item['target_point'])
                relative_target_point = original_target_point - reference_pos
                
                # 处理图像 - image_transform已包含Resize + ToTensor + Normalize
                img_bytes = seq_start_item['rgb_hist_jpg'][-1]
                img = Image.open(io.BytesIO(img_bytes))
                img_tensor = self.image_transform(img)  # 返回已归一化的tensor (C, H, W)
                img_data = img_tensor.numpy()  # 转换为numpy以便缓存
                
                # 构建序列数据
                sequence_data = {
                    'town_name': seq_start_item['town_name'],
                    'speed': seq_start_item['speed'] / 12.0,  # 归一化
                    'command': np.array(seq_start_item['command']),
                    'next_command': np.array(seq_start_item['next_command']),
                    'target_point': relative_target_point,
                    'ego_waypoints': seq_start_item['ego_waypoints'],
                    'image': img_data,
                    'image_orig_shape': np.array(img.size),
                    'agent_pos': agent_pos
                }
                
                sequences_data.append(sequence_data)
        
        self._file_cache[file_idx] = sequences_data
        return sequences_data


    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):

        metadata = self.sequence_metadata[idx]
        file_idx = metadata['file_idx']
        start_idx = metadata['start_idx']
        
        file_sequences = self._load_file_data(file_idx)
        
        if start_idx < len(file_sequences):
            sequence_data = file_sequences[start_idx]
        else:
            raise IndexError(f"Sequence index {start_idx} out of range for file {file_idx}")
        
        sample = {
            'town_name': sequence_data['town_name'],
            'speed': np.repeat(sequence_data['speed'], self.obs_horizon),  
            'command': np.tile(sequence_data['command'], (self.obs_horizon, 1)),
            'next_command': np.tile(sequence_data['next_command'], (self.obs_horizon, 1)),
            'target_point': np.tile(sequence_data['target_point'], (self.obs_horizon, 1)),
            'ego_waypoints': sequence_data['ego_waypoints'],
            'image': np.tile(sequence_data['image'], (self.obs_horizon, 1, 1, 1)),  
            'image_orig_shape': np.tile(sequence_data['image_orig_shape'], (self.obs_horizon, 1)),
            'agent_pos': sequence_data['agent_pos']  
        }
        
        for key in ['speed', 'command', 'next_command', 'target_point', 'image', 'image_orig_shape', 'agent_pos']:
            if key in sample:
                if isinstance(sample[key], np.ndarray):
                    sample[key] = torch.from_numpy(sample[key]).float()
                    if key == 'image_orig_shape':
                        sample[key] = sample[key].int()
        
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
            max_files=10
        )
    if True:
        print(f"\n总样本数: {len(dataset)}")
        all_samples = list(range(len(dataset)))

        if len(all_samples) > 0:
            rand_idx = random.choice(all_samples)
            rand_sample = dataset[rand_idx]
            print(f"\n随机选择的样本索引: {rand_idx}")
            agent_pos = rand_sample['agent_pos'] if 'agent_pos' in rand_sample else None
            target_point = rand_sample['target_point'] if 'target_point' in rand_sample else None
            
            if agent_pos is not None:
                if isinstance(agent_pos, torch.Tensor):
                    agent_pos = agent_pos.numpy()
                
                if agent_pos.ndim == 3 and agent_pos.shape[1] == 1:
                    agent_pos = agent_pos.squeeze(1)
                obs_agent_pos = agent_pos[:obs_horizon]
                pred_agent_pos = agent_pos[obs_horizon:]
                
                plt.figure(figsize=(10, 8))
                
                if len(obs_agent_pos) > 0:
                    plt.plot(obs_agent_pos[:, 1], obs_agent_pos[:, 0], 'bo-', label='Observed agent_pos', markersize=8, linewidth=2)

                if len(pred_agent_pos) > 0:
                    plt.plot(pred_agent_pos[:, 1], pred_agent_pos[:, 0], 'ro--', label='Predicted agent_pos', markersize=8, linewidth=2)
                
                if target_point is not None:
                    if isinstance(target_point, torch.Tensor):
                        target_point = target_point.numpy()
                    
                    if target_point.ndim == 2: 
                        target_point = target_point[0]
                    plt.plot(target_point[1], target_point[0], 'g*', markersize=15, label='Target point (relative)', markeredgecolor='black', markeredgewidth=1)
                    
                    if len(obs_agent_pos) > 0:
                        last_obs = obs_agent_pos[-1]
                        plt.plot([last_obs[1], target_point[1]], [last_obs[0], target_point[0]], 'g--', alpha=0.5, linewidth=1, label='Direction to target')
                
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

            if 'image' in rand_sample:
                images = rand_sample['image'] 
                
                # ImageNet反归一化
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                
                for t, img_arr in enumerate(images):
                    if isinstance(img_arr, torch.Tensor):
                        img_arr = img_arr.numpy()
                    
                    # 反归一化：denormalized = normalized * std + mean
                    img_denorm = img_arr * std + mean
                    img_denorm = np.clip(img_denorm, 0, 1)
                    
                    if img_denorm.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                        img_vis = np.moveaxis(img_denorm, 0, -1)
                    else:
                        img_vis = img_denorm
                    
                    plt.figure(figsize=(8, 2))
                    plt.imshow(img_vis)
                    plt.title(f'Random Sample {rand_idx} - Obs Image t={t} (Denormalized)')
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
