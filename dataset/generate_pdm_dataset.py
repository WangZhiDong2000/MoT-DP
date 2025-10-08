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
import time


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
                 use_gpu_processing: bool = True,
                 sample_interval: int = 2,
                 image_base_path: str = None):  # 新增：图像文件的基础路径

        self.device = device
        self.use_gpu_processing = use_gpu_processing and torch.cuda.is_available()
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.sample_interval = sample_interval
        self.image_base_path = image_base_path  # 保存图像基础路径
        
        
        
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
        print(f"\n{'='*60}")
        print(f"Found {len(pkl_files)} PKL files")
        
        # 警告：文件数量过多可能导致初始化时间很长
        if len(pkl_files) > 100:
            print(f"\n⚠️  WARNING: Large number of files detected!")
            print(f"   - Indexing {len(pkl_files)} files may take several minutes")
            print(f"   - Consider using max_files parameter to limit dataset size")
            print(f"   - Recommended: max_files=50-100 for initial testing")
        
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
        corrupted_files = []

        # TODO : 并行化文件加载和索引
        for file_idx, pkl_file in enumerate(tqdm(pkl_files, desc='Indexing files')):
            try:
                with open(pkl_file, 'rb') as f:
                    episode_data = pickle.load(f)
                
                # 应用采样间隔进行降采样
                episode_data = episode_data[::self.sample_interval]
                
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
                    # print(f"File {os.path.basename(pkl_file)}: {file_sequence_count} sequences")
            except (pickle.UnpicklingError, EOFError, Exception) as e:
                print(f"Warning: Skipping corrupted file {os.path.basename(pkl_file)}: {e}")
                corrupted_files.append(pkl_file)
                continue
            # break
        
        if corrupted_files:
            print(f"\n⚠️ Skipped {len(corrupted_files)} corrupted files:")
            for f in corrupted_files[:5]:  # 只显示前5个
                print(f"  - {os.path.basename(f)}")
            if len(corrupted_files) > 5:
                print(f"  ... and {len(corrupted_files) - 5} more")
        
        print(f"\n{'='*60}")
        print(f"DATASET STATISTICS:")
        print(f"  Total sequences available: {total_sequences}")
        print(f"  Files loaded: {len([f for f in pkl_files if f not in corrupted_files])}")
        print(f"  Sequence length (pred_horizon): {pred_horizon}")
        print(f"  Observation steps (obs_horizon): {obs_horizon}")
        print(f"  Sample interval: {sample_interval}")
        
        # 估计内存占用
        estimated_memory_per_seq_mb = 4.0
        estimated_total_memory_gb = (total_sequences * estimated_memory_per_seq_mb) / 1024
        
        print(f"\nMEMORY ESTIMATES:")
        print(f"  ~{estimated_memory_per_seq_mb} MB per sequence")
        print(f"  ~{estimated_total_memory_gb:.2f} GB total dataset size")
        
        if estimated_total_memory_gb > 50:
            print(f"\n⚠️⚠️⚠️  CRITICAL WARNING  ⚠️⚠️⚠️")
            print(f"Dataset is VERY LARGE (~{estimated_total_memory_gb:.1f} GB)!")
            print(f"This will cause:")
            print(f"  1. Long initialization time (5-30 minutes)")
            print(f"  2. High memory usage during training")
            print(f"  3. DataLoader worker processes may hang/freeze")
            print(f"  4. First batch loading may take 10-30 minutes")
            print(f"\nRECOMMENDATIONS:")
            print(f"  - Reduce max_files (current: unlimited)")
            print(f"  - Increase sample_interval (current: {sample_interval})")
            print(f"  - Set num_workers=0 or 1 in DataLoader config")
        elif estimated_total_memory_gb > 20:
            print(f"\n⚠️  WARNING: Large dataset (~{estimated_total_memory_gb:.1f} GB)")
            print(f"  - Initialization may take 2-5 minutes")
            print(f"  - First batch loading may take 2-5 minutes")
            print(f"  - Consider reducing num_workers in DataLoader")
        
        print(f"{'='*60}\n")
        

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
        try:
            with open(pkl_file, 'rb') as f:
                episode_data = pickle.load(f)
            
            # 应用采样间隔进行降采样
            episode_data = episode_data[::self.sample_interval]
            
            valid_items = [item for item in episode_data if len(item['ego_waypoints']) >= 1]
        except (pickle.UnpicklingError, EOFError, Exception) as e:
            print(f"Error loading file {os.path.basename(pkl_file)}: {e}")
            self._file_cache[file_idx] = []
            return []
        
        sequences_data = []
        time1=time.time()
        if len(valid_items) >= self.pred_horizon:
            for start_idx in range(len(valid_items) - self.pred_horizon + 1):
                sequence_positions = []
                
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
                
                last_obs_idx = start_idx + self.obs_horizon - 1
                if last_obs_idx < len(valid_items):
                    last_obs_item = valid_items[last_obs_idx]
                    last_ego_wp = np.array(last_obs_item['ego_waypoints'])
                    
                    base_waypoint_offset = 1  
                    for pred_step in range(self.action_horizon):
                        waypoint_idx = base_waypoint_offset + (pred_step + 1) * self.sample_interval
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
                
                reference_pos = sequence_positions[self.obs_horizon - 1]
                relative_positions = []
                for pos in sequence_positions:
                    relative_pos = pos - reference_pos
                    relative_positions.append(relative_pos)
                
                agent_pos = np.array(relative_positions)
                seq_start_item = valid_items[start_idx]
                original_target_point = np.array(seq_start_item['target_point'])
                relative_target_point = original_target_point - reference_pos
                
                # 处理图像 - image_transform已包含Resize + ToTensor + Normalize
                img_bytes = seq_start_item['rgb_hist_jpg'][-1]
                
                try:
                    if isinstance(img_bytes, str):
                        # 如果是文件路径字符串，需要拼接完整路径
                        if self.image_base_path is not None:
                            # 使用提供的基础路径
                            img_path = os.path.join(self.image_base_path, img_bytes)
                        else:
                            # 尝试使用相对于PKL文件的路径
                            pkl_dir = os.path.dirname(pkl_file)
                            img_path = os.path.join(pkl_dir, img_bytes)
                        
                        # 从文件读取
                        with open(img_path, 'rb') as f:
                            img_bytes = f.read()
                    elif isinstance(img_bytes, bytes):
                        # 如果已经是字节数据，直接使用
                        pass
                    else:
                        # 其他类型，尝试转换
                        raise ValueError(f"Unexpected image data type: {type(img_bytes)}")
                    
                    img = Image.open(io.BytesIO(img_bytes))
                    img_tensor = self.image_transform(img)  # 返回已归一化的tensor (C, H, W)
                    img_data = img_tensor.numpy()  # 转换为numpy以便缓存
                    
                except Exception as e:
                    print(f"Warning: Failed to load image for sequence {start_idx} in file {os.path.basename(pkl_file)}: {e}")
                    print(f"  Image data type: {type(seq_start_item['rgb_hist_jpg'][-1])}")
                    if isinstance(seq_start_item['rgb_hist_jpg'][-1], str):
                        print(f"  Image relative path: {seq_start_item['rgb_hist_jpg'][-1]}")
                        if self.image_base_path:
                            attempted_path = os.path.join(self.image_base_path, seq_start_item['rgb_hist_jpg'][-1])
                        else:
                            attempted_path = os.path.join(os.path.dirname(pkl_file), seq_start_item['rgb_hist_jpg'][-1])
                        print(f"  Attempted full path: {attempted_path}")
                        print(f"  Path exists: {os.path.exists(attempted_path)}")
                    # 使用默认的空图像
                    img_data = np.zeros((3, 256, 928), dtype=np.float32)
                
                # 构建序列数据
                sequence_data = {
                    'town_name': seq_start_item['town_name'],
                    'speed': seq_start_item['speed'] / 12.0,  # 归一化
                    'command': np.array(seq_start_item['command']),
                    'next_command': np.array(seq_start_item['next_command']),
                    'target_point': relative_target_point,
                    'ego_waypoints': seq_start_item['ego_waypoints'],
                    'image': img_data,
                    'image_orig_shape': np.array([928, 256]) if isinstance(img_bytes, str) else np.array(img.size),
                    'agent_pos': agent_pos
                }
                
                sequences_data.append(sequence_data)
        
        self._file_cache[file_idx] = sequences_data
        time2=time.time()
        print(f"Loaded file {os.path.basename(pkl_file)} with {len(sequences_data)} sequences in {time2-time1:.2f} seconds")
        return sequences_data


    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        # duandian
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
    dataset_path = '/home/wang/dataset/output/tmp_data'
    actual_dataset_path = '/home/wang/dataset/data'  # 图像文件的实际位置
    pred_horizon = 7  # 总的序列长度（包含观测+预测）
    obs_horizon = 2   # 观测的时间步数
    action_horizon = 4  # 动作预测的时间步数
    sample_interval = 2  #
    dataset = CARLAImageDataset(
            dataset_path=dataset_path,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            max_files=10,
            sample_interval=sample_interval,
            image_base_path=actual_dataset_path  # 传入图像基础路径
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
