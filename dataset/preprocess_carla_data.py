import os
import glob
import pickle
import torch
import numpy as np
from PIL import Image
import io
import argparse
from tqdm import tqdm
import torchvision.transforms as transforms

# 运行方式:
# python dataset/preprocess_carla_data.py \
#   --input-dir /path/to/raw/carla_dataset \
#   --output-dir /path/to/processed_carla_dataset \
#   --pred-horizon 6 \
#   --obs-horizon 2 \
#   --action-horizon 4


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

def process_and_save_data(
    pkl_files,
    output_path,
    pred_horizon,
    obs_horizon,
    action_horizon,
    image_transform,
    sample_interval=1,
):
    """
    Core processing function: Reads raw pkl files, splits them into samples, and saves them.
    """
    global_sample_counter = 0
    # 使用 os.path.basename(output_path) 来获取 'train' 或 'val'
    file_iterator = tqdm(pkl_files, desc=f"Processing files for {os.path.basename(output_path)}")
    if sample_interval > 1:
        print(f"sample interval is {sample_interval}")
    for pkl_file in file_iterator:
        try:
            with open(pkl_file, 'rb') as f:
                episode_data = pickle.load(f)

                # TODO: split the dataset to sample according to meta action interval
                if sample_interval > 1:
                    episode_data = episode_data[::sample_interval]

        except Exception as e:
            print(f"Warning: Could not load file {pkl_file}. Error: {e}")
            continue

        source_scene_name = os.path.basename(pkl_file).replace('.pkl', '')
        valid_items = [item for item in episode_data if len(item['ego_waypoints']) >= 1]
        
        if len(valid_items) < pred_horizon:
            continue

        # 遍历文件，切分出所有可能的序列
        for start_idx in range(len(valid_items) - pred_horizon + 1):
            
            # ==================== 核心修改点 ====================
            # 1. 收集多帧观测数据 (图像路径, 速度等)
            obs_image_paths = []  # 改为存储图像路径而不是图像tensor
            obs_lidar_bev_paths = []  # 存储lidar BEV图像路径
            obs_speeds = []
            obs_headings = []
            # ... 可以根据需要添加其他需要观测历史的状态 ...

            for i in range(obs_horizon):
                obs_item_idx = start_idx + i
                obs_item = valid_items[obs_item_idx]
                
                # 存储图像路径而不是加载图像
                img_data = obs_item['rgb_hist_jpg'][-1]
                # Check if img_data is bytes or a file path
                if isinstance(img_data, bytes):
                    # 如果是bytes，我们需要先保存为文件（或者跳过这种情况）
                    print(f"Warning: Found bytes image data in {pkl_file}, skipping this type")
                    obs_image_paths.append(None)
                else:
                    # img_data is a relative path
                    obs_image_paths.append(img_data)
                
                # 存储lidar BEV图像路径（处理方式与RGB图像相同）
                if 'lidar_bev_hist' in obs_item and obs_item['lidar_bev_hist']:
                    bev_data = obs_item['lidar_bev_hist'][-1]
                    if isinstance(bev_data, bytes):
                        print(f"Warning: Found bytes BEV image data in {pkl_file}, skipping this type")
                        obs_lidar_bev_paths.append(None)
                    else:
                        # bev_data is a relative path or None
                        obs_lidar_bev_paths.append(bev_data)
                else:
                    # 如果没有lidar_bev_hist字段，则添加None
                    obs_lidar_bev_paths.append(None)
                
                # low dimension states
                obs_speeds.append(obs_item['speed'] / 12.0) # 归一化
                obs_headings.append(obs_item['theta'])
                #

            # 将列表转换为Array
            obs_speeds_array = np.array(obs_speeds)     # Shape: (obs_horizon,)
            obs_headings_array = np.array(obs_headings) # Shape: (obs_horizon,)

            # 2. 提取 agent_pos (未来轨迹) 和 target_point (导航目标)
            # agent_pos: 前 obs_horizon 个为 [0,0], 后 action_horizon 个从 ego_waypoints 截取
            
            sequence_positions = []
            
            # 观测部分: 全部为 [0, 0]
            for obs_step in range(obs_horizon):
                sequence_positions.append(np.array([0.0, 0.0]))
            
            # 预测部分: 从当前帧的 ego_waypoints 中截取
            seq_start_item = valid_items[start_idx]
            ego_waypoints = np.array(seq_start_item['ego_waypoints'])
            
            # 从 ego_waypoints[1] 开始取 action_horizon 个点
            for action_step in range(action_horizon):
                waypoint_idx = action_step + 1
                if len(ego_waypoints) > waypoint_idx:
                    action_pos = ego_waypoints[waypoint_idx].copy()
                elif len(ego_waypoints) > 0:
                    action_pos = ego_waypoints[-1].copy()
                else:
                    action_pos = np.array([0.0, 0.0])
                sequence_positions.append(action_pos)
            
            agent_pos = np.array(sequence_positions)
            
            # target_point 和 command 来自于观测序列的起点
            original_target_point = np.array(seq_start_item['target_point'])

            # 3. 构建更完整的样本字典
            sample_data = {
                'town_name': seq_start_item['town_name'],
                'source_scene': source_scene_name,
                
                # 多帧观测数据
                'image_paths': obs_image_paths,  # 存储图像路径列表而不是tensor
                'lidar_bev_paths': obs_lidar_bev_paths,  # 存储lidar BEV图像路径列表
                'speed': obs_speeds_array,       # Shape: (obs_horizon,)  
                'heading': obs_headings_array,   # Shape: (obs_horizon,)  
                # ... 其他观测历史状态 ...
                
                # 静态或单帧数据 (根据需要可以扩展为多帧)
                'command': np.tile(np.array(seq_start_item['command']), (obs_horizon, 1)),  
                'next_command': np.tile(np.array(seq_start_item['next_command']), (obs_horizon, 1)),
                
                # 标签和目标
                'target_point': np.tile(original_target_point, (obs_horizon, 1)),
                'agent_pos': agent_pos,
            }

            # 4. 保存为独立文件
            sample_filename = f"sample_{global_sample_counter:08d}.pkl"
            sample_filepath = os.path.join(output_path, sample_filename)
            with open(sample_filepath, 'wb') as f:
                pickle.dump(sample_data, f)
            
            global_sample_counter += 1

    print(f"Finished processing for {os.path.basename(output_path)}. Total samples: {global_sample_counter}")

def main(args):
    # 创建输出目录
    train_output_dir = os.path.join(args.output_dir, 'train')
    val_output_dir = os.path.join(args.output_dir, 'val')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)

    # 获取所有原始pkl文件
    all_pkl_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pkl")))
    if not all_pkl_files:
        print(f"Error: No .pkl files found in {args.input_dir}")
        return

    print(f"Found {len(all_pkl_files)} total .pkl files.")

    # 划分训练集和验证集
    num_train_files = int(len(all_pkl_files) * args.train_split)
    train_files = all_pkl_files[:num_train_files]
    val_files = all_pkl_files[num_train_files:]

    print(f"Splitting into {len(train_files)} training files and {len(val_files)} validation files.")

    # 处理训练集
    process_and_save_data(
        train_files,
        train_output_dir,
        args.pred_horizon,
        args.obs_horizon,
        args.action_horizon,
        None,  # image_transform不再需要
        args.sample_interval
    )

    # 处理验证集
    process_and_save_data(
        val_files,
        val_output_dir,
        args.pred_horizon,
        args.obs_horizon,
        args.action_horizon,
        None,  # image_transform不再需要
        args.sample_interval
    )

    print("\nPreprocessing complete!")
    print(f"Processed data saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CARLA pkl dataset into individual samples.")
    parser.add_argument('--input-dir', type=str, default='/home/wang/dataset/output/tmp_data', help='Directory containing raw .pkl files.')
    parser.add_argument('--output-dir', type=str, default='/home/wang/projects/diffusion_policy_z/data', help='Directory to save processed sample files.')
    parser.add_argument('--pred-horizon', type=int, default=6, help='Total prediction horizon length.')
    parser.add_argument('--obs-horizon', type=int, default=2, help='Observation horizon length.')
    parser.add_argument('--action-horizon', type=int, default=4, help='Action horizon length.')
    parser.add_argument('--train-split', type=float, default=0.8, help='Ratio of files to use for training.')
    parser.add_argument('--sample-interval', type=int, default=4, help='Interval for downsampling episode data.')
    args = parser.parse_args()
    main(args)