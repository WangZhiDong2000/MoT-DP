import os
from os.path import join
import gzip, json, pickle
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
import cv2
import multiprocessing
import argparse

# All data in the Bench2Drive dataset are in the left-handed coordinate system.
# This code converts all coordinate systems (world coordinate system, vehicle coordinate system,
# camera coordinate system, and lidar coordinate system) to the right-handed coordinate system
# consistent with the nuscenes dataset.

MAX_DISTANCE = 75              # Filter bounding boxes that are too far from the vehicle
FILTER_Z_SHRESHOLD = 10        # Filter bounding boxes that are too high/low from the vehicle
FILTER_INVISINLE = True        # Filter bounding boxes based on visibility
NUM_VISIBLE_SHRESHOLD = 1      # Filter bounding boxes with fewer visible vertices than this value
NUM_OUTPOINT_SHRESHOLD = 7     # Filter bounding boxes where the number of vertices outside the frame is greater than this value in all cameras
CAMERA_TO_FOLDER_MAP = {'CAM_FRONT':'rgb_front', 'CAM_FRONT_LEFT':'rgb_front_left', 'CAM_FRONT_RIGHT':'rgb_front_right', 'CAM_BACK':'rgb_back', 'CAM_BACK_LEFT':'rgb_back_left', 'CAM_BACK_RIGHT':'rgb_back_right'}

stand_to_ue4_rotate = np.array([[ 0, 0, 1, 0],
                                [ 1, 0, 0, 0],
                                [ 0,-1, 0, 0],
                                [ 0, 0, 0, 1]])

lidar_to_righthand_ego = np.array([[  0, 1, 0, 0],
                                   [ -1, 0, 0, 0],
                                   [  0, 0, 1, 0],
                                   [  0, 0, 0, 1]])

lefthand_ego_to_lidar = np.array([[ 0, 1, 0, 0],
                                  [ 1, 0, 0, 0],
                                  [ 0, 0, 1, 0],
                                  [ 0, 0, 0, 1]])

left2right = np.eye(4)
left2right[1,1] = -1

def apply_trans(vec,world2ego):
    vec = np.concatenate((vec,np.array([1])))
    t = world2ego @ vec
    return t[0:3]

def get_pose_matrix(dic):
    new_matrix = np.zeros((4,4))
    new_matrix[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=dic['theta']-np.pi/2).rotation_matrix
    new_matrix[0,3] = dic['x']
    new_matrix[1,3] = dic['y']
    new_matrix[3,3] = 1
    return new_matrix

def get_npc2world(npc):
    for key in ['world2vehicle','world2ego','world2sign','world2ped']:
        if key in npc.keys():
            npc2world = np.linalg.inv(np.array(npc[key]))
            yaw_from_matrix = np.arctan2(npc2world[1,0], npc2world[0,0])
            yaw = npc['rotation'][-1] / 180 * np.pi
            if abs(yaw-yaw_from_matrix)> 0.01:
                npc2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=yaw).rotation_matrix
            npc2world = left2right @ npc2world @ left2right
            return npc2world
    npc2world = np.eye(4)
    npc2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=npc['rotation'][-1]/180*np.pi).rotation_matrix
    npc2world[0:3,3] = np.array(npc['location'])
    return left2right @ npc2world @ left2right

def get_global_trigger_vertex(center,extent,yaw_in_degree):
    x,y = center[0],-center[1]
    dx,dy = extent[0],extent[1]
    yaw_in_radians = -yaw_in_degree/180*np.pi
    vertex_in_self = np.array([[ dx, dy],
                               [-dx, dy],
                               [-dx,-dy],
                               [ dx,-dy]])
    rotate_matrix = np.array([[np.cos(yaw_in_radians),-np.sin(yaw_in_radians)],
                              [np.sin(yaw_in_radians), np.cos(yaw_in_radians)]])
    rotated_vertex = (rotate_matrix @ vertex_in_self.T).T
    vertex_in_global = np.array([[x,y]]).repeat(4,axis=0) + rotated_vertex
    return vertex_in_global

def get_image_point(loc, K, w2c):
    point = np.array([loc[0], loc[1], loc[2], 1])
    point_camera = np.dot(w2c, point)
    point_camera = point_camera[0:3]
    depth = point_camera[2]
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2], depth

def command_to_one_hot(command):
    if command < 0:
        command = 4
    command -= 1
    if command not in [0, 1, 2, 3, 4, 5]:
        command = 3
    cmd_one_hot = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cmd_one_hot[command] = 1.0
    return np.array(cmd_one_hot)

def get_waypoints(measurements, action_horizon, y_augmentation=0.0, yaw_augmentation=0.0):
    """
    Transform waypoints to be origin at current ego position.
    
    Args:
        measurements: List of measurement dicts, where measurements[0] is the current frame
        action_horizon: Number of future waypoints to extract (not including current)
        y_augmentation: Lateral augmentation
        yaw_augmentation: Yaw augmentation in degrees
    
    Returns:
        waypoints_aug: List of waypoints in ego frame (BEV, 2D), length = action_horizon + 1
                      First waypoint is current position [0, 0], followed by future waypoints
    """
    origin = measurements[0]
    origin_matrix = np.array(origin['ego_matrix'])[:3]
    origin_translation = origin_matrix[:, 3:4]
    origin_rotation = origin_matrix[:, :3]

    waypoints = []
    # First waypoint is current position (always [0, 0] in ego frame)
    waypoints.append(np.array([0.0, 0.0]))
    
    # Extract future waypoints: measurements[1:action_horizon+1]
    for index in range(1, action_horizon + 1):
        if index < len(measurements):
            waypoint = np.array(measurements[index]['ego_matrix'])[:3, 3:4]
            waypoint_ego_frame = origin_rotation.T @ (waypoint - origin_translation)
            # Drop the height dimension because we predict waypoints in BEV
            waypoints.append(waypoint_ego_frame[:2, 0])
        else:
            # If we run out of measurements, pad with the last valid waypoint
            if len(waypoints) > 1:
                waypoints.append(waypoints[-1].copy())
            else:
                # This case should be rare if we have at least one future frame
                waypoints.append(np.array([0.0, 0.0]))

    # Final check to ensure correct length
    while len(waypoints) < action_horizon + 1:
        print(f"Warning: Not enough future frames, padding with last waypoint. Have {len(waypoints)}, need {action_horizon+1}")
        if len(waypoints) > 1:
            waypoints.append(waypoints[-1].copy())
        else:
            print("Warning: No future frames available, using current position for all waypoints")
            waypoints.append(np.array([0.0, 0.0]))
    
    # Data augmentation
    waypoints_aug = []
    aug_yaw_rad = np.deg2rad(yaw_augmentation)
    rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], 
                                [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)]])
    translation = np.array([[0.0], [y_augmentation]])
    
    for waypoint in waypoints:
        pos = np.expand_dims(waypoint, axis=1)
        waypoint_aug = rotation_matrix.T @ (pos - translation)
        waypoints_aug.append(np.squeeze(waypoint_aug))

    return waypoints_aug

def preprocess(folder_list, idx, tmp_dir, data_root, out_dir, 
               obs_horizon, action_horizon, sample_interval, hz_interval, save_mode):
    """
    Preprocess PDM Lite data into pkl files.
    
    Args:
        folder_list: List of folders to process
        idx: Worker index
        tmp_dir: Temporary directory name
        train_or_val: 'train' or 'val'
        data_root: Root directory of raw data
        out_dir: Output directory
        obs_horizon: Number of observation history frames.
        action_horizon: Number of future action/waypoint frames to predict.
        sample_interval: Interval between training samples (e.g., 10 frames).
        hz_interval: Interval within a sample to match frequency (e.g., 2 for 2Hz from 4Hz data).
        save_mode: 'scene' to save all frames per scene in one file, 
                   'frame' to save each frame separately.
    """
    final_data = []

    # Check if folder_list contains direct routes or parent directories
    # In Structure 2 (<ScenarioName>/<RouteFolder>), folder_list already contains the complete paths
    # In Structure 1 (TownXX/data/<route>), we may need to expand
    folder_list_new = []
    for folder in folder_list:
        folder_path = join(data_root, folder)
        # Check if this folder directly contains 'measurements' or 'lidar'
        if os.path.exists(join(folder_path, 'measurements')) or os.path.exists(join(folder_path, 'lidar')):
            # This is already a route folder with data
            folder_list_new.append(folder)
        else:
            # This might be a parent folder, try to find route folders inside
            if os.path.isdir(folder_path):
                all_scenes = os.listdir(folder_path)
                for scene in all_scenes:
                    scene_path = join(folder_path, scene)
                    if os.path.exists(join(scene_path, 'measurements')) or os.path.exists(join(scene_path, 'lidar')):
                        folder_list_new.append(join(folder, scene))

    if idx == 0:
        folders = tqdm(folder_list_new)
    else:
        folders = folder_list_new

    for folder_name in folders:
        folder_path = join(data_root, folder_name)
        
        # Check if measurements directory exists
        measurements_dir = join(folder_path, 'measurements')
        if not os.path.exists(measurements_dir):
            if idx == 0:
                print(f"\nWarning: measurements directory not found in {folder_path}, skipping...")
            continue
            
        measurements_files = sorted(os.listdir(measurements_dir))
        num_seq = len(measurements_files)
        
        # Calculate the last valid frame index based on the required future frames
        last_valid_future_frame_offset = (action_horizon) * hz_interval
        last_frame_idx = num_seq - last_valid_future_frame_offset - 1

        scene_data = []
        STORE_BYTES = False # Whether to store image bytes or image paths
        rgb_dir = join(folder_path, 'rgb')
        
        # Start processing from a frame that allows for a full observation history
        scen_start_frame_offset = (obs_horizon - 1) * hz_interval
        
        for ii in range(scen_start_frame_offset, last_frame_idx, sample_interval):
            # --- Load all measurements needed for this sample ---
            # This includes history, current, and future frames, all sampled with hz_interval
            
            # 1. Determine the range of raw frame indices to load
            history_start_idx = ii - (obs_horizon - 1) * hz_interval
            future_end_idx = ii + (action_horizon + 1) * hz_interval    # +1 to include the current frame to calculate relative waypoints 
            
            # 2. Load the measurement files within this range, stepping by hz_interval
            loaded_measurements = []
            for i in range(history_start_idx, future_end_idx, hz_interval):
                if i < 0 or i >= num_seq:
                    continue
                try:
                    measurement_file = measurements_files[i]
                    with gzip.open(join(folder_path, 'measurements', measurement_file), 'rt', encoding='utf-8') as gz_file:
                        anno = json.load(gz_file)
                    # Inject frame_id from filename for consistency
                    anno['frame_id'] = int(measurement_file.split('.')[0])
                    loaded_measurements.append(anno)
                except (FileNotFoundError, IndexError, json.JSONDecodeError):
                    pass
            
            # 3. Find the current frame in our down-sampled list
            current_anno_in_list = next((anno for anno in loaded_measurements if anno['frame_id'] == ii), None)
            
            # We need at least obs_horizon frames in our list, and the current frame must be present
            if current_anno_in_list is None or len(loaded_measurements) < obs_horizon:
                print(f"warning: skipping frame {ii} in {folder_name} due to insufficient data")
                continue
            
            current_idx_in_list = loaded_measurements.index(current_anno_in_list)

            # --- Create the data dictionary for this sample ---
            frame_data = {}
            
            # Metadata
            frame_data['town_name'] = folder_name.split('/')[0]
            frame_data['event_name'] = folder_name.split('/')[2] if len(folder_name.split('/'))>2 else 'None'
            frame_data['route_name'] = folder_name.split('/')[-1]
            frame_data['frame_id'] = ii
            
            # Current state from current_anno
            # frame_data['speed'] = current_anno_in_list['speed']        # need history for speed
            frame_data['throttle'] = current_anno_in_list['throttle']
            frame_data['steer'] = current_anno_in_list['steer']
            frame_data['brake'] = current_anno_in_list['brake']
            # frame_data['theta'] = current_anno_in_list['theta']       # need history for theta
            frame_data['command'] = command_to_one_hot(current_anno_in_list['command'])
            frame_data['next_command'] = command_to_one_hot(current_anno_in_list['next_command'])
            frame_data['target_point'] = np.array(current_anno_in_list['target_point'])

            # --- future waypoints and route ---
            # Waypoints: Pass the down-sampled future measurements to get_waypoints
            # The list starts from the current frame
            future_measurements = loaded_measurements[current_idx_in_list:]       
            waypoints = get_waypoints(future_measurements, action_horizon=action_horizon)
            frame_data['ego_waypoints'] = np.array(waypoints)
            assert frame_data['ego_waypoints'].shape == (action_horizon + 1, 2)
            
            # Route information
            route = current_anno_in_list['route']
            if len(route) < 20:
                num_missing = 20 - len(route)
                route = np.array(route)
                route = np.vstack((route, np.tile(route[-1], (num_missing, 1))))
            else:
                route = np.array(route[:20])
            frame_data['route'] = route

            # --- Low dimensional state History ---
            sped_hist = []
            theta_hist = []
            
            for j in range(0, current_idx_in_list + 1):          # no need for hz_interval here because the loaded measurements already account for it
                anno_j = loaded_measurements[j]
                if anno_j is not None:
                    sped_hist.append(anno_j['speed'])
                    theta_hist.append(anno_j['theta'])
                else:
                    # If the specific frame is missing, repeat the last valid state
                    if sped_hist:
                        sped_hist.append(sped_hist[-1])
                        theta_hist.append(theta_hist[-1])
                    else:
                        sped_hist.append(0.0)
                        theta_hist.append(0.0)
            frame_data['speed_hist'] = np.array(sped_hist)
            frame_data['theta_hist'] = np.array(theta_hist)
            assert frame_data['speed_hist'].shape == (obs_horizon,)
            assert frame_data['theta_hist'].shape == (obs_horizon,)

            # --- Image History ---
            rgb_hist = []
            obs_start_idx = ii - (obs_horizon - 1) * hz_interval
            for j in range(obs_start_idx, ii + 1, hz_interval):
                if j < 0: continue
                img_path = join(rgb_dir, f"{j:04d}.jpg")
                if STORE_BYTES:
                    try:
                        with open(img_path, 'rb') as f:
                            rgb_hist.append(f.read())
                    except FileNotFoundError:
                        rgb_hist.append(b"") # Placeholder for missing image
                else:
                    # Store relative path
                    rgb_hist.append(join(folder_name, 'rgb', f"{j:04d}.jpg"))
            
            # Pad if we don't have enough history frames (e.g., at the start of a scene)
            while len(rgb_hist) < obs_horizon:
                if rgb_hist:
                    rgb_hist.insert(0, rgb_hist[0])
                else:
                    # This should not happen if scen_start_frame_offset is correct, but as a fallback:
                    rgb_hist.insert(0, join(folder_name, 'rgb', f"{obs_start_idx:04d}.jpg"))
                    
            frame_data['rgb_hist_jpg'] = rgb_hist
            
            # --- Lidar BEV History ---
            lidar_bev_dir = join(folder_path, 'lidar_bev')
            lidar_bev_hist = []
            if os.path.exists(lidar_bev_dir):
                for j in range(obs_start_idx, ii + 1, hz_interval):
                    if j < 0: continue
                    bev_path = join(lidar_bev_dir, f"{j:04d}.png")
                    if STORE_BYTES:
                        try:
                            with open(bev_path, 'rb') as f:
                                lidar_bev_hist.append(f.read())
                        except FileNotFoundError:
                            lidar_bev_hist.append(b"")
                    else:
                        if os.path.exists(bev_path):
                            lidar_bev_hist.append(join(folder_name, 'lidar_bev', f"{j:04d}.png"))
                        else:
                            lidar_bev_hist.append(None) # Mark missing BEV image
                
                # Pad if we don't have enough history frames
                while len(lidar_bev_hist) < obs_horizon:
                    if lidar_bev_hist:
                        lidar_bev_hist.insert(0, lidar_bev_hist[0])
                    else:
                        lidar_bev_hist.insert(0, None)
            
            # If lidar_bev directory doesn't exist or no images were found, fill with None
            if not lidar_bev_hist:
                lidar_bev_hist = [None] * obs_horizon
                
            frame_data['lidar_bev_hist'] = lidar_bev_hist

            # --- VQA Features (dp_vl_feature) ---
            # dp_vl_feature contains .pt files at frame indices that are multiples of 5
            # For other frames, map to the previous multiple of 5
            dp_vl_feature_dir = join(folder_path, 'dp_vl_feature')
            vqa_hist = []
            if os.path.exists(dp_vl_feature_dir):
                for j in range(obs_start_idx, ii + 1, hz_interval):
                    if j < 0: 
                        continue
                    # Find the corresponding frame index that is a multiple of 5
                    vqa_frame_idx = (j // 5) * 5
                    vqa_path = join(dp_vl_feature_dir, f"{vqa_frame_idx:04d}.pt")
                    
                    if os.path.exists(vqa_path):
                        vqa_hist.append(join(folder_name, 'dp_vl_feature', f"{vqa_frame_idx:04d}.pt"))
                    else:
                        vqa_hist.append(None)
                
                # Pad if we don't have enough history frames
                while len(vqa_hist) < obs_horizon:
                    if vqa_hist:
                        vqa_hist.insert(0, vqa_hist[0])
                    else:
                        vqa_hist.insert(0, None)
            
            # If dp_vl_feature directory doesn't exist or no features were found, fill with None
            if not vqa_hist:
                vqa_hist = [None] * obs_horizon
                
            frame_data['vqa'] = vqa_hist

            scene_data.append(frame_data)
            
        # --- Save data for the entire scene ---
        if not scene_data:
            continue

        scene_name = folder_name.replace('/', '_')
        out_path_dir = join(out_dir, tmp_dir)
        os.makedirs(out_path_dir, exist_ok=True)
        
        if save_mode == 'scene':
            out_path = join(out_path_dir, f"{scene_name}.pkl")
            with open(out_path, 'wb') as f:
                pickle.dump(scene_data, f)
            if idx == 0:
                print(f"Saved {len(scene_data)} frames to {out_path}")
        elif save_mode == 'frame':
            for frame_data_item in scene_data:
                frame_id = frame_data_item['frame_id']
                out_path = join(out_path_dir, f"{scene_name}_{frame_id:04d}.pkl")
                with open(out_path, 'wb') as f:
                    pickle.dump(frame_data_item, f)
            if idx == 0 and scene_data:
                print(f"Saved {len(scene_data)} frames separately for scene {scene_name}")
        else:
            raise ValueError(f"Invalid save_mode: {save_mode}. Must be 'scene' or 'frame'.")

    # This part is not used for creating final info files anymore
    # os.makedirs(join(out_dir, tmp_dir), exist_ok=True)
    # with open(join(out_dir, tmp_dir, 'b2d_infos_' + train_or_val + '_' + str(idx) + '.pkl'), 'wb') as f:
    #    pickle.dump(final_data, f)

def generate_infos(folder_list, workers, tmp_dir, data_root, out_dir,
                   obs_horizon, action_horizon, sample_interval, hz_interval, save_mode):
    """
    Generate dataset info using multiple workers.
    """
    folder_num = len(folder_list)
    devide_list = [(folder_num // workers) * i for i in range(workers)]
    devide_list.append(folder_num)
    
    process_list = []
    for i in range(workers):
        sub_folder_list = folder_list[devide_list[i]:devide_list[i+1]]
        process = multiprocessing.Process(
            target=preprocess, 
            args=(sub_folder_list, i, tmp_dir, data_root, out_dir,
                  obs_horizon, action_horizon, sample_interval, hz_interval, save_mode)
        )
        process.start()
        process_list.append(process)
    
    # Wait for all processes to finish
    for process in process_list:
        process.join()

def split_train_val(in_dir, out_dir, val_ratio=0.1):
    """
    Split processed data into train and val sets.
    
    Args:
        in_dir: Directory with processed pkl files
        out_dir: Output directory for split datasets
        val_ratio: Ratio of validation set
    """
    import shutil
    all_files = [f for f in os.listdir(in_dir) if f.endswith('.pkl')]
    np.random.shuffle(all_files)
    
    num_val = int(len(all_files) * val_ratio)
    val_files = all_files[:num_val]
    train_files = all_files[num_val:]
    
    train_dir = join(out_dir, 'train')
    val_dir = join(out_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    for f in train_files:
        os.rename(join(in_dir, f), join(train_dir, f))
    
    for f in val_files:
        os.rename(join(in_dir, f), join(val_dir, f))
    
    print(f"Split {len(all_files)} files into {len(train_files)} train and {len(val_files)} val files.")
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Preprocess PDM Lite dataset')
    argparser.add_argument('--data-root', type=str, default='/home/wang/Project/carla_garage/data/' , help='Root directory of raw PDM Lite data')
    argparser.add_argument('--out-dir', type=str, default='/home/wang/Project/carla_garage/tmp_data', help='Output directory for processed data')
    argparser.add_argument('--obs-horizon', type=int, default=2, help='Number of observation history frames')
    argparser.add_argument('--action-horizon', type=int, default=8, help='Number of future action/waypoint frames to predict (e.g., 8 for 4s at 2Hz)')
    argparser.add_argument('--sample-interval', type=int, default=2, help='Interval between training samples (e.g., 10 frames)')
    argparser.add_argument('--hz-interval', type=int, default=2, help='Interval within a sample to match frequency (e.g., 2 for 2Hz from 4Hz data)')
    argparser.add_argument('--save-mode', type=str, default='frame', choices=['scene', 'frame'], help='Save mode: "scene" to save all frames per scene in one file, "frame" to save each frame separately')
    argparser.add_argument('--workers', type=int, default=4, help='Number of workers for parallel processing')
    argparser.add_argument('--tmp-dir', default="tmp_data", help='Temporary directory name for intermediate files')

    args = argparser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # --- Flexible logic to find all route folders from the top-level data_root ---
    # Supports two structures:
    # Structure 1: <data_root>/TownXX/data/<route_folder>
    # Structure 2: <data_root>/<ScenarioName>/<route_folder>
    
    train_list = []
    
    # Try Structure 1: TownXX/data/route_folder
    all_towns = [d for d in os.listdir(args.data_root) 
                 if os.path.isdir(join(args.data_root, d)) and d.startswith('Town')]
    
    if all_towns:
        print(f"Found {len(all_towns)} Town folders, using Structure 1: TownXX/data/...")
        for town_folder in all_towns:
            data_folder_path = join(args.data_root, town_folder, 'data')
            if os.path.isdir(data_folder_path):
                route_folders = os.listdir(data_folder_path)
                for route_folder in route_folders:
                    relative_route_path = join(town_folder, 'data', route_folder)
                    train_list.append(relative_route_path)
    else:
        # Try Structure 2: <ScenarioName>/<route_folder> (where route_folder contains 'lidar' or 'measurements')
        print(f"No Town folders found, trying Structure 2: <ScenarioName>/<RouteFolder>/...")
        all_scenarios = [d for d in os.listdir(args.data_root) 
                        if os.path.isdir(join(args.data_root, d))]
        
        for scenario_folder in all_scenarios:
            scenario_path = join(args.data_root, scenario_folder)
            route_folders = [d for d in os.listdir(scenario_path) 
                           if os.path.isdir(join(scenario_path, d))]
            
            for route_folder in route_folders:
                route_path = join(scenario_path, route_folder)
                # Check if this folder contains required data (lidar or measurements)
                if (os.path.exists(join(route_path, 'lidar')) or 
                    os.path.exists(join(route_path, 'measurements'))):
                    relative_route_path = join(scenario_folder, route_folder)
                    train_list.append(relative_route_path)

    if not train_list:
        print(f"Warning: No route folders found in {args.data_root}. Please check the directory structure.")
        print("Expected structure 1: --data-root/TownXX/data/route_folder_name")
        print("Expected structure 2: --data-root/<ScenarioName>/<RouteFolder>/ (with lidar/ or measurements/ subdirectory)")
    else:
        print(f"Found {len(train_list)} route folders to process.")
    
    print(f'Processing data with parameters:')
    print(f'  Data root: {args.data_root}')
    print(f'  Output dir: {args.out_dir}')
    print(f'  Obs horizon: {args.obs_horizon}')
    print(f'  Action horizon: {args.action_horizon}')
    print(f'  Sample interval: {args.sample_interval}')
    print(f'  Hz interval: {args.hz_interval}')
    print(f'  Save mode: {args.save_mode}')
    print(f'  Workers: {args.workers}')
    print(f'  Number of routes found: {len(train_list)}')
    print()
    
    print('Processing train data...')
    generate_infos(train_list, args.workers, args.tmp_dir, 
                   args.data_root, args.out_dir,
                   args.obs_horizon, args.action_horizon, args.sample_interval,
                   args.hz_interval, args.save_mode)
    
    print('Finished!')

    print("start split train/val dataset")
    split_train_val(join(args.out_dir, args.tmp_dir), args.out_dir, val_ratio=0.1)