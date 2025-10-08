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

DATAROOT = '/home/wang/dataset/data'  # Bench2Drive raw data root
OUT_DIR = '/home/wang/dataset/pkl'

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

def get_waypoints(measurements, seq_len, y_augmentation=0.0, yaw_augmentation=0.0):
    """transform waypoints to be origin at ego_matrix"""
    origin = measurements[0]
    origin_matrix = np.array(origin['ego_matrix'])[:3]
    origin_translation = origin_matrix[:, 3:4]
    origin_rotation = origin_matrix[:, :3]

    waypoints = []
    for index in range(0, len(measurements)):
        waypoint = np.array(measurements[index]['ego_matrix'])[:3, 3:4]
        waypoint_ego_frame = origin_rotation.T @ (waypoint - origin_translation)
        # Drop the height dimension because we predict waypoints in BEV
        waypoints.append(waypoint_ego_frame[:2, 0])
    # Data augmentation
    waypoints_aug = []
    aug_yaw_rad = np.deg2rad(yaw_augmentation)
    rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad),
                                                                                np.cos(aug_yaw_rad)]])

    translation = np.array([[0.0], [y_augmentation]])
    for waypoint in waypoints:
        pos = np.expand_dims(waypoint, axis=1)
        waypoint_aug = rotation_matrix.T @ (pos - translation)
        waypoints_aug.append(np.squeeze(waypoint_aug))

    return waypoints_aug

def preprocess(folder_list, idx, tmp_dir, train_or_val):

    data_root = DATAROOT
    pred_len = 2 * 4  # 2s
    seq_len = 1 * 4  # 1s
    final_data = []

    folder_list_new = []
    for folder in folder_list:
        all_scenes = os.listdir(join(data_root, folder))
        for scene in all_scenes:
            if 'Town12' in scene or 'Town13' in scene:
                folder_list_new.append(join(folder, scene))

    if idx == 0:
        folders = tqdm(folder_list_new)
    else:
        folders = folder_list_new

    for folder_name in folders:
        folder_path = join(data_root, folder_name)
        num_seq = len(os.listdir(join(folder_path,'measurements')))
        last_frame = num_seq - (seq_len - 1) - pred_len

        scene_data = []
        STORE_BYTES = True # Whether to store image bytes or image paths
        rgb_dir = join(folder_path, 'rgb')

        for ii in range(0, last_frame):
            loaded_measurements = []
            for i in range(ii, ii + seq_len + pred_len):
                with gzip.open(join(folder_path, 'measurements', str(i).zfill(4) + '.json.gz'), 'rt', encoding='utf-8') as gz_file:
                    anno = json.load(gz_file)
                loaded_measurements.append(anno)
            
            current_anno = loaded_measurements[seq_len-1]
            frame_data = {}
            
            frame_data['town_name'] =  folder_name.split('/')[1].split('_')[0]
            frame_data['speed'] = current_anno['speed']
            frame_data['throttle'] = current_anno['throttle']
            frame_data['steer'] = current_anno['steer']
            frame_data['brake'] = current_anno['brake']
            frame_data['theta'] = current_anno['theta']
            frame_data['command'] = command_to_one_hot(current_anno['command'])
            frame_data['next_command'] = command_to_one_hot(current_anno['next_command'])
            frame_data['target_point'] = np.array(current_anno['target_point'])

            waypoints = get_waypoints(loaded_measurements[seq_len - 1:], seq_len=seq_len)
            frame_data['ego_waypoints'] = np.array(waypoints)
            
            route = current_anno['route']
            if len(route) < 20:
                num_missing = 20 - len(route)
                route = np.array(route)
                route = np.vstack((route, np.tile(route[-1], (num_missing, 1))))
            else:
                route = np.array(route[:20])
            frame_data['route'] = route

            rgb_hist = []
            for j in range(ii, ii + seq_len):
                img_path = join(rgb_dir, f"{j:04d}.jpg")
                if STORE_BYTES:
                    try:
                        with open(img_path, 'rb') as f:
                            rgb_hist.append(f.read())
                    except FileNotFoundError:
                        rgb_hist.append(b"")
                else:
                    rgb_hist.append(img_path)
            frame_data['rgb_hist_jpg'] = rgb_hist

            scene_data.append(frame_data)
            
        scene_name = folder_name.replace('/', '_')
        out_dir = join(OUT_DIR, tmp_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = join(out_dir, f"{scene_name}_{train_or_val}.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(scene_data, f)
        print(f"Saved {len(scene_data)} frames to {out_path}")

    os.makedirs(join(OUT_DIR,tmp_dir),exist_ok=True)
    with open(join(OUT_DIR,tmp_dir,'b2d_infos_'+train_or_val+'_'+str(idx)+'.pkl'),'wb') as f:
        pickle.dump(final_data,f)


def generate_infos(folder_list,workers,train_or_val,tmp_dir):

    folder_num = len(folder_list)
    devide_list = [(folder_num//workers)*i for i in range(workers)]
    devide_list.append(folder_num)
    for i in range(workers):
        sub_folder_list = folder_list[devide_list[i]:devide_list[i+1]]
        # preprocess(sub_folder_list,i,tmp_dir,train_or_val)
        process = multiprocessing.Process(target=preprocess, args=(sub_folder_list,i,tmp_dir,train_or_val))
        process.start()
        process_list.append(process)
    # for i in range(workers):
    #     process_list[i].join()

    # union_data = []
    # for i in range(workers):
    #     with open(join(OUT_DIR,tmp_dir,'b2d_infos_'+train_or_val+'_'+str(i)+'.pkl'),'rb') as f:
    #         data = pickle.load(f)
    #     union_data.extend(data)
    # with open(join(OUT_DIR,'b2d_infos_'+train_or_val+'.pkl'),'wb') as f:
    #     pickle.dump(union_data,f)

if __name__ == "__main__":


    os.makedirs(OUT_DIR,exist_ok=True)
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--workers',type=int, default=4, help='num of workers to prepare dataset')
    argparser.add_argument('--tmp_dir', default="tmp_data", )
    args = argparser.parse_args()    
    workers = args.workers
    process_list = []
    # with open('../../data/splits/bench2drive_base_train_val_split.json','r') as f:
    #     train_val_split = json.load(f)
        
    all_folder = os.listdir(DATAROOT)
    train_list = []
    for foldername in all_folder:
        if 'training_jsons' in foldername:
            continue
        else:
            train_list.append(foldername)
    print('processing train data...')
    generate_infos(train_list,workers,'train',args.tmp_dir)
    # process_list = []
    # print('processing val data...')
    # generate_infos(train_val_split['val'],workers,'val',args.tmp_dir)
    print('finish!')