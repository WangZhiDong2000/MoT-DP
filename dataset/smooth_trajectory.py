# -*- coding: utf-8 -*-
import os
from os.path import join
import json, pickle, argparse, multiprocessing, re, math, gzip
import numpy as np
from tqdm import tqdm

DATAROOT = '/home/wang/Dataset/b2d_10scene'
OUT_DIR  = '/home/wang/Dataset/b2d_10scene'
SAMPLE_INTERVAL = 1                       # 每隔多少帧采样一次
IMAGE_ROOT = DATAROOT                      # 用于拼接图片路径（默认等于 DATAROOT）

CAMERA_SUBDIR = 'camera/rgb_front'         # 每帧 jpg 位于 <scene>/camera/rgb_front/00000.jpg
FRAME_NAME_RE = re.compile(r'^\d{5}\.json\.gz$')

def command_to_one_hot(command):
    if command < 0:
        command = 4
    command -= 1
    if command not in [0, 1, 2, 3, 4, 5]:
        command = 3
    cmd_one_hot = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cmd_one_hot[command] = 1.0
    return np.array(cmd_one_hot)

def _is_num(x):
    try:
        return isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False

def _bilateral_1d(arr, sigma_t=2.0, sigma_s=0.8, radius=5):
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    out = np.empty_like(arr)
    offsets = np.arange(-radius, radius + 1)
    time_kernel = np.exp(-0.5 * (offsets / float(sigma_t)) ** 2)
    for i in range(n):
        j0, j1 = max(0, i - radius), min(n - 1, i + radius)
        js = np.arange(j0, j1 + 1)
        tk = time_kernel[(js - i) + radius]
        diff = arr[js] - arr[i]
        sk = np.exp(-0.5 * (diff / float(sigma_s)) ** 2)
        w = tk * sk
        s = w.sum()
        out[i] = arr[i] if s <= 1e-12 else (w @ arr[js]) / s
    return out

def smooth_traj(traj_xyz, sigma_t=2.0, sigma_s_xy=0.8, sigma_s_z=None, radius=5):
    """
    traj_xyz: (N,3) 的 np.ndarray
    """
    traj = np.asarray(traj_xyz, dtype=np.float64)
    assert traj.ndim == 2 and traj.shape[1] in (2,3)
    if traj.shape[1] == 2:
        traj = np.concatenate([traj, np.zeros((traj.shape[0],1), dtype=np.float64)], axis=1)
    out = traj.copy()
    out[:,0] = _bilateral_1d(out[:,0], sigma_t, sigma_s_xy, radius)
    out[:,1] = _bilateral_1d(out[:,1], sigma_t, sigma_s_xy, radius)
    ss = sigma_s_xy if sigma_s_z is None else sigma_s_z
    out[:,2] = _bilateral_1d(out[:,2], sigma_t, ss, radius)
    return out

def compute_theta_from_trajectory(traj_xy):
    """
    计算从轨迹中导出的theta (heading angle)
    traj_xy: (N, 2) 的 np.ndarray，包含平滑后的x, y坐标
    返回: (N,) 的数组，包含每个点的heading角度
    """
    traj = np.asarray(traj_xy, dtype=np.float64)
    assert traj.ndim == 2 and traj.shape[1] == 2
    
    n = traj.shape[0]
    theta = np.zeros(n)
    
    # 对于每个点，计算指向下一个点的方向
    for i in range(n - 1):
        dx = traj[i + 1, 0] - traj[i, 0]
        dy = traj[i + 1, 1] - traj[i, 1]
        theta[i] = np.arctan2(dy, dx) + np.pi / 2  # 转换为CARLA坐标系
    
    # 最后一个点使用前一个点的方向
    if n > 1:
        theta[-1] = theta[-2]
    
    return theta

def list_scene_dirs(data_root):
    scenes = []
    for d in os.listdir(data_root):
        scene_path = join(data_root, d)
        if os.path.isdir(scene_path) and os.path.isdir(join(scene_path, 'anno')):
            scenes.append(d)
    return sorted(scenes)

def list_frame_jsons(scene_dir):
    files = [f for f in os.listdir(scene_dir) if FRAME_NAME_RE.match(f)]
    files.sort()
    return files

def extract_ego_world_from_obj(obj):

    x = obj.get("x", None)
    y = obj.get("y", None)
    z = 0.0

    bbs = obj.get("bounding_boxes", [])
    ego_z = None
    ego_xy = None
    for bb in bbs:
        if bb.get("class") == "ego_vehicle":
            c = bb.get("center")
            loc = bb.get("location")
            if isinstance(c, list) and len(c) >= 3 and _is_num(c[2]):
                ego_z = float(c[2])
            elif isinstance(loc, list) and len(loc) >= 3 and _is_num(loc[2]):
                ego_z = float(loc[2])
            if isinstance(loc, list) and len(loc) >= 2 and _is_num(loc[0]) and _is_num(loc[1]):
                ego_xy = (float(loc[0]), float(loc[1]))
            break
    if ego_z is not None:
        z = ego_z

    if _is_num(x) and _is_num(y):
        return [float(x), float(y), float(z)]
    if ego_xy is not None:
        return [ego_xy[0], ego_xy[1], float(z)]
    return None

def read_optional(obj, key, default=None):
    v = obj.get(key, default)
    return v

def read_json_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def preprocess_b2d(scene_list, idx, tmp_dir, train_or_val,
                   sigma_t=5, sigma_s_xy=4, sigma_s_z=None, radius=15):
    data_root = DATAROOT
    out_dir  = join(OUT_DIR, tmp_dir)
    os.makedirs(out_dir, exist_ok=True)

    iterator = tqdm(scene_list) if idx == 0 else scene_list
    for scene_id in iterator:
        scene_path = join(data_root, scene_id, 'anno')
        frame_files = list_frame_jsons(scene_path)
        if not frame_files:
            continue
        
        fids, traj_raw = [], []
        json_cache = {}
        for fname in frame_files:
            fid = int(fname.split('.')[0])
            obj = read_json_gz(join(scene_path, fname))
            json_cache[fid] = obj
            pt = extract_ego_world_from_obj(obj)
            if pt is None:
                continue
            fids.append(fid)
            traj_raw.append(pt)
        if len(traj_raw) == 0:
            continue

        # 对原始轨迹进行平滑处理
        traj_raw = np.asarray(traj_raw, dtype=np.float64)
        traj_smooth = smooth_traj(traj_raw, sigma_t=sigma_t, sigma_s_xy=sigma_s_xy, sigma_s_z=sigma_s_z, radius=radius)
        
        # 从平滑后的轨迹计算theta (heading angle)
        traj_smooth_xy = traj_smooth[:, :2]
        theta_smooth = compute_theta_from_trajectory(traj_smooth_xy)
        
        # 创建包含平滑后x, y, theta的映射
        fid2smooth = {}
        for i, fid in enumerate(fids):
            fid2smooth[fid] = {
                'x': float(traj_smooth[i, 0]),
                'y': float(traj_smooth[i, 1]),
                'z': float(traj_smooth[i, 2]),
                'theta': float(theta_smooth[i])
            }

        # 保存平滑后的轨迹到文件，供preprocess_b2d.py使用
        smooth_traj_file = join(out_dir, f"{scene_id}_smooth_traj.pkl")
        with open(smooth_traj_file, 'wb') as f:
            pickle.dump(fid2smooth, f)

        for fname in frame_files:
            fid = int(fname.split('.')[0])
            if fid % SAMPLE_INTERVAL != 0:
                continue
            obj = json_cache.get(fid)
            if obj is None:
                try:
                    obj = read_json_gz(join(scene_path, fname))
                except Exception:
                    continue

            frame_data = {}
            frame_data['scene_id'] = scene_id
            for k in ['speed','throttle','steer','brake']:
                if k in obj: frame_data[k] = obj[k]

            if 'command' in obj:
                frame_data['command'] = command_to_one_hot(obj['command'])
            if 'next_command' in obj:
                frame_data['next_command'] = command_to_one_hot(obj['next_command'])

            if 'target_point' in obj:
                frame_data['target_point'] = np.array(obj['target_point'])
            if 'ego_matrix' in obj:
                try:
                    m = np.array(obj['ego_matrix'])
                    if m.shape[0] >= 2 and m.shape[1] >= 4:
                        frame_data['ego_waypoint'] = m[:2, 3]
                except Exception:
                    pass
            if 'route' in obj:
                route = obj['route']
                if isinstance(route, list):
                    if len(route) < 20:
                        num_missing = 20 - len(route)
                        route = np.array(route)
                        route = np.vstack((route, np.tile(route[-1], (num_missing, 1))))
                    else:
                        route = np.array(route[:20])
                    frame_data['route'] = route

            # 图像路径（前视）
            frame_data['rgb'] = join(scene_id, CAMERA_SUBDIR, f"{fid:05d}.jpg")
            # 平滑后的当前帧自车世界系坐标和theta
            if fid in fid2smooth:
                frame_data['ego_world_smooth'] = [fid2smooth[fid]['x'], fid2smooth[fid]['y'], fid2smooth[fid]['z']]
                frame_data['theta_smooth'] = fid2smooth[fid]['theta']
            else:
                frame_data['ego_world_smooth'] = None
                frame_data['theta_smooth'] = None

            out_name = f"{scene_id}_{fid:05d}_{train_or_val}.pkl"
            out_path = join(out_dir, out_name)
            with open(out_path, 'wb') as f:
                pickle.dump(frame_data, f)

def generate_infos_b2d(scene_list, workers, train_or_val, tmp_dir):
    folder_num = len(scene_list)
    devide_list = [(folder_num//workers)*i for i in range(workers)]
    devide_list.append(folder_num)
    process_list = []
    for i in range(workers):
        sub_scenes = scene_list[devide_list[i]:devide_list[i+1]]
        # preprocess_b2d(sub_scenes, i, tmp_dir, train_or_val)
        p = multiprocessing.Process(
            target=preprocess_b2d,
            args=(sub_scenes, i, tmp_dir, train_or_val)
        )
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--tmp_dir', default="smoothed_data_b2d")
    args = parser.parse_args()

    all_scenes = list_scene_dirs(DATAROOT)

    print('processing Bench2Drive train data...')
    generate_infos_b2d(all_scenes, args.workers, 'train', args.tmp_dir)
    print('finish!')