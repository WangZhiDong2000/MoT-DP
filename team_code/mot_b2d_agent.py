import os
import sys
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
import imageio

# Add the project directories to the Python path
project_root = str(pathlib.Path(__file__).parent.parent.parent)
leaderboard_root = os.path.join(project_root, 'leaderboard')
scenario_runner_root = os.path.join(project_root, 'scenario_runner')
mot_dp_root = os.path.join(project_root, 'MoT-DP')
carla_api_root = os.path.join(project_root.replace('Bench2Drive', 'carla'), 'PythonAPI', 'carla')

for path in [project_root, leaderboard_root, scenario_runner_root, mot_dp_root, carla_api_root]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

from leaderboard.autoagents import autonomous_agent
from policy.diffusion_dit_carla_policy import DiffusionDiTCarlaPolicy
from team_code.planner import RoutePlanner
from dataset.generate_lidar_bev_b2d import generate_lidar_bev_images
from scipy.optimize import fsolve


SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)
PLANNER_TYPE = os.environ.get('PLANNER_TYPE', None)
print('*'*10)
print(PLANNER_TYPE)
print('*'*10)
EARTH_RADIUS_EQUA = 6378137.0


def get_entry_point():
	return 'MOTAgent'

def create_carla_config(config_path=None):
    if config_path is None:
        config_path = "/root/z_projects/code/MoT-DP-1/config/pdm_server.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_best_model(checkpoint_path, config, device):
    print(f"Loading best model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    action_stats = {
    'min': torch.tensor([-11.77335262298584, -59.26432800292969]),
    'max': torch.tensor([98.34003448486328, 55.585079193115234]),
    'mean': torch.tensor([9.755727767944336, 0.03559679538011551]),
    'std': torch.tensor([14.527670860290527, 3.224050521850586]),
    }

    policy = DiffusionDiTCarlaPolicy(config, action_stats=action_stats).to(device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Validation Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  - Training Loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    
    if 'val_metrics' in checkpoint:
        print(f"  - Validation Metrics:")
        for key, value in checkpoint['val_metrics'].items():
            if isinstance(value, (int, float)):
                print(f"    {key}: {value:.4f}")

    return policy



class MOTAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file):
		self.track = autonomous_agent.Track.SENSORS
		if IS_BENCH2DRIVE:
			self.save_name = path_to_conf_file.split('+')[-1]
			self.config_path = path_to_conf_file.split('+')[0]
		else:
			self.config_path = path_to_conf_file
			self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		self.config = create_carla_config()
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		checkpoint_base_path = self.config.get('training', {}).get('checkpoint_dir', "/root/z_projects/code/MoT-DP-1/checkpoints/carla_dit_best")
		checkpoint_path = os.path.join(checkpoint_base_path, "carla_policy_best.pt")
		self.net = load_best_model(checkpoint_path, self.config, device)

		self.takeover = False
		self.stop_time = 0
		self.takeover_time = 0

		self.save_path = None
		self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

		# self.lat_ref, self.lon_ref = 42.0, 2.0
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			# string = pathlib.Path(os.environ['ROUTES']).stem + '_'
			# string += self.save_name
			string = self.save_name
			print (string)

		self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
		self.save_path.mkdir(parents=True, exist_ok=False)

		(self.save_path / 'rgb_front').mkdir()
		(self.save_path / 'meta').mkdir()
		(self.save_path / 'bev').mkdir()
		(self.save_path / 'lidar_bev').mkdir()
		
		# Initialize lidar buffer for combining two frames
		self.lidar_buffer = deque(maxlen=2)
		self.lidar_step_counter = 0
		self.last_ego_transform = None
		self.last_lidar = None
		
		# Initialize observation history buffers for accumulating historical observations
		obs_horizon = self.config.get('obs_horizon', 4)   
		self.obs_horizon = obs_horizon
		self.lidar_bev_history = deque(maxlen=obs_horizon*10) # tick frequency 20hz -> 2hz
		self.rgb_history = deque(maxlen=obs_horizon*10)
		self.speed_history = deque(maxlen=obs_horizon*10)
		self.theta_history = deque(maxlen=obs_horizon*10)
		self.throttle_history = deque(maxlen=obs_horizon*10)
		self.next_command_history = deque(maxlen=obs_horizon*10)
		self.target_point_history = deque(maxlen=obs_horizon*10)
		self.waypoint_history = deque(maxlen=obs_horizon*10)
		self.last_throttle = deque(maxlen=obs_horizon*10) 
		self.last_brake = deque(maxlen=obs_horizon*10) 

		
		# Frame skip counter for 2Hz obs_dict construction (20Hz tick -> 2Hz obs_dict)
		self.obs_accumulate_counter = 0

	def _init(self):
		try:
			locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
			lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
			E = EARTH_RADIUS_EQUA
			def equations(vars):
				x, y = vars
				eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * E) - math.cos(x * math.pi / 180) * y
				eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * E * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * E * math.log(math.tan((90 + x) * math.pi / 360))
				return [eq1, eq2]
			initial_guess = [0, 0]
			solution = fsolve(equations, initial_guess)
			self.lat_ref, self.lon_ref = solution[0], solution[1]
		except Exception as e:
			print(e, flush=True)
			self.lat_ref, self.lon_ref = 0, 0
		print(self.lat_ref, self.lon_ref, self.save_name)
		#
		self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
		self._route_planner.set_route(self._global_plan, True)
		self.initialized = True
		self.metric_info = {}

	def sensors(self):
		sensors =  [
				{
					'type': 'sensor.camera.rgb',
					'x': -1.50, 'y': 0.0, 'z': 2.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 1024, 'height': 512, 'fov': 110,
					'id': 'CAM_FRONT'
					},
				# lidar
				{
          			'type': 'sensor.lidar.ray_cast',
          			'x': 0.0, 'y': 0.0, 'z': 2.5,
          			'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
          			'id': 'LIDAR'
      				},
				# imu
				{
					'type': 'sensor.other.imu',
					'x': -1.4, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'IMU'
					},
				# gps
				{
					'type': 'sensor.other.gnss',
					'x': -1.4, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'GPS'
					},
				# speed
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'SPEED'
					},
				]
		
		if IS_BENCH2DRIVE:
			sensors += [
					{	
						'type': 'sensor.camera.rgb',
						'x': 0.0, 'y': 0.0, 'z': 50.0,
						'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
						'width': 512, 'height': 512, 'fov': 5 * 10.0,
						'id': 'bev'
					}]
		return sensors

	def tick(self, input_data):
		self.step += 1
		# Process camera
		rgb_front =  cv2.cvtColor(input_data['CAM_FRONT'][1][:, :, :3], cv2.COLOR_BGR2RGB)

		# Process lidar - convert to ego coordinate and buffer two frames to get complete point cloud
		lidar_ego = lidar_to_ego_coordinate(input_data['LIDAR'])
		
		# Get current pose information
		gps = input_data['GPS'][1][:2]
		compass = input_data['IMU'][1][-1]
		
		# Combine two frames of lidar data using algin_lidar
		if self.last_lidar is not None and self.last_ego_transform is not None:
			# Calculate relative transformation between current and last frame
			current_pos = np.array([gps[0], gps[1], 0.0])
			last_pos = np.array([self.last_ego_transform['gps'][0], self.last_ego_transform['gps'][1], 0.0])
			relative_translation = current_pos - last_pos
			
			# Calculate relative rotation
			current_yaw = compass
			last_yaw = self.last_ego_transform['compass']
			relative_rotation = current_yaw - last_yaw
			
			# Rotate difference vector from global to local coordinate system
			orientation_target = np.deg2rad(current_yaw)
			rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
										[np.sin(orientation_target), np.cos(orientation_target), 0.0], 
										[0.0, 0.0, 1.0]])
			relative_translation_local = rotation_matrix.T @ relative_translation
			
			# Align the last lidar to current coordinate system
			lidar_last = algin_lidar(self.last_lidar, relative_translation_local, relative_rotation)
			# Combine lidar frames
			lidar_combined = np.concatenate((lidar_ego, lidar_last), axis=0)
		else:
			lidar_combined = lidar_ego
		
		# Store current frame for next iteration
		self.last_lidar = lidar_ego
		self.last_ego_transform = {'gps': gps, 'compass': compass}
		
		# Generate lidar BEV image from combined lidar data
		lidar_bev_img = generate_lidar_bev_images(
			np.copy(lidar_combined), 
			saving_name=None, 
			img_height=448, 
			img_width=448
		)
		# Convert BEV image to tensor format for interfuser_bev_encoder backbone
		lidar_bev_tensor = torch.from_numpy(lidar_bev_img).permute(2, 0, 1).float() / 255.0
		
		#Process other sensors
		bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		speed = input_data['SPEED'][1]['speed']

		if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
			compass = 0.0

		result = {
				'rgb_front': rgb_front,
				'lidar_bev': lidar_bev_tensor,
				'gps': gps,
				'speed': speed,
				'compass': compass,
				'bev': bev
				}
		pos = self.gps_to_location(result['gps'])
		result['gps'] = pos
		next_wp, next_cmd = self._route_planner.run_step(pos)
		result['next_command'] = next_cmd.value
		theta = compass - np.pi/2
		R = np.array([
			[np.cos(theta), np.sin(theta)],
			[-np.sin(theta),  np.cos(theta)]
			])

		local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
		local_command_point = R.dot(local_command_point)
		result['target_point'] = tuple(local_command_point)
		result['theta']=theta

		return result
	
	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()
		tick_data = self.tick(input_data)
		if self.step < 1:
			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0
			return control

		gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
		command = tick_data['next_command']
		if command < 0:
			command = 4
		command -= 1
		assert command in [0, 1, 2, 3, 4, 5]
		cmd_one_hot = [0] * 6
		cmd_one_hot[command] = 1
		cmd_one_hot = torch.tensor(cmd_one_hot).view(1, 6).to('cuda', dtype=torch.float32)
		speed = torch.FloatTensor([float(tick_data['speed'])]).view(1,1).to('cuda', dtype=torch.float32)
		theta = torch.FloatTensor([float(tick_data['theta'])]).view(1,1).to('cuda', dtype=torch.float32)
		lidar = tick_data['lidar_bev'].to('cuda', dtype=torch.float32)
		rgb_front = tick_data['rgb_front'].to('cuda', dtype=torch.float32)
		waypoint = tick_data['gps'].to('cuda', dtype=torch.float32)		
		tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
										torch.FloatTensor([tick_data['target_point'][1]])]
		target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)


		# Accumulate observation history into buffers 
		self.lidar_bev_history.append(lidar)
		self.rgb_history.append(rgb_front)
		self.speed_history.append(speed)
		self.target_point_history.append(target_point)
		self.next_command_history.append(cmd_one_hot)
		self.theta_history.append(theta)
		self.waypoint_history.append(waypoint)
		
		# Build obs_dict from historical observations
		lidar_history_list = list(self.lidar_bev_history)
		speed_history_list = list(self.speed_history)
		target_point_history_list = list(self.target_point_history)
		cmd_history_list = list(self.next_command_history)
		theta_history_list = list(self.theta_history)
		throttle_history_list = list(self.last_throttle)
		brake_history_list = list(self.last_brake)
		rgb_history_list = list(self.rgb_history)
		waypoint_history_list = list(self.waypoint_history)
		
		# Sample every 10th element: indices 9, 19, 29, ..., (obs_horizon*10-1)
		lidar_list = [lidar_history_list[i*10 - 1] for i in range(1, self.obs_horizon + 1) if i*10 - 1 < len(lidar_history_list)]
		speed_list = [speed_history_list[i*10 - 1] for i in range(1, self.obs_horizon + 1) if i*10 - 1 < len(speed_history_list)]
		target_point_list = [target_point_history_list[i*10 - 1] for i in range(1, self.obs_horizon + 1) if i*10 - 1 < len(target_point_history_list)]
		cmd_list = [cmd_history_list[i*10 - 1] for i in range(1, self.obs_horizon + 1) if i*10 - 1 < len(cmd_history_list)]
		theta_list = [theta_history_list[i*10 - 1] for i in range(1, self.obs_horizon + 1) if i*10 - 1 < len(theta_history_list)]
		throttle_list = [throttle_history_list[i*10 - 1] for i in range(1, self.obs_horizon + 1) if i*10 - 1 < len(throttle_history_list)]
		brake_list = [brake_history_list[i*10 - 1] for i in range(1, self.obs_horizon + 1) if i*10 - 1 < len(brake_history_list)]
		waypoint_list = [waypoint_history_list[i*10 - 1] for i in range(1, self.obs_horizon + 1) if i*10 - 1 < len(waypoint_history_list)]
		# sample every 5 for rgb
		rgb_list = [rgb_history_list[-1 - i*5] for i in range(5) if -1 - i*5 >= -len(rgb_history_list)]
		rgb_list = rgb_list[::-1]  # Reverse to get chronological order (oldest to newest)
		
		# If sampled list is empty or not full, pad with the current observation
		if len(lidar_list) == 0:
			# First few frames - initialize buffer with current observation
			lidar_list = [lidar] * self.obs_horizon
			speed_list = [speed] * self.obs_horizon
			target_point_list = [target_point] * self.obs_horizon
			cmd_list = [cmd_one_hot] * self.obs_horizon
			rgb_list = [rgb_front] * 5
			theta_list = [theta] * self.obs_horizon
			throttle_list = [torch.tensor(0.0).view(1, 1).to('cuda')] * self.obs_horizon
			brake_list = [torch.tensor(0.0).view(1, 1).to('cuda')] * self.obs_horizon
			waypoint_list = [waypoint] * self.obs_horizon
		elif len(lidar_list) < self.obs_horizon:
			# Buffer not yet full - pad with the oldest observation
			pad_size = self.obs_horizon - len(lidar_list)
			lidar_list = [lidar_list[0]] * pad_size + lidar_list
			speed_list = [speed_list[0]] * pad_size + speed_list
			target_point_list = [target_point_list[0]] * pad_size + target_point_list
			cmd_list = [cmd_list[0]] * pad_size + cmd_list
			theta_list = [theta_list[0]] * pad_size + theta_list if len(theta_list) > 0 else [theta] * self.obs_horizon
			throttle_list = [throttle_list[0]] * pad_size + throttle_list if len(throttle_list) > 0 else [torch.tensor(0.0).view(1, 1).to('cuda')] * self.obs_horizon
			brake_list = [brake_list[0]] * pad_size + brake_list if len(brake_list) > 0 else [torch.tensor(0.0).view(1, 1).to('cuda')] * self.obs_horizon
			rgb_list = [rgb_front] * 5 if len(rgb_list) == 0 else rgb_list
			waypoint_list = [waypoint_list[0]] * pad_size + waypoint_list if len(waypoint_list) > 0 else [waypoint] * self.obs_horizon
		
		# Stack along time dimension
		lidar_stacked = torch.cat(lidar_list, dim=0).unsqueeze(0)  # (1, obs_horizon, C, H, W)
		speed_stacked = torch.cat(speed_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 1)
		theta_stacked = torch.cat(theta_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 1)
		throttle_stacked = torch.cat(throttle_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 1)
		brake_stacked = torch.cat(brake_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 1)
		cmd_stacked = torch.cat(cmd_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 6)
		target_point_stacked = torch.cat(target_point_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 2)
		waypoint_stacked = torch.cat(waypoint_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 2)
		rgb_stacked = torch.cat(rgb_list, dim=0).unsqueeze(0)  # (1, 5, C, H, W)
		
		# Transform historical waypoints and target_points to be relative to current position
		# Current frame info for transformation
		current_pos = tick_data['gps']  # numpy array [x, y]
		current_theta = tick_data['theta']  # theta = compass - np.pi/2
		current_R = np.array([
			[np.cos(current_theta), np.sin(current_theta)],
			[-np.sin(current_theta), np.cos(current_theta)]
		])
		
		# Transform each historical waypoint to current frame
		waypoint_relative_list = []
		for i in range(self.obs_horizon):
			past_waypoint = waypoint_stacked[0, i].cpu().numpy()  # [x, y] in global coordinates
			# Transform: current_R.T @ (past - current)
			relative_waypoint = current_R.T @ (past_waypoint - current_pos)
			waypoint_relative_list.append(torch.from_numpy(relative_waypoint).float().to('cuda'))
		
		# Stack the transformed waypoints
		waypoint_relative_stacked = torch.stack(waypoint_relative_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 2)
		
		# Transform each historical target_point to current frame
		# target_point_list contains points that are relative to their respective historical frames
		# We need to transform them to be relative to the current frame
		target_point_relative_list = []
		theta_history_list_np = [theta_list[i].cpu().numpy()[0, 0] for i in range(self.obs_horizon)]
		waypoint_history_list_np = [waypoint_stacked[0, i].cpu().numpy() for i in range(self.obs_horizon)]
		
		for i in range(self.obs_horizon):
			# Get target point in past frame's ego coordinate
			target_in_past_frame = target_point_stacked[0, i].cpu().numpy()  # [x, y] relative to past frame
			
			# Get past frame's position and rotation
			past_pos = waypoint_history_list_np[i]  # [x, y] in global coordinates
			past_theta = theta_history_list_np[i]
			past_R = np.array([
				[np.cos(past_theta), np.sin(past_theta)],
				[-np.sin(past_theta), np.cos(past_theta)]
			])
			
			# Transform target point from past frame's ego coordinate to world coordinate
			# target_world = past_R @ target_in_past_frame + past_pos
			target_world = past_R @ target_in_past_frame + past_pos
			
			# Transform from world coordinate to current frame's ego coordinate
			# target_in_current_frame = current_R.T @ (target_world - current_pos)
			target_in_current_frame = current_R.T @ (target_world - current_pos)
			
			target_point_relative_list.append(torch.from_numpy(target_in_current_frame).float().to('cuda'))
		
		# Stack the transformed target points
		target_point_relative_stacked = torch.stack(target_point_relative_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 2)
		
		# Concatenate all ego status features: speed + theta + throttle + brake + cmd + target_point_relative + waypoint_relative
		# ego_status_stacked: (1, obs_horizon, 1+1+1+1+6+2+2) = (1, obs_horizon, 14)
		ego_status_stacked = torch.cat([
			speed_stacked,                  # (1, obs_horizon, 1)
			theta_stacked,                  # (1, obs_horizon, 1)
			throttle_stacked,               # (1, obs_horizon, 1)
			brake_stacked,                  # (1, obs_horizon, 1)
			cmd_stacked,                    # (1, obs_horizon, 6)
			target_point_relative_stacked,  # (1, obs_horizon, 2) - target points in current frame
			waypoint_relative_stacked       # (1, obs_horizon, 2) - ego positions in current frame
		], dim=-1)  # Concatenate along feature dimension
		
		

		mot_obs_dict = {
				'rgb_hist': rgb_stacked,  # (1, 5, C, H, W)
				'lidar_bev': lidar_list[-1].unsqueeze(0),  # (1, C, H, W)
			}
		dp_obs_dict = {
                'lidar_bev': lidar_stacked,  # (B, obs_horizon, C, H, W) = (1, obs_horizon, 3, 448, 448)
                'ego_status': ego_status_stacked,  # (B, obs_horizon, 14) = (1, obs_horizon, 1+1+1+1+6+2+2)
            }
		result = self.net.predict_action(dp_obs_dict)
        
		pred = torch.from_numpy(result['action'])
		print(pred)
		steer_traj, throttle_traj, brake_traj, metadata_traj = self.net.control_pid(pred, gt_velocity, target_point)
		# if brake_traj < 0.05: brake_traj = 0.0
		# if throttle_traj > brake_traj: brake_traj = 0.0

		control = carla.VehicleControl()

		self.pid_metadata = metadata_traj
		self.pid_metadata['agent'] = 'only_traj'
		control.steer = np.clip(float(steer_traj), -1, 1)
		control.throttle = np.clip(float(throttle_traj), 0, 0.75)
		control.brake = np.clip(float(brake_traj), 0, 1)
		
		self.pid_metadata['steer_traj'] = float(steer_traj)
		self.pid_metadata['throttle_traj'] = float(throttle_traj)
		self.pid_metadata['brake_traj'] = float(brake_traj)

		if abs(control.steer) > 0.07: ## In turning
			speed_threshold = 1.0 ## Avoid stuck during turning
		else:
			speed_threshold = 1.5 ## Avoid pass stop/red light/collision
		if float(tick_data['speed']) > speed_threshold:
			max_throttle = 0.05
		else:
			max_throttle = 0.5
		control.throttle = np.clip(control.throttle, a_min=0.0, a_max=max_throttle)


		self.pid_metadata['steer'] = control.steer
		self.pid_metadata['throttle'] = control.throttle
		self.pid_metadata['brake'] = control.brake
		metric_info = self.get_metric_info()
		self.metric_info[self.step] = metric_info
		self.last_brake.append(control.brake)
		self.last_throttle.append(control.throttle)
		if SAVE_PATH is not None and self.step % 1 == 0:
			self.save(tick_data)
		return control

	def save(self, tick_data):
		frame = self.step
		Image.fromarray(tick_data['rgb_front']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
		Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))
		
		if 'lidar_bev' in tick_data:
			lidar_bev_tensor = tick_data['lidar_bev']
			if isinstance(lidar_bev_tensor, torch.Tensor):
				lidar_bev_tensor = lidar_bev_tensor.cpu().numpy()
			lidar_bev_img = (lidar_bev_tensor.transpose(1, 2, 0) * 255).astype(np.uint8)
			imageio.imwrite(str(self.save_path / 'lidar_bev' / (f'{frame:04d}.png')), lidar_bev_img)

		outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
		json.dump(self.pid_metadata, outfile, indent=4)
		outfile.close()

		# metric info
		outfile = open(self.save_path / 'metric_info.json', 'w')
		json.dump(self.metric_info, outfile, indent=4)
		outfile.close()

	def destroy(self):
		del self.net
		torch.cuda.empty_cache()

	def gps_to_location(self, gps):
		# gps content: numpy array: [lat, lon, alt]
		lat, lon = gps
		scale = math.cos(self.lat_ref * math.pi / 180.0)
		my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
		mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
		y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
		x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
		return np.array([x, y])


def lidar_to_ego_coordinate(lidar):
	"""
	Converts the LiDAR points given by the simulator into the ego agents
	coordinate system
	:param lidar: the LiDAR point cloud as provided in the input of run_step
	:return: lidar where the points are w.r.t. 0/0/0 of the car and the carla
	coordinate system.
	"""
	yaw = np.deg2rad(-90.0)
	rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])

	translation = np.array([0.0, 0.0, 2.5])

	# The double transpose is a trick to compute all the points together.
	ego_lidar = (rotation_matrix @ lidar[1][:, :3].T).T + translation

	return ego_lidar


def algin_lidar(lidar, translation, yaw):
	"""
	Translates and rotates a LiDAR into a new coordinate system.
	Rotation is inverse to translation and yaw
	:param lidar: numpy LiDAR point cloud (N,3)
	:param translation: translations in meters
	:param yaw: yaw angle in radians
	:return: numpy LiDAR point cloud in the new coordinate system.
	"""
	rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])

	aligned_lidar = (rotation_matrix.T @ (lidar - translation).T).T

	return aligned_lidar


if __name__ == "__main__":
	test_config = create_carla_config()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	checkpoint_base_path = test_config.get('training', {}).get('checkpoint_dir')
	checkpoint_path = os.path.join(checkpoint_base_path, "carla_policy_best.pt")
	policy = load_best_model(checkpoint_path, test_config, device)