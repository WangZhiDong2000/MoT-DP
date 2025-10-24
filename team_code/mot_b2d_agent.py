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
from collections import OrderedDict
import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T
import imageio
import laspy

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
from dataset.generate_lidar_bev_pdm import (
    generate_lidar_bev_images, 
    removePoints, 
    makeBVFeature,
    BOUNDARY
)
from dataset.generate_lidar_bev_b2d import generate_lidar_bev_images as generate_lidar_bev_images_b2d
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
        config_path = "/home/wang/Project/MoT-DP/config/carla.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_best_model(checkpoint_path, config, device):
    print(f"Loading best model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    action_stats = {
    'min': torch.tensor([-11.77335262298584, -59.26432800292969]),
    'max': torch.tensor([98.34003448486328, 55.585079193115234]),
    'mean': torch.tensor([10.975193977355957, 0.04004639387130737]),
    'std': torch.tensor([14.96833324432373, 3.419595956802368]),
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
		self.steer_step = 0
		self.last_moving_status = 0
		self.last_moving_step = -1
		self.last_steers = deque()
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
		checkpoint_base_path = self.config.get('logging', {}).get('checkpoint_dir', "/root/z_projects/code/MoT-DP/checkpoints/pdm_linearnorm_2obs_8pred")
		checkpoint_path = os.path.join(checkpoint_base_path, "carla_policy_best.pt")
		self.net = load_best_model(checkpoint_path, self.config, device)

		self.takeover = False
		self.stop_time = 0
		self.takeover_time = 0

		self.save_path = None
		self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])


		self.last_steers = deque()
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
		(self.save_path / 'lidar').mkdir()
		
		# Initialize lidar buffer for combining two frames
		self.lidar_buffer = deque(maxlen=2)
		self.lidar_step_counter = 0
		self.last_ego_transform = None
		self.last_lidar = None

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
					'x': 0.80, 'y': 0.0, 'z': 1.60,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 1600, 'height': 900, 'fov': 70,
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
		lidar_raw = input_data['LIDAR'][1]
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
		lidar_bev_img = generate_lidar_bev_images_b2d(
			np.copy(lidar_combined), 
			saving_name=None, 
			img_height=448, 
			img_width=448
		)
		# Convert BEV image to tensor format for interfuser_bev_encoder backbone
		# Input shape should be (C, H, W) where C=3 (RGB), H=448, W=448
		lidar_bev_tensor = torch.from_numpy(lidar_bev_img).permute(2, 0, 1).float() / 255.0
		
		#Process other sensors
		bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		speed = input_data['SPEED'][1]['speed']

		if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
			compass = 0.0

		result = {
				'rgb_front': rgb_front,
				'lidar_bev': lidar_bev_tensor,
				'lidar_combined': lidar_combined,
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
		lidar = tick_data['lidar_bev'].to('cuda', dtype=torch.float32)
		# Add batch dimension to lidar_bev: (C, H, W) -> (1, C, H, W)
		lidar = lidar.unsqueeze(0)
		tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
										torch.FloatTensor([tick_data['target_point'][1]])]
		target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)

		obs_horizon = self.config.get('obs_horizon', 2)
		obs_dict = {
                'lidar_bev': lidar.repeat(1, obs_horizon, 1, 1, 1),  # (B, obs_horizon, C, H, W) = (1, obs_horizon, 3, 448, 448)
                'speed': speed.repeat(1, obs_horizon, 1),  # (B, obs_horizon, 1) - 观测步的speed
                'target_point': target_point.repeat(1, obs_horizon, 1),  # (B, obs_horizon, 2) - 观测步的target_point
                'next_command': cmd_one_hot.repeat(1, obs_horizon, 1), 
            }
		result = self.net.predict_action(obs_dict)
        
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
		# self.pid_metadata['steer'] = control.steer
		# self.pid_metadata['throttle'] = control.throttle
		# self.pid_metadata['brake'] = control.brake
		

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

		if control.brake > 0:
			control.brake = 1.0
		if control.brake > 0.5:
			control.throttle = float(0)

		self.pid_metadata['steer'] = control.steer
		self.pid_metadata['throttle'] = control.throttle
		self.pid_metadata['brake'] = control.brake
		metric_info = self.get_metric_info()
		self.metric_info[self.step] = metric_info
		if SAVE_PATH is not None and self.step % 1 == 0:
			self.save(tick_data)
		return control

	def save(self, tick_data):
		frame = self.step

		Image.fromarray(tick_data['rgb_front']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))

		Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))


		# Save combined lidar data in LAZ format (compressed point cloud)
		if 'lidar_combined' in tick_data and len(tick_data['lidar_combined']) > 0:
			lidar_combined = tick_data['lidar_combined']
			# Use LAZ compression format similar to data_agent.py
			header = laspy.LasHeader(point_format=0)
			header.offsets = np.min(lidar_combined, axis=0)
			header.scales = np.array([0.01, 0.01, 0.01])

			laz_path = self.save_path / 'lidar' / (f'{frame:04d}.laz')
			with laspy.open(laz_path, mode='w', header=header) as writer:
				point_record = laspy.ScaleAwarePointRecord.zeros(lidar_combined.shape[0], header=header)
				point_record.x = lidar_combined[:, 0]
				point_record.y = lidar_combined[:, 1]
				point_record.z = lidar_combined[:, 2]

				writer.write_points(point_record)
			
			# Read LAZ file and generate BEV image using generate_lidar_bev_images
			try:
				# Use laspy.read() method similar to get_lidar_pts() in generate_lidar_bev_b2d
				las = laspy.read(str(laz_path))
				lidar_data = np.vstack((las.x, las.y, las.z)).transpose()
				
				# Generate BEV image from LAZ data using generate_lidar_bev_images function
				lidar_bev_img = generate_lidar_bev_images(
					lidar_data,
					saving_name=str(self.save_path / 'lidar_bev' / (f'{frame:04d}.png')),
					img_height=448,
					img_width=448
				)
			except Exception as e:
				print(f"Error reading LAZ file or generating BEV image: {e}")

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
	checkpoint_base_path = test_config.get('logging', {}).get('checkpoint_dir')
	checkpoint_path = os.path.join(checkpoint_base_path, "carla_policy_best.pt")
	policy = load_best_model(checkpoint_path, test_config, device)