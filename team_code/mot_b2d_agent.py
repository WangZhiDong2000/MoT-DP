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


class LiDARBEVConverter:
	"""
	Real-time LiDAR to BEV (Bird's Eye View) image converter.
	Converts raw CARLA LiDAR point clouds to BEV images for model input.
	Outputs normalized torch tensors ready for InterfuserBEVEncoder.
	"""
	def __init__(self, img_height=448, img_width=448, normalize=True, return_tensor=True):
		"""
		Initialize the LiDAR BEV converter.
		
		Args:
			img_height: Height of the output BEV image
			img_width: Width of the output BEV image
			normalize: Whether to normalize the output to [0, 1] range
			return_tensor: Whether to return PyTorch tensor (B, 3, H, W) or numpy array (H, W, 3)
		"""
		self.img_height = img_height
		self.img_width = img_width
		self.discretization = (BOUNDARY["maxX"] - BOUNDARY["minX"]) / img_height
		self.normalize = normalize
		self.return_tensor = return_tensor
	
	def process_lidar(self, lidar_pc, return_numpy=False):
		"""
		Convert raw LiDAR point cloud to BEV image ready for InterfuserBEVEncoder.
		
		Args:
			lidar_pc: Raw LiDAR point cloud from CARLA sensor (Nx3 or Nx4 array)
			return_numpy: If True, return numpy array instead of torch tensor
			
		Returns:
			bev_image: 
				- If return_tensor=True: PyTorch tensor (1, 3, H, W) normalized to [0, 1]
				- If return_tensor=False: numpy array (H, W, 3) with values in [0, 255]
		"""
		try:
			# Ensure input is numpy array
			if not isinstance(lidar_pc, np.ndarray):
				lidar_pc = np.array(lidar_pc)
			
			# Handle empty point cloud
			if lidar_pc.shape[0] == 0:
				# Return black image if no points
				empty_image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
				return self._format_output(empty_image, return_numpy)
			
			# Make a copy to avoid modifying original data
			lidar_pc = np.copy(lidar_pc)
			
			# Step 1: Coordinate transformation (flip x-axis for CARLA convention)
			lidar_pc[:, 0] = lidar_pc[:, 0] * -1
			
			# Step 2: Handle 3-channel to 4-channel (add intensity if not present)
			if lidar_pc.shape[-1] == 3:
				# If only XYZ, add a default intensity channel
				lidar_pc = np.concatenate([lidar_pc, np.ones((*lidar_pc.shape[:-1], 1))], axis=-1)
			
			# Step 3: Remove points outside boundary and center region
			lidar_pc = removePoints(lidar_pc, BOUNDARY)
			
			# Handle case when all points are filtered out
			if lidar_pc.shape[0] == 0:
				empty_image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
				return self._format_output(empty_image, return_numpy)
			
			# Step 4: Translate points to positive coordinates for BEV generation
			lidar_pc_filtered = np.copy(lidar_pc)
			lidar_pc_filtered[:, 0] = lidar_pc_filtered[:, 0] + BOUNDARY["maxX"]
			lidar_pc_filtered[:, 1] = lidar_pc_filtered[:, 1] + BOUNDARY["maxY"]
			
			# Step 5: Generate BEV image (uint8, [0, 255])
			bev_image = makeBVFeature(
				lidar_pc_filtered, 
				BOUNDARY, 
				self.img_height, 
				self.img_width, 
				self.discretization
			)
			
			# Step 6: Zero out the blue channel as per original implementation
			bev_image[:, :, 2] = 0.0
			
			return self._format_output(bev_image, return_numpy)
			
		except Exception as e:
			print(f"Error processing LiDAR: {e}", flush=True)
			# Return black image on error
			empty_image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
			return self._format_output(empty_image, return_numpy)
	
	def _format_output(self, bev_image_np, return_numpy):
		"""
		Format the output according to settings.
		
		Args:
			bev_image_np: numpy array (H, W, 3) with uint8 values [0, 255]
			return_numpy: Whether to return numpy or torch tensor
			
		Returns:
			Formatted output (numpy array or torch tensor)
		"""
		# Convert to float [0, 1] if normalize is True
		if self.normalize:
			bev_image_np = bev_image_np.astype(np.float32) / 255.0
		
		# Return numpy if requested
		if return_numpy or not self.return_tensor:
			return bev_image_np
		
		# Convert to torch tensor (B, C, H, W) format for model input
		# Input: (H, W, 3) -> (1, 3, H, W)
		bev_tensor = torch.from_numpy(bev_image_np).permute(2, 0, 1).unsqueeze(0).float()
		
		return bev_tensor


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

		# Initialize LiDAR BEV converter - returns normalized torch tensors (1, 3, 448, 448)
		# ready for direct input to InterfuserBEVEncoder
		self.lidar_bev_converter = LiDARBEVConverter(
			img_height=448, 
			img_width=448, 
			normalize=True,      # Normalize to [0, 1]
			return_tensor=True   # Return (1, 3, H, W) torch tensor
		)

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

			(self.save_path / 'rgb').mkdir()
			(self.save_path / 'rgb_front').mkdir()
			(self.save_path / 'meta').mkdir()
			(self.save_path / 'bev').mkdir()

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
				{
					'type': 'sensor.camera.rgb',
					'x': 0.27, 'y': -0.55, 'z': 1.60,
					'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
					'width': 1600, 'height': 900, 'fov': 70,
					'id': 'CAM_FRONT_LEFT'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': 0.27, 'y': 0.55, 'z': 1.60,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
					'width': 1600, 'height': 900, 'fov': 70,
					'id': 'CAM_FRONT_RIGHT'
					},
				# lidar
				{
          			'type': 'sensor.lidar.ray_cast',
          			'x': 0.0, 'y': 0.0, 'z': 2.5,
          			'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
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
		front_img = cv2.cvtColor(input_data['CAM_FRONT'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		front_left_img = cv2.cvtColor(input_data['CAM_FRONT_LEFT'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		front_right_img = cv2.cvtColor(input_data['CAM_FRONT_RIGHT'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_front =  cv2.cvtColor(input_data['CAM_FRONT'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]

		_, front_img = cv2.imencode('.jpg', front_img, encode_param)
		front_img = cv2.imdecode(front_img, cv2.IMREAD_COLOR)

		_, front_left_img = cv2.imencode('.jpg', front_left_img, encode_param)
		front_left_img = cv2.imdecode(front_left_img, cv2.IMREAD_COLOR)

		_, front_right_img = cv2.imencode('.jpg', front_right_img, encode_param)
		front_right_img = cv2.imdecode(front_right_img, cv2.IMREAD_COLOR)
		front_img = front_img[:, 200:1400, :]
		front_left_img = front_left_img[:, :1400, :]
		front_right_img = front_right_img[:, 200:, :]

		rgb = np.concatenate((front_left_img, front_img, front_right_img), axis=1)
		rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
		rgb = torch.nn.functional.interpolate(rgb, size=(256, 900), mode='bilinear', align_corners=False)
		rgb = rgb.squeeze(0).permute(1, 2, 0).byte().numpy()

		# Process lidar 
		lidar_raw = input_data['LIDAR'][1]
		lidar_bev_tensor = self.lidar_bev_converter.process_lidar(lidar_raw)
		
		#Process other sensors
		bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		gps = input_data['GPS'][1][:2]
		speed = input_data['SPEED'][1]['speed']
		compass = input_data['IMU'][1][-1]

		if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
			compass = 0.0

		result = {
				'rgb': rgb,
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

		return result
	
	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()
		tick_data = self.tick(input_data)
		if self.step < 1:
			rgb = self._im_transform(tick_data['rgb']).unsqueeze(0)

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
		speed = speed / 12
		rgb = self._im_transform(tick_data['rgb']).unsqueeze(0).to('cuda', dtype=torch.float32)
		lidar= tick_data['lidar_bev'].to('cuda', dtype=torch.float32)
		tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
										torch.FloatTensor([tick_data['target_point'][1]])]
		target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)

		obs_horizon = self.config.get('obs_horizon', 2)
		obs_dict = {
                'lidar_bev': lidar.repeat(1, obs_horizon, 1, 1, 1),  # (B, obs_horizon, 3, 448, 448)
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

		Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))

		Image.fromarray(tick_data['rgb_front']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))

		Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))

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


if __name__ == "__main__":
	test_config = create_carla_config()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	checkpoint_base_path = test_config.get('logging', {}).get('checkpoint_dir')
	checkpoint_path = os.path.join(checkpoint_base_path, "carla_policy_best.pt")
	policy = load_best_model(checkpoint_path, test_config, device)