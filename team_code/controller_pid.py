import numpy as np
from collections import deque


class PIDController(object):
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D

		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0

	def step(self, error):
		self._window.append(error)
		self._max = max(self._max, abs(error))
		self._min = -abs(self._max)

		if len(self._window) >= 2:
			integral = np.mean(self._window)
			derivative = (self._window[-1] - self._window[-2])
		else:
			integral = 0.0
			derivative = 0.0

		return self._K_P * error + self._K_I * integral + self._K_D * derivative


class WaypointPIDController:
	"""
	PID controller for waypoint-based vehicle control.
	Handles both lateral (steering) and longitudinal (speed) control.
	"""
	def __init__(self, config):
		"""
		Initialize PID controllers with configuration parameters.
		
		Args:
			config: Configuration dictionary containing controller parameters
		"""
		self.config = config
		control_cfg = config.get('controller', {})
		
		# Initialize PID controllers
		self.turn_controller = PIDController(
			K_P=control_cfg.get('turn_KP', 0.75), 
			K_I=control_cfg.get('turn_KI', 0.75), 
			K_D=control_cfg.get('turn_KD', 0.3), 
			n=control_cfg.get('turn_n', 40)
		)
		self.speed_controller = PIDController(
			K_P=control_cfg.get('speed_KP', 5.0),
			K_I=control_cfg.get('speed_KI', 0.5),
			K_D=control_cfg.get('speed_KD', 1.0),
			n=control_cfg.get('speed_n', 40)
		)
		
		# Read hyperparameters from config
		self.aim_dist = control_cfg.get('aim_dist', 4.0)
		self.angle_thresh = control_cfg.get('angle_thresh', 0.3)
		self.dist_thresh = control_cfg.get('dist_thresh', 10.0)
		self.brake_speed = control_cfg.get('brake_speed', 0.4)
		self.brake_ratio = control_cfg.get('brake_ratio', 1.1)
		self.clip_delta = control_cfg.get('clip_delta', 0.25)
		self.max_throttle = control_cfg.get('max_throttle', 0.75)
	
	def control_pid(self, waypoints, velocity, target):
		"""
		Predicts vehicle control with a PID controller.
		
		Args:
			waypoints (tensor): predicted waypoints from the model
			velocity (tensor): current vehicle speed
			target (tensor): target waypoint
			
		Returns:
			steer (float): steering angle
			throttle (float): throttle value
			brake (float): brake value
			metadata (dict): control metadata for debugging
		"""
		assert(waypoints.size(0) == 1)
		waypoints = waypoints[0].data.cpu().numpy()
		target = target.squeeze().data.cpu().numpy()

		# Swap x and y coordinates (coordinate system transformation)
		waypoints[:, [0, 1]] = waypoints[:, [1, 0]]  
		target[[0, 1]] = target[[1, 0]]

		# Iterate over vectors between predicted waypoints
		num_pairs = len(waypoints) - 1
		best_norm = 1e5
		desired_speed = 0
		aim = waypoints[0]
		for i in range(num_pairs):
			# Magnitude of vectors, used for speed
			desired_speed += np.linalg.norm(
				waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs    
			# Norm of vector midpoints, used for steering
			norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
			if abs(self.aim_dist - best_norm) > abs(self.aim_dist - norm):
				aim = waypoints[i]
				best_norm = norm

		aim_last = waypoints[-1] - waypoints[-2]

		angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
		angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
		angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

		# Choice of point to aim for steering, removing outlier predictions
		# Use target point if it has a smaller angle or if error is large
		# Predicted point otherwise
		# (reduces noise in eg. straight roads, helps with sudden turn commands)
		use_target_to_aim = np.abs(angle_target) < np.abs(angle)
		use_target_to_aim = use_target_to_aim or (np.abs(angle_target - angle_last) > self.angle_thresh and target[1] < self.dist_thresh)
		if use_target_to_aim:
			angle_final = angle_target
		else:
			angle_final = angle

		steer = self.turn_controller.step(angle_final)
		steer = np.clip(steer, -1.0, 1.0)

		speed = velocity[0].data.cpu().numpy()
		
		# Modified braking logic to help recovery from stops
		# Only brake if significantly overspeeding OR desired speed is extremely low
		should_brake = (desired_speed < self.brake_speed) or (speed / max(desired_speed, 0.5)) > self.brake_ratio
		
		# Calculate throttle based on speed difference
		delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
		throttle = self.speed_controller.step(delta)
		throttle = np.clip(throttle, 0.0, self.max_throttle)
		
		# Smarter braking logic - avoid premature braking during recovery
		if should_brake:
			# Only stop throttle if we're really overspeeding or need to stop
			if speed > desired_speed * 1.5:  # Significant overspeed
				throttle = 0.0
			elif speed < 2.0:  # Low speed - keep some throttle to help recovery
				throttle = max(throttle * 0.5, 0.1)  # Reduce but don't eliminate
			else:
				throttle = throttle * 0.3  # Reduce throttle moderately
			
			# Progressive braking
			if speed > desired_speed:
				overspeed = speed - desired_speed
				brake_intensity = min(1.0, overspeed / max(desired_speed * 0.7, 1.5))
				brake = float(np.clip(brake_intensity, 0.15, 1.0))
			else:
				brake = 0.3  # Light brake for very low desired speed
		else:
			brake = 0.0
			throttle = throttle  # Use PID output directly

		metadata = {
			'speed': float(speed.astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'wp_4': tuple(waypoints[3].astype(np.float64)),
			'wp_3': tuple(waypoints[2].astype(np.float64)),
			'wp_2': tuple(waypoints[1].astype(np.float64)),
			'wp_1': tuple(waypoints[0].astype(np.float64)),
			'aim': tuple(aim.astype(np.float64)),
			'target': tuple(target.astype(np.float64)),
			'desired_speed': float(desired_speed.astype(np.float64)),
			'angle': float(angle.astype(np.float64)),
			'angle_last': float(angle_last.astype(np.float64)),
			'angle_target': float(angle_target.astype(np.float64)),
			'angle_final': float(angle_final.astype(np.float64)),
			'delta': float(delta.astype(np.float64)),
		}

		return steer, throttle, brake, metadata
