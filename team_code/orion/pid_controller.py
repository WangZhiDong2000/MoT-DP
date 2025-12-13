from collections import deque
import numpy as np

class PID(object):
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



class PIDController(object):
    
    def __init__(self, turn_KP=1.45, turn_KI=0.5, turn_KD=0.4, turn_n=40, speed_KP=3.8, speed_KI=0.4,speed_KD=0.8, speed_n = 40,max_throttle=0.75, brake_speed=0.02,brake_ratio=1.15, clip_delta=0.20, aim_dist=3.2, angle_thresh=0.3, dist_thresh=10):
        
        self.turn_controller = PID(K_P=turn_KP, K_I=turn_KI, K_D=turn_KD, n=turn_n)
        self.speed_controller = PID(K_P=speed_KP, K_I=speed_KI, K_D=speed_KD, n=speed_n)
        self.max_throttle = max_throttle
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio
        self.clip_delta = clip_delta
        self.aim_dist = aim_dist
        self.angle_thresh = angle_thresh
        self.dist_thresh = dist_thresh
        
        # Speed-adaptive steering parameters
        self.low_speed_threshold = 5.0   # Below this speed, use enhanced steering gain
        self.high_speed_threshold = 12.0  # Above this speed, use minimum steering gain
        self.min_steer_scale = 0.55      # Minimum steering scale at high speed
        self.low_speed_boost = 1.15      # Boost factor for low-speed turns (especially for intersections)

    def control_pid(self, waypoints, speed, target):
        ''' Predicts vehicle control with a PID controller.
        Args:
            waypoints (array): numpy array of waypoints
            speed (float or array): current vehicle speed (can be scalar or array)
            target (array): target waypoint
        '''

        # iterate over vectors between predicted waypoints
        num_pairs = len(waypoints) - 1
        # num_pairs = 6 
        best_norm = 1e5
        desired_speed = 0
        aim = waypoints[0]
        
        # ============ Improved Speed Calculation ============
        # Use weights based on prediction L2 error: 1s: 0.37, 2s: 0.99, 3s: 1.88
        # Lower error -> higher weight (inverse error weighting)
        # Near-term segments are more reliable, far-term have more uncertainty
        
        # Calculate weighted speed based on all waypoint segments
        segment_speeds = []
        
        for i in range(num_pairs):
            # Speed from segment distance
            segment_dist = np.linalg.norm(waypoints[i+1] - waypoints[i])
            segment_speeds.append(segment_dist * 2.0)  # Convert to m/s (assume 0.5s per step)
        
        # Inverse error weights with moderate, distributed trust across waypoints
        # Assuming 6 waypoints at 0.5s intervals: 0.5s, 1s, 1.5s, 2s, 2.5s, 3s
        # L2 errors: 1s: 0.37, 2s: 0.99, 3s: 1.88 (from known data)
        # Interpolated: ~0.2, ~0.37, ~0.68, ~0.99, ~1.44, ~1.88
        l2_errors = [0.20, 0.37, 0.68, 0.99, 1.44, 1.88][:num_pairs]
        
        # Apply moderate, smoothly increasing penalty to distribute trust more evenly
        # Gradual decay: 1.0, 1.15, 1.25, 1.35, 1.45, 1.6
        trust_decay = [1.0, 1.15, 1.25, 1.35, 1.45, 1.6][:num_pairs]
        adjusted_errors = [err * decay for err, decay in zip(l2_errors, trust_decay)]
        
        # Inverse error weights
        weights = np.array([1.0 / err for err in adjusted_errors])
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        # Weighted average speed
        desired_speed = np.sum(np.array(segment_speeds) * weights)
        
        # ============ Steering Aim Point Selection ============
        for i in range(num_pairs):
            # norm of vector points, used for steering
            norm = np.linalg.norm((waypoints[i]))
            if abs(self.aim_dist-best_norm) > abs(self.aim_dist-norm):
                aim = waypoints[i]
                best_norm = norm
            # norm of vector midpoints, used for steering
            norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
            if abs(self.aim_dist-best_norm) > abs(self.aim_dist-norm):
                aim = (waypoints[i+1] + waypoints[i]) / 2.0
                best_norm = norm

        aim_last = waypoints[-1] - waypoints[-2]
        # Numerical stability at ultra-low speeds
        # max speed of 0.1 if aim = waypoints[0]
        if aim[1] <= 0.02: 
            angle = np.array(0.0)
        else: 
            angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
        angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

        # ============ Balanced Steering Control ============
        # Blend model prediction with target point to prevent:
        # 1. Oscillation during straight driving (model noise)
        # 2. Overshooting at end of turns (late correction)
        
        angle_diff = abs(angle - angle_target)
        
        # Check if we're in straight driving scenario (small steering angles)
        is_straight = abs(angle) < 0.08 and abs(angle_target) < 0.08
        
        # Determine weights based on agreement and situation
        if is_straight:
            # During straight driving, heavily favor target point to reduce oscillation
            model_weight = 0.55
            target_weight = 0.45
        elif angle_diff < 0.08:  # Good agreement - mostly trust model but add target stability
            model_weight = 0.75
            target_weight = 0.25
        elif angle_diff < 0.2:  # Moderate disagreement - balanced blend
            model_weight = 0.55
            target_weight = 0.45
        elif angle_diff < 0.35:  # Significant disagreement - lean towards target
            model_weight = 0.4
            target_weight = 0.6
        else:  # Large disagreement - mostly use target for safety
            model_weight = 0.25
            target_weight = 0.75
        
        # Additional check: if trajectory end direction differs from aim, reduce model trust
        # This helps with turn completion - when angle_last is returning towards straight
        if abs(angle_last - angle) > self.angle_thresh:
            model_weight *= 0.8
            target_weight = 1.0 - model_weight
        
        # Weighted blend
        angle_blended = model_weight * angle + target_weight * angle_target
        
        # Safety constraint: limit deviation from target when target is close
        if target[1] < self.dist_thresh and target[1] > 0.5:
            max_deviation = 0.25  # Maximum allowed deviation from target angle
            if abs(angle_blended - angle_target) > max_deviation:
                if angle_blended > angle_target:
                    angle_blended = angle_target + max_deviation
                else:
                    angle_blended = angle_target - max_deviation
        
        # Use blended angle
        angle_final = np.clip(angle_blended, -0.8, 0.8)

        steer = self.turn_controller.step(angle_final)
        steer = np.clip(steer, -1.0, 1.0)

        # Convert speed to float if it's a numpy array
        if isinstance(speed, np.ndarray):
            speed_scalar = float(speed.astype(np.float64))
        else:
            speed_scalar = float(speed)
        
        # ============ Speed-adaptive steering scaling ============
        # At low speeds (intersections), boost steering for tighter turns
        # At high speeds, reduce steering sensitivity to prevent oscillations
        if speed_scalar <= self.low_speed_threshold:
            # Apply boost at low speeds to handle tight intersection turns
            steer_scale = self.low_speed_boost
        elif speed_scalar >= self.high_speed_threshold:
            steer_scale = self.min_steer_scale
        else:
            # Linear interpolation between low_speed_boost and min_steer_scale
            t = (speed_scalar - self.low_speed_threshold) / (self.high_speed_threshold - self.low_speed_threshold)
            steer_scale = self.low_speed_boost - t * (self.low_speed_boost - self.min_steer_scale)
        
        steer = steer * steer_scale

        brake = desired_speed < self.brake_speed or (speed_scalar / desired_speed) > self.brake_ratio

        delta = np.clip(desired_speed - speed_scalar, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed_scalar),
            'steer': float(steer),
            'steer_scale': float(steer_scale),
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
            'angle_blended': float(angle_blended),
            'angle_final': float(angle_final),
            'model_weight': float(model_weight),
            'target_weight': float(target_weight),
            'is_straight': bool(is_straight),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata