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
import random

# Add the project directories to the Python path
project_root = str(pathlib.Path(__file__).parent.parent.parent)
leaderboard_root = str(os.path.join(project_root, 'leaderboard'))
scenario_runner_root = str(os.path.join(project_root, 'scenario_runner'))
mot_dp_root = str(os.path.join(project_root, 'MoT-DP'))
carla_api_root = str(os.path.join(project_root.replace('Bench2Drive', 'carla'), 'PythonAPI', 'carla'))

for path in [project_root, leaderboard_root, scenario_runner_root, mot_dp_root, carla_api_root]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

# Clean sys.path to ensure all entries are strings (fix for torch.compile issue)
sys.path = [str(p) for p in sys.path]

from leaderboard.autoagents import autonomous_agent
from policy.diffusion_dit_carla_policy import DiffusionDiTCarlaPolicy
from team_code.planner import RoutePlanner
from team_code.controller_pid import WaypointPIDController
from dataset.generate_lidar_bev_b2d import generate_lidar_bev_images
from scipy.optimize import fsolve
# mot dependencies
project_root = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
# Add MoT-DP paths
mot_dp_path = str(os.path.join(os.path.dirname(os.path.dirname(project_root)), 'MoT-DP'))
mot_path = str(os.path.join(mot_dp_path, 'mot'))
sys.path.append(mot_dp_path)
sys.path.append(mot_path)

# Clean sys.path to ensure all entries are strings (fix for torch.compile issue)
sys.path = [str(p) for p in sys.path]

from transformers import HfArgumentParser
import json
from dataclasses import dataclass, field
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from PIL import Image
from safetensors.torch import load_file
import glob
from data.reasoning.data_utils import add_special_tokens
from mot.modeling.automotive import (
    AutoMoTConfig, AutoMoT,
    Qwen3VLTextConfig, Qwen3VLTextModel, Qwen3VLForConditionalGenerationMoT
)
from dataset.unified_carla_dataset import CARLAImageDataset
from policy.diffusion_dit_carla_policy import DiffusionDiTCarlaPolicy
from mot.evaluation.inference import InterleaveInferencer
from transformers import AutoTokenizer


SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)
PLANNER_TYPE = os.environ.get('PLANNER_TYPE', None)
print('*'*10)
print(PLANNER_TYPE)
print('*'*10)
EARTH_RADIUS_EQUA = 6378137.0

# dp utils
def get_entry_point():
	return 'MOTAgent'

def create_carla_config(config_path=None):
    if config_path is None:
        config_path = "/home/wang/Project/MoT-DP/config/pdm_local.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_best_model(checkpoint_path, config, device):
    print(f"Loading best model from: {checkpoint_path}")
    
    # Fix for numpy compatibility issue
    import sys
    import numpy as np
    if not hasattr(np, '_core'):
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
    
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

# mot utils
@dataclass
class ModelArguments:
    model_path: str = field(
        default="/home/wang/Project/MoT-DP/checkpoints/mot",
        metadata={"help": "Path to the converted AutoMoT model checkpoint"}
    )
    qwen3vl_path: str = field(
        default="/home/wang/Project/MoT-DP/config/mot_config",
        metadata={"help": "Path to the Qwen3VL base model for config loading"}
    )
    max_latent_size: int = field(
        default=64,
        metadata={"help": "Maximum size of latent representations"}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Patch size for latent space processing"}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of patches per side for vision transformer"}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function for connector layers"}
    )
    mot_num_attention_heads: int = field(
        default=16,
        metadata={"help": "Number of attention heads for MoT attention components. Defaults to half of regular attention heads if not specified."}
    )
    mot_num_key_value_heads: int = field(
        default=4,
        metadata={"help": "Number of key-value heads for MoT attention components. Defaults to half of regular KV heads if not specified."}
    )
    mot_intermediate_size: int = field(
        default=4864,
        metadata={"help": "Intermediate size for MoT MLP components. Defaults to same as regular intermediate_size if not specified."}
    )
    reasoning_query_dim: int = field(
        default=588,
        metadata={"help": "Dimension of the reasoning query embedding."}
    )
    reasoning_query_max_num_tokens: int = field(
        default=8, #256, #64,
        metadata={"help": "Maximum number of tokens in the reasoning query."}
    )
    action_query_dim: int = field(
        default=588,
        metadata={"help": "Dimension of the action query embedding."}
    )
    action_query_tokens: int = field(
        default=1,
        metadata={"help": "Number of tokens in the action query."}
    )

@dataclass
class InferenceArguments:
    dataset_jsonl: str = field(
        default="b2d_data_val.jsonl",
        metadata={"help": "Path to the input dataset JSONL file"}
    )
    output_jsonl: str = field(
        default="",
        metadata={"help": "Path to the output JSONL file. If empty, will use input filename with .pred.jsonl suffix"}
    )
    base_path: str = field(
        default="/share-data/pdm_lite",
        metadata={"help": "Base path for resolving relative image paths. If empty, use current working directory"}
    )
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Enable visual generation capabilities"}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Enable visual understanding capabilities"}
    )
    max_num_tokens: int = field(
        default=16384,
        metadata={"help": "Maximum number of tokens for inference"}
    )
    start_idx: int = field(
        default=0,
        metadata={"help": "Starting index for processing dataset samples"}
    )
    max_samples: int = field(
        default=-1,
        metadata={"help": "Maximum number of samples to process. -1 means process all samples"}
    )

def load_safetensors_weights(model_path):
    """Load weights from single or multiple safetensors files."""
    # Try single file first (like AutoMoT 2B)
    single_file = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single_file):
        print(f"Loading from single file: {single_file}")
        return load_file(single_file)
    
    # Try multiple files (like Qwen3VL-4B)
    pattern = os.path.join(model_path, "model-*.safetensors")
    safetensor_files = sorted(glob.glob(pattern))
    
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")
    
    print(f"Loading from multiple files: {safetensor_files}")
    combined_state_dict = {}
    
    for file_path in safetensor_files:
        file_state_dict = load_file(file_path)
        combined_state_dict.update(file_state_dict)
        print(f"Loaded {len(file_state_dict)} parameters from {os.path.basename(file_path)}")
    
    print(f"Total loaded parameters: {len(combined_state_dict)}")
    return combined_state_dict

def convert_model_dtype_with_exceptions(model, target_dtype, exclude_buffer_patterns=None):
    if exclude_buffer_patterns is None:
        exclude_buffer_patterns = []
    
    for name, param in model.named_parameters():
        param.data = param.data.to(target_dtype)

    for name, buffer in model.named_buffers():
        should_exclude = any(pattern in name for pattern in exclude_buffer_patterns)
        
        if should_exclude:
            print(f"⊗ Skipped buffer: {name} (kept as {buffer.dtype})")
        else:
            buffer.data = buffer.data.to(target_dtype)
            print(f"✓ Converted buffer: {name} to {target_dtype}")   
    return model

def load_model_mot(device):

    # Parse arguments - use empty args to avoid conflicts with leaderboard args
    parser = HfArgumentParser((ModelArguments, InferenceArguments))
    model_args, inference_args = parser.parse_args_into_dataclasses(args=[])

    assert torch.cuda.is_available(), "CUDA is required"

    # Set random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load configs from Qwen3VL model
    qwen3vl_config_path = model_args.qwen3vl_path
    
    # Load the unified config and extract text_config and vision_config
    with open(f"{qwen3vl_config_path}/config.json", "r") as f:
        full_config = json.load(f)
    
    # Extract and create LLM config using Qwen3VL LLM with mRoPE support
    text_config_dict = full_config["text_config"]
    llm_config = Qwen3VLTextConfig(**text_config_dict)
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = True #False
    llm_config.layer_module = "Qwen3VLMoTDecoderLayer"  # Disable MoT for debugging
    # llm_config.layer_module = "Qwen3VLMoTDecoderLayer"  
    llm_config.mot_num_attention_heads = model_args.mot_num_attention_heads
    llm_config.mot_num_key_value_heads = model_args.mot_num_key_value_heads
    llm_config.mot_intermediate_size = model_args.mot_intermediate_size

    # Extract and create Vision config
    vision_config_dict = full_config["vision_config"]
    vit_config = Qwen3VLVisionConfig(**vision_config_dict)

    config = AutoMoTConfig(
        visual_gen=inference_args.visual_gen,
        visual_und=inference_args.visual_und,
        llm_config=llm_config,
        vision_config=vit_config,  # Changed from vit_config to vision_config
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        connector_act=model_args.connector_act,
        interpolate_pos=False,
        reasoning_query_dim=model_args.reasoning_query_dim,
        reasoning_query_tokens=model_args.reasoning_query_max_num_tokens,
        action_query_dim=model_args.action_query_dim,
        action_query_tokens=model_args.action_query_tokens,
    )

    # Initialize model with Qwen3VL LLM (supports mRoPE)
    language_model = Qwen3VLForConditionalGenerationMoT(llm_config)
    vit_model = Qwen3VLVisionModel(vit_config)
        
    model = AutoMoT(language_model, vit_model, config)

    device_map = {"": "cuda:0"}

    # Load converted AutoMoT checkpoint manually (accelerate has weight issues)
    print(f"Loading converted AutoMoT checkpoint from {model_args.model_path}...")
    
    state_dict = load_safetensors_weights(model_args.model_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Filter out lm_head.weight from missing keys if tie_word_embeddings=True
    actual_missing_keys = [k for k in missing_keys if k != 'language_model.lm_head.weight']
    print(f"Loaded weights: {len(actual_missing_keys)} missing, {len(unexpected_keys)} unexpected")
    
    if actual_missing_keys:
        print(f"Missing keys: {actual_missing_keys[:10]}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys[:5]}")
        
    # Move to device and siet dtype
    model = model.to(device_map[""]).eval()
    model = convert_model_dtype_with_exceptions(
        model,
        torch.bfloat16,
        exclude_buffer_patterns=['inv_freq']
    )
    
    # Verify tie_word_embeddings is working correctly
    embed_weight = model.language_model.model.embed_tokens.weight
    lm_head_weight = model.language_model.lm_head.weight
    weights_tied = embed_weight is lm_head_weight
    weights_equal = torch.equal(embed_weight, lm_head_weight)
    embed_norm = torch.norm(embed_weight).item()
    
    if not weights_tied or not weights_equal:
        print("WARNING: tie_word_embeddings may not be working correctly!")
    elif embed_norm < 10:
        print("WARNING: embed_tokens weights appear to be randomly initialized!")
    else:
        print("✓ tie_word_embeddings is working correctly")
    
    # Explicitly move any remaining CPU tensors to GPU
    print("Ensuring all parameters are on CUDA...")
    for name, param in model.named_parameters():
        if param.device.type == 'cpu':
            print(f"Moving {name} from CPU to CUDA")
            param.data = param.data.to("cuda:0")
    
    for name, buffer in model.named_buffers():
        if buffer.device.type == 'cpu':
            print(f"Moving buffer {name} from CPU to CUDA")
            buffer.data = buffer.data.to("cuda:0")

    print("Model loaded successfully")


    return model

def attach_debugger():
    import debugpy
    debugpy.listen(5683)
    print("Waiting for debugger!")
    debugpy.wait_for_client()
    print("Attached!")

def build_cleaned_prompt_and_modes(target_point):

    if isinstance(target_point, torch.Tensor):
        tp = target_point.detach().cpu().view(-1)
        x, y = float(tp[0].item()), float(tp[1].item())
    elif isinstance(target_point, np.ndarray):
        tp = target_point.reshape(-1)
        x, y = float(tp[0]), float(tp[1])
    elif isinstance(target_point, (list, tuple)):
        assert len(target_point) >= 2
        x, y = float(target_point[0]), float(target_point[1])
    else:
        raise TypeError(f"Unsupported type for target_point: {type(target_point)}")

    x_str = f"{x:.6f}"
    y_str = f"{y:.6f}"

    prompt = f"Your target point is ({x_str}, {y_str}), what's your driving suggestion?"

    understanding_output = False
    reasoning_output = True

    return prompt, understanding_output, reasoning_output

class MOTAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file):
		self.track = autonomous_agent.Track.SENSORS
		if IS_BENCH2DRIVE:
			self.save_name = path_to_conf_file.split('+')[-1]
			self.config_path = path_to_conf_file.split('+')[0]
		else:
			now = datetime.datetime.now()
			self.config_path = path_to_conf_file
			self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		# Load diffusion policy
		print("Loading diffusion policy...")
		self.config = create_carla_config()
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		checkpoint_base_path = self.config.get('training', {}).get('checkpoint_dir', "/home/wang/Project/MoT-DP/checkpoints/carla_dit_best")
		checkpoint_path = os.path.join(checkpoint_base_path, "carla_policy_best.pt")
		
		# Load diffusion policy in bfloat16 to save memory
		self.net = load_best_model(checkpoint_path, self.config, device)
		# Convert to bfloat16 to reduce memory usage (ensure all submodules are converted)
		self.net = self.net.to(torch.bfloat16)
		# Explicitly convert obs_encoder to bfloat16 to ensure compatibility
		if hasattr(self.net, 'obs_encoder'):
			self.net.obs_encoder = self.net.obs_encoder.to(torch.bfloat16)
		print("✓ Diffusion policy loaded (bfloat16).")
		
		
		# Clear cache after loading diffusion policy
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			import gc
			gc.collect()


		# Load MoT model
		print("Loading MoT model...")
		parser = HfArgumentParser((ModelArguments, InferenceArguments))
		# Parse empty args to use defaults, avoiding conflicts with leaderboard args
		model_args, inference_args = parser.parse_args_into_dataclasses(args=[])
		self.inference_args = inference_args  # Store as instance variable
		self.AutoMoT = load_model_mot(device)
		tokenizer = AutoTokenizer.from_pretrained(model_args.qwen3vl_path)
		tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
		self.AutoMoT.language_model.tokenizer = tokenizer
		self.inferencer = InterleaveInferencer(
        model=self.AutoMoT,
        vae_model=None,
        tokenizer=tokenizer,
        vae_transform=None,
        vit_transform=None,  # Not used for Qwen3VL, handled internally by model
        new_token_ids=new_token_ids,
        max_num_tokens=inference_args.max_num_tokens,
    	)
		print("✓ MoT model loaded.")
		
		# Initialize PID controller
		self.pid_controller = WaypointPIDController(self.config)

		self.save_path = None
		self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

		# Store previous control for odd steps
		self.prev_control = None

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
		self.throttle_history = deque(maxlen=obs_horizon*10) 
		self.brake_history = deque(maxlen=obs_horizon*10) 

		
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

	def _build_obs_dict(self, tick_data, lidar, rgb_front, speed, theta, target_point, cmd_one_hot, waypoint):
		"""
		Build observation dictionaries from historical data.
		
		Returns:
			lidar_stacked: (1, obs_horizon, C, H, W) stacked lidar BEV images
			ego_status_stacked: (1, obs_horizon, 14) concatenated ego status features
			rgb_stacked: (1, 5, C, H, W) stacked RGB images
		"""
		# Build obs_dict from historical observations
		lidar_history_list = list(self.lidar_bev_history)
		speed_history_list = list(self.speed_history)
		target_point_history_list = list(self.target_point_history)
		cmd_history_list = list(self.next_command_history)
		theta_history_list = list(self.theta_history)
		throttle_history_list = list(self.throttle_history)
		brake_history_list = list(self.brake_history)
		rgb_history_list = list(self.rgb_history)
		waypoint_history_list = list(self.waypoint_history)
		
		# Sample every 10th element from the end: -1, -11, -21, -31 (corresponding to 0s, -0.5s, -1s, -1.5s)
		# Then reverse to get chronological order: oldest to newest
		lidar_list = [lidar_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(lidar_history_list)]
		speed_list = [speed_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(speed_history_list)]
		target_point_list = [target_point_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(target_point_history_list)]
		cmd_list = [cmd_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(cmd_history_list)]
		theta_list = [theta_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(theta_history_list)]
		throttle_list = [throttle_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(throttle_history_list)]
		brake_list = [brake_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(brake_history_list)]
		waypoint_list = [waypoint_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(waypoint_history_list)]
		
		# Reverse to get chronological order (oldest to newest)
		lidar_list = lidar_list[::-1]
		speed_list = speed_list[::-1]
		target_point_list = target_point_list[::-1]
		cmd_list = cmd_list[::-1]
		theta_list = theta_list[::-1]
		throttle_list = throttle_list[::-1]
		brake_list = brake_list[::-1]
		waypoint_list = waypoint_list[::-1]
		
		# sample every 5 for rgb from the end
		rgb_list = [rgb_history_list[-1 - i*5] for i in range(5) if -1 - i*5 >= -len(rgb_history_list)]
		rgb_list = rgb_list[::-1]  # Reverse to get chronological order (oldest to newest)
		
		# Stack along time dimension
		lidar_stacked = torch.stack(lidar_list, dim=0).unsqueeze(0)  # (1, obs_horizon, C, H, W)
		speed_stacked = torch.cat(speed_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 1)
		theta_stacked = torch.cat(theta_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 1)
		throttle_stacked = torch.cat(throttle_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 1)
		brake_stacked = torch.cat(brake_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 1)
		cmd_stacked = torch.cat(cmd_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 6)
		target_point_stacked = torch.cat(target_point_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 2)
		waypoint_stacked = torch.stack(waypoint_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 2)
		rgb_stacked = torch.stack(rgb_list, dim=0).unsqueeze(0)  # (1, 5, C, H, W)
		
		# Transform historical waypoints and target_points to be relative to current position
		# Current frame info for transformation
		current_pos = tick_data['gps']  # numpy array [x, y]
		current_theta = tick_data['theta']  # theta = compass - np.pi/2
		# Rotation matrix from world frame to ego frame (inverse rotation by current_theta)
		current_R = np.array([
			[np.cos(current_theta), np.sin(current_theta)],
			[-np.sin(current_theta), np.cos(current_theta)]
		])
		
		# Transform each historical waypoint to current frame
		waypoint_relative_list = []
		for i in range(self.obs_horizon):
			past_waypoint = waypoint_stacked[0, i].cpu().numpy()  # [x, y] in global coordinates
			# Transform from world to ego frame: R @ (past - current)
			relative_waypoint = current_R @ (past_waypoint - current_pos)
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
			# Rotation matrix from world to ego for the past frame
			past_R = np.array([
				[np.cos(past_theta), np.sin(past_theta)],
				[-np.sin(past_theta), np.cos(past_theta)]
			])
			
			# Transform target point from past frame's ego coordinate to world coordinate
			# Use past_R.T (ego to world) to transform the relative target point
			target_world = past_R.T @ target_in_past_frame + past_pos
			
			# Transform from world coordinate to current frame's ego coordinate
			# Use current_R (world to ego) to transform to current frame
			target_in_current_frame = current_R @ (target_world - current_pos)
			
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
		
		return lidar_stacked, ego_status_stacked, rgb_stacked


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
		
		# Prepare current observations
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
		
		# Convert rgb_front from numpy to tensor
		rgb_front = torch.from_numpy(tick_data['rgb_front']).permute(2, 0, 1).float() / 255.0
		rgb_front = rgb_front.to('cuda', dtype=torch.float32)
		
		# Convert gps from numpy to tensor
		waypoint = torch.from_numpy(tick_data['gps']).float().to('cuda', dtype=torch.float32)
		
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
		
		# Append throttle and brake from previous step (or 0 for first step)
		if self.step < 1:
			# First step: initialize with 0
			self.throttle_history.append(torch.tensor(0.0).view(1, 1).to('cuda'))
			self.brake_history.append(torch.tensor(0.0).view(1, 1).to('cuda'))
		else:
			# Use control from previous step
			prev_control = self.prev_control if self.prev_control is not None else carla.VehicleControl()
			self.throttle_history.append(torch.tensor(prev_control.throttle).view(1, 1).to('cuda'))
			self.brake_history.append(torch.tensor(prev_control.brake).view(1, 1).to('cuda'))
		
		# Check if buffer is full (need at least 31 steps for sampling -1, -11, -21, -31)
		buffer_size_needed = 31  # For obs_horizon=4, need at least 31 elements
		buffer_is_full = len(self.lidar_bev_history) >= buffer_size_needed
		
		if not buffer_is_full:
			# Buffer not full yet - use simple control to fill buffer
			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.75
			control.brake = 0.0
			self.prev_control = control
			self.pid_metadata = {}
			self.pid_metadata['agent'] = 'filling_buffer'
		elif (self.step+1) % 2 == 0:
			# Buffer is full and it's time to update (even steps)
			# Build observation dictionaries from historical data
			lidar_stacked, ego_status_stacked, rgb_stacked = self._build_obs_dict(
				tick_data, lidar, rgb_front, speed, theta, target_point, 
				cmd_one_hot, waypoint
			)
			
			# Convert tensors to PIL Images for MoT inferencer
			# Following the format from visualize_mot_open_loop.py
			# rgb_stacked shape: (1, 5, C, H, W)
			rgb_pil_list = []
			for i in range(rgb_stacked.shape[1]):  # Iterate over 5 frames
				rgb_tensor = rgb_stacked[0, i]  # (C, H, W)
				# Convert from tensor to PIL Image
				rgb_np = (rgb_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
				rgb_pil = Image.fromarray(rgb_np, mode='RGB')
				rgb_pil_list.append(rgb_pil)
			
			# Convert lidar_bev to PIL Image and put in a list (as expected by inferencer)
			# lidar_stacked shape: (1, obs_horizon, C, H, W)
			lidar_tensor = lidar_stacked[0, -1]  # (C, H, W) - last frame
			lidar_np = (lidar_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
			lidar_pil = Image.fromarray(lidar_np, mode='RGB')
			lidar_pil_list = [lidar_pil]  # Put in a list like visualize_mot_open_loop.py
			
			# Get MoT reasoning - matching visualize_mot_open_loop.py format
			prompt_cleaned , understanding_output, reasoning_output = build_cleaned_prompt_and_modes(target_point)
			predicted_answer = self.inferencer(
                image=rgb_pil_list,
                lidar=lidar_pil_list,  # Pass as list
                text=prompt_cleaned,
                understanding_output=understanding_output,
                reasoning_output=reasoning_output,
                max_think_token_n=self.inference_args.max_num_tokens,
                do_sample=False,
                text_temperature=0.0,
            )

			# Get diffusion policy action
			# Convert all inputs to bfloat16 to match model dtype
			dp_obs_dict = {
	                'lidar_bev': lidar_stacked.to(torch.bfloat16),  # (B, obs_horizon, C, H, W) = (1, obs_horizon, 3, 448, 448)
	                'ego_status': ego_status_stacked.to(torch.bfloat16),  # (B, obs_horizon, 14) = (1, obs_horizon, 1+1+1+1+6+2+2)
					'gen_vit_tokens': predicted_answer["gen_vit_tokens"].unsqueeze(0).to('cuda', dtype=torch.bfloat16),  
                	'reasoning_query_tokens': predicted_answer["reasoning_query_tokens"].unsqueeze(0).to('cuda', dtype=torch.bfloat16),  
	            }
			result = self.net.predict_action(dp_obs_dict)
			pred = torch.from_numpy(result['action'])
			print(pred)

			# Control with PID controller
			steer_traj, throttle_traj, brake_traj, metadata_traj = self.pid_controller.control_pid(pred, gt_velocity, target_point)

			control = carla.VehicleControl()

			self.pid_metadata = metadata_traj
			self.pid_metadata['agent'] = 'only_traj'
			
			# Use PID outputs with reasonable limits
			control.steer = np.clip(float(steer_traj), -1, 1)
			control.throttle = np.clip(float(throttle_traj), 0.0, 0.75)
			control.brake = np.clip(float(brake_traj), 0, 1)
			
			# Reduce micro-braking noise
			if control.brake < 0.05:
				control.brake = 0.0
			
			self.pid_metadata['steer_traj'] = float(steer_traj)
			self.pid_metadata['throttle_traj'] = float(throttle_traj)
			self.pid_metadata['brake_traj'] = float(brake_traj)

			# Safety speed limits based on steering angle
			current_speed = float(tick_data['speed'])
			abs_steer = abs(control.steer)
			
			# Balanced approach: moderate limits for turns, high trust for straight
			if abs_steer > 0.3:  ## Sharp turn - strict control needed
				speed_threshold = 4.5  ## ~16.2 km/h for sharp turns
				if current_speed > speed_threshold:
					overspeed_ratio = (current_speed - speed_threshold) / speed_threshold
					if overspeed_ratio > 0.5:
						max_throttle = 0.1
					elif overspeed_ratio > 0.3:
						max_throttle = 0.3
					elif overspeed_ratio > 0.1:
						max_throttle = 0.5
					else:
						max_throttle = 0.7
				else:
					max_throttle = 0.7
					
			elif abs_steer > 0.15:  ## Medium turn - moderate control
				speed_threshold = 7.0  ## ~25.2 km/h for medium turns
				if current_speed > speed_threshold:
					overspeed_ratio = (current_speed - speed_threshold) / speed_threshold
					if overspeed_ratio > 0.4:
						max_throttle = 0.3
					elif overspeed_ratio > 0.2:
						max_throttle = 0.6
					else:
						max_throttle = 0.8
				else:
					max_throttle = 0.8
					
			else:  ## Straight or gentle turn - very high trust, minimal limits
				speed_threshold = 15.0  ## ~54 km/h for straight (increased from 10.0 for more aggressive driving)
				if current_speed > speed_threshold:
					overspeed_ratio = (current_speed - speed_threshold) / speed_threshold
					# Extremely tolerant overspeed handling for straight driving
					if overspeed_ratio > 1.0:  # 100%+ overspeed (>30m/s / 108km/h)
						max_throttle = 0.6
					elif overspeed_ratio > 0.6:  # 60-100% overspeed
						max_throttle = 0.8
					else:  # <60% overspeed (up to ~24m/s / 86km/h)
						max_throttle = 1.0
				else:
					max_throttle = 1.0  # Full throttle when under limit
			
			control.throttle = np.clip(control.throttle, a_min=0.0, a_max=max_throttle)
			
			self.pid_metadata['steer'] = control.steer
			self.pid_metadata['throttle'] = control.throttle
			self.pid_metadata['brake'] = control.brake
			
			# Store control for next odd step
			self.prev_control = control
		else:
			# On odd steps, use previous control
			control = self.prev_control if self.prev_control is not None else carla.VehicleControl()
			self.pid_metadata = {}
			self.pid_metadata['agent'] = 'reused_control'
		
		metric_info = self.get_metric_info()
		self.metric_info[self.step] = metric_info
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
	print("="*60)
	print("Testing MOTAgent setup method")
	print("="*60)
	
	# Test setup method
	try:
		# Create agent instance
		agent = MOTAgent()
		print("✓ MOTAgent instance created successfully")
		
		# Test with a mock config file path
		# Set environment variable to avoid SAVE_PATH issues
		os.environ['SAVE_PATH'] = '/tmp/mot_test'
		
		# Create a test config path
		test_config_path = "/home/wang/Project/MoT-DP/config/pdm_local.yaml"
		
		print(f"\nTesting setup with config: {test_config_path}")
		print("-"*60)
		
		# Call setup method
		agent.setup(test_config_path)
		
		print("\n" + "="*60)
		print("✓ Setup completed successfully!")
		print("="*60)
		
		# Verify key attributes are initialized
		print("\nVerifying initialized attributes:")
		print(f"  - config loaded: {agent.config is not None}")
		print(f"  - diffusion policy loaded: {agent.net is not None}")
		print(f"  - MoT model loaded: {agent.AutoMoT is not None}")
		print(f"  - inferencer initialized: {agent.inferencer is not None}")
		print(f"  - PID controller initialized: {agent.pid_controller is not None}")
		print(f"  - obs_horizon: {agent.obs_horizon}")
		print(f"  - initialized flag: {agent.initialized}")
		
	except Exception as e:
		print("\n" + "="*60)
		print("✗ Setup failed with error:")
		print("="*60)
		import traceback
		traceback.print_exc()
		print("\nError details:")
		print(f"  Type: {type(e).__name__}")
		print(f"  Message: {str(e)}")
	
	print("\n" + "="*60)
	print("Test completed")
	print("="*60)