#!/usr/bin/env python3
import os
import sys
import torch
import time
import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
from transformers import HfArgumentParser
import json
from dataclasses import dataclass, field
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from PIL import Image
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'mot'))
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

@dataclass
class ModelArguments:
    model_path: str = field(
        default="/mnt/data/qihang_projects/load_qwen3_vl_4b_fast_thinking_origin/0007200",
        metadata={"help": "Path to the converted AutoMoT model checkpoint"}
    )
    qwen3vl_path: str = field(
        default="/mnt/data/models/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17",
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


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

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

def load_model(config, checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint['config']
    action_stats = {
    'min': torch.tensor([-11.77335262298584, -59.26432800292969]),
    'max': torch.tensor([98.34003448486328, 55.585079193115234]),
    'mean': torch.tensor([9.755727767944336, 0.03559679538011551]),
    'std': torch.tensor([14.527670860290527, 3.224050521850586]),
    }
    
    policy = DiffusionDiTCarlaPolicy(config, action_stats=action_stats).to(device)
    
    if 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model weights from checkpoint")
        if 'epoch' in checkpoint:
            print(f"  Checkpoint epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    else:
        print("⚠ No model_state_dict found in checkpoint")
    
    policy.eval()
    return policy

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

    parser = HfArgumentParser((ModelArguments, InferenceArguments))
    model_args, inference_args = parser.parse_args_into_dataclasses()
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
        reasoning_query_dim=model_args.reasoning_query_dim,
        reasoning_query_tokens=model_args.reasoning_query_max_num_tokens,
        interpolate_pos=False,
    )

    # Initialize model with Qwen3VL LLM (supports mRoPE)
    language_model = Qwen3VLForConditionalGenerationMoT(llm_config)
    vit_model = Qwen3VLVisionModel(vit_config)
    # print(f"Debug - Official vision config: depth={vit_config.depth}, hidden_size={vit_config.hidden_size}, out_hidden_size={vit_config.out_hidden_size}")
    model = AutoMoT(language_model, vit_model, config)
    model.to(device)
    state_dict = load_safetensors_weights(model_args.model_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Filter out lm_head.weight from missing keys if tie_word_embeddings=True
    actual_missing_keys = [k for k in missing_keys if k != 'language_model.lm_head.weight']
    print(f"Loaded weights: {len(actual_missing_keys)} missing, {len(unexpected_keys)} unexpected")
    device_map = {"": "cuda:0"}
    if actual_missing_keys:
        print(f"Missing keys: {actual_missing_keys[:10]}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys[:5]}")
    # Move to device and siet dtype
    # model = model.to(device_map[""]).eval().to(torch.bfloat16)
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
    
    print(f"Weight tying verification:")
    print(f"  - Weights are tied (same object): {weights_tied}")
    print(f"  - Weight values are equal: {weights_equal}")
    print(f"  - embed_tokens weight norm: {embed_norm:.4f} (should be > 100, not random)")
    
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

def main():
    # Configuration
    # attach_debugger()
    config_path = os.path.join(project_root, "config", "pdm_server.yaml")
    checkpoint_path = os.path.join(project_root, "checkpoints", "carla_dit_best", "carla_policy_best.pt")
    config = load_config(config_path)
    parser = HfArgumentParser((ModelArguments, InferenceArguments))
    model_args, inference_args = parser.parse_args_into_dataclasses()
    # Dataset paths
    dataset_path = os.path.join(
        config.get('training', {}).get('dataset_path', ''), 'val'
    )
    image_data_root = config.get('training', {}).get('image_data_root', '')
    
    print(f"Dataset path: {dataset_path}")
    print(f"Image data root: {image_data_root}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        image_data_root=image_data_root,
        mode='val'
    )
    print(f"✓ Loaded {len(dataset)} validation samples")
    AutoMoT = load_model_mot(device)
    tokenizer = AutoTokenizer.from_pretrained(model_args.qwen3vl_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    AutoMoT.language_model.tokenizer = tokenizer
    policy = load_model(config, checkpoint_path, device)
    inferencer = InterleaveInferencer(
        model=AutoMoT,
        vae_model=None,
        tokenizer=tokenizer,
        vae_transform=None,
        vit_transform=None,  # Not used for Qwen3VL, handled internally by model
        new_token_ids=new_token_ids,
        max_num_tokens=inference_args.max_num_tokens,
    )
    random.seed(44)  # For reproducibility
    sample_indices = random.sample(range(len(dataset)), 6)
    print(f"Selected sample indices: {sample_indices}")
    
    samples = []
    predictions = []
    rgb_images = []
    rgb_last_frames = []
    lidar_last_frames = []
    lidar_bev_images = []
    # Process each sample
    with torch.no_grad():
        for idx in sample_indices:
            sample = dataset[idx]
            
            # Load RGB history
            rgb_hist_paths = sample['rgb_hist_jpg']
            loaded_rgb_frames = []
            for path in rgb_hist_paths:
                full_path = os.path.join(image_data_root, path)
                img = Image.open(full_path).convert("RGB")   # 保持 PIL.Image
                loaded_rgb_frames.append(img)

            rgb_hist = loaded_rgb_frames
            print(f"  Sample {idx}: Loaded {len(loaded_rgb_frames)} RGB frames (PIL), first frame size: {rgb_hist[0].size}")
            last_frame = rgb_hist[-1]
            rgb_hist.append(last_frame)
            ## add for 2nd transformer
            rgb_hist.append(last_frame)
            rgb_images.append(rgb_hist)
            rgb_last_frames.append(last_frame)

            # Load LiDAR BEV history
            lidar_bev_paths = sample['lidar_bev_hist']
            loaded_bev_frames = []
            for path in lidar_bev_paths:
                full_path = os.path.join(image_data_root, path)
                img = Image.open(full_path).convert("RGB")
                loaded_bev_frames.append(img)
            
            lidar_bev_hist = loaded_bev_frames
            print(f"  Sample {idx}: Loaded {len(loaded_bev_frames)} LiDAR BEV frames (PIL), first frame size: {lidar_bev_hist[0].size}")
            lidar_bev_images.append(lidar_bev_hist)
            lidar_last_frames.append(lidar_bev_hist[-1])

            
            # Prepare observation dict for dp model
            mot_dict ={
                'lidar_bev': lidar_last_frames,  # ( H, W, C)
                'rgb_hist_jpg': rgb_hist,  # (obs_horizon, H, W, C)
                'target_point': sample['target_point'].unsqueeze(0).to(device),  # (1, 2)
            }
            # gen_vit_tokens, reasoning_query_tokens= mot_model.forward(mot_dict)
            prompt_cleaned , understanding_output, reasoning_output = build_cleaned_prompt_and_modes(mot_dict['target_point'])
            t1 = time.time()
            output = inferencer(
                image=mot_dict["rgb_hist_jpg"],
                lidar=mot_dict["lidar_bev"],
                text=prompt_cleaned,
                understanding_output=understanding_output,
                reasoning_output=reasoning_output,
                max_think_token_n=inference_args.max_num_tokens,
                do_sample=False,
                #frame_idx=line_idx,
                text_temperature=0.0,
                #resolved_lidar_paths=resolved_lidar_paths,
            )
            inference_time = time.time() - t1
            print(f"  Sample {idx}: MoT Inference time: {inference_time:.3f} seconds")
            predicted_answer = output
            print('predicted_answer:\n', predicted_answer)
            obs_dict = {
                'lidar_token': sample['lidar_token'].unsqueeze(0).to(device),  # (1, obs_horizon, seq_len, 512)
                'lidar_token_global': sample['lidar_token_global'].unsqueeze(0).to(device),  # (1, obs_horizon, 1, 512)
                'ego_status': sample['ego_status'].unsqueeze(0).to(device),  # (1, obs_horizon, feature_dim)
                'gen_vit_tokens': output["gen_vit_tokens"].unsqueeze(0).to(device),  # (1, ...)
                'reasoning_query_tokens': output["reasoning_query_tokens"].unsqueeze(0).to(device),  # (1, ...)
            }

            # Predict
            t0 = time.time()
            result = policy.predict_action(obs_dict)
            inference_time = time.time() - t0
            print(f"  Sample {idx}: DP Inference time: {inference_time:.3f} seconds")
            pred = result['action'][0]  # (pred_horizon, 2)
            
            samples.append(sample)
            predictions.append(pred)
            
            print(f"  Sample {idx}: Prediction shape {pred.shape}")
    
    save_dir = os.path.join(project_root, "image")
    os.makedirs(save_dir, exist_ok=True)
    data_save_path = os.path.join(save_dir, "trajectory_data.npz")
    
    # Save trajectory data
    np.savez(data_save_path,
             sample_indices=sample_indices,
             predictions=[p for p in predictions],
             waypoints_hist=[s['waypoints_hist'].cpu().numpy() if isinstance(s['waypoints_hist'], torch.Tensor) else s['waypoints_hist'] for s in samples],
             agent_pos=[s['agent_pos'].cpu().numpy() if isinstance(s['agent_pos'], torch.Tensor) else s['agent_pos'] for s in samples])
    print(f"✓ Saved trajectory data to: {data_save_path}")
    
    # Save RGB images (individual frames)
    for i, (sample_idx, rgb_image) in enumerate(zip(sample_indices, rgb_images)):
        # Convert different types to numpy
        if isinstance(rgb_image, torch.Tensor):
            rgb_array = rgb_image.cpu().numpy()
        elif isinstance(rgb_image, list):
            rgb_array = np.array(rgb_image)
        else:
            rgb_array = rgb_image
        
        # If it's a sequence of frames, save the last one
        if rgb_array.ndim == 4:  # (T, H, W, C) or similar
            rgb_array = rgb_array[-1]
        
        # Handle different possible shapes (C, H, W) or (H, W, C)
        if rgb_array.ndim == 3:
            if rgb_array.shape[0] in [3, 4]:  # Likely (C, H, W) format
                rgb_array = np.transpose(rgb_array, (1, 2, 0))
            
            # Normalize to 0-255 if values are in 0-1 range
            if rgb_array.max() <= 1.0:
                rgb_array = (rgb_array * 255).astype(np.uint8)
            else:
                rgb_array = rgb_array.astype(np.uint8)
            
            # Save image
            if rgb_array.shape[2] >= 3:
                img = Image.fromarray(rgb_array[:, :, :3])
            else:
                img = Image.fromarray(rgb_array)
            image_save_path = os.path.join(save_dir, f"rgb_sample_{sample_idx}.png")
            img.save(image_save_path)
            print(f"✓ Saved RGB image to: {image_save_path}")