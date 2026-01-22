"""
MoT (Multimodal Trajectory) utilities for model loading and inference.
Contains functions related to MoT model initialization, configuration, and prompt processing.
"""

import os
import sys
import json
import random
import glob
from dataclasses import dataclass, field
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import HfArgumentParser
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

from mot.modeling.automotive import (
    AutoMoTConfig, AutoMoT,
    Qwen3VLTextConfig, Qwen3VLTextModel, Qwen3VLForConditionalGenerationMoT
)


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
        default=8,
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
    """Convert model parameters to target dtype, with memory-efficient approach."""
    if exclude_buffer_patterns is None:
        exclude_buffer_patterns = []
    
    # Convert parameters in chunks to avoid memory spikes
    for name, param in model.named_parameters():
        if param.device.type == 'cuda':
            # For CUDA tensors, convert in-place to avoid extra memory allocation
            param.data = param.data.to(target_dtype)
        else:
            # For CPU tensors, convert and then move to avoid duplicating on GPU
            param.data = param.data.to(target_dtype)

    for name, buffer in model.named_buffers():
        should_exclude = any(pattern in name for pattern in exclude_buffer_patterns)
        
        if should_exclude:
            pass  # Keep original dtype
        else:
            if buffer.device.type == 'cuda':
                buffer.data = buffer.data.to(target_dtype)
            else:
                buffer.data = buffer.data.to(target_dtype)
    
    return model


def load_model_mot(device):
    """Load and initialize the MoT model with proper configuration."""
    parser = HfArgumentParser((ModelArguments, InferenceArguments))
    model_args, inference_args = parser.parse_args_into_dataclasses(args=[])

    assert torch.cuda.is_available(), "CUDA is required"
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
    llm_config.tie_word_embeddings = True
    llm_config.layer_module = "Qwen3VLMoTDecoderLayer"
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
        vision_config=vit_config,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        connector_act=model_args.connector_act,
        interpolate_pos=False,
        reasoning_query_dim=model_args.reasoning_query_dim,
        reasoning_query_tokens=model_args.reasoning_query_max_num_tokens,
        action_query_dim=model_args.action_query_dim,
        action_query_tokens=model_args.action_query_tokens,
    )

    # Initialize model on CPU first to save GPU memory during loading
    print("Initializing model on CPU...")
    language_model = Qwen3VLForConditionalGenerationMoT(llm_config)
    vit_model = Qwen3VLVisionModel(vit_config)
    model = AutoMoT(language_model, vit_model, config)

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
    
    # Free state_dict memory immediately
    del state_dict
    import gc
    gc.collect()
    
    # Convert to bfloat16 on CPU first (saves GPU memory during transfer)
    print("Converting model to bfloat16 on CPU...")
    model = convert_model_dtype_with_exceptions(
        model,
        torch.bfloat16,
        exclude_buffer_patterns=['inv_freq']
    )
    
    # Clear any cached memory before moving to GPU
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Now move to GPU (already in bfloat16, so less memory needed)
    print("Moving model to GPU...")
    model = model.to("cuda:0").eval()
    
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
    
    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Model loaded successfully")

    return model


def build_cleaned_prompt_and_modes(target_point_speed):
    """
    Build cleaned prompt and output modes from target point and speed data.
    
    Args:
        target_point_speed: Tensor, numpy array, or list containing [speed, x, y]
    
    Returns:
        tuple: (prompt, understanding_output, reasoning_output)
    """
    if isinstance(target_point_speed, torch.Tensor):
        tp = target_point_speed.detach().cpu().view(-1)
        speed, x, y = float(tp[0].item()), float(tp[1].item()), float(tp[2].item())
    elif isinstance(target_point_speed, np.ndarray):
        tp = target_point_speed.reshape(-1)
        speed, x, y = float(tp[0]), float(tp[1]), float(tp[2])
    elif isinstance(target_point_speed, (list, tuple)):
        assert len(target_point_speed) >= 3
        speed, x, y = float(target_point_speed[0]), float(target_point_speed[1]), float(target_point_speed[2])
    else:
        raise TypeError(f"Unsupported type for target_point: {type(target_point_speed)}")

    x_str = f"{x:.6f}"
    y_str = f"{y:.6f}"
    prompt = f"Your target point is ({x_str}, {y_str}), and your current velocity is {speed:.2f} m/s. Predict the driving actions ( now, +1s, +2s) and plan the trajectory for the next 3 seconds."

    understanding_output = False
    reasoning_output = True

    return prompt, understanding_output, reasoning_output


def parse_decision_sequence(decision_str):
    """
    Parse decision sequence from model output.
    
    Args:
        decision_str: String like '<|im_start|> stop, accelerate, stop<|im_end|>'
                      or 'stop, accelerate, stop'
    
    Returns:
        tuple: (decision_now, decision_1s, decision_2s) - three decisions as strings
               Returns (None, None, None) if parsing fails
    
    Example:
        >>> parse_decision_sequence('<|im_start|> stop, accelerate, stop<|im_end|>')
        ('stop', 'accelerate', 'stop')
    """
    if not decision_str or not isinstance(decision_str, str):
        return (None, None, None)
    
    # Remove special tokens
    cleaned = decision_str.replace('<|im_start|>', '').replace('<|im_end|>', '')
    # Strip whitespace
    cleaned = cleaned.strip()
    
    # Split by comma
    parts = [p.strip() for p in cleaned.split(',')]
    
    # Ensure we have exactly 3 decisions
    if len(parts) >= 3:
        return (parts[0], parts[1], parts[2])
    elif len(parts) == 2:
        return (parts[0], parts[1], None)
    elif len(parts) == 1:
        return (parts[0], None, None)
    else:
        return (None, None, None)


def split_prompt(prompt_cleaned):
    """
    Split the prompt into two sentences.
    
    Args:
        prompt_cleaned: String like 'Your target point is (53.101654, 0.201010), and your current velocity is 0.00 m/s. Predict the driving actions ( now, +1s, +2s) and plan the trajectory for the next 3 seconds.'
    
    Returns:
        tuple: (sentence1, sentence2)
    
    Example:
        >>> split_prompt('Your target point is (...). Predict the driving actions...')
        ('Your target point is (...).', 'Predict the driving actions...')
    """
    if not prompt_cleaned or not isinstance(prompt_cleaned, str):
        return (None, None)
    
    # Find the first period followed by space (end of first sentence)
    # Pattern: "...m/s. Predict..."
    split_marker = "m/s. "
    if split_marker in prompt_cleaned:
        idx = prompt_cleaned.find(split_marker)
        sentence1 = prompt_cleaned[:idx + 4]  # Include "m/s."
        sentence2 = prompt_cleaned[idx + 5:]  # Skip "m/s. "
        return (sentence1.strip(), sentence2.strip())
    
    # Fallback: split by ". " if marker not found
    if ". " in prompt_cleaned:
        idx = prompt_cleaned.find(". ")
        sentence1 = prompt_cleaned[:idx + 1]
        sentence2 = prompt_cleaned[idx + 2:]
        return (sentence1.strip(), sentence2.strip())
    
    return (prompt_cleaned, None)
