import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Callable, Union
from collections import defaultdict
import numpy as np
import math
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from model.transformer_for_diffusion import TransformerForDiffusion, LowdimMaskGenerator
from model.interfuser_bev_encoder import InterfuserBEVEncoder
from model.interfuser_bev_encoder import load_lidar_submodules
import os
from collections import OrderedDict
from collections import deque


class DDIMScheduler_with_logprob(DDIMScheduler):
    """
    DDIMScheduler with multiplicative and additive noise support.
    Based on DiffusionDriveV2 implementation.
    """
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 1.0,  # 1.0 for ddpm, 0.0 for ddim
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        prev_sample: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[Tuple, torch.Tensor]:
        """
        Predict the sample from the previous timestep by reversing the SDE.
        Implements multiplicative and additive noise following DiffusionDriveV2.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 1. get previous step value (=t-1)
        prev_timestep = (
            timestep - self.config.num_train_timesteps // self.num_inference_steps
        )

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)"
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = (eta * variance ** (0.5)).clamp_(min=1e-10)

        if use_clipped_model_output:
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t"
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2).clamp_(min=0) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise"
        prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if prev_sample_mean is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
                " `prev_sample` stays `None`."
            )
        
        # 8. Add multiplicative and additive noise (DiffusionDriveV2 style)
        if eta > 0:
            std_dev_t_mul = torch.clip(std_dev_t, min=0.04)
            std_dev_t_add = torch.tensor(0.0).to(std_dev_t.device)
        else:
            std_dev_t_mul = torch.tensor(0.0).to(std_dev_t.device)
            std_dev_t_add = torch.tensor(0.0).to(std_dev_t.device)
        
        if prev_sample is None:
            # Multiplicative noise - applied per sample, shared across time steps
            # Shape: (B, 1, 1, 2) for x and y separately
            if model_output.dim() == 3:
                # Shape: (B, T, 2) -> need (B, 1, 2) noise
                variance_noise_x = randn_tensor(
                    [model_output.shape[0], 1, 1], generator=generator, device=model_output.device, dtype=model_output.dtype
                ) * std_dev_t_mul + 1.0
                variance_noise_y = randn_tensor(
                    [model_output.shape[0], 1, 1], generator=generator, device=model_output.device, dtype=model_output.dtype
                ) * std_dev_t_mul + 1.0
                variance_noise_mul = torch.cat((variance_noise_x, variance_noise_y), dim=-1)
                variance_noise_mul = variance_noise_mul.expand(-1, model_output.shape[1], -1)
                
                # Additive noise
                variance_noise_x_add = randn_tensor(
                    [model_output.shape[0], 1, 1], generator=generator, device=model_output.device, dtype=model_output.dtype
                )
                variance_noise_y_add = randn_tensor(
                    [model_output.shape[0], 1, 1], generator=generator, device=model_output.device, dtype=model_output.dtype
                )
                variance_noise_add = torch.cat((variance_noise_x_add, variance_noise_y_add), dim=-1)
                variance_noise_add = variance_noise_add.expand(-1, model_output.shape[1], -1)
            else:
                # 4D case: (B, G, T, 2) for multi-modal
                variance_noise_horizon = randn_tensor(
                    [model_output.shape[0], model_output.shape[1], 1, 1], generator=generator, device=model_output.device, dtype=model_output.dtype
                ) * std_dev_t_mul + 1.0
                variance_noise_vert = randn_tensor(
                    [model_output.shape[0], model_output.shape[1], 1, 1], generator=generator, device=model_output.device, dtype=model_output.dtype
                ) * std_dev_t_mul + 1.0
                variance_noise_mul = torch.cat((variance_noise_horizon, variance_noise_vert), dim=-1)
                variance_noise_mul = variance_noise_mul.repeat(1, 1, model_output.shape[2], 1)
                
                variance_noise_x = randn_tensor(
                    [model_output.shape[0], model_output.shape[1], 1, 1], generator=generator, device=model_output.device, dtype=model_output.dtype
                )
                variance_noise_y = randn_tensor(
                    [model_output.shape[0], model_output.shape[1], 1, 1], generator=generator, device=model_output.device, dtype=model_output.dtype
                )
                variance_noise_add = torch.cat((variance_noise_x, variance_noise_y), dim=-1)
                variance_noise_add = variance_noise_add.repeat(1, 1, model_output.shape[2], 1)

            prev_sample = prev_sample_mean * variance_noise_mul + std_dev_t_add * variance_noise_add
        
        # Compute log probability for potential RL usage
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t_mul**2 + 1e-8))
            - torch.log(std_dev_t_mul + 1e-8)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        if log_prob.dim() == 3:
            log_prob = log_prob.sum(dim=(-2, -1))
        else:
            log_prob = log_prob.sum(dim=(-3, -2, -1))
        
        return prev_sample.type(sample.dtype), log_prob, prev_sample_mean.type(sample.dtype)



def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

class DiffusionDiTCarlaPolicy(nn.Module):
    def __init__(self, config: Dict, action_stats: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        
        # config
        self.cfg = config
        policy_cfg = config['policy']
        noise_scheduler_cfg = config['noise_scheduler']

        obs_as_global_cond = policy_cfg.get('obs_as_global_cond', True)
        self.obs_as_global_cond = obs_as_global_cond
        shape_meta = config['shape_meta']
        action_shape = shape_meta['action']['shape']
        action_dim = action_shape[0]
        
        # Action normalization settings
        self.enable_action_normalization = config.get('enable_action_normalization', False)
        self.action_stats = action_stats
        if self.enable_action_normalization and self.action_stats is not None:
            print(f"✓ Action normalization enabled with stats:")
            print(f"  Action min: {self.action_stats['min']}")
            print(f"  Action max: {self.action_stats['max']}")
        else:
            print("⚠ Action normalization disabled")
            self.action_stats = None
        

        self.n_obs_steps = policy_cfg.get('n_obs_steps', config.get('obs_horizon', 1))
        
        # Load BEV encoder configuration from config file
        bev_encoder_cfg = config.get('bev_encoder', {})
        obs_encoder = InterfuserBEVEncoder(
            perception_backbone=None,
            state_dim=bev_encoder_cfg.get('state_dim', 13),  # 修改为13维以支持拼接后的ego_status
            feature_dim=bev_encoder_cfg.get('feature_dim', 256),
            use_group_norm=bev_encoder_cfg.get('use_group_norm', True),
            freeze_backbone=bev_encoder_cfg.get('freeze_backbone', False),
            bev_input_size=tuple(bev_encoder_cfg.get('bev_input_size', [448, 448]))
        )
        
        # Load pretrained weights from config
        pretrained_path = bev_encoder_cfg.get('pretrained_path', None)
        if pretrained_path is not None and os.path.exists(pretrained_path):
            load_lidar_submodules(obs_encoder, pretrained_path, strict=False, logger=None)
            print(f"✓ BEV encoder loaded from: {pretrained_path}")
        else:
            print(f"⚠ BEV encoder pretrained_path not found or not specified: {pretrained_path}")
            print("  Continuing with random initialization...")
        
        self.obs_encoder = obs_encoder

        vlm_feature_dim = 2560  # 隐藏层维度
        self.feature_encoder = nn.Linear(vlm_feature_dim, 1536)

        obs_feature_dim = 256  

        # Get status_dim from bev_encoder config
        status_dim = bev_encoder_cfg.get('state_dim', 15)
        
        # Get ego_status_seq_len from policy config (defaults to n_obs_steps)
        ego_status_seq_len = policy_cfg.get('ego_status_seq_len', self.n_obs_steps)

        model = TransformerForDiffusion(
            input_dim=policy_cfg.get('input_dim', 2),
            output_dim=policy_cfg.get('output_dim', 2),
            horizon=policy_cfg.get('horizon', 16),
            n_obs_steps=self.n_obs_steps,  
            cond_dim=256,   
            n_layer=policy_cfg.get('n_layer', 8),
            n_head=policy_cfg.get('n_head', 8),
            n_emb=policy_cfg.get('n_emb', 512),
            p_drop_emb=policy_cfg.get('p_drop_emb', 0.1),
            p_drop_attn=policy_cfg.get('p_drop_attn', 0.1),
            causal_attn=policy_cfg.get('causal_attn', True),
            obs_as_cond=obs_as_global_cond,
            n_cond_layers=policy_cfg.get('n_cond_layers', 4),
            status_dim=status_dim,  # 传入 ego_status 维度
            ego_status_seq_len=ego_status_seq_len  # 传入 ego_status 序列长度
        )

        self.model = model
        
        # ========== Truncated Diffusion Configuration (DiffusionDriveV2 style) ==========
        # Use DDIM scheduler for both training and inference with truncated diffusion
        diffusion_cfg = config.get('truncated_diffusion', {})
        self.use_truncated_diffusion = diffusion_cfg.get('enabled', True)
        self.num_train_timesteps = diffusion_cfg.get('num_train_timesteps', 1000)
        self.trunc_timesteps = diffusion_cfg.get('trunc_timesteps', 8)  # Truncated timestep for anchor during inference
        self.train_trunc_timesteps = diffusion_cfg.get('train_trunc_timesteps', 50)  # Max timestep during training (DiffusionDrive uses 50)
        self.num_diffusion_steps = diffusion_cfg.get('num_diffusion_steps', 2)  # Number of denoising steps
        self.diffusion_eta = diffusion_cfg.get('eta', 0.0)  # 0.0 for DDIM, 1.0 for DDPM
        
        # Normalization parameters (DiffusionDriveV2 style: linear mapping to [-1, 1])
        # x: 2*(x + x_offset)/x_range - 1
        # y: 2*(y + y_offset)/y_range - 1
        self.norm_x_offset = diffusion_cfg.get('norm_x_offset', 2.0)  # x range: [-2, 78]
        self.norm_x_range = diffusion_cfg.get('norm_x_range', 80.0)
        self.norm_y_offset = diffusion_cfg.get('norm_y_offset', 20.0)  # y range: [-20, 36]
        self.norm_y_range = diffusion_cfg.get('norm_y_range', 56.0)
        
        # Training scheduler (DDIMScheduler for add_noise)
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            steps_offset=1,
            beta_schedule="scaled_linear",
            prediction_type="sample",  # Predict clean sample directly
        )
        
        # Inference scheduler with multiplicative/additive noise
        self.diffusionrl_scheduler = DDIMScheduler_with_logprob(
            num_train_timesteps=self.num_train_timesteps,
            steps_offset=1,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )
        
        # Legacy scheduler for backward compatibility (when truncated diffusion is disabled)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_cfg.get('num_diffusion_steps', 100),
            beta_start=noise_scheduler_cfg.get('beta_start', 0.0001),
            beta_end=noise_scheduler_cfg.get('beta_end', 0.02),
            beta_schedule=noise_scheduler_cfg.get('beta_schedule', "squaredcos_cap_v2"),
            clip_sample=noise_scheduler_cfg.get('clip_sample', False),
            prediction_type=noise_scheduler_cfg.get('prediction_type', "epsilon"),
        )

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_global_cond) else obs_feature_dim,
            max_n_obs_steps=self.n_obs_steps,  
            fix_obs_steps=True,
            action_visible=False
        )

        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.horizon = policy_cfg.get('horizon', 16)
        self.n_action_steps = policy_cfg.get('action_horizon', 8)
        self.num_inference_steps = policy_cfg.get('num_inference_steps', 100)
    
    # ========== DiffusionDriveV2-style Normalization Functions ==========
    def norm_odo(self, odo_info_fut: torch.Tensor) -> torch.Tensor:
        """
        Normalize trajectory coordinates to [-1, 1] range.
        Following DiffusionDriveV2: 2*(x + offset)/range - 1
        """
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        
        # Linear mapping to [-1, 1]
        odo_info_fut_x = 2 * (odo_info_fut_x + self.norm_x_offset) / self.norm_x_range - 1
        odo_info_fut_y = 2 * (odo_info_fut_y + self.norm_y_offset) / self.norm_y_range - 1
        
        return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)
    
    def denorm_odo(self, odo_info_fut: torch.Tensor) -> torch.Tensor:
        """
        Denormalize trajectory from [-1, 1] back to original scale.
        Following DiffusionDriveV2: (x + 1)/2 * range - offset
        """
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        
        # Inverse linear mapping from [-1, 1]
        odo_info_fut_x = (odo_info_fut_x + 1) / 2 * self.norm_x_range - self.norm_x_offset
        odo_info_fut_y = (odo_info_fut_y + 1) / 2 * self.norm_y_range - self.norm_y_offset
        
        return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Normalize action from original range to [-1, 1]
        Args:
            action: tensor of shape (..., action_dim)
        Returns:
            normalized action in range [-1, 1]
        """
        if not self.enable_action_normalization or self.action_stats is None:
            return action
        
        device = action.device
        action_min = self.action_stats['min'].to(device)
        action_max = self.action_stats['max'].to(device)
        
        # Normalize to [0, 1]
        normalized = (action - action_min) / (action_max - action_min + 1e-8)
        # Normalize to [-1, 1]
        normalized = normalized * 2 - 1
        return normalized

    def unnormalize_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize action from [-1, 1] back to original range
        Args:
            normalized_action: tensor of shape (..., action_dim) in range [-1, 1]
        Returns:
            action in original range
        """
        if not self.enable_action_normalization or self.action_stats is None:
            return normalized_action
        
        device = normalized_action.device
        action_min = self.action_stats['min'].to(device)
        action_max = self.action_stats['max'].to(device)
        
        # Unnormalize from [-1, 1] to [0, 1]
        unnormalized = (normalized_action + 1) / 2
        # Unnormalize to original range
        unnormalized = unnormalized * (action_max - action_min) + action_min
        return unnormalized

    def extract_tcp_features(self, obs_dict, return_attention=False):
        """
        使用InterfuserBEVEncoder提取特征
        支持两种模式：
        1. 使用预处理好的BEV特征（推荐，快速）
        2. 使用原始lidar_bev图像（兼容模式，慢）
        
        Args:
            obs_dict: 观测字典，应包含：
                - 'lidar_token': (B, seq_len, 512) 预处理的空间特征，或
                - 'lidar_token_global': (B, 1, 512) 预处理的全局特征，或
                - 'lidar_bev': (B, 3, 448, 448) 原始BEV图像（兼容模式）
                - 'ego_status' (B, 13): [accel(3), rot_rate(3), vel(3), steer(1), command(3)]
                  已经拼接好的13维状态向量，直接作为state输入
            return_attention: 是否返回attention map
            
        Returns:
            如果return_attention=False: j_ctrl特征 (B, 256)
            如果return_attention=True: (j_ctrl特征, attention_map) 
        """
        try:
            device = next(self.parameters()).device
            # Get the dtype of the model parameters
            model_dtype = next(self.parameters()).dtype
            state = obs_dict['ego_status'].to(device=device, dtype=model_dtype)  # Use model dtype instead of float32
            use_precomputed = 'lidar_token' in obs_dict and 'lidar_token_global' in obs_dict
            if use_precomputed:
                lidar_token = obs_dict['lidar_token'].to(device=device, dtype=model_dtype)
                lidar_token_global = obs_dict['lidar_token_global'].to(device=device, dtype=model_dtype)
                
                if return_attention:
                    j_ctrl, attention_map = self.obs_encoder(
                        state=state,
                        lidar_token=lidar_token,
                        lidar_token_global=lidar_token_global,
                        normalize=True,
                        return_attention=True
                    )
                else:
                    j_ctrl = self.obs_encoder(
                        state=state,
                        lidar_token=lidar_token,
                        lidar_token_global=lidar_token_global,
                        normalize=True,
                        return_attention=False
                    )
                    attention_map = None
            else:
                if 'lidar_bev' not in obs_dict:
                    raise KeyError("Neither pre-computed features (lidar_token, lidar_token_global) nor raw BEV images (lidar_bev) found in obs_dict")
                
                lidar_bev_img = obs_dict['lidar_bev'].to(device=device, dtype=model_dtype)  # Use model dtype instead of float32
                if return_attention:
                    j_ctrl, attention_map = self.obs_encoder(
                        image=lidar_bev_img,
                        state=state,
                        normalize=True,
                        return_attention=True
                    )
                else:
                    j_ctrl = self.obs_encoder(
                        image=lidar_bev_img,
                        state=state,
                        normalize=True,
                        return_attention=False
                    )
                    attention_map = None
            
            if return_attention:
                return j_ctrl, attention_map
            else:
                return j_ctrl
                
        except KeyError as e:
            raise KeyError(f"Missing required field in obs_dict for TCP feature extraction: {e}")
        except Exception as e:
            raise RuntimeError(f"Error in TCP feature extraction: {e}")

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        batch: {
            # BEV特征（二选一）
            'lidar_token': (B, obs_horizon, seq_len, 512) - 预处理的空间特征（推荐）
            'lidar_token_global': (B, obs_horizon, 1, 512) - 预处理的全局特征（推荐）
            'lidar_bev': (B, obs_horizon, 3, 448, 448) - 原始LiDAR BEV图像（兼容模式）
            
            'agent_pos': (B, horizon, 2) - 未来轨迹点
            'ego_status': (B, obs_horizon, 13) - 车辆状态 [accel(3), rot_rate(3), vel(3), steer(1), command(3)]
            'anchor': (B, horizon, 2) - anchor轨迹点（用于truncated diffusion）
        }
        """
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype  # Get model dtype
        nobs = {}
        required_fields = ['lidar_token', 'lidar_token_global', 'lidar_bev', 'ego_status', 'agent_pos', 'reasoning_query_tokens', 'gen_vit_tokens', 'anchor']
        
        for field in required_fields:
            if field in batch:
                if field in ['lidar_bev', 'lidar_token', 'lidar_token_global']:
                    nobs[field] = batch[field].to(device=device, dtype=model_dtype)  # Use model dtype
                else:
                    nobs[field] = batch[field].to(device)

        raw_agent_pos = batch['agent_pos'].to(device)

        # (B, horizon, 2)
        To = self.n_obs_steps
        nactions = raw_agent_pos
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        cond = None
        
        # Get ground truth trajectory
        trajectory = nactions.to(dtype=model_dtype)  # (B, horizon, 2)
        
        # Get anchor trajectory for truncated diffusion
        anchor = batch.get('anchor', None)
        if anchor is not None:
            anchor = anchor.to(device=device, dtype=model_dtype)  # (B, horizon, 2)
        
        batch_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))  # (B*To, ...)
        batch_features = self.extract_tcp_features(batch_nobs)  # (B*To, feature_dim)
        feature_dim = batch_features.shape[-1]
        cond = batch_features.reshape(batch_size, To, feature_dim)  # Already in model_dtype

        # Prepare VL tokens
        gen_vit_tokens = batch['gen_vit_tokens']
        reasoning_query_tokens = batch['reasoning_query_tokens']
        gen_vit_tokens = gen_vit_tokens.to(device=device, dtype=model_dtype)
        gen_vit_tokens = self.feature_encoder(gen_vit_tokens)
        reasoning_query_tokens = reasoning_query_tokens.to(device=device, dtype=model_dtype)
        reasoning_query_tokens = self.feature_encoder(reasoning_query_tokens)
        
        # Get ego_status
        ego_status = nobs['ego_status'].to(dtype=model_dtype)

        # ========== Compute Loss Based on Diffusion Mode ==========
        if self.use_truncated_diffusion and anchor is not None:
            # Truncated Diffusion Training (DiffusionDriveV2 style)
            loss = self._compute_truncated_diffusion_loss(
                trajectory=trajectory,
                anchor=anchor,
                cond=cond,
                gen_vit_tokens=gen_vit_tokens,
                reasoning_query_tokens=reasoning_query_tokens,
                ego_status=ego_status,
                device=device,
                model_dtype=model_dtype
            )
        else:
            # Legacy DDPM Training (backward compatible)
            loss = self._compute_ddpm_loss(
                trajectory=trajectory,
                cond=cond,
                gen_vit_tokens=gen_vit_tokens,
                reasoning_query_tokens=reasoning_query_tokens,
                ego_status=ego_status,
                device=device,
                model_dtype=model_dtype
            )
        
        return loss
    
    def _compute_truncated_diffusion_loss(
        self,
        trajectory: torch.Tensor,
        anchor: torch.Tensor,
        cond: torch.Tensor,
        gen_vit_tokens: torch.Tensor,
        reasoning_query_tokens: torch.Tensor,
        ego_status: torch.Tensor,
        device: torch.device,
        model_dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Compute loss using truncated diffusion (DiffusionDriveV2 style).
        Instead of starting from pure noise, we start from anchor with small noise.
        The model predicts the clean sample directly.
        
        Training: Add noise at random timesteps [0, train_trunc_timesteps)
        Inference: Add noise at fixed trunc_timesteps (smaller value)
        """
        batch_size = trajectory.shape[0]
        
        # 1. Normalize trajectories using DiffusionDriveV2 style normalization
        trajectory_norm = self.norm_odo(trajectory)  # (B, T, 2)
        anchor_norm = self.norm_odo(anchor)  # (B, T, 2)
        
        # 2. Sample random timesteps within truncated range (like DiffusionDrive training)
        timesteps = torch.randint(
            0, self.train_trunc_timesteps,  # Training uses larger range [0, 50)
            (batch_size,), device=device
        ).long()
        
        # 3. Add noise to anchor
        noise = torch.randn_like(anchor_norm, device=device)
        noisy_anchor = self.diffusion_scheduler.add_noise(
            original_samples=anchor_norm,
            noise=noise,
            timesteps=timesteps
        )
        
        # 4. Clamp to valid range
        noisy_anchor = torch.clamp(noisy_anchor, min=-1, max=1)
        
        # 5. Denormalize for model input (model expects denormalized coordinates)
        noisy_trajectory_denorm = self.denorm_odo(noisy_anchor)
        
        # 6. Predict clean sample
        pred = self.model(
            noisy_trajectory_denorm,
            timesteps,  # Use the sampled timesteps
            cond,
            gen_vit_tokens=gen_vit_tokens,
            reasoning_query_tokens=reasoning_query_tokens,
            ego_status=ego_status
        )
        
        # 7. Compute loss - predict clean sample (not noise)
        # Target is the ground truth trajectory (denormalized)
        # DiffusionDrive uses L1 loss
        target = trajectory
        
        loss = F.l1_loss(pred, target, reduction='none')
        
        if loss.shape[-1] > 2:
            loss = loss[..., :2]
        
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss
    
    def _compute_ddpm_loss(
        self,
        trajectory: torch.Tensor,
        cond: torch.Tensor,
        gen_vit_tokens: torch.Tensor,
        reasoning_query_tokens: torch.Tensor,
        ego_status: torch.Tensor,
        device: torch.device,
        model_dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Compute loss using standard DDPM (legacy, backward compatible).
        """
        batch_size = trajectory.shape[0]
        
        # Normalize trajectory for training
        if self.enable_action_normalization and self.action_stats is not None:
            trajectory = self.normalize_action(trajectory)
        if torch.isnan(trajectory).any() or torch.isinf(trajectory).any():
            print("Warning: NaN or Inf detected in normalized trajectory")
            trajectory = torch.nan_to_num(trajectory, nan=0.0, posinf=1.0, neginf=-1.0)

        condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.get('num_train_timesteps', 100), 
            (batch_size,), device=trajectory.device
        ).long()
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask
        
        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        pred = self.model(
            noisy_trajectory,
            timesteps,
            cond,
            gen_vit_tokens=gen_vit_tokens,
            reasoning_query_tokens=reasoning_query_tokens,
            ego_status=ego_status
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        
        if loss.shape[-1] > 2:
            loss = loss[..., :2]  
            
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
    

    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None, gen_vit_tokens=None, 
            reasoning_query_tokens=None, ego_status=None,
            anchor=None,  # Anchor trajectory for truncated diffusion
            # keyword arguments to scheduler.step
            **kwargs
            ):
        """
        Generate trajectory samples using diffusion.
        Supports both truncated diffusion (with anchor) and standard diffusion.
        """
        model = self.model
        device = condition_data.device
        bs = condition_data.shape[0]
        model_dtype = condition_data.dtype
        
        # ===== OPTIMIZATION: Cache encoder outputs =====
        with torch.no_grad():
            memory, vl_features, reasoning_features, vl_padding_mask, reasoning_padding_mask = \
                model.encode_conditions(
                    cond=cond,
                    gen_vit_tokens=gen_vit_tokens,
                    reasoning_query_tokens=reasoning_query_tokens
                )
        
        # ========== Choose Diffusion Mode ==========
        if self.use_truncated_diffusion and anchor is not None:
            # Truncated Diffusion with Anchor (DiffusionDriveV2 style)
            trajectory = self._truncated_diffusion_sample(
                anchor=anchor,
                memory=memory,
                vl_features=vl_features,
                reasoning_features=reasoning_features,
                cond=cond,
                ego_status=ego_status,
                vl_padding_mask=vl_padding_mask,
                reasoning_padding_mask=reasoning_padding_mask,
                device=device,
                model_dtype=model_dtype,
                generator=generator
            )
        else:
            # Standard Diffusion (legacy)
            trajectory = self._standard_diffusion_sample(
                condition_data=condition_data,
                condition_mask=condition_mask,
                memory=memory,
                vl_features=vl_features,
                reasoning_features=reasoning_features,
                cond=cond,
                ego_status=ego_status,
                vl_padding_mask=vl_padding_mask,
                reasoning_padding_mask=reasoning_padding_mask,
                device=device,
                generator=generator,
                **kwargs
            )

        return trajectory
    
    def _truncated_diffusion_sample(
        self,
        anchor: torch.Tensor,
        memory: torch.Tensor,
        vl_features: torch.Tensor,
        reasoning_features: torch.Tensor,
        cond: torch.Tensor,
        ego_status: torch.Tensor,
        vl_padding_mask: torch.Tensor,
        reasoning_padding_mask: torch.Tensor,
        device: torch.device,
        model_dtype: torch.dtype,
        generator=None
    ) -> torch.Tensor:
        """
        Truncated diffusion sampling (DiffusionDriveV2 style).
        Start from anchor with small noise, denoise for few steps.
        """
        bs = anchor.shape[0]
        
        # Set up scheduler
        self.diffusionrl_scheduler.set_timesteps(self.num_train_timesteps, device)
        
        # Compute rollout timesteps
        step_ratio = 20 / self.num_diffusion_steps
        roll_timesteps = (np.arange(0, self.num_diffusion_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)
        
        # 1. Normalize anchor
        diffusion_output = self.norm_odo(anchor)  # (B, T, 2)
        
        # 2. Add truncated noise
        noise = torch.randn(diffusion_output.shape, device=device, dtype=model_dtype)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * self.trunc_timesteps
        diffusion_output = self.diffusion_scheduler.add_noise(
            original_samples=diffusion_output,
            noise=noise,
            timesteps=trunc_timesteps
        )
        
        # 3. Denoising loop
        for i, k in enumerate(roll_timesteps):
            # Clamp and denormalize
            x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)  # (B, T, 2)
            
            # Get timestep
            timesteps = k
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(device)
            timesteps = timesteps.expand(bs)
            
            # Predict clean sample
            pred = self.model.decode_with_cache(
                sample=noisy_traj_points,
                timestep=timesteps,
                memory=memory,
                vl_features=vl_features,
                reasoning_features=reasoning_features,
                cond=cond,
                ego_status=ego_status,
                vl_padding_mask=vl_padding_mask,
                reasoning_padding_mask=reasoning_padding_mask
            )
            
            # Normalize prediction
            x_start = self.norm_odo(pred)  # (B, T, 2)
            
            # DDIM step with multiplicative/additive noise
            diffusion_output, _, _ = self.diffusionrl_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=diffusion_output,
                eta=self.diffusion_eta,
                generator=generator
            )
        
        # 4. Final denormalization
        trajectory = self.denorm_odo(torch.clamp(diffusion_output, min=-1, max=1))
        
        return trajectory
    
    def _standard_diffusion_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        memory: torch.Tensor,
        vl_features: torch.Tensor,
        reasoning_features: torch.Tensor,
        cond: torch.Tensor,
        ego_status: torch.Tensor,
        vl_padding_mask: torch.Tensor,
        reasoning_padding_mask: torch.Tensor,
        device: torch.device,
        generator=None,
        **kwargs
    ) -> torch.Tensor:
        """
        Standard diffusion sampling (legacy DDPM).
        """
        scheduler = self.noise_scheduler
        
        # Sample from Gaussian noise
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=device,
            generator=generator
        )
    
        # Set step values
        scheduler.set_timesteps(self.num_inference_steps)
        
        for t in scheduler.timesteps:
            # Apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # Predict model output using cached encoder outputs
            model_output = self.model.decode_with_cache(
                sample=trajectory,
                timestep=t,
                memory=memory,
                vl_features=vl_features,
                reasoning_features=reasoning_features,
                cond=cond,
                ego_status=ego_status,
                vl_padding_mask=vl_padding_mask,
                reasoning_padding_mask=reasoning_padding_mask
            )

            # Compute previous sample: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
            ).prev_sample
        
        # Enforce conditioning
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype  # Get model dtype
        nobs = dict_apply(obs_dict, lambda x: x.to(device))

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        batch_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))  # (B*To, ...)
        batch_features = self.extract_tcp_features(batch_nobs)  # (B*To, feature_dim)
        feature_dim = batch_features.shape[-1]
        cond = batch_features.reshape(B, To, feature_dim)  # (B, To, feature_dim)
        shape = (B, T, Da)
       
        cond_data = torch.zeros(size=shape, device=device, dtype=model_dtype)  # Use model dtype
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        gen_vit_tokens = nobs['gen_vit_tokens']
        reasoning_query_tokens = nobs['reasoning_query_tokens']
        
        # Process gen_vit_tokens through feature_encoder
        gen_vit_tokens = gen_vit_tokens.to(device=device, dtype=model_dtype)  # Use model dtype
        gen_vit_tokens = self.feature_encoder(gen_vit_tokens)  # Project to 1536 dim
        
        # Process reasoning_query_tokens through feature_encoder
        reasoning_query_tokens = reasoning_query_tokens.to(device=device, dtype=model_dtype)  # Use model dtype
        reasoning_query_tokens = self.feature_encoder(reasoning_query_tokens)  # Project to 1536 dim
        
        # Use current ego_status instead of full history
        ego_status = nobs['ego_status']  # (B, ego_status_dim)
        # Ensure ego_status has the correct dtype
        ego_status = ego_status.to(dtype=model_dtype)
        
        # Get anchor for truncated diffusion (if available)
        anchor = nobs.get('anchor', None)
        if anchor is not None:
            anchor = anchor.to(device=device, dtype=model_dtype)
            # Ensure anchor has correct shape (B, T, 2)
            if anchor.dim() == 2:
                anchor = anchor.unsqueeze(0)  # Add batch dim
            if anchor.shape[0] != B:
                anchor = anchor.expand(B, -1, -1)
        
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            gen_vit_tokens=gen_vit_tokens,
            reasoning_query_tokens=reasoning_query_tokens,
            ego_status=ego_status,
            anchor=anchor
        )
        
        naction_pred = nsample[...,:Da]
        
        # For truncated diffusion, the output is already in original scale
        # For legacy mode, we may need to unnormalize
        if not self.use_truncated_diffusion:
            # Clamp normalized predictions to [-1, 1] to prevent extreme values
            naction_pred = torch.clamp(naction_pred, -1.0, 1.0)
            
            # Unnormalize action predictions back to original range
            if self.enable_action_normalization and self.action_stats is not None:
                naction_pred = self.unnormalize_action(naction_pred)
        
        action_pred = naction_pred.detach().cpu().numpy()
        action = action_pred
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result