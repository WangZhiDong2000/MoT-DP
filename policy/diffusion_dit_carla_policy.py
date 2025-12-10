import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Callable
from collections import defaultdict
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from model.transformer_for_diffusion import TransformerForDiffusion, LowdimMaskGenerator
from model.interfuser_bev_encoder import InterfuserBEVEncoder
from model.interfuser_bev_encoder import load_lidar_submodules
import os
from collections import OrderedDict
from collections import deque

VLMDriveBackbone = None
VLM_AVAILABLE = False

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
        }
        """
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype  # Get model dtype
        nobs = {}
        required_fields = ['lidar_token', 'lidar_token_global', 'lidar_bev', 'ego_status', 'agent_pos', 'reasoning_query_tokens', 'gen_vit_tokens']
        
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
        
        # Normalize trajectory for training
        trajectory = nactions.to(dtype=model_dtype)  # Use model dtype instead of .float()
        if self.enable_action_normalization and self.action_stats is not None:
            trajectory = self.normalize_action(trajectory)
        if torch.isnan(trajectory).any() or torch.isinf(trajectory).any():
            print("Warning: NaN or Inf detected in normalized trajectory")
            trajectory = torch.nan_to_num(trajectory, nan=0.0, posinf=1.0, neginf=-1.0)

        batch_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))  # (B*To, ...)
        batch_features = self.extract_tcp_features(batch_nobs)  # (B*To, feature_dim)
        feature_dim = batch_features.shape[-1]
        cond = batch_features.reshape(batch_size, To, feature_dim)  # Already in model_dtype, no need for .float()  

        condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.get('num_train_timesteps', 100), 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        gen_vit_tokens = batch['gen_vit_tokens']
        reasoning_query_tokens = batch['reasoning_query_tokens']
        
        # Process gen_vit_tokens through feature_encoder
        gen_vit_tokens = gen_vit_tokens.to(device=device, dtype=model_dtype)  # Use model dtype
        gen_vit_tokens = self.feature_encoder(gen_vit_tokens)  # Project to 1536 dim
        
        # Process reasoning_query_tokens through feature_encoder
        reasoning_query_tokens = reasoning_query_tokens.to(device=device, dtype=model_dtype)  # Use model dtype
        reasoning_query_tokens = self.feature_encoder(reasoning_query_tokens)  # Project to 1536 dim
        
        # Use current ego_status instead of full history
        ego_status = nobs['ego_status']  # (B, 13)
        # Ensure ego_status has the correct dtype
        ego_status = ego_status.to(dtype=model_dtype)
        
        pred = self.model(noisy_trajectory, timesteps, cond, gen_vit_tokens=gen_vit_tokens, reasoning_query_tokens=reasoning_query_tokens, ego_status=ego_status)

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
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler
        device = condition_data.device
        bs = condition_data.shape[0]

        # 直接从高斯分布采样
        trajectory = torch.randn(
                size=condition_data.shape, 
                dtype=condition_data.dtype,
                device=device,
                generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        
        # ===== OPTIMIZATION: Cache encoder outputs =====
        # Encoder output doesn't depend on timestep or trajectory,
        # so we only compute it once and reuse for all diffusion steps
        with torch.no_grad():
            memory, vl_features, reasoning_features, vl_padding_mask, reasoning_padding_mask = \
                model.encode_conditions(
                    cond=cond,
                    gen_vit_tokens=gen_vit_tokens,
                    reasoning_query_tokens=reasoning_query_tokens
                )

        
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output using cached encoder outputs
            
            model_output = model.decode_with_cache(
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

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
       
        
        # finally make sure conditioning is enforced
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
        ego_status = nobs['ego_status']  # (B, 14)
        # Ensure ego_status has the correct dtype
        ego_status = ego_status.to(dtype=model_dtype)
        
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            gen_vit_tokens=gen_vit_tokens,
            reasoning_query_tokens=reasoning_query_tokens,
            ego_status=ego_status
            )
        
        naction_pred = nsample[...,:Da]
        
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

