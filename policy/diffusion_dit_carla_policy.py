import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Callable, Union
from collections import defaultdict
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from model.transformer_for_diffusion import TransformerForDiffusion
# from model.transformer_for_diffusion_lite import TransformerForDiffusionLite as TransformerForDiffusion
from model.interfuser_bev_encoder import InterfuserBEVEncoder
from model.interfuser_bev_encoder import load_lidar_submodules
import os



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




class DiffusionDiTCarlaPolicy(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        
        # config
        self.cfg = config
        policy_cfg = config['policy']

        obs_as_global_cond = policy_cfg.get('obs_as_global_cond', True)
        self.obs_as_global_cond = obs_as_global_cond
        shape_meta = config['shape_meta']
        action_shape = shape_meta['action']['shape']
        action_dim = action_shape[0]
        
        # Action normalization settings
        self.enable_action_normalization = config.get('enable_action_normalization', True)
        

        self.n_obs_steps = policy_cfg.get('n_obs_steps', config.get('obs_horizon', 1))
        
        # Load BEV encoder configuration from config file
        bev_encoder_cfg = config.get('bev_encoder', {})
        obs_encoder = InterfuserBEVEncoder(
            perception_backbone=None,
            state_dim=bev_encoder_cfg.get('state_dim', 13),  # 修改为13维以支持拼接后的ego_status
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
        
        # Number of waypoints for route prediction
        num_waypoints = policy_cfg.get('num_waypoints', 20)
        self.num_waypoints = num_waypoints

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
            status_dim=status_dim,
            ego_status_seq_len=ego_status_seq_len,
            bev_dim=512,  # BEV token dimension from InterfuserBEVEncoder
            state_token_dim=128,  # State token dimension from InterfuserBEVEncoder
            num_waypoints=num_waypoints,  # Number of route waypoints
        )

        self.model = model
        
        # ========== Truncated Diffusion Configuration (DiffusionDriveV2 style) ==========
        diffusion_cfg = config.get('truncated_diffusion', {})
        self.num_train_timesteps = diffusion_cfg.get('num_train_timesteps', 1000)
        self.trunc_timesteps = diffusion_cfg.get('trunc_timesteps', 8)  # Truncated timestep for anchor during inference
        self.train_trunc_timesteps = diffusion_cfg.get('train_trunc_timesteps', 50)  # Max timestep during training (DiffusionDrive uses 50)
        self.num_diffusion_steps = diffusion_cfg.get('num_diffusion_steps', 2)  # Number of denoising steps
        self.diffusion_eta = diffusion_cfg.get('eta', 1.0)  # 1.0 for stochastic multiplicative noise
        
        # Normalization parameters (DiffusionDrive v1 style: linear mapping to [-1, 1])
        # x: 2*(x + x_offset)/x_range - 1
        # y: 2*(y + y_offset)/y_range - 1
        self.norm_x_offset = diffusion_cfg.get('norm_x_offset', 2.0)  # x range: [-2, 78]
        self.norm_x_range = diffusion_cfg.get('norm_x_range', 80.0)
        self.norm_y_offset = diffusion_cfg.get('norm_y_offset', 20.0)  # y range: [-20, 36]
        self.norm_y_range = diffusion_cfg.get('norm_y_range', 56.0)
        
        # Route prediction auxiliary loss weight (横向控制重要性)
        self.route_loss_weight = diffusion_cfg.get('route_loss_weight', 0.5)
        
        # DDIMScheduler for variance computation (DiffusionDriveV2 style)
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            steps_offset=1,
            beta_schedule="scaled_linear",
            prediction_type="sample",  # Predict clean sample directly
        )

        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.horizon = policy_cfg.get('horizon', 16)
        self.n_action_steps = policy_cfg.get('action_horizon', 8)
    
    # ========== Normalization Functions ==========
    def norm_odo(self, odo_info_fut: torch.Tensor) -> torch.Tensor:
        """
        Normalize trajectory coordinates to [-1, 1] range.
        Following DiffusionDrive v1: 2*(x + offset)/range - 1
        
        For our data (x: [-0.066, 74.045], y: [-17.526, 32.736]):
        - x: 2*(x + 1)/76 - 1, maps [-1, 75] to [-1, 1]
        - y: 2*(y + 18)/52 - 1, maps [-18, 34] to [-1, 1]
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
        Following DiffusionDrive v1: (x + 1)/2 * range - offset
        """
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        
        # Inverse linear mapping from [-1, 1]
        odo_info_fut_x = (odo_info_fut_x + 1) / 2 * self.norm_x_range - self.norm_x_offset
        odo_info_fut_y = (odo_info_fut_y + 1) / 2 * self.norm_y_range - self.norm_y_offset
        
        return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)

    

    def add_multiplicative_noise_scheduled(
        self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, int],
        eta: float = 1.0,
        std_min: float = 0.04
    ) -> torch.Tensor:
        """
        Add multiplicative noise with scheduler-based variance (DiffusionDriveV2 style).
        The noise level is determined by the diffusion scheduler's variance at the given timestep.
        
        DiffusionDriveV2 formula:
            prev_sample = prev_sample_mean * variance_noise_mul + std_dev_t_add * variance_noise_add
        
        When eta > 0:
            - std_dev_t_mul = clip(std_dev_t, min=0.04) for multiplicative noise
            - std_dev_t_add = 0.0 (no additive noise)
        
        Multiplicative noise is applied separately to x (horizon) and y (vert) directions,
        then combined: sample * noise_mul
        
        Args:
            sample: (B, T, 2) normalized trajectory
            timestep: current diffusion timestep (scalar or tensor)
            eta: scaling factor for variance (0.0 = deterministic, 1.0 = full stochasticity)
            std_min: minimum standard deviation to prevent zero noise (V2 uses 0.04)
            
        Returns:
            Noisy sample with timestep-scheduled multiplicative noise applied
        """
        device = sample.device
        dtype = sample.dtype
        bs = sample.shape[0]
        T = sample.shape[1]  # trajectory length (num_points)
        
        # Get timestep as integer
        if torch.is_tensor(timestep):
            t = timestep.item() if timestep.numel() == 1 else timestep[0].item()
        else:
            t = timestep
        t = int(t)
        
        # Compute variance from scheduler (DDIM style)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        prev_t = t - self.num_train_timesteps // max(self.num_diffusion_steps, 1)
        prev_t = max(prev_t, 0)
        
        alpha_prod_t = self.diffusion_scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.diffusion_scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.diffusion_scheduler.final_alpha_cumprod
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Variance formula from DDIM
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        variance = max(variance.item(), 1e-10)
        
        # std_dev_t with eta scaling
        std_dev_t = eta * (variance ** 0.5)
        
        # DiffusionDriveV2 style: std_dev_t_mul = clip(std_dev_t, min=0.04)
        std_dev_t_mul = max(std_dev_t, std_min)
        
        # Generate multiplicative noise for horizon (x) and vert (y) separately
        # DiffusionDriveV2: variance_noise_horizon/vert shape is (B, G, 1, 1), then repeat
        # Our shape: (B, 1, 1) for horizon and vert, then cat to (B, 1, 2), then repeat to (B, T, 2)
        
        # variance_noise_horizon = randn * std_dev_t_mul + 1.0  (for x direction)
        variance_noise_horizon = torch.randn([bs, 1, 1], device=device, dtype=dtype) * std_dev_t_mul + 1.0
        # variance_noise_vert = randn * std_dev_t_mul + 1.0  (for y direction)
        variance_noise_vert = torch.randn([bs, 1, 1], device=device, dtype=dtype) * std_dev_t_mul + 1.0
        
        # Concatenate horizon and vert: (B, 1, 1) + (B, 1, 1) -> (B, 1, 2)
        variance_noise_mul = torch.cat([variance_noise_horizon, variance_noise_vert], dim=-1)
        
        # Repeat across trajectory length: (B, 1, 2) -> (B, T, 2)
        variance_noise_mul = variance_noise_mul.expand(-1, T, -1)
        
        # Apply multiplicative noise: sample * variance_noise_mul
        # This matches DiffusionDriveV2: prev_sample = prev_sample_mean * variance_noise_mul
        # (when std_dev_t_add = 0, the additive term is zero)
        noisy_sample = sample * variance_noise_mul
        
        return noisy_sample

    def add_multiplicative_noise_scheduled_batch(
        self, 
        sample: torch.Tensor, 
        timesteps: torch.Tensor,
        eta: float = 1.0,
        std_min: float = 0.04
    ) -> torch.Tensor:
        """
        Add multiplicative noise with per-sample timesteps (batch version).
        Each sample in the batch gets noise corresponding to its own timestep.
        
        This is the correct implementation for training where each sample should have
        noise added according to its own sampled timestep.
        
        Args:
            sample: (B, T, 2) normalized trajectory
            timesteps: (B,) tensor of timesteps, one per sample
            eta: scaling factor for variance (0.0 = deterministic, 1.0 = full stochasticity)
            std_min: minimum standard deviation to prevent zero noise (V2 uses 0.04)
            
        Returns:
            Noisy sample with per-sample timestep-scheduled multiplicative noise applied
        """
        device = sample.device
        dtype = sample.dtype
        bs = sample.shape[0]
        T = sample.shape[1]  # trajectory length (num_points)
        
        # Compute variance for each sample based on its timestep
        # Pre-compute alpha_cumprod values on CPU then move to device
        alphas_cumprod = self.diffusion_scheduler.alphas_cumprod
        
        # Get prev_t for each sample
        step_ratio = self.num_train_timesteps // max(self.num_diffusion_steps, 1)
        prev_timesteps = (timesteps - step_ratio).clamp(min=0)
        
        # Gather alpha_prod values for each sample
        alpha_prod_t = alphas_cumprod[timesteps.cpu()].to(device=device, dtype=dtype)  # (B,)
        alpha_prod_t_prev = alphas_cumprod[prev_timesteps.cpu()].to(device=device, dtype=dtype)  # (B,)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Variance formula from DDIM: (B,)
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        variance = variance.clamp(min=1e-10)
        
        # std_dev_t with eta scaling: (B,)
        std_dev_t = eta * (variance ** 0.5)
        
        # DiffusionDriveV2 style: std_dev_t_mul = clip(std_dev_t, min=0.04)
        std_dev_t_mul = std_dev_t.clamp(min=std_min)  # (B,)
        
        # Reshape for broadcasting: (B,) -> (B, 1, 1)
        std_dev_t_mul = std_dev_t_mul.view(bs, 1, 1)
        
        # Generate multiplicative noise for horizon (x) and vert (y) separately
        # variance_noise_horizon = randn * std_dev_t_mul + 1.0  (for x direction)
        variance_noise_horizon = torch.randn([bs, 1, 1], device=device, dtype=dtype) * std_dev_t_mul + 1.0
        # variance_noise_vert = randn * std_dev_t_mul + 1.0  (for y direction)
        variance_noise_vert = torch.randn([bs, 1, 1], device=device, dtype=dtype) * std_dev_t_mul + 1.0
        
        # Concatenate horizon and vert: (B, 1, 1) + (B, 1, 1) -> (B, 1, 2)
        variance_noise_mul = torch.cat([variance_noise_horizon, variance_noise_vert], dim=-1)
        
        # Repeat across trajectory length: (B, 1, 2) -> (B, T, 2)
        variance_noise_mul = variance_noise_mul.expand(-1, T, -1)
        
        # Apply multiplicative noise: sample * variance_noise_mul
        noisy_sample = sample * variance_noise_mul
        
        return noisy_sample



    

    def extract_bev_state_tokens(self, obs_dict):
        """
        使用InterfuserBEVEncoder提取BEV tokens和State tokens (Decoder-Only架构)
        
        支持两种模式：
        1. 使用预处理好的BEV特征（推荐，快速）
        2. 使用原始lidar_bev图像（兼容模式，慢）
        
        Args:
            obs_dict: 观测字典，应包含：
                - 'lidar_token': (B, seq_len, 512) 预处理的空间特征，或
                - 'lidar_token_global': (B, 1, 512) 预处理的全局特征，或
                - 'lidar_bev': (B, 3, 448, 448) 原始BEV图像（兼容模式）
                - 'ego_status' (B, state_dim): 状态向量
                
        Returns:
            bev_tokens: (B, seq_len+1, 512) - BEV空间tokens + global token
            state_tokens: (B, 1, 128) - 状态编码tokens
        """
        try:
            device = next(self.parameters()).device
            model_dtype = next(self.parameters()).dtype
            state = obs_dict['ego_status'].to(device=device, dtype=model_dtype)
            
            use_precomputed = 'lidar_token' in obs_dict and 'lidar_token_global' in obs_dict
            
            if use_precomputed:
                lidar_token = obs_dict['lidar_token'].to(device=device, dtype=model_dtype)
                lidar_token_global = obs_dict['lidar_token_global'].to(device=device, dtype=model_dtype)
                
                bev_tokens, state_tokens = self.obs_encoder(
                    state=state,
                    lidar_token=lidar_token,
                    lidar_token_global=lidar_token_global,
                )
            else:
                if 'lidar_bev' not in obs_dict:
                    raise KeyError("Neither pre-computed features (lidar_token, lidar_token_global) nor raw BEV images (lidar_bev) found in obs_dict")
                
                lidar_bev_img = obs_dict['lidar_bev'].to(device=device, dtype=model_dtype)
                
                bev_tokens, state_tokens = self.obs_encoder(
                    image=lidar_bev_img,
                    state=state,
                )
            
            return bev_tokens, state_tokens
                
        except KeyError as e:
            raise KeyError(f"Missing required field in obs_dict for BEV/State token extraction: {e}")
        except Exception as e:
            raise RuntimeError(f"Error in TCP feature extraction: {e}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward method for DDP compatibility.
        DDP only synchronizes gradients when forward() is called, not for other methods.
        This method simply calls compute_loss() to enable proper gradient synchronization
        in distributed training.
        """
        return self.compute_loss(batch)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        batch: {
            # BEV特征（二选一）
            'lidar_token': (B, obs_horizon, seq_len, 512) - 预处理的空间特征（推荐）
            'lidar_token_global': (B, obs_horizon, 1, 512) - 预处理的全局特征（推荐）
            'lidar_bev': (B, obs_horizon, 3, 448, 448) - 原始LiDAR BEV图像（兼容模式）
            
            'agent_pos': (B, horizon, 2) - 未来轨迹点
            'ego_status': (B, obs_horizon, state_dim) - 车辆状态
            'anchor': (B, horizon, 2) - anchor轨迹点（用于truncated diffusion）
            'route': (B, num_waypoints, 2) - 路线waypoints（可选，用于route预测辅助任务）
        }
        """
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        nobs = {}
        required_fields = ['lidar_token', 'lidar_token_global', 'lidar_bev', 'ego_status', 'agent_pos', 'reasoning_query_tokens', 'gen_vit_tokens', 'anchor', 'route']
        
        for field in required_fields:
            if field in batch:
                if field in ['lidar_bev', 'lidar_token', 'lidar_token_global']:
                    nobs[field] = batch[field].to(device=device, dtype=model_dtype)
                else:
                    nobs[field] = batch[field].to(device)

        raw_agent_pos = batch['agent_pos'].to(device)

        To = self.n_obs_steps
        nactions = raw_agent_pos
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        
        # Get ground truth trajectory
        trajectory = nactions.to(dtype=model_dtype)  # (B, horizon, 2)
        
        # Get anchor trajectory for truncated diffusion
        anchor = batch.get('anchor', None)
        if anchor is not None:
            anchor = anchor.to(device=device, dtype=model_dtype)  # (B, horizon, 2)
        
        # Get route ground truth for auxiliary task
        route_gt = batch.get('route', None)
        if route_gt is not None:
            route_gt = route_gt.to(device=device, dtype=model_dtype)  # (B, num_waypoints, 2)
        
        # Extract BEV tokens and State tokens (Decoder-Only架构)
        # Flatten temporal dimension for feature extraction
        batch_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))  # (B*To, ...)
        bev_tokens, state_tokens = self.extract_bev_state_tokens(batch_nobs)
        
        # Use only the last timestep's features (or average over time)
        # bev_tokens: (B*To, seq_len+1, 512) -> use last timestep
        # state_tokens: (B*To, 1, 128) -> use last timestep
        bev_tokens = bev_tokens.reshape(batch_size, To, -1, 512)[:, -1, :, :]  # (B, seq_len+1, 512)
        state_tokens = state_tokens.reshape(batch_size, To, 1, 128)[:, -1, :, :]  # (B, 1, 128)

        # Prepare VL tokens
        gen_vit_tokens = batch['gen_vit_tokens']
        reasoning_query_tokens = batch['reasoning_query_tokens']
        gen_vit_tokens = gen_vit_tokens.to(device=device, dtype=model_dtype)
        gen_vit_tokens = self.feature_encoder(gen_vit_tokens)
        reasoning_query_tokens = reasoning_query_tokens.to(device=device, dtype=model_dtype)
        reasoning_query_tokens = self.feature_encoder(reasoning_query_tokens)
        
        # Get ego_status
        ego_status = nobs['ego_status'].to(dtype=model_dtype)

        # ========== Compute Loss (Truncated Diffusion DiffusionDriveV2 style) ==========
        loss = self._compute_truncated_diffusion_loss(
            trajectory=trajectory,
            anchor=anchor,
            bev_tokens=bev_tokens,
            state_tokens=state_tokens,
            gen_vit_tokens=gen_vit_tokens,
            reasoning_query_tokens=reasoning_query_tokens,
            ego_status=ego_status,
            route_gt=route_gt,
            device=device,
            model_dtype=model_dtype
        )
        
        return loss
    
    def _compute_truncated_diffusion_loss(
        self,
        trajectory: torch.Tensor,
        anchor: torch.Tensor,
        bev_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        gen_vit_tokens: torch.Tensor,
        reasoning_query_tokens: torch.Tensor,
        ego_status: torch.Tensor,
        device: torch.device,
        model_dtype: torch.dtype,
        route_gt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss using truncated diffusion (DiffusionDriveV2 style).
        Instead of starting from pure noise, we start from anchor with multiplicative noise.
        The model predicts the clean sample directly.
        
        Training: Add scheduler-based multiplicative noise (noise level depends on timestep)
        
        Args:
            trajectory: (B, horizon, 2) - ground truth trajectory
            anchor: (B, horizon, 2) - anchor trajectory for truncated diffusion
            bev_tokens: (B, seq_len+1, 512) - BEV spatial tokens from encoder
            state_tokens: (B, 1, 128) - state tokens from encoder
            gen_vit_tokens: (B, seq_len, dim) - VIT tokens
            reasoning_query_tokens: (B, seq_len, dim) - reasoning tokens
            ego_status: (B, To, status_dim) - ego vehicle status
            device: torch device
            model_dtype: model dtype (e.g., bfloat16)
            route_gt: (B, num_waypoints, 2) - optional ground truth route for auxiliary loss
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
        
        # 3. Add scheduler-based multiplicative noise to anchor (DiffusionDriveV2 style)
        noisy_anchor = self.add_multiplicative_noise_scheduled_batch(
            anchor_norm,
            timesteps=timesteps,
            eta=1.0,
            std_min=0.04
        )
        
        # 4. Clamp to valid range
        noisy_anchor = torch.clamp(noisy_anchor, min=-1, max=1)
        
        # 5. Denormalize for model input (model expects denormalized coordinates)
        noisy_trajectory_denorm = self.denorm_odo(noisy_anchor)
        
        # 6. Create dummy cond for API compatibility (not used in decoder-only)
        cond = torch.zeros(batch_size, ego_status.shape[1], 256, device=device, dtype=model_dtype)
        
        # 7. Predict clean sample and route using decoder-only model
        pred, route_pred = self.model(
            sample=noisy_trajectory_denorm,
            timestep=timesteps,
            cond=cond,  # For API compatibility
            bev_tokens=bev_tokens,
            state_tokens=state_tokens,
            gen_vit_tokens=gen_vit_tokens,
            reasoning_query_tokens=reasoning_query_tokens,
            ego_status=ego_status
        )
        
        # 8. Compute trajectory loss - predict clean sample (not noise)
        target = trajectory
        
        traj_loss = F.l1_loss(pred, target, reduction='none')
        
        if traj_loss.shape[-1] > 2:
            traj_loss = traj_loss[..., :2]
        
        traj_loss = reduce(traj_loss, 'b ... -> b (...)', 'mean')
        traj_loss = traj_loss.mean()
        
        # 9. Compute route loss (auxiliary task)
        total_loss = traj_loss
        if route_gt is not None:
            route_loss = F.l1_loss(route_pred, route_gt, reduction='none')
            route_loss = reduce(route_loss, 'b ... -> b (...)', 'mean')
            route_loss = route_loss.mean()
            
            route_loss_weight = getattr(self, 'route_loss_weight', 0.1)
            total_loss = traj_loss + route_loss_weight * route_loss
        
        return total_loss

    def conditional_sample(self, 
            bev_tokens: torch.Tensor,
            state_tokens: torch.Tensor,
            gen_vit_tokens: torch.Tensor,
            reasoning_query_tokens: torch.Tensor,
            ego_status: torch.Tensor,
            anchor: torch.Tensor,
            device: torch.device,
            model_dtype: torch.dtype,
            generator=None,
            **kwargs
            ):
        """
        Generate trajectory samples using truncated diffusion (DiffusionDriveV2 style).
        
        Args:
            bev_tokens: (B, seq_len+1, 512) - BEV spatial tokens
            state_tokens: (B, 1, 128) - state tokens
            gen_vit_tokens: (B, seq_len, 1536) - VL tokens
            reasoning_query_tokens: (B, seq_len, 1536) - reasoning tokens
            ego_status: (B, To, status_dim) - ego status history
            anchor: (B, T, 2) - anchor trajectory
            
        Returns:
            (trajectory, route_pred) tuple - trajectory (B, T, 2), route_pred (B, 20, 2)
        """
        result = self._truncated_diffusion_sample(
            anchor=anchor,
            bev_tokens=bev_tokens,
            state_tokens=state_tokens,
            ego_status=ego_status,
            gen_vit_tokens=gen_vit_tokens,
            reasoning_query_tokens=reasoning_query_tokens,
            device=device,
            model_dtype=model_dtype,
            generator=generator
        )

        return result
    
    def _truncated_diffusion_sample(
        self,
        anchor: torch.Tensor,
        bev_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        ego_status: torch.Tensor,
        gen_vit_tokens: torch.Tensor,
        reasoning_query_tokens: torch.Tensor,
        device: torch.device,
        model_dtype: torch.dtype,
        generator=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Truncated diffusion sampling (DiffusionDriveV2 style with multiplicative noise).
        Start from anchor with multiplicative noise, denoise for few steps.
        
        Key insight:
        - The model predicts the CLEAN trajectory directly (not noise, not residual)
        - Uses multiplicative noise with scheduler-based variance (timestep-dependent)
        - Final output is the model's direct prediction
            
        Returns:
            (trajectory, route_pred) tuple - trajectory (B, T, 2), route_pred (B, 20, 2)
        """
        bs = anchor.shape[0]
        
        # Set up scheduler
        self.diffusion_scheduler.set_timesteps(self.num_train_timesteps, device)
        
        # Compute rollout timesteps
        step_ratio = 20 / self.num_diffusion_steps
        roll_timesteps = (np.arange(0, self.num_diffusion_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)
        
        # Create dummy cond for API compatibility
        cond = torch.zeros(bs, ego_status.shape[1], 256, device=device, dtype=model_dtype)
        
        # 1. Normalize anchor
        diffusion_output = self.norm_odo(anchor)  # (B, T, 2)
        
        # 2. Add initial multiplicative noise using truncated timestep (scheduler-based)
        diffusion_output = self.add_multiplicative_noise_scheduled(
            diffusion_output, 
            timestep=self.trunc_timesteps,
            eta=1.0,
            std_min=0.04
        )
        
        # 3. Denoising loop
        pred = None
        route_pred = None
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
            
            # Predict clean sample using decoder-only model
            pred, route_pred = self.model(
                sample=noisy_traj_points.to(dtype=model_dtype),
                timestep=timesteps,
                cond=cond,
                bev_tokens=bev_tokens,
                state_tokens=state_tokens,
                gen_vit_tokens=gen_vit_tokens,
                reasoning_query_tokens=reasoning_query_tokens,
                ego_status=ego_status,
            )
            
            # For next iteration, use the normalized prediction as input
            x_start = self.norm_odo(pred)  # (B, T, 2)
            
            # Add noise for next iteration based on the next timestep
            if i < len(roll_timesteps) - 1:
                next_k = roll_timesteps[i + 1]
                diffusion_output = self.add_multiplicative_noise_scheduled(
                    x_start,
                    timestep=next_k,
                    eta=1.0,
                    std_min=0.02
                )
            else:
                diffusion_output = x_start
        
        # 4. Return the model's direct prediction
        return pred, route_pred

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        nobs = dict_apply(obs_dict, lambda x: x.to(device))

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # Extract BEV tokens and State tokens
        batch_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))  # (B*To, ...)
        bev_tokens, state_tokens = self.extract_bev_state_tokens(batch_nobs)
        
        # Use only the last timestep's features
        bev_tokens = bev_tokens.reshape(B, To, -1, 512)[:, -1, :, :]  # (B, seq_len+1, 512)
        state_tokens = state_tokens.reshape(B, To, 1, 128)[:, -1, :, :]  # (B, 1, 128)

        # Process VL tokens
        gen_vit_tokens = nobs['gen_vit_tokens']
        reasoning_query_tokens = nobs['reasoning_query_tokens']
        
        gen_vit_tokens = gen_vit_tokens.to(device=device, dtype=model_dtype)
        gen_vit_tokens = self.feature_encoder(gen_vit_tokens)
        
        reasoning_query_tokens = reasoning_query_tokens.to(device=device, dtype=model_dtype)
        reasoning_query_tokens = self.feature_encoder(reasoning_query_tokens)
        
        # Get ego_status
        ego_status = nobs['ego_status']
        ego_status = ego_status.to(dtype=model_dtype)
        
        # Get anchor for truncated diffusion
        anchor = nobs.get('anchor', None)
        if anchor is not None:
            anchor = anchor.to(device=device, dtype=model_dtype)
            if anchor.dim() == 2:
                anchor = anchor.unsqueeze(0)
            if anchor.shape[0] != B:
                anchor = anchor.expand(B, -1, -1)
        
        # Generate samples using truncated diffusion
        nsample, route_pred = self.conditional_sample(
            bev_tokens=bev_tokens,
            state_tokens=state_tokens,
            gen_vit_tokens=gen_vit_tokens,
            reasoning_query_tokens=reasoning_query_tokens,
            ego_status=ego_status,
            anchor=anchor,
            device=device,
            model_dtype=model_dtype,
        )
        
        naction_pred = nsample[...,:Da]
        
        # Convert to float32 before numpy
        action_pred = naction_pred.detach().float().cpu().numpy()
        action = action_pred
        result = {
            'action': action,
            'action_pred': action_pred,
            'route_pred': route_pred,
        }
        
        return result