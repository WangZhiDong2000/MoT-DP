import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Callable
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from model.transformer_for_diffusion import TransformerForDiffusion, LowdimMaskGenerator
from model.interfuser_bev_encoder import InterfuserBEVEncoder
from model.interfuser_bev_encoder import load_lidar_submodules
import os

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
        self.obs_encoder.cuda()

        # TODO load vlm and vlm encoder model）
        self.vlm_backbone = None
        self.feature_encoder = None
        
        if VLM_AVAILABLE and VLMDriveBackbone is not None:
            try:
                vlm_device = 'cpu'  
                self.vlm_backbone = VLMDriveBackbone(
                    model_type='qwen',
                    checkpoint_path='Qwen/Qwen2.5-VL-3B-Instruct',
                    device=vlm_device
                )
                print("✓ VLM backbone initialized successfully on CPU")
            except Exception as e:
                print(f"⚠ VLM backbone initialization failed: {e}")
                self.vlm_backbone = None
        else:
            print("⚠ VLM backbone not available, using simulated features")
        self._init_fixed_vlm_features()

        obs_feature_dim = 256  

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
            n_cond_layers=policy_cfg.get('n_cond_layers', 4)
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

        # anchor
        plan_anchor_path = '/root/z_projects/code/MoT-DP-1/data/kmeans/kmeans_plan_vocab_6.npy'
        plan_anchor = np.load(plan_anchor_path)

        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) # 6,8,2

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self.enable_action_normalization or self.action_stats is None:
            return action
        
        device = action.device
        action_min = self.action_stats['min'].to(device)
        action_max = self.action_stats['max'].to(device)   
        normalized = (action - action_min) / (action_max - action_min + 1e-8)
        normalized = normalized * 2 - 1
        return normalized

    def unnormalize_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        if not self.enable_action_normalization or self.action_stats is None:
            return normalized_action
        
        device = normalized_action.device
        action_min = self.action_stats['min'].to(device)
        action_max = self.action_stats['max'].to(device)
        unnormalized = (normalized_action + 1) / 2
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
            state = obs_dict['ego_status'].to(device='cuda', dtype=torch.float32)  # (B, 13)
            use_precomputed = 'lidar_token' in obs_dict and 'lidar_token_global' in obs_dict
            if use_precomputed:
                lidar_token = obs_dict['lidar_token'].to(device='cuda', dtype=torch.float32)
                lidar_token_global = obs_dict['lidar_token_global'].to(device='cuda', dtype=torch.float32)
                
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
                
                lidar_bev_img = obs_dict['lidar_bev'].to(device='cuda', dtype=torch.float32)
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
            
            # nuScenes必需字段
            'agent_pos': (B, horizon, 2) - 未来轨迹点
            'ego_status': (B, obs_horizon, 13) - 车辆状态 [accel(3), rot_rate(3), vel(3), steer(1), command(3)]
        }
        """
        device = next(self.parameters()).device
        nobs = {}
        required_fields = ['lidar_token', 'lidar_token_global', 'lidar_bev', 'ego_status', 'agent_pos']
        
        for field in required_fields:
            if field in batch:
                if field in ['lidar_bev', 'lidar_token', 'lidar_token_global']:
                    nobs[field] = batch[field].to(device=device, dtype=torch.float32)
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
        trajectory = nactions.float()
        if self.enable_action_normalization and self.action_stats is not None:
            trajectory = self.normalize_action(trajectory)
            if torch.isnan(trajectory).any() or torch.isinf(trajectory).any():
                print("Warning: NaN or Inf detected in normalized trajectory")
                trajectory = torch.nan_to_num(trajectory, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.obs_as_global_cond:
            batch_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))  # (B*To, ...)
            batch_features = self.extract_tcp_features(batch_nobs)  # (B*To, feature_dim)
            feature_dim = batch_features.shape[-1]
            cond = batch_features.reshape(batch_size, To, feature_dim).float()  # (B, To, feature_dim)  
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.extract_tcp_features(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)

        condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)

        # Sample noise that we'll add to the images
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
        vqa = batch.get('vqa', None)
        if vqa is None:
            # 如果vqa为None或不存在，使用初始化中生成的模板
            vl_features, vl_mask = self.generate_simulated_vlm_outputs(
                batch_size=noisy_trajectory.shape[0], 
                device=device,
                max_seq_len=None
            )
        else:
            vl_features = vqa.to(device=device, dtype=torch.float32)  # (B, seq_len, feat_dim)
            vl_mask = torch.ones(vl_features.shape[:2], dtype=torch.bool, device=device)
        vl_embeds = self.feature_encoder(vl_features)
        pred = self.model(noisy_trajectory, timesteps, vl_embeds, cond, vl_mask=vl_mask)

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
            cond=None, generator=None, vl_features=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        
        # Use provided vl_features or generate simulated ones
        if vl_features is None:
            vl_features, vl_mask = self.generate_simulated_vlm_outputs(trajectory.shape[0], trajectory.device)
        else:
            # vl_features provided, create mask for valid positions
            vl_mask = torch.ones(vl_features.shape[:2], dtype=torch.bool, device=vl_features.device)
        
        vl_embeds = self.feature_encoder(vl_features)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, vl_embeds, cond, vl_mask=vl_mask)


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
        if self.obs_as_global_cond:
            batch_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))  # (B*To, ...)
            batch_features = self.extract_tcp_features(batch_nobs)  # (B*To, feature_dim)
            feature_dim = batch_features.shape[-1]
            cond = batch_features.reshape(B, To, feature_dim)  # (B, To, feature_dim)
            shape = (B, T, Da)
           
            cond_data = torch.zeros(size=shape, device=device, dtype=torch.float32)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            batch_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))  # (B*To, ...)
            batch_features = self.extract_tcp_features(batch_nobs)  # (B*To, feature_dim)
            feature_dim = batch_features.shape[-1]
            nobs_features = batch_features.reshape(B, To, feature_dim)  # (B, To, feature_dim)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=torch.float32)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        # Extract VQA features from obs_dict if available
        if 'vqa' not in nobs or nobs['vqa'] is None:
            vl_feat, _ = self.generate_simulated_vlm_outputs(
                batch_size=B, 
                device=device,
                max_seq_len=None
            )
        else:
            vl_feat = nobs['vqa'].to(dtype=torch.float32)
        
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            vl_features=vl_feat
            )
        
        naction_pred = nsample[...,:Da]
        
        # Clamp normalized predictions to [-1, 1] to prevent extreme values
        naction_pred = torch.clamp(naction_pred, -1.0, 1.0)
        
        # Unnormalize action predictions back to original range
        if self.enable_action_normalization and self.action_stats is not None:
            naction_pred = self.unnormalize_action(naction_pred)
            
            # Additional safety clamp after unnormalization to prevent extreme outliers
            device = naction_pred.device
            action_min = self.action_stats['min'].to(device)
            action_max = self.action_stats['max'].to(device)
            
            # Expand to reasonable bounds (20% beyond training range)
            safety_margin = 0.2
            range_size = action_max - action_min
            expanded_min = action_min - safety_margin * range_size
            expanded_max = action_max + safety_margin * range_size
            
            naction_pred = torch.clamp(naction_pred, expanded_min, expanded_max)
        
        action_pred = naction_pred.detach().cpu().numpy()
        # 直接返回整个预测序列，因为horizon=action_horizon
        action = action_pred
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def extract_attention_map(self, obs_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        device = next(self.parameters()).device
        nobs = dict_apply(obs_dict, lambda x: x.to(device))
        last_obs = dict_apply(nobs, lambda x: x[:, -1, ...])
        with torch.no_grad():
            _, attention_map = self.extract_tcp_features(last_obs, return_attention=True)
        attention_map_np = attention_map[0].cpu().numpy()  # (H, W)
        
        return attention_map_np

    def predict_action_with_steps(self, obs_dict: Dict[str, torch.Tensor]):
  
        device = next(self.parameters()).device
        nobs = dict_apply(obs_dict, lambda x: x.to(device))

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # 准备条件
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_global_cond:
            batch_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))  # (B*To, ...)
            batch_features = self.extract_tcp_features(batch_nobs)  # (B*To, feature_dim)
            feature_dim = batch_features.shape[-1]
            cond = batch_features.reshape(B, To, feature_dim)  # (B, To, feature_dim)
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=torch.float32)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            batch_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))  # (B*To, ...)
            batch_features = self.extract_tcp_features(batch_nobs)  # (B*To, feature_dim)
            feature_dim = batch_features.shape[-1]
            nobs_features = batch_features.reshape(B, To, feature_dim)  # (B, To, feature_dim)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=torch.float32)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True


        if 'vqa' not in nobs or nobs['vqa'] is None:
            vl_feat, _ = self.generate_simulated_vlm_outputs(
                batch_size=B, 
                device=device,
                max_seq_len=None
            )
        else:
            vl_feat = nobs['vqa'].to(dtype=torch.float32)
        
        nsample, denoising_steps = self.conditional_sample_with_steps(
            cond_data, 
            cond_mask,
            cond=cond,
            vl_features=vl_feat
        )
        
        naction_pred = nsample[...,:Da]
        naction_pred = torch.clamp(naction_pred, -1.0, 1.0)
        

        denoising_steps_unnormalized = []
        if self.enable_action_normalization and self.action_stats is not None:
            for step in denoising_steps:
                step_actions = step[..., :Da]
                step_actions_clamped = torch.clamp(step_actions, -1.0, 1.0)
                step_actions_unnorm = self.unnormalize_action(step_actions_clamped)
                denoising_steps_unnormalized.append(step_actions_unnorm)
            
            # 反归一化最终预测
            naction_pred = self.unnormalize_action(naction_pred)
            device = naction_pred.device
            action_min = self.action_stats['min'].to(device)
            action_max = self.action_stats['max'].to(device)
            safety_margin = 0.2
            range_size = action_max - action_min
            expanded_min = action_min - safety_margin * range_size
            expanded_max = action_max + safety_margin * range_size
            naction_pred = torch.clamp(naction_pred, expanded_min, expanded_max)
        else:
            denoising_steps_unnormalized = [step[..., :Da] for step in denoising_steps]
        
        action_pred = naction_pred.detach().cpu().numpy()
        action = action_pred
        result = {
            'action': action,
            'action_pred': action_pred
        }
        
        # 转换去噪步骤为numpy（已经反归一化）
        denoising_steps_np = [step[0].detach().cpu().numpy() for step in denoising_steps_unnormalized]
        
        return result, denoising_steps_np

    def conditional_sample_with_steps(self, 
            condition_data, condition_mask,
            cond=None, generator=None, vl_features=None,
            **kwargs):
        """
        条件采样并返回中间去噪步骤
        噪声加在归一化后的plan_anchor上（参考transfuser_model_v2.py的forward_train方法）
        """
        model = self.model
        scheduler = self.noise_scheduler
        device = condition_data.device
        bs = condition_data.shape[0]

        # 1. 准备 plan_anchor：加噪声到归一化后的 plan_anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)  # (bs, 6, 8, 2)
        odo_info_fut = self.normalize_action(plan_anchor)  # 归一化到[-1, 1]
        
        # 添加噪声到归一化后的 plan_anchor
        noise = torch.randn(odo_info_fut.shape, device=device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.get('num_train_timesteps', 100),
            (bs,), device=device
        ).long()
        noisy_trajectory = scheduler.add_noise(
            original_samples=odo_info_fut,
            noise=noise,
            timesteps=timesteps
        ).float()
        noisy_trajectory = torch.clamp(noisy_trajectory, min=-1, max=1)
        trajectory = self.unnormalize_action(noisy_trajectory)  # 反归一化回原始范围
    
        scheduler.set_timesteps(self.num_inference_steps)
        
        # Use provided vl_features or generate simulated ones
        if vl_features is None:
            print("No VLM features provided, generating simulated features...")
            vl_features, vl_mask = self.generate_simulated_vlm_outputs(trajectory.shape[0], trajectory.device)
        else:
            # vl_features provided, create mask for valid positions
            vl_mask = torch.ones(vl_features.shape[:2], dtype=torch.bool, device=vl_features.device)
        
        vl_embeds = self.feature_encoder(vl_features)

        # 保存去噪步骤
        denoising_steps = []
        
        total_steps = len(scheduler.timesteps)
        for i, t in enumerate(scheduler.timesteps):
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, vl_embeds, cond, vl_mask=vl_mask)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
            
        
        # 最终结果
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory, denoising_steps

    # ========= VLM feature simulati, temporary! TODO   ============

    def _init_loaded_vlm_features(self):
        """
        从预先保存的文件中加载VLM特征
        """
        print("Loading VLM features from file...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        template_path = os.path.join(project_root, 'fixed_vlm_template.pt')
        if os.path.exists(template_path):
            self.fixed_vlm_template = torch.load(template_path)
            self.fixed_seq_len = self.fixed_vlm_template.shape[0]
            print(f"✓ VLM features loaded: shape={self.fixed_vlm_template.shape}, seq_len={self.fixed_seq_len}")
            # 根据实际VLM特征维度创建特征编码器
            vlm_feature_dim = self.fixed_vlm_template.shape[1]  # 隐藏层维度
            self.feature_encoder = nn.Linear(vlm_feature_dim, 1536)
            self.feature_encoder.eval()
            print(f"✓ Feature encoder created: {vlm_feature_dim} -> 1536")
        else:
            print("⚠ VLM feature file not found, using simulated features")
            self._init_fixed_vlm_features()


    def _init_fixed_vlm_features(self):
        print("Initializing VLM features...")
        
        if self.vlm_backbone is not None:
            try:
                print("Attempting to use real VLM model...")
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                # 创建一个简单的图像输入来获取VLM特征
                # 使用PIL创建一个测试图像，这是Qwen VL模型所期望的格式
                import numpy as np
                from PIL import Image
                
                # 创建一个固定的测试图像 (224x224 RGB)
                test_image_array = np.random.RandomState(42).randint(0, 255, (224, 224, 3), dtype=np.uint8)
                test_image = Image.fromarray(test_image_array)
                test_text = "What actions should be taken based on this scene?"
                
                try:
                    # 使用VLM的processor来正确处理图像和文本
                    messages = [
                        {
                            "role": "user", 
                            "content": [
                                {"type": "image", "image": test_image},
                                {"type": "text", "text": test_text}
                            ]
                        }
                    ]
                    
                    # 应用聊天模板
                    text_inputs = self.vlm_backbone.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    # 处理输入
                    inputs = self.vlm_backbone.tokenizer(
                        text=[text_inputs],
                        images=[test_image], 
                        return_tensors="pt",
                        padding=True
                    )
                    
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    if device == 'cuda' and hasattr(self.vlm_backbone, 'model'):
                        self.vlm_backbone.model = self.vlm_backbone.model.to(device)
                    
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # 获取VLM模型的隐藏状态
                    with torch.no_grad():
                        outputs = self.vlm_backbone.model(
                            **inputs,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        
                        # 获取最后一层隐藏状态
                        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                            hidden_states = outputs.hidden_states[-1]  
                            
                            # 移除batch维度并转换为float32
                            self.fixed_vlm_template = hidden_states.squeeze(0).float()  # (seq_len, hidden_size)
                            self.fixed_seq_len = self.fixed_vlm_template.shape[0]
                            
                            print(f"✓ Real VLM features initialized: shape={self.fixed_vlm_template.shape}, seq_len={self.fixed_seq_len}")
                            
                            # 根据实际VLM特征维度创建特征编码器
                            vlm_feature_dim = self.fixed_vlm_template.shape[1]  # 隐藏层维度
                            self.feature_encoder = nn.Linear(vlm_feature_dim, 1536)
                            self.feature_encoder.eval()
                            print(f"✓ Feature encoder created: {vlm_feature_dim} -> 1536")
                            self.fixed_vlm_template = self.fixed_vlm_template.cpu()
                            
                            if device == 'cuda':
                                self.vlm_backbone.model = self.vlm_backbone.model.cpu()
                                del self.vlm_backbone.model
                                self.vlm_backbone.model = None
                                torch.cuda.empty_cache()
                                print("✓ VLM model moved to CPU and GPU memory cleared")
                            
                            return
                        else:
                            print("⚠ VLM model output does not contain hidden_states, falling back to simulated features")
                            
                except Exception as inner_e:
                    print(f"⚠ Error in VLM processing: {inner_e}")
                    
            except Exception as e:
                print(f"⚠ Failed to initialize real VLM features: {e}")
        
        # 备用方案：使用固定的随机特征
        print("Using simulated VLM features...")
        F = 2560  # VLM隐藏层维度
        self.fixed_seq_len = 8  # 固定序列长度
        
        generator = torch.Generator()
        generator.manual_seed(42)  
        
        # 生成单个固定的VLM特征模板 (seq_len, F)
        self.fixed_vlm_template = torch.randn(
            self.fixed_seq_len, F, 
            generator=generator, 
            dtype=torch.float32
        )
        
        print(f"✓ Simulated VLM features initialized: shape={self.fixed_vlm_template.shape}, seq_len={self.fixed_seq_len}")
        
        # 根据VLM特征维度创建特征编码器
        if self.feature_encoder is None:
            vlm_feature_dim = self.fixed_vlm_template.shape[1]  # 隐藏层维度 
            self.feature_encoder = nn.Linear(vlm_feature_dim, 1536)
            self.feature_encoder.eval()
            print(f"✓ Feature encoder created: {vlm_feature_dim} -> 1536")
    
    def generate_simulated_vlm_outputs(self, batch_size, device, max_seq_len=None):
        """
        生成真实的VLM输出特征, 支持可变序列长度和padding
        
        Args:
            batch_size: 批次大小 (B)
            device: 设备 ('cuda' 或 'cpu')
            max_seq_len: 最大序列长度, 用于padding (可选)
            
        Returns:
            vlm_features: 形状为 (B, seq_len, F) 的张量
            vl_mask: 形状为 (B, seq_len) 的bool张量,True表示有效位置, False表示padding
        """
        # 将固定模板移动到指定设备
        template_on_device = self.fixed_vlm_template.to(device)
        current_seq_len = template_on_device.shape[0]
        
        # 如果指定了最大序列长度且当前序列较短，则进行padding
        if max_seq_len is not None and current_seq_len < max_seq_len:
            # 创建padding
            padding_size = max_seq_len - current_seq_len
            feature_dim = template_on_device.shape[1]
            padding = torch.zeros(padding_size, feature_dim, device=device, dtype=template_on_device.dtype)
            
            # 添加padding到模板
            padded_template = torch.cat([template_on_device, padding], dim=0)
            
            # 创建mask：True为有效位置，False为padding位置
            vl_mask = torch.ones(max_seq_len, dtype=torch.bool, device=device)
            vl_mask[current_seq_len:] = False
            
            seq_len = max_seq_len
            template_to_use = padded_template
        else:
            # 不需要padding，所有位置都是有效的
            vl_mask = torch.ones(current_seq_len, dtype=torch.bool, device=device)
            seq_len = current_seq_len
            template_to_use = template_on_device
        
        # 为批次重复
        if batch_size == 1:
            vlm_features = template_to_use.unsqueeze(0)  # (1, seq_len, F)
            vl_mask_batch = vl_mask.unsqueeze(0)  # (1, seq_len)
        else:
            vlm_features = template_to_use.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, seq_len, F)
            vl_mask_batch = vl_mask.unsqueeze(0).repeat(batch_size, 1)  # (B, seq_len)
        
        return vlm_features, vl_mask_batch
