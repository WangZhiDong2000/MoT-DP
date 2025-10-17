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
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    # unnormalize from [-1, 1] to [0, 1]
    ndata = (ndata + 1) / 2
    # unnormalize to original range
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
        obs_encoder = InterfuserBEVEncoder(
            perception_backbone=None,
            state_dim=9,
            feature_dim=256,
            use_group_norm=True,
            freeze_backbone=False,  # 设为False以便加载权重
            bev_input_size=(448, 448)
        )
        pretrained_path = '/home/wang/Project/MoT-DP/model/interfuser/lidar_bev_encoder_only.pth'
        load_lidar_submodules(obs_encoder, pretrained_path, strict=False, logger=None)
        self.obs_encoder = obs_encoder
        self.obs_encoder.cuda()

        # TODO load vlm and vlm encoder model）
        self.vlm_backbone = None
        
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
        self.feature_encoder = None
        self._init_loaded_vlm_features()


        # create diffusion model
        # TCP模型输出特征维度（j_ctrl）为256，修改相应维度
        obs_feature_dim = 256  
        
        # Optional GroupNorm for j_ctrl features (recommended for training stability)
        # self.use_j_ctrl_norm = policy_cfg.get('use_j_ctrl_norm', False)
        # if self.use_j_ctrl_norm:
        #     self.j_ctrl_norm = nn.GroupNorm(num_groups=8, num_channels=256)
        #     print("✓ GroupNorm enabled for j_ctrl features")  

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
        使用SimplifiedTCPEncoder提取特征
        现在使用lidar_bev作为视觉输入
        
        Args:
            obs_dict: 观测字典
            return_attention: 是否返回attention map
            
        Returns:
            如果return_attention=False: j_ctrl特征 (B, 256)
            如果return_attention=True: (j_ctrl特征, attention_map) 
        """
        try:
            # 使用lidar_bev替代原来的front_img (RGB image)
            lidar_bev_img = obs_dict['lidar_bev'].to(device='cuda', dtype=torch.float32)  
            speed = obs_dict['speed'].to(dtype=torch.float32).view(-1,1) / 12.
            target_point = obs_dict['target_point'].to(dtype=torch.float32)
            command = obs_dict['next_command'].to(dtype=torch.float32)
            state = torch.cat([speed, target_point, command], 1).to('cuda')
            
            if return_attention:
                j_ctrl, attention_map = self.obs_encoder(lidar_bev_img, state, normalize=True, return_attention=True)
            else:
                j_ctrl = self.obs_encoder(lidar_bev_img, state, normalize=True, return_attention=False)
                attention_map = None
                
            # Optional GroupNorm for better training stability
            # if self.use_j_ctrl_norm and self.training:
            #     j_ctrl = self.j_ctrl_norm(j_ctrl.unsqueeze(-1)).squeeze(-1)
            
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
            'lidar_bev': (B, obs_horizon, 3, 448, 448) - LiDAR BEV图像
            'agent_pos': (B, obs_horizon, 2)
            'next_command': (B, obs_horizon, 6)
            'speed': (B, obs_horizon)
            'target_point': (B, obs_horizon, 2)
        }
        """
        device = next(self.parameters()).device
        nobs = {}
        carla_fields = ['lidar_bev', 'next_command', 'speed', 'target_point','agent_pos']
        for field in carla_fields:
            if field in batch:
                if field == 'lidar_bev':
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
            obs_features_list = []
            for t in range(To):
                this_step_nobs = dict_apply(nobs, lambda x: x[:, t, ...])
                step_features = self.extract_tcp_features(this_step_nobs)  # (B, feature_dim)
                obs_features_list.append(step_features)
            
            # 堆叠所有时间步的特征: (B, To, feature_dim)
            cond = torch.stack(obs_features_list, dim=1).float()  
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.extract_tcp_features(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)

        # generate impainting mask
        # condition_mask = self.mask_generator(trajectory.shape)
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
        vl_features, vl_mask = self.generate_simulated_vlm_outputs(batch_size, trajectory.device)
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
            cond=None, generator=None,
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
        vl_features, vl_mask = self.generate_simulated_vlm_outputs(trajectory.shape[0],trajectory.device)
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
            obs_features_list = []
            for t in range(To):
                this_step_nobs = dict_apply(nobs, lambda x: x[:, t, ...])
                step_features = self.extract_tcp_features(this_step_nobs)  # (B, feature_dim)
                obs_features_list.append(step_features)
            cond = torch.stack(obs_features_list, dim=1)
            shape = (B, T, Da)
           
            cond_data = torch.zeros(size=shape, device=device, dtype=torch.float32)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            obs_features_list = []
            for t in range(To):
                this_step_nobs = dict_apply(nobs, lambda x: x[:, t, ...])
                step_features = self.extract_tcp_features(this_step_nobs)  # (B, feature_dim)
                obs_features_list.append(step_features)
            
            # 堆叠所有时间步的特征: (B, To, feature_dim)
            nobs_features = torch.stack(obs_features_list, dim=1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=torch.float32)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
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
        """
        提取最后一帧观测的attention map
        
        Args:
            obs_dict: 观测字典
            
        Returns:
            attention_map: (H, W) numpy数组
        """
        device = next(self.parameters()).device
        nobs = dict_apply(obs_dict, lambda x: x.to(device))
        
        # 获取最后一帧观测
        last_obs = dict_apply(nobs, lambda x: x[:, -1, ...])
        
        # 提取特征和attention map
        with torch.no_grad():
            _, attention_map = self.extract_tcp_features(last_obs, return_attention=True)
        
        # 转换为numpy
        attention_map_np = attention_map[0].cpu().numpy()  # (H, W)
        
        return attention_map_np

    def predict_action_with_steps(self, obs_dict: Dict[str, torch.Tensor]):
        """
        预测动作并返回去噪过程的中间步骤
        
        Returns:
            result: 预测结果字典
            denoising_steps: 去噪过程中的轨迹列表
        """
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
            obs_features_list = []
            for t in range(To):
                this_step_nobs = dict_apply(nobs, lambda x: x[:, t, ...])
                step_features = self.extract_tcp_features(this_step_nobs)
                obs_features_list.append(step_features)
            cond = torch.stack(obs_features_list, dim=1)
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=torch.float32)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            obs_features_list = []
            for t in range(To):
                this_step_nobs = dict_apply(nobs, lambda x: x[:, t, ...])
                step_features = self.extract_tcp_features(this_step_nobs)
                obs_features_list.append(step_features)
            nobs_features = torch.stack(obs_features_list, dim=1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=torch.float32)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # 运行采样并捕获中间步骤
        nsample, denoising_steps = self.conditional_sample_with_steps(
            cond_data, 
            cond_mask,
            cond=cond,
        )
        
        naction_pred = nsample[...,:Da]
        naction_pred = torch.clamp(naction_pred, -1.0, 1.0)
        
        # 反归一化去噪步骤用于可视化
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
            cond=None, generator=None,
            **kwargs):
        """
        条件采样并返回中间去噪步骤
        """
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        scheduler.set_timesteps(self.num_inference_steps)
        vl_features, vl_mask = self.generate_simulated_vlm_outputs(trajectory.shape[0], trajectory.device)
        vl_embeds = self.feature_encoder(vl_features)

        # 保存去噪步骤
        denoising_steps = []
        denoising_steps.append(trajectory.clone())  # 初始噪声
        
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
            
            # 保存关键步骤（初始、33%、66%、最终）
            if i == total_steps // 3 or i == 2 * total_steps // 3:
                denoising_steps.append(trajectory.clone())
        
        # 最终结果
        trajectory[condition_mask] = condition_data[condition_mask]
        denoising_steps.append(trajectory.clone())

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
        """
        初始化真实的VLM特征,从实际的VLM模型中提取隐藏层特征
        如果VLM模型不可用,则使用固定的随机特征作为备用
        """
        print("Initializing VLM features...")
        
        # 检查VLM backbone是否可用
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
        F = 3584  # VLM隐藏层维度
        self.fixed_seq_len = 25  # 固定序列长度
        
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



def use_dummy_test():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path, "config/carla.yaml")
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)  
    
    # 对于测试，禁用action normalization或提供dummy stats
    config['enable_action_normalization'] = False
    model = DiffusionDiTCarlaPolicy(config)
    # print(model)

    # dummy input - 使用CARLA驾驶数据集格式，obs_horizon=1
    obs_dict = {
        'lidar_bev': torch.randn(4, 1, 3, 448, 448),  # LiDAR BEV图像格式 (448x448)
        'agent_pos': torch.randn(4, 1, 2),      # 代理位置
        'speed': torch.randn(4, 1),             # 速度
        'theta': torch.randn(4, 1),             # 朝向角
        'throttle': torch.randn(4, 1),          # 油门
        'steer': torch.randn(4, 1),             # 转向
        'brake': torch.randn(4, 1),             # 刹车
        'target_point': torch.randn(4, 1, 2),  # 目标点
        'next_command': torch.randn(4, 1, 6),  # 指令
    }
    batch_input = {
        'lidar_bev': obs_dict['lidar_bev'],
        'agent_pos': obs_dict['agent_pos'],
        'speed': obs_dict['speed'],
        'theta': obs_dict['theta'],
        'throttle': obs_dict['throttle'],
        'steer': obs_dict['steer'],
        'brake': obs_dict['brake'],
        'target_point': obs_dict['target_point'],
        'next_command': obs_dict['next_command'],
        'action': torch.randn(4, 16, 2)  
    }
    out = model.compute_loss(batch_input)
    #print(out)

if __name__ == "__main__":
    use_dummy_test()