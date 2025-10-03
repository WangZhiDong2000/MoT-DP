import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Callable
from collections import defaultdict
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from model.transformer_for_diffusion import TransformerForDiffusion, LowdimMaskGenerator
from model.simplified_tcp_encoder import SimplifiedTCPEncoder
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
        
        # 从配置文件获取obs_horizon
        self.n_obs_steps = policy_cfg.get('n_obs_steps', config.get('obs_horizon', 1))
        
        # obs encoder - 使用TCP模型
        obs_encoder = SimplifiedTCPEncoder(
            state_dim=9,
            feature_dim=256,
            use_group_norm=policy_cfg.get('use_j_ctrl_norm', False),
            freeze_backbone=True
        )
            
        self.obs_encoder = obs_encoder

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
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * self.n_obs_steps
        

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

    def extract_tcp_features(self, obs_dict):
        
        front_img = obs_dict['image'].to(device='cuda', dtype=torch.float32)  
        speed = obs_dict['speed'].to(dtype=torch.float32).view(-1,1) / 12.
        target_point = obs_dict['target_point'].to(dtype=torch.float32)
        command = obs_dict['next_command'].to(dtype=torch.float32)
        state = torch.cat([speed, target_point, command], 1).to('cuda')
        j_ctrl = self.obs_encoder(front_img, state)
                
        return j_ctrl

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        batch: {
            'image': (B, obs_horizon, C, H, W)
            'agent_pos': (B, obs_horizon, 2)
            'action': (B, action_horizon, action_dim)
            'speed': (B, obs_horizon)
            'target_point': (B, obs_horizon, 2)
        }
        """
        device = next(self.parameters()).device
        nobs = {}
        carla_fields = ['image', 'next_command', 'speed', 'target_point','agent_pos']
        for field in carla_fields:
            if field in batch:
                if field == 'image':
                    nobs[field] = batch[field].to(device=device, dtype=torch.float32)
                else:
                    nobs[field] = batch[field].to(device)

        raw_agent_pos = batch['agent_pos'].to(device)

        # (B, horizon, 2)
        To = self.n_obs_steps
        nactions = raw_agent_pos[:, To:, :]
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
        'image': torch.randn(4, 1, 3, 96, 96),  # CARLA图像格式
        'agent_pos': torch.randn(4, 1, 2),      # 代理位置
        'speed': torch.randn(4, 1),             # 速度
        'theta': torch.randn(4, 1),             # 朝向角
        'throttle': torch.randn(4, 1),          # 油门
        'steer': torch.randn(4, 1),             # 转向
        'brake': torch.randn(4, 1),             # 刹车
        'target_point': torch.randn(4, 1, 2),  # 目标点
    }
    batch_input = {
        'image': obs_dict['image'],
        'agent_pos': obs_dict['agent_pos'],
        'speed': obs_dict['speed'],
        'theta': obs_dict['theta'],
        'throttle': obs_dict['throttle'],
        'steer': obs_dict['steer'],
        'brake': obs_dict['brake'],
        'target_point': obs_dict['target_point'],
        'action': torch.randn(4, 16, 2)  
    }
    out = model.compute_loss(batch_input)
    #print(out)

if __name__ == "__main__":
    use_dummy_test()