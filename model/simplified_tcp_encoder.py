
import torch
import torch.nn as nn
from typing import Tuple, Optional
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from model.TCP.resnet import resnet34

class SimplifiedTCPEncoder(nn.Module):
    """
    简化版TCP编码器，专门用于扩散策略的观测编码
    
    输入:
        - image: (B, 3, H, W) - RGB图像，推荐尺寸 256x928
        - state: (B, state_dim) - 状态信息 [speed(1), target_point(2), command(6)]
    
    输出:
        - feature: (B, 256) - 编码后的特征向量
    
    相比完整TCP的简化:
        1. 移除了PID控制器 (不需要控制输出)
        2. 移除了轨迹预测分支 (join_traj, decoder_traj等)
        3. 移除了动作预测头 (policy_head, action_head等)
        4. 移除了价值估计分支 (value_branch)
        5. 只保留核心的视觉感知 + 测量融合 + 空间注意力机制
    """
    
    def __init__(
        self, 
        perception_backbone: Optional[nn.Module] = None,
        state_dim: int = 9,  # speed(1) + target_point(2) + command(6)
        feature_dim: int = 256,
        use_group_norm: bool = True,
        freeze_backbone: bool = True
    ):
        """
        Args:
            perception_backbone: 视觉backbone (如ResNet34)，如果为None则自动创建
            state_dim: 状态维度 (速度+目标点+指令)
            feature_dim: 输出特征维度
            use_group_norm: 是否使用GroupNorm标准化输出特征
            freeze_backbone: 是否冻结backbone参数
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.freeze_backbone = freeze_backbone
        
        if perception_backbone is None:
            self.perception = resnet34(pretrained=True)
        else:
            self.perception = perception_backbone

        # 2. 测量编码器 (state -> measurement features)
        self.measurements = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        
        # 3. 空间注意力模块
        self.init_att = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 8 * 29),  
            nn.Softmax(dim=1)
        )
        
        # 4. 特征融合模块
        # 融合视觉特征(经过注意力加权)和测量特征
        self.join_ctrl = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # 5. GroupNorm 
        self.use_group_norm = use_group_norm
        if use_group_norm:
            # 256 channels, 8 groups -> 32 channels per group
            self.feature_norm = nn.GroupNorm(num_groups=8, num_channels=feature_dim)
        
        if freeze_backbone and self.perception is not None:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        for param in self.perception.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.perception.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
    
    def forward(
        self, 
        image: torch.Tensor, 
        state: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        
        if self.perception is None:
            raise RuntimeError("Perception backbone not initialized. Set self.perception before forward.")
        
        batch_size = image.shape[0]
        
        # Step 1: 视觉感知
        if self.freeze_backbone:
            with torch.no_grad():
                feature_emb, cnn_feature = self.perception(image)
        else:
            feature_emb, cnn_feature = self.perception(image)
        
        # Step 2: 测量编码
        measurement_feature = self.measurements(state)
        
        # Step 3: 空间注意力
        init_att_flat = self.init_att(measurement_feature)
        init_att = init_att_flat.view(batch_size, 1, 8, 29)
        feature_emb_att = torch.sum(cnn_feature * init_att, dim=(2, 3))
        
        # Step 4: 特征融合
        fused_feature = torch.cat([feature_emb_att, measurement_feature], dim=1)
        output_feature = self.join_ctrl(fused_feature)
        
        # Step 5: 可选的特征标准化
        if normalize is None:
            normalize = self.use_group_norm and self.training
        
        if normalize and self.use_group_norm:
            output_feature = self.feature_norm(output_feature.unsqueeze(-1)).squeeze(-1)
        
        return output_feature
    
    def extract_features_batch(
        self, 
        images: torch.Tensor, 
        states: torch.Tensor
    ) -> torch.Tensor:
        
        B, T = images.shape[:2]
        
        # Reshape: (B, T, ...) -> (B*T, ...)
        images_flat = images.reshape(B * T, *images.shape[2:])
        states_flat = states.reshape(B * T, *states.shape[2:])
        
        # Extract features
        features_flat = self.forward(images_flat, states_flat)
        
        # Reshape back: (B*T, feature_dim) -> (B, T, feature_dim)
        features = features_flat.reshape(B, T, self.feature_dim)
        
        return features
    
    def get_feature_stats(self, feature: torch.Tensor) -> dict:

        return {
            'mean': feature.mean().item(),
            'std': feature.std().item(),
            'min': feature.min().item(),
            'max': feature.max().item(),
            'shape': tuple(feature.shape),
        }


# def create_simplified_tcp_encoder(
#     pretrained_backbone_path: str = None,
#     state_dim: int = 9,
#     feature_dim: int = 256,
#     use_group_norm: bool = True,
#     freeze_backbone: bool = True
# ) -> SimplifiedTCPEncoder:
#     """
#     创建简化版TCP编码器的工厂函数
    
#     Args:
#         pretrained_backbone_path: 预训练backbone权重路径
#         state_dim: 状态维度
#         feature_dim: 输出特征维度
#         use_group_norm: 是否使用GroupNorm
#         freeze_backbone: 是否冻结backbone
    
#     Returns:
#         SimplifiedTCPEncoder实例
#     """
#     import sys
#     import os
#     # Add project root to path for imports
#     project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     if project_root not in sys.path:
#         sys.path.insert(0, project_root)
    
#     from model.TCP.resnet import resnet34
    
#     # 创建ResNet34 backbone
#     perception = resnet34(pretrained=True)
#     print("✓ ResNet34 backbone created with ImageNet pretrained weights")
    
#     # 如果提供了完整TCP的预训练权重，加载部分参数
#     if pretrained_backbone_path is not None:
#         try:
#             from collections import OrderedDict
#             import os
            
#             if os.path.exists(pretrained_backbone_path):
#                 print(f"Loading pretrained weights from {pretrained_backbone_path}")
#                 ckpt = torch.load(pretrained_backbone_path, map_location='cpu')
                
#                 # 处理不同的checkpoint格式
#                 if 'state_dict' in ckpt:
#                     state_dict = ckpt['state_dict']
#                 else:
#                     state_dict = ckpt
                
#                 # 过滤出perception相关的权重
#                 perception_state_dict = OrderedDict()
#                 for key, value in state_dict.items():
#                     if 'perception' in key:
#                         new_key = key.replace('model.perception.', '').replace('perception.', '')
#                         perception_state_dict[new_key] = value
                
#                 if len(perception_state_dict) > 0:
#                     perception.load_state_dict(perception_state_dict, strict=False)
#                     print(f"✓ Loaded {len(perception_state_dict)} pretrained weights for perception")
#                 else:
#                     print("⚠ No perception weights found in checkpoint, using ImageNet weights")
#             else:
#                 print(f"⚠ Checkpoint not found: {pretrained_backbone_path}")
#         except Exception as e:
#             print(f"⚠ Failed to load pretrained weights: {e}")
#             print("  Using ImageNet pretrained weights instead")
    
#     # 创建编码器
#     encoder = SimplifiedTCPEncoder(
#         perception_backbone=perception,
#         state_dim=state_dim,
#         feature_dim=feature_dim,
#         use_group_norm=use_group_norm,
#         freeze_backbone=freeze_backbone
#     )
    
#     return encoder


# if __name__ == "__main__":
#     # 测试代码
#     print("="*80)
#     print("Testing Simplified TCP Encoder")
#     print("="*80)
    
#     # 创建编码器
#     encoder = create_simplified_tcp_encoder(
#         pretrained_backbone_path='/home/wang/projects/Bench2DriveZoo/tcp_b2d.ckpt',
#         use_group_norm=True,
#         freeze_backbone=True
#     )
#     encoder = encoder.cuda()
#     encoder.eval()
    
#     # 测试单个样本
#     print("\n" + "="*80)
#     print("Test 1: Single sample")
#     print("="*80)
    
#     batch_size = 4
#     image = torch.randn(batch_size, 3, 256, 928).cuda()
#     state = torch.cat([
#         torch.randn(batch_size, 1).cuda() / 12.0,  # speed
#         torch.randn(batch_size, 2).cuda(),  # target_point
#         torch.randn(batch_size, 6).cuda(),  # command (one-hot)
#     ], dim=1)
    
#     print(f"Input shapes:")
#     print(f"  - image: {image.shape}")
#     print(f"  - state: {state.shape}")
    
#     with torch.no_grad():
#         feature = encoder(image, state)
    
#     print(f"\nOutput:")
#     print(f"  - feature shape: {feature.shape}")
#     print(f"  - feature stats: {encoder.get_feature_stats(feature)}")
    
#     # 测试批量时间步
#     print("\n" + "="*80)
#     print("Test 2: Multiple timesteps")
#     print("="*80)
    
#     batch_size = 2
#     timesteps = 3
#     images = torch.randn(batch_size, timesteps, 3, 256, 928).cuda()
#     states = torch.cat([
#         torch.randn(batch_size, timesteps, 1).cuda() / 12.0,
#         torch.randn(batch_size, timesteps, 2).cuda(),
#         torch.randn(batch_size, timesteps, 6).cuda(),
#     ], dim=2)
    
#     print(f"Input shapes:")
#     print(f"  - images: {images.shape}")
#     print(f"  - states: {states.shape}")
    
#     with torch.no_grad():
#         features = encoder.extract_features_batch(images, states)
    
#     print(f"\nOutput:")
#     print(f"  - features shape: {features.shape}")
#     print(f"  - features stats: {encoder.get_feature_stats(features)}")
    
#     print("\n" + "="*80)
#     print("✓ All tests passed!")
#     print("="*80)
    
#     # 统计参数量
#     total_params = sum(p.numel() for p in encoder.parameters())
#     trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
#     print(f"\nModel statistics:")
#     print(f"  - Total parameters: {total_params:,}")
#     print(f"  - Trainable parameters: {trainable_params:,}")
#     print(f"  - Frozen parameters: {total_params - trainable_params:,}")
