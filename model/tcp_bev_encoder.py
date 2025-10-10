
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
        - image: (B, 3, H, W) - LiDAR BEV图像，推荐尺寸 336x336
        - state: (B, state_dim) - 状态信息 [speed(1), target_point(2), command(6)]
    
    输出:
        - feature: (B, 256) - 编码后的特征向量
    
    相比完整TCP的简化:
        1. 移除了PID控制器 (不需要控制输出)
        2. 移除了轨迹预测分支 (join_traj, decoder_traj等)
        3. 移除了动作预测头 (policy_head, action_head等)
        4. 移除了价值估计分支 (value_branch)
        5. 只保留核心的视觉感知 + 测量融合 + 空间注意力机制
        6. 适配LiDAR BEV图像输入 (336x336)，对应的ResNet34特征图为11x11
    """
    
    def __init__(
        self, 
        perception_backbone: Optional[nn.Module] = None,
        state_dim: int = 9,  # speed(1) + target_point(2) + command(6)
        feature_dim: int = 256,
        use_group_norm: bool = True,
        freeze_backbone: bool = True,
        bev_input_size: Tuple[int, int] = (336, 336)  # LiDAR BEV图像尺寸
    ):
        """
        Args:
            perception_backbone: 视觉backbone (如ResNet34)，如果为None则自动创建
            state_dim: 状态维度 (速度+目标点+指令)
            feature_dim: 输出特征维度
            use_group_norm: 是否使用GroupNorm标准化输出特征
            freeze_backbone: 是否冻结backbone参数
            bev_input_size: BEV图像输入尺寸，默认(336, 336)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.freeze_backbone = freeze_backbone
        self.bev_input_size = bev_input_size
        
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
        # 计算ResNet34输出特征图尺寸: 对于336x336输入 -> 11x11特征图
        # ResNet34: conv1(stride=2) -> maxpool(stride=2) -> layer2(stride=2) -> layer3(stride=2) -> layer4(stride=2)
        # 336 -> 168 -> 84 -> 42 -> 21 -> 11
        feat_h = bev_input_size[0] // 32  # 336 // 32 = 10.5 ≈ 11
        feat_w = bev_input_size[1] // 32  # 336 // 32 = 10.5 ≈ 11
        
        # 更精确的计算
        if bev_input_size == (336, 336):
            feat_h, feat_w = 11, 11  # 实际输出为11x11
        else:
            # 通用计算: input_size -> (input_size // 32)
            feat_h = (bev_input_size[0] + 31) // 32
            feat_w = (bev_input_size[1] + 31) // 32
        
        self.feat_h = feat_h
        self.feat_w = feat_w
        spatial_size = feat_h * feat_w
        
        self.init_att = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, spatial_size),  # 11 * 11 = 121 for 336x336 input
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
        # 使用动态计算的特征图尺寸
        init_att_flat = self.init_att(measurement_feature)
        init_att = init_att_flat.view(batch_size, 1, self.feat_h, self.feat_w)
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



