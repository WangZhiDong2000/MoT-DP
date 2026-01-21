import torch
import torch.nn as nn
from typing import Tuple, Optional
import sys
import os
try:
    import torch.distributed as dist
except ImportError:
    dist = None

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.interfuser.resnet import resnet18d
from model.interfuser.modeling_utils import MLPconnector, PositionEmbeddingSine, HybridEmbed
from functools import partial


class InterfuserBEVEncoder(nn.Module):
    """LiDAR BEV图像和状态信息的编码器 (Decoder-Only版本)
    
    输入:
        - image: (B, 3, H, W) - LiDAR BEV图像
        - state: (B, state_dim) - 状态信息
    
    输出 (forward):
        - bev_tokens: (B, seq_len+1, 512) - BEV空间tokens + global token (不池化)
        - state_tokens: (B, 1, 128) - 状态编码tokens (独立于BEV)
    
    设计理念:
        - 不再使用低维状态引导的BEV特征池化
        - BEV tokens和State tokens作为独立特征传递给decoder-only架构
        - 保留完整的空间信息用于sequential cross-attention
    """
    
    def __init__(
        self, 
        perception_backbone: Optional[nn.Module] = None,
        state_dim: int = 9,  # speed(1) + target_point(2) + command(6)
        freeze_backbone: bool = True,
        bev_input_size: Tuple[int, int] = (448, 448)
    ):
        """
        Args:
            perception_backbone: 视觉backbone (可选，默认使用resnet18d)
            state_dim: 状态维度
            freeze_backbone: 是否冻结backbone参数
            bev_input_size: BEV图像输入尺寸
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.freeze_backbone = freeze_backbone
        self.bev_input_size = bev_input_size
        
        # 1. BEV backbone
        if perception_backbone is None:
            lidar_backbone = resnet18d(
                pretrained=False,
                in_chans=3,
                features_only=True,
                out_indices=[4],   
            )
            self.perception = lidar_backbone
            embed_dim = 256
            self.lidar_embed_layer = partial(HybridEmbed, backbone=lidar_backbone)
            self.lidar_connector = MLPconnector(256, 512, 'gelu')
            self.bev_model_mot = self.lidar_embed_layer(
                img_size=224,
                patch_size=16,
                in_chans=3,
                embed_dim=embed_dim,
            )
            self.position_encoding = PositionEmbeddingSine(embed_dim // 2, normalize=True)
        else:
            self.perception = perception_backbone

        # 2. 测量编码器 (state -> measurement features)
        # 输出维度128，作为独立的state tokens
        self.measurements = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        
        if freeze_backbone and self.perception is not None:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """冻结backbone参数"""
        for param in self.perception.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """解冻backbone参数"""
        for param in self.perception.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
    
    def forward(
        self, 
        image: torch.Tensor = None, 
        state: torch.Tensor = None,
        lidar_token: torch.Tensor = None,
        lidar_token_global: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass返回独立的BEV tokens和state tokens。
        
        Args:
            image: Raw BEV image (B, 3, H, W) - optional if using pre-computed features
            state: State information (B, state_dim)
            lidar_token: Pre-computed local tokens (B, seq_len, 512) - optional
            lidar_token_global: Pre-computed global token (B, 1, 512) - optional
            
        Returns:
            bev_tokens: (B, seq_len+1, 512) - BEV spatial tokens + global token
            state_tokens: (B, 1, 128) - Encoded state as token sequence
        """
        use_precomputed = lidar_token is not None and lidar_token_global is not None
        
        if use_precomputed:
            # Mode 1: Use pre-computed features (fast path)
            batch_size = lidar_token.shape[0]
        else:
            # Mode 2: Extract features from raw image
            if self.perception is None:
                raise RuntimeError("Perception backbone not initialized.")
            if image is None:
                raise ValueError("image is required when not using pre-computed features")
            
            batch_size = image.shape[0]
            
            # 使用 bev_model_mot 处理原始图像
            lidar_token_raw, lidar_token_global_raw = self.bev_model_mot(image)
            
            # Ensure outputs match the model's dtype
            target_dtype = next(self.parameters()).dtype
            lidar_token_raw = lidar_token_raw.to(dtype=target_dtype)
            lidar_token_global_raw = lidar_token_global_raw.to(dtype=target_dtype)
            
            # 添加位置编码
            pos_encoding = self.position_encoding(lidar_token_raw).to(dtype=target_dtype)
            lidar_token_raw = lidar_token_raw + pos_encoding
            
            # 重塑为序列格式
            lidar_token_raw = lidar_token_raw.flatten(2).permute(2, 0, 1)  # (seq_len, batch, embed_dim)
            lidar_token_global_raw = lidar_token_global_raw.permute(2, 0, 1)  # (1, batch, embed_dim)
            
            # 连接局部和全局token
            lidar_tokens = torch.cat([lidar_token_raw, lidar_token_global_raw], dim=0)
            
            # 通过connector
            lidar_tokens = self.lidar_connector(lidar_tokens)
            
            # 转换为 (batch, seq_len, 512) 格式
            lidar_token = lidar_tokens[:-1].permute(1, 0, 2)  # (batch, seq_len, 512)
            lidar_token_global = lidar_tokens[-1:].permute(1, 0, 2)  # (batch, 1, 512)
        
        # Concatenate local and global BEV tokens
        # bev_tokens: (B, seq_len + 1, 512)
        bev_tokens = torch.cat([lidar_token, lidar_token_global], dim=1)
        
        # 编码state (不与BEV融合)
        if state is None:
            raise ValueError("state is required for forward pass")
        
        # state_tokens: (B, 1, 128)
        state_feature = self.measurements(state)  # (B, 128)
        state_tokens = state_feature.unsqueeze(1)  # (B, 1, 128)
        
        return bev_tokens, state_tokens

    def forward_batch(
        self, 
        images: torch.Tensor = None, 
        states: torch.Tensor = None,
        lidar_tokens: torch.Tensor = None,
        lidar_tokens_global: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract BEV and state tokens for a batch with temporal dimension.
        
        Args:
            images: (B, T, C, H, W) - optional if using pre-computed features
            states: (B, T, state_dim)
            lidar_tokens: (B, T, seq_len, 512) - optional pre-computed
            lidar_tokens_global: (B, T, 1, 512) - optional pre-computed
            
        Returns:
            bev_tokens: (B, T, seq_len+1, 512) - BEV tokens per timestep
            state_tokens: (B, T, 1, 128) - State tokens per timestep
        """
        use_precomputed = lidar_tokens is not None and lidar_tokens_global is not None
        
        if use_precomputed:
            B, T = lidar_tokens.shape[:2]
            lidar_tokens_flat = lidar_tokens.reshape(B * T, *lidar_tokens.shape[2:])
            lidar_tokens_global_flat = lidar_tokens_global.reshape(B * T, *lidar_tokens_global.shape[2:])
            states_flat = states.reshape(B * T, states.shape[-1])
            
            bev_tokens_flat, state_tokens_flat = self.forward(
                state=states_flat,
                lidar_token=lidar_tokens_flat,
                lidar_token_global=lidar_tokens_global_flat
            )
        else:
            if images is None or states is None:
                raise ValueError("Either (images, states) or (lidar_tokens, lidar_tokens_global, states) must be provided")
            
            B, T = images.shape[:2]
            images_flat = images.reshape(B * T, *images.shape[2:])
            states_flat = states.reshape(B * T, states.shape[-1])
            
            bev_tokens_flat, state_tokens_flat = self.forward(images_flat, states_flat)
        
        # Reshape back
        bev_seq_len = bev_tokens_flat.shape[1]
        bev_tokens = bev_tokens_flat.reshape(B, T, bev_seq_len, -1)
        state_tokens = state_tokens_flat.reshape(B, T, 1, -1)
        
        return bev_tokens, state_tokens


def load_lidar_submodules(model: torch.nn.Module, ckpt_path: str, strict: bool = False, logger=None):
    """
    加载LiDAR子模块的预训练权重，支持键名映射
    """
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        if logger:
            logger.warning(f"[LiDAR] ckpt 不存在: {ckpt_path}")
        else:
            print(f"[LiDAR] ckpt 不存在: {ckpt_path}")
        return
    
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    
    # 清理模块名称
    clean = {}
    for k, v in state.items():
        kk = k.replace("module.", "")
        clean[kk] = v
    
    # 键名映射规则
    ckpt2model_prefix = {
        "lidar_backbone.": "bev_model_mot.backbone.",
        "lidar_patch_embed.proj.": "bev_model_mot.proj.",
    }
    ckpt_src_prefixes = tuple(ckpt2model_prefix.keys())
    model_dst_prefixes = tuple(ckpt2model_prefix.values())

    # 重映射键名
    remapped = {} 
    for ckpt_k, v in clean.items():
        if ckpt_k.startswith(ckpt_src_prefixes):
            for src_prefix, dst_prefix in ckpt2model_prefix.items():
                if ckpt_k.startswith(src_prefix):
                    model_k = dst_prefix + ckpt_k[len(src_prefix):]
                    remapped[model_k] = (v, ckpt_k)
                    break

    # 检查形状匹配
    model_sd = model.state_dict()
    to_load = {}
    for model_k, (tensor, ckpt_k) in remapped.items():
        if model_k in model_sd:
            if model_sd[model_k].shape == tensor.shape:
                to_load[model_k] = tensor

    # 加载权重
    msg = model.load_state_dict(to_load, strict=False)

    # 打印统计
    rank0 = True
    if dist is not None and dist.is_initialized():
        rank0 = dist.get_rank() == 0
    
    if rank0:
        expected_keys = [k for k in model_sd.keys() if k.startswith(model_dst_prefixes)]
        if logger:
            logger.info(f"[LiDAR] 成功加载 {len(to_load)}/{len(expected_keys)} 个权重")
        else:
            print(f"[LiDAR] 成功加载 {len(to_load)}/{len(expected_keys)} 个权重")


if __name__ == "__main__":
    print("Testing InterfuserBEVEncoder (Decoder-Only Version)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = InterfuserBEVEncoder(
        perception_backbone=None,
        state_dim=9,
        freeze_backbone=True,
        bev_input_size=(448, 448)
    ).to(device)
    
    # Test forward
    B = 4
    image = torch.randn(B, 3, 448, 448).to(device)
    state = torch.randn(B, 9).to(device)
    
    with torch.no_grad():
        bev_tokens, state_tokens = encoder(image, state)
    
    print(f"BEV tokens shape: {bev_tokens.shape}")  # (B, 197, 512)
    print(f"State tokens shape: {state_tokens.shape}")  # (B, 1, 128)
    
    # Test batch forward
    images = torch.randn(2, 3, 3, 448, 448).to(device)
    states = torch.randn(2, 3, 9).to(device)
    
    with torch.no_grad():
        bev_batch, state_batch = encoder.forward_batch(images, states)
    
    print(f"BEV batch shape: {bev_batch.shape}")  # (2, 3, 197, 512)
    print(f"State batch shape: {state_batch.shape}")  # (2, 3, 1, 128)
    
    print("\n✓ All tests passed!")