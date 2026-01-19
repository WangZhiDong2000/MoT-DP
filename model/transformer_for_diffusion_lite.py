"""
Decoder-Only Transformer for Diffusion-based Trajectory Prediction (Lite Version).

This module implements a simplified decoder-only architecture for trajectory prediction,
with the following optimizations compared to the full version:
- No self-attention in decoder blocks
- No separate cross-attention to history (history info only for AdaLN modulation)
- Spatial BEV attention using grid sampling based on trajectory positions
- FiLM modulation: Reasoning tokens modulate VL tokens, then single cross-attention

Key design choices:
- Sequential: Spatial BEV Attention -> FiLM-modulated VL Cross-Attention
- AdaLN modulation: timestep + current_status + GRU(history)
- Unified decoder with heterogeneous queries for trajectory (horizon) + route (num_waypoints)
- MLP output heads (no GRU)
"""

from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

logger = logging.getLogger(__name__)


# =============================================================================
# Basic Components
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """
    Generate sine position embedding for 2D positions.
    Adapted from DAB-DETR.
    
    Args:
        pos_tensor: (B, N, 2) or (B, N, T, 2) position tensor with (x, y) coordinates
        hidden_dim: embedding dimension (output will be 2 * hidden_dim)
    
    Returns:
        pos_embed: position embedding tensor
    """
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos


class ModuleAttrMixin(nn.Module):
    """Mixin for device/dtype properties."""
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter(torch.zeros(1))

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# =============================================================================
# History Encoder (Multiple Options)
# =============================================================================

class HistoryEncoderGRU(nn.Module):
    """Encodes the history sequence of ego status into a single vector using GRU."""
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        _, h_n = self.gru(x)
        h = h_n[-1]
        return self.out_proj(self.act(h))


class HistoryEncoderAttention(nn.Module):
    """
    Encodes history sequence using attention-based pooling.
    
    This preserves information from all timesteps while learning
    importance weights for each timestep.
    
    Advantages over GRU:
    - Better information preservation (no forgetting)
    - Interpretable attention weights
    - More efficient computation
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # Project input to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # Temporal position encoding (learnable)
        self.max_seq_len = 16
        self.temporal_pos_emb = nn.Parameter(torch.zeros(1, self.max_seq_len, hidden_dim))
        
        # Attention pooling: query is learnable, keys/values are projected history
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        
        # Output projection with residual path
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.temporal_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.pool_query, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) - history sequence
            
        Returns:
            pooled: (B, hidden_dim) - pooled history representation
        """
        B, T, _ = x.shape
        
        # Project to hidden dimension
        h = self.input_proj(x)  # (B, T, hidden_dim)
        
        # Add temporal position encoding
        pos_emb = self.temporal_pos_emb[:, :T, :]
        h = h + pos_emb
        
        # Attention pooling: learnable query attends to all history
        query = self.pool_query.expand(B, -1, -1)  # (B, 1, hidden_dim)
        pooled, _ = self.attn(query, h, h)  # (B, 1, hidden_dim)
        pooled = pooled.squeeze(1)  # (B, hidden_dim)
        
        # Output projection
        return self.out_proj(pooled)


class HistoryEncoderMLP(nn.Module):
    """
    Simple MLP encoder that flattens and projects history.
    
    Fastest option, but loses explicit temporal structure.
    Best for very short sequences (T <= 4).
    """
    def __init__(self, input_dim: int, hidden_dim: int, max_seq_len: int = 4):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * max_seq_len, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) - history sequence
            
        Returns:
            pooled: (B, hidden_dim) - pooled history representation
        """
        B, T, D = x.shape
        # Pad or truncate to max_seq_len
        if T < self.max_seq_len:
            pad = torch.zeros(B, self.max_seq_len - T, D, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif T > self.max_seq_len:
            x = x[:, -self.max_seq_len:, :]
        
        # Flatten and project
        x_flat = x.reshape(B, -1)  # (B, T * input_dim)
        return self.mlp(x_flat)


# Alias for backward compatibility
HistoryEncoder = HistoryEncoderAttention


# =============================================================================
# FiLM Modulation Layer
# =============================================================================

class FiLMModulation(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    
    Uses conditioning (e.g., reasoning tokens) to modulate target features (e.g., VL tokens)
    via scale and shift: output = gamma * features + beta
    """
    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Project condition to generate gamma and beta
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, feature_dim * 2),
            nn.SiLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize to identity transform (gamma=1, beta=0)
        nn.init.zeros_(self.condition_proj[-1].weight)
        nn.init.zeros_(self.condition_proj[-1].bias)
    
    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation.
        
        Args:
            features: (B, T_feat, feature_dim) - features to modulate (e.g., VL tokens)
            condition: (B, T_cond, condition_dim) - conditioning (e.g., reasoning tokens)
            
        Returns:
            modulated: (B, T_feat, feature_dim) - modulated features
        """
        # Pool condition across sequence to get (B, condition_dim)
        cond_pooled = condition.mean(dim=1)  # (B, condition_dim)
        
        # Generate gamma and beta
        gamma_beta = self.condition_proj(cond_pooled)  # (B, feature_dim * 2)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # Each: (B, feature_dim)
        
        # Apply FiLM: (1 + gamma) * features + beta
        gamma = gamma.unsqueeze(1)  # (B, 1, feature_dim)
        beta = beta.unsqueeze(1)    # (B, 1, feature_dim)
        
        return features * (1 + gamma) + beta


# =============================================================================
# BEV Cross-Attention (supports 1D sequence BEV tokens)
# =============================================================================

class BEVCrossAttention(nn.Module):
    """
    BEV Cross-Attention that supports 1D sequence BEV tokens.
    
    Uses standard cross-attention with trajectory position-based query modulation.
    This is more flexible than grid sampling when BEV tokens are already 1D sequences.
    """
    def __init__(
        self, 
        embed_dims: int, 
        num_heads: int = 8,
        in_bev_dims: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        
        # Value projection (from BEV features)
        self.bev_proj = nn.Sequential(
            nn.Linear(in_bev_dims, embed_dims),
            nn.LayerNorm(embed_dims),
        )
        
        # Position embedding projection for trajectory points
        self.pos_embed_proj = nn.Linear(embed_dims, embed_dims)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self, 
        queries: torch.Tensor,        # (B, N, embed_dims) - query features
        traj_points: torch.Tensor,    # (B, N, T, 2) or (B, N, 2) - trajectory positions
        bev_feature: torch.Tensor,    # (B, seq_len, in_bev_dims) - 1D BEV tokens
        bev_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: Query features (B, N, embed_dims)
            traj_points: Trajectory positions (B, N, T, 2) or (B, N, 2) - used for position modulation
            bev_feature: BEV features (B, seq_len, in_bev_dims) - 1D sequence
            bev_padding_mask: Optional mask for BEV tokens (B, seq_len)
            
        Returns:
            output: (B, N, embed_dims) - attended features
        """
        B, N, _ = queries.shape
        
        # Project BEV features to embed_dims
        bev_proj = self.bev_proj(bev_feature)  # (B, seq_len, embed_dims)
        
        # Handle trajectory points shape
        if traj_points.dim() == 4:
            # (B, N, T, 2) -> use mean position for modulation
            traj_pos = traj_points.mean(dim=2)  # (B, N, 2)
        else:
            traj_pos = traj_points  # (B, N, 2)
        
        # Generate position embeddings for query modulation
        pos_embed = gen_sineembed_for_position(traj_pos, self.embed_dims)  # (B, N, embed_dims)
        
        # Modulate queries with position information
        queries_pos = queries + self.pos_embed_proj(pos_embed)
        
        # Cross-attention: queries attend to BEV features
        attn_out, _ = self.cross_attn(
            queries_pos, bev_proj, bev_proj,
            key_padding_mask=bev_padding_mask
        )
        
        # Output projection with residual
        output = self.output_proj(attn_out)
        return self.dropout(output) + queries


# =============================================================================
# Trajectory Head (MLP version)
# =============================================================================

class TrajectoryMLPHead(nn.Module):
    """
    MLP-based Trajectory Head for decoding trajectory.
    Replaces GRU with simple MLP layers.
    """
    def __init__(self, n_emb: int, output_dim: int, p_drop: float = 0.1):
        super().__init__()
        self.ln_f = nn.LayerNorm(n_emb)
        
        # MLP layers with conditioning
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(n_emb, n_emb),
            nn.GELU(),
            nn.Dropout(p_drop),
        )
        
        # Conditioning projection
        self.cond_proj = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.SiLU(),
        )
        
        # Output projection
        self.output_head = nn.Linear(n_emb, output_dim)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_emb) - decoder output
            conditioning: (B, n_emb) - timestep + ego_status conditioning
        Returns:
            (B, T, output_dim) - trajectory prediction
        """
        x = self.ln_f(x)
        
        # Add conditioning as bias
        cond = self.cond_proj(conditioning).unsqueeze(1)  # (B, 1, n_emb)
        x = x + cond
        
        # MLP processing
        x = x + self.mlp(x)
        
        # Output projection
        return self.output_head(x)


# =============================================================================
# Simplified Decoder Block (Lite Version)
# =============================================================================

class DecoderBlockLite(nn.Module):
    """
    Simplified Decoder Block without self-attention and history cross-attention.
    
    Architecture (per block):
        1. BEV Cross-Attention (standard cross-attention with position modulation)
        2. Cross-Attention to FiLM-modulated VL tokens
        3. Feed-Forward Network
        
    Compared to full version:
        - No self-attention
        - No history cross-attention (history only for AdaLN modulation)
        - No separate reasoning cross-attention (reasoning modulates VL via FiLM)
        
    All operations use AdaLN modulation conditioned on timestep + current_ego_status + GRU(history).
    """
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        batch_first: bool = True,
        in_bev_dims: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # 1. BEV Cross-Attention (supports 1D sequence BEV)
        self.bev_cross_attn = BEVCrossAttention(
            embed_dims=d_model,
            num_heads=nhead,
            in_bev_dims=in_bev_dims,
            dropout=dropout,
        )
        
        # 2. Cross-Attention to VL tokens (modulated by reasoning)
        self.vl_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=batch_first
        )
        
        # 3. FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Layer Norms (3 groups: bev, vl, ffn)
        self.norm_bev = nn.LayerNorm(d_model)
        self.norm_vl = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        
        self.activation = nn.GELU()
        
        # AdaLN modulation - 9 params: 3 groups x 3 (shift, scale, gate)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 9 * d_model, bias=True)
        )
    
    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
    def forward(
        self,
        x: torch.Tensor,                    # (B, N, d_model) - queries
        traj_points: torch.Tensor,          # (B, N, T, 2) - trajectory positions for position modulation
        bev_feature: torch.Tensor,          # (B, seq_len, bev_dim) - BEV features (1D sequence)
        vl_tokens_modulated: torch.Tensor,  # (B, T_vl, d_model) - FiLM-modulated VL tokens
        conditioning: torch.Tensor,         # (B, d_model) - AdaLN conditioning
        bev_padding_mask: Optional[torch.Tensor] = None,
        vl_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Query features (B, N, d_model)
            traj_points: Trajectory positions for position modulation (B, N, T, 2)
            bev_feature: BEV features (B, seq_len, bev_dim) - 1D sequence
            vl_tokens_modulated: Pre-modulated VL tokens (by FiLM with reasoning)
            conditioning: AdaLN conditioning (timestep + status + history)
            bev_padding_mask: Optional mask for BEV tokens
            vl_padding_mask: Optional mask for VL tokens
            
        Returns:
            x: Updated query features (B, N, d_model)
        """
        # Generate modulation parameters
        mod_params = self.adaLN_modulation(conditioning)
        (shift_bev, scale_bev, gate_bev,
         shift_vl, scale_vl, gate_vl,
         shift_ffn, scale_ffn, gate_ffn) = mod_params.chunk(9, dim=1)
        
        # 1. BEV Cross-Attention
        x_norm = self.modulate(self.norm_bev(x), shift_bev, scale_bev)
        x_bev = self.bev_cross_attn(x_norm, traj_points, bev_feature, bev_padding_mask)
        x = x + gate_bev.unsqueeze(1) * (x_bev - x_norm)  # Residual already in bev_cross_attn
        
        # 2. Cross-Attention to VL (modulated by reasoning)
        x_norm = self.modulate(self.norm_vl(x), shift_vl, scale_vl)
        x_vl, _ = self.vl_cross_attn(x_norm, vl_tokens_modulated, vl_tokens_modulated, 
                                      key_padding_mask=vl_padding_mask)
        x = x + gate_vl.unsqueeze(1) * x_vl
        
        # 3. FFN
        x_norm = self.modulate(self.norm_ffn(x), shift_ffn, scale_ffn)
        x_ffn = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = x + gate_ffn.unsqueeze(1) * x_ffn
        
        return x


# =============================================================================
# Unified Decoder (Lite Version)
# =============================================================================

class UnifiedDecoderLite(nn.Module):
    """
    Simplified Unified Decoder with heterogeneous queries.
    
    Combines trajectory prediction (horizon points) and route prediction (num_waypoints)
    into a single decoder, using:
    - Segment embeddings to distinguish query types
    - FiLM modulation: Reasoning tokens modulate VL tokens
    - BEV cross-attention with trajectory position modulation
    - Single cross-attention to modulated VL tokens
    
    Query structure: [trajectory (horizon) | route (num_waypoints)]
    Attention order: BEV Cross-Attn (pos modulated) -> Modulated VL Cross-Attention -> FFN
    """
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 8,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        bev_dim: int = 512,
        vl_dim: int = 1536,
        reasoning_dim: int = 1536,
        horizon: int = 8,
        num_waypoints: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.horizon = horizon
        self.num_waypoints = num_waypoints
        
        # Projection layers for conditions
        self.vl_proj = nn.Sequential(nn.Linear(vl_dim, d_model), nn.LayerNorm(d_model))
        self.reasoning_proj = nn.Sequential(nn.Linear(reasoning_dim, d_model), nn.LayerNorm(d_model))
        
        # FiLM modulation: Reasoning modulates VL
        self.film_modulation = FiLMModulation(d_model, d_model)
        
        # Route queries (learnable)
        self.route_queries = nn.Parameter(torch.randn(1, num_waypoints, d_model))
        
        # Position embeddings for queries
        self.route_pos_emb = nn.Parameter(torch.zeros(1, num_waypoints, d_model))
        
        # Segment embeddings to distinguish query types
        self.traj_segment_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.route_segment_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Default route positions for position modulation
        # These represent typical route waypoint positions (used for position encoding)
        self.route_default_positions = nn.Parameter(
            torch.linspace(0, 30, num_waypoints).unsqueeze(0).unsqueeze(-1).expand(-1, -1, 2).clone()
        )  # (1, num_waypoints, 2)
        
        # Decoder blocks
        self.layers = nn.ModuleList([
            DecoderBlockLite(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                in_bev_dims=bev_dim,
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.route_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.route_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.traj_segment_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.route_segment_emb, mean=0.0, std=0.02)
        for layer in self.layers:
            nn.init.zeros_(layer.adaLN_modulation[-1].weight)
            nn.init.zeros_(layer.adaLN_modulation[-1].bias)
    
    def forward(
        self,
        traj_emb: torch.Tensor,           # (B, horizon, d_model) - trajectory query embeddings
        traj_positions: torch.Tensor,     # (B, horizon, 2) - noisy trajectory positions for BEV sampling
        bev_feature: torch.Tensor,        # (B, H*W, bev_dim) or (B, C, H, W)
        vl_tokens: torch.Tensor,          # (B, T_vl, vl_dim) - VL tokens
        reasoning_tokens: torch.Tensor,   # (B, T_r, reasoning_dim) - reasoning tokens
        conditioning: torch.Tensor,       # (B, d_model) - AdaLN conditioning
        vl_padding_mask: Optional[torch.Tensor] = None,
        reasoning_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with unified queries.
        
        Args:
            traj_emb: (B, horizon, d_model) - trajectory query embeddings (noisy traj embedded)
            traj_positions: (B, horizon, 2) - noisy trajectory positions for spatial BEV attention
            bev_feature: BEV features for spatial attention
            vl_tokens: VL tokens to be modulated by reasoning
            reasoning_tokens: Reasoning tokens for FiLM modulation
            conditioning: AdaLN conditioning (timestep + status + history)
            
        Returns:
            traj_out: (B, horizon, d_model) - trajectory output
            route_out: (B, num_waypoints, d_model) - route output
        """
        B = traj_emb.shape[0]
        T_traj = traj_emb.shape[1]
        
        # Add segment embedding to trajectory
        traj_emb = traj_emb + self.traj_segment_emb
        
        # Route queries with position and segment embeddings
        route_emb = self.route_queries.expand(B, -1, -1) + self.route_pos_emb + self.route_segment_emb
        
        # Concatenate queries: [trajectory | route]
        x = torch.cat([traj_emb, route_emb], dim=1)  # (B, horizon + num_waypoints, d_model)
        
        # Prepare trajectory positions for spatial BEV attention
        # For trajectory: use noisy positions
        # For route: use default positions (learnable)
        route_positions = self.route_default_positions.expand(B, -1, -1)  # (B, num_waypoints, 2)
        
        # Combine positions for unified queries
        # Each query will sample from its corresponding position
        traj_points_unified = torch.cat([
            traj_positions.unsqueeze(2),  # (B, horizon, 1, 2)
            route_positions.unsqueeze(2),  # (B, num_waypoints, 1, 2)
        ], dim=1)  # (B, horizon + num_waypoints, 1, 2)
        
        # Project and modulate VL tokens with reasoning
        vl_proj = self.vl_proj(vl_tokens)
        reasoning_proj = self.reasoning_proj(reasoning_tokens)
        
        # FiLM: Reasoning modulates VL
        vl_modulated = self.film_modulation(vl_proj, reasoning_proj)
        
        # Decoder layers
        for layer in self.layers:
            x = layer(
                x, traj_points_unified, bev_feature, vl_modulated,
                conditioning, bev_padding_mask=None, vl_padding_mask=vl_padding_mask
            )
        
        x = self.final_norm(x)
        
        # Split outputs
        traj_out = x[:, :T_traj, :]
        route_out = x[:, T_traj:, :]
        
        return traj_out, route_out


# =============================================================================
# Main Model: TransformerForDiffusionLite
# =============================================================================

class TransformerForDiffusionLite(ModuleAttrMixin):
    """
    Simplified Decoder-Only Transformer for Diffusion-based Trajectory Prediction.
    
    Key differences from full version:
    - No self-attention in decoder blocks
    - No history cross-attention (history only for AdaLN modulation)
    - Spatial BEV attention using grid sampling at trajectory positions
    - FiLM modulation: Reasoning tokens modulate VL tokens
    - Single cross-attention to modulated VL tokens
    
    Query structure: [trajectory (horizon) | route (num_waypoints)]
    Attention order: Spatial BEV -> Modulated VL Cross-Attention -> FFN
    
    Loss design:
    - Trajectory loss: for longitudinal control (speed/acceleration)
    - Route loss: for lateral control (steering direction)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 2,
        n_layer: int = 8,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        obs_as_cond: bool = False,
        n_cond_layers: int = 4,
        vl_emb_dim: int = 1536,
        reasoning_emb_dim: int = 1536,
        status_dim: int = 15,
        ego_status_seq_len: int = 1,
        bev_dim: int = 512,
        state_token_dim: int = 128,
        num_waypoints: int = 20,
        history_encoder_type: str = "attention",  # "attention", "gru", or "mlp"
    ) -> None:
        super().__init__()
        
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.num_waypoints = num_waypoints
        self.status_dim = status_dim
        self.vl_emb_dim = vl_emb_dim
        self.reasoning_emb_dim = reasoning_emb_dim
        self.bev_dim = bev_dim
        self.T = horizon
        
        # Input embedding for noisy trajectory
        self.input_emb = nn.Linear(input_dim, n_emb)
        
        # Position embeddings for trajectory queries
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        
        self.drop = nn.Dropout(p_drop_emb)
        self.pre_decoder_norm = nn.LayerNorm(n_emb)
        
        # Conditioning: timestep + current_status + encoded history
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.ego_status_proj = nn.Linear(status_dim, n_emb)  # For current status only
        
        # History encoder (selectable type)
        if history_encoder_type == "attention":
            self.history_encoder = HistoryEncoderAttention(status_dim, n_emb)
        elif history_encoder_type == "gru":
            self.history_encoder = HistoryEncoderGRU(status_dim, n_emb)
        elif history_encoder_type == "mlp":
            self.history_encoder = HistoryEncoderMLP(status_dim, n_emb, max_seq_len=n_obs_steps)
        else:
            raise ValueError(f"Unknown history_encoder_type: {history_encoder_type}")
        
        # Unified Decoder (Lite version)
        self.decoder = UnifiedDecoderLite(
            d_model=n_emb,
            nhead=n_head,
            num_layers=n_layer,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            bev_dim=bev_dim,
            vl_dim=vl_emb_dim,
            reasoning_dim=reasoning_emb_dim,
            horizon=horizon,
            num_waypoints=num_waypoints,
        )
        
        self.causal_attn = causal_attn  # Kept for API compatibility
        
        # Output heads
        self.trajectory_head = TrajectoryMLPHead(n_emb, output_dim, p_drop_emb)
        self.route_head = nn.Sequential(
            nn.LayerNorm(n_emb),
            nn.Linear(n_emb, n_emb),
            nn.GELU(),
            nn.Dropout(p_drop_emb),
            nn.Linear(n_emb, 2),  # Route outputs (x, y) waypoints
        )
        
        self.apply(self._init_weights)
        
        logger.info("TransformerForDiffusionLite (Simplified Decoder-Only) - parameters: %e", 
                   sum(p.numel() for p in self.parameters()))
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param.data)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusionLite):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
    
    def get_optim_groups(self, weight_decay: float = 1e-3):
        decay = set()
        no_decay = set()
        whitelist = (nn.Linear, nn.MultiheadAttention, nn.Conv1d, nn.GRU)
        blacklist = (nn.LayerNorm, nn.Embedding, RMSNorm)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias") or "bias" in pn:
                    no_decay.add(fpn)
                elif "weight" in pn and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        # Parameters that should not decay
        no_decay_patterns = ['pos_emb', '_dummy_variable', 'segment_emb', 'route_queries', 'route_default_positions']
        for name in param_dict:
            if any(pattern in name for pattern in no_decay_patterns):
                no_decay.add(name)
                # Remove from decay if already added
                decay.discard(name)
        
        # Ensure all params are in one of the sets
        for name in param_dict:
            if name not in decay and name not in no_decay:
                # Default to decay for unclassified parameters
                if 'weight' in name:
                    decay.add(name)
                else:
                    no_decay.add(name)
        
        inter_params = decay & no_decay
        union_params = decay | no_decay
        
        # Debug: print problematic params
        if len(inter_params) > 0:
            print(f"Warning: params in both decay and no_decay: {inter_params}")
            # Remove from decay to resolve
            for p in inter_params:
                decay.discard(p)
        
        missing_params = param_dict.keys() - union_params
        if len(missing_params) > 0:
            print(f"Warning: missing params: {missing_params}")
            for p in missing_params:
                no_decay.add(p)
        
        return [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
    
    def configure_optimizers(self, learning_rate: float = 1e-4, weight_decay: float = 1e-3,
                            betas: Tuple[float, float] = (0.9, 0.95)):
        return torch.optim.AdamW(self.get_optim_groups(weight_decay), lr=learning_rate, betas=betas)
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: torch.Tensor,
        bev_tokens: torch.Tensor,
        gen_vit_tokens: torch.Tensor,
        reasoning_query_tokens: torch.Tensor,
        ego_status: torch.Tensor,
        bev_padding_mask: Optional[torch.Tensor] = None,
        vl_mask: Optional[torch.Tensor] = None,
        reasoning_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass with simplified decoder.
        
        Args:
            sample: (B, T, input_dim) - noisy trajectory (positions)
            timestep: diffusion timestep
            cond: (B, T_obs, cond_dim) - for API compatibility (not used)
            bev_tokens: (B, H*W, bev_dim) - BEV spatial tokens
            gen_vit_tokens: (B, T_vl, vl_dim) - VL tokens
            reasoning_query_tokens: (B, T_r, reasoning_dim) - reasoning tokens
            ego_status: (B, T_obs, status_dim) - ego status history (4 frames)
            
        Returns:
            trajectory: (B, T, output_dim) - for longitudinal control
            route_pred: (B, num_waypoints, 2) - for lateral control
            
        Architecture:
            1. AdaLN conditioning: timestep + current_status + GRU(history)
            2. FiLM: Reasoning tokens modulate VL tokens
            3. Decoder layers: Spatial BEV Attention -> Modulated VL Cross-Attention -> FFN
        """
        model_dtype = next(self.parameters()).dtype
        
        sample = sample.contiguous().to(dtype=model_dtype)
        bev_tokens = bev_tokens.contiguous().to(dtype=model_dtype)
        vl_tokens = gen_vit_tokens.contiguous().to(dtype=model_dtype)
        reasoning_tokens = reasoning_query_tokens.contiguous().to(dtype=model_dtype)
        ego_status = ego_status.to(dtype=model_dtype)
        
        B = sample.shape[0]
        T_traj = sample.shape[1]
        
        # Timestep handling
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timesteps = timestep.expand(B)
        
        # ========== Conditioning ==========
        # 1. Timestep embedding
        time_emb = self.time_emb(timesteps).to(dtype=model_dtype)
        
        # 2. Current status embedding (last frame only for AdaLN)
        current_status = ego_status[:, -1, :]  # (B, status_dim)
        status_emb = self.ego_status_proj(current_status)
        
        # 3. GRU-encoded global history for AdaLN
        hist_global_emb = self.history_encoder(ego_status)  # (B, n_emb)
        
        # Combined conditioning for AdaLN modulation
        conditioning = time_emb + status_emb + hist_global_emb
        
        # ========== Trajectory Query Embedding ==========
        traj_emb = self.input_emb(sample)  # (B, T_traj, n_emb)
        pos_emb = self.pos_emb[:, :T_traj, :]
        traj_emb = traj_emb + pos_emb
        traj_emb = self.drop(traj_emb)
        traj_emb = self.pre_decoder_norm(traj_emb)
        
        # ========== Extract trajectory positions for spatial BEV attention ==========
        # sample contains (x, y) positions
        traj_positions = sample[..., :2]  # (B, T_traj, 2)
        
        # ========== Padding Masks ==========
        vl_padding_mask = ~vl_mask if vl_mask is not None else (torch.norm(vl_tokens, dim=-1) == 0)
        # reasoning_padding_mask not directly used (absorbed in FiLM)
        
        # ========== Unified Decoder ==========
        traj_out, route_out = self.decoder(
            traj_emb, traj_positions,
            bev_tokens, vl_tokens, reasoning_tokens,
            conditioning, vl_padding_mask
        )
        
        # ========== Output Heads ==========
        trajectory = self.trajectory_head(traj_out, conditioning)
        route_pred = self.route_head(route_out)
        
        return trajectory, route_pred


# =============================================================================
# Test
# =============================================================================

def test():
    """Test the lite decoder-only architecture."""
    print("=" * 60)
    print("Testing TransformerForDiffusionLite")
    print("Simplified: No Self-Attn, No History Cross-Attn, FiLM, Spatial BEV")
    print("=" * 60)
    
    transformer = TransformerForDiffusionLite(
        input_dim=2,
        output_dim=2,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        n_layer=6,
        n_head=8,
        n_emb=512,
        causal_attn=False,
        vl_emb_dim=1536,
        reasoning_emb_dim=1536,
        status_dim=13,
        bev_dim=512,
        num_waypoints=20,
        history_encoder_type="attention",  # Test attention-based history encoder
    )
    
    print(f"Model parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    B = 4
    timestep = torch.tensor(0)
    sample = torch.randn((B, 8, 2))
    cond = torch.zeros((B, 4, 10))
    bev_tokens = torch.randn((B, 394, 512))  # 1D BEV sequence (seq_len + global token)
    vl_tokens = torch.randn((B, 36, 1536))
    reasoning_tokens = torch.randn((B, 10, 1536))
    ego_status = torch.randn((B, 4, 13))  # 4 frames of history
    
    print("\nTest 1: Basic forward pass")
    trajectory, route_pred = transformer(
        sample=sample, timestep=timestep, cond=cond,
        bev_tokens=bev_tokens,
        gen_vit_tokens=vl_tokens, reasoning_query_tokens=reasoning_tokens,
        ego_status=ego_status,
    )
    print(f"  Trajectory: {trajectory.shape} (for longitudinal control)")
    print(f"  Route: {route_pred.shape} (for lateral control)")
    assert trajectory.shape == (B, 8, 2), f"Expected (B, 8, 2), got {trajectory.shape}"
    assert route_pred.shape == (B, 20, 2), f"Expected (B, 20, 2), got {route_pred.shape}"
    
    print("\nTest 2: Different BEV tokens affect output")
    bev_tokens2 = torch.randn((B, 394, 512))
    traj2, _ = transformer(sample=sample, timestep=timestep, cond=cond,
                          bev_tokens=bev_tokens2,
                          gen_vit_tokens=vl_tokens, reasoning_query_tokens=reasoning_tokens,
                          ego_status=ego_status)
    diff = torch.abs(trajectory - traj2).mean()
    print(f"  Difference: {diff:.6f}")
    assert diff > 0, "BEV should affect output"
    
    print("\nTest 3: Different reasoning affects output (via FiLM)")
    reasoning_tokens2 = torch.randn((B, 10, 1536))
    traj3, _ = transformer(sample=sample, timestep=timestep, cond=cond,
                          bev_tokens=bev_tokens,
                          gen_vit_tokens=vl_tokens, reasoning_query_tokens=reasoning_tokens2,
                          ego_status=ego_status)
    diff_reason = torch.abs(trajectory - traj3).mean()
    print(f"  Difference: {diff_reason:.6f}")
    assert diff_reason > 0, "Reasoning should affect output via FiLM"
    
    print("\nTest 4: Different history affects output")
    ego_status2 = torch.randn((B, 4, 13))
    traj4, _ = transformer(sample=sample, timestep=timestep, cond=cond,
                          bev_tokens=bev_tokens,
                          gen_vit_tokens=vl_tokens, reasoning_query_tokens=reasoning_tokens,
                          ego_status=ego_status2)
    diff_hist = torch.abs(trajectory - traj4).mean()
    print(f"  Difference: {diff_hist:.6f}")
    assert diff_hist > 0, "History should affect output"
    
    print("\nTest 5: Trajectory positions affect position encoding")
    sample2 = torch.randn((B, 8, 2)) * 10  # Different positions
    traj5, _ = transformer(sample=sample2, timestep=timestep, cond=cond,
                          bev_tokens=bev_tokens,
                          gen_vit_tokens=vl_tokens, reasoning_query_tokens=reasoning_tokens,
                          ego_status=ego_status)
    diff_pos = torch.abs(trajectory - traj5).mean()
    print(f"  Difference: {diff_pos:.6f}")
    assert diff_pos > 0, "Trajectory positions should affect output"
    
    print("\nTest 6: Optimizer")
    opt = transformer.configure_optimizers()
    print(f"  Optimizer created: {type(opt).__name__}")
    
    # Test different history encoder types
    print("\n" + "=" * 60)
    print("Test 7: Compare history encoder types")
    print("=" * 60)
    
    for enc_type in ["attention", "gru", "mlp"]:
        model = TransformerForDiffusionLite(
            input_dim=2, output_dim=2, horizon=8, n_obs_steps=4,
            n_layer=2, n_head=4, n_emb=256,
            vl_emb_dim=1536, reasoning_emb_dim=1536, status_dim=13, bev_dim=512,
            history_encoder_type=enc_type,
        )
        params = sum(p.numel() for p in model.history_encoder.parameters())
        
        # Test forward
        with torch.no_grad():
            out, _ = model(
                sample=sample, timestep=timestep, cond=cond,
                bev_tokens=bev_tokens, gen_vit_tokens=vl_tokens,
                reasoning_query_tokens=reasoning_tokens, ego_status=ego_status,
            )
        
        print(f"  {enc_type:10s}: history_encoder params = {params:,}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nArchitecture summary:")
    print("  - Queries: [trajectory (8) | route (20)]")
    print("  - Per-layer: BEV Cross-Attn (pos modulated) -> FiLM-modulated VL Cross-Attn -> FFN")
    print("  - AdaLN conditioning: timestep + current_status + encoded_history")
    print("  - NO self-attention, NO history cross-attention")
    print("  - FiLM: Reasoning modulates VL tokens")
    print("  - BEV: Standard cross-attention with trajectory position modulation")
    print("\nHistory encoder options:")
    print("  - attention: Attention pooling (recommended, best information preservation)")
    print("  - gru: GRU encoding (captures temporal dynamics)")
    print("  - mlp: Simple MLP (fastest, for very short sequences)")
    print("=" * 60)


if __name__ == "__main__":
    test()
