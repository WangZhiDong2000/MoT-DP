"""
Decoder-Only Transformer for Diffusion-based Trajectory Prediction.

This module implements a decoder-only architecture for trajectory prediction,
using sequential cross-attention to multiple condition sources:
- BEV spatial tokens (from InterfuserBEVEncoder)
- State tokens (low-dim ego status)  
- VL tokens (vision-language features)
- Reasoning tokens

Key design choices:
- No encoder: removes attention pooling that loses information
- Sequential cross-attention in each decoder layer
- All condition sources maintain full sequence information
- Unified decoder with heterogeneous queries for trajectory (6) + route (20)
- MLP output heads (no GRU)
"""

from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) module."""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta

        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = position_ids.max().item() + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device)

        B, seq_len = position_ids.shape
        cos = self.cos_cached[:, :, position_ids[0], :].expand(B, -1, -1, -1)
        sin = self.sin_cached[:, :, position_ids[0], :].expand(B, -1, -1, -1)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MultiheadAttentionWithQKNorm(nn.Module):
    """MultiheadAttention with Query-Key Normalization and optional RoPE."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        norm_type: str = "rmsnorm",
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, 
            bias=bias, batch_first=batch_first
        )
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        
        if norm_type == "rmsnorm":
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_position_embeddings, rope_theta)
        
        self.batch_first = batch_first
    
    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q_norm(q), self.k_norm(k)
    
    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, seq_len_q, seq_len_k = q.shape[0], q.shape[2], k.shape[2]
        if seq_len_q != seq_len_k:
            return q, k
        
        position_ids = torch.arange(seq_len_q, device=q.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rope(q, position_ids)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None, need_weights: bool = True,
                attn_mask: Optional[torch.Tensor] = None, average_attn_weights: bool = True):
        if self.batch_first:
            B, seq_len_q, embed_dim = query.shape
            _, seq_len_k, _ = key.shape
            
            q = query.view(B, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
            k = key.view(B, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
            
            q, k = self._apply_qk_norm(q, k)
            if self.use_rope:
                q, k = self._apply_rope(q, k)
            
            query = q.transpose(1, 2).reshape(B, seq_len_q, embed_dim)
            key = k.transpose(1, 2).reshape(B, seq_len_k, embed_dim)
        
        return self.attn(query=query, key=key, value=value, key_padding_mask=key_padding_mask,
                        need_weights=need_weights, attn_mask=attn_mask, average_attn_weights=average_attn_weights)


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
# History Encoder
# =============================================================================

class HistoryEncoder(nn.Module):
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
# Decoder-Only Block
# =============================================================================

class DecoderOnlyBlock(nn.Module):
    """
    Decoder-Only Block with Sequential Cross-Attention to Multiple Condition Sources.
    
    Architecture (per block):
        1. Self-Attention on query (trajectory + route) with causal mask
        2. Cross-Attention to History status tokens (complete low-dim ego status sequence)
        3. Cross-Attention to BEV spatial tokens
        4. Cross-Attention to VL tokens (vision-language features)
        5. Cross-Attention to Reasoning tokens
        6. Feed-Forward Network
        
    Note: State tokens from BEV encoder are NOT used here since History already
    contains complete low-dim status information (speed, theta, command, etc.)
        
    All operations use AdaLN modulation conditioned on timestep + current_ego_status.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, batch_first: bool = True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # 1. Self-Attention
        self.self_attn = MultiheadAttentionWithQKNorm(
            d_model, nhead, dropout=dropout, batch_first=batch_first, norm_type="rmsnorm"
        )
        
        # 2-5. Cross-Attention layers (no State cross-attn since History contains status)
        self.hist_cross_attn = MultiheadAttentionWithQKNorm(
            d_model, nhead, dropout=dropout, batch_first=batch_first, norm_type="rmsnorm"
        )
        self.bev_cross_attn = MultiheadAttentionWithQKNorm(
            d_model, nhead, dropout=dropout, batch_first=batch_first, norm_type="rmsnorm"
        )
        self.vl_cross_attn = MultiheadAttentionWithQKNorm(
            d_model, nhead, dropout=dropout, batch_first=batch_first, norm_type="rmsnorm"
        )
        self.reasoning_cross_attn = MultiheadAttentionWithQKNorm(
            d_model, nhead, dropout=dropout, batch_first=batch_first, norm_type="rmsnorm"
        )
        
        # 6. FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Layer Norms (6 groups: self, hist, bev, vl, reasoning, ffn)
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_hist = nn.LayerNorm(d_model)
        self.norm_bev = nn.LayerNorm(d_model)
        self.norm_vl = nn.LayerNorm(d_model)
        self.norm_reasoning = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        
        self.activation = nn.GELU()
        
        # AdaLN modulation - 18 params: 6 groups x 3 (shift, scale, gate)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 18 * d_model, bias=True)
        )
    
    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
    def forward(
        self,
        x: torch.Tensor,
        hist_tokens: torch.Tensor,  # (B, To, d_model) - history status sequence
        bev_tokens: torch.Tensor,
        vl_tokens: torch.Tensor,
        reasoning_tokens: torch.Tensor,
        conditioning: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        hist_padding_mask: Optional[torch.Tensor] = None,
        bev_padding_mask: Optional[torch.Tensor] = None,
        vl_padding_mask: Optional[torch.Tensor] = None,
        reasoning_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Generate modulation parameters
        mod_params = self.adaLN_modulation(conditioning)
        (shift_self, scale_self, gate_self,
         shift_hist, scale_hist, gate_hist,
         shift_bev, scale_bev, gate_bev,
         shift_vl, scale_vl, gate_vl,
         shift_reason, scale_reason, gate_reason,
         shift_ffn, scale_ffn, gate_ffn) = mod_params.chunk(18, dim=1)
        
        # 1. Self-Attention
        x_norm = self.modulate(self.norm_self(x), shift_self, scale_self)
        x_self, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=self_attn_mask)
        x = x + gate_self.unsqueeze(1) * x_self
        
        # 2. Cross-Attention to History status (contains complete low-dim state info)
        x_norm = self.modulate(self.norm_hist(x), shift_hist, scale_hist)
        x_hist, _ = self.hist_cross_attn(x_norm, hist_tokens, hist_tokens, key_padding_mask=hist_padding_mask)
        x = x + gate_hist.unsqueeze(1) * x_hist
        
        # 3. Cross-Attention to BEV
        x_norm = self.modulate(self.norm_bev(x), shift_bev, scale_bev)
        x_bev, _ = self.bev_cross_attn(x_norm, bev_tokens, bev_tokens, key_padding_mask=bev_padding_mask)
        x = x + gate_bev.unsqueeze(1) * x_bev
        
        # 4. Cross-Attention to VL
        x_norm = self.modulate(self.norm_vl(x), shift_vl, scale_vl)
        x_vl, _ = self.vl_cross_attn(x_norm, vl_tokens, vl_tokens, key_padding_mask=vl_padding_mask)
        x = x + gate_vl.unsqueeze(1) * x_vl
        
        # 5. Cross-Attention to Reasoning
        x_norm = self.modulate(self.norm_reasoning(x), shift_reason, scale_reason)
        x_reason, _ = self.reasoning_cross_attn(x_norm, reasoning_tokens, reasoning_tokens, 
                                                 key_padding_mask=reasoning_padding_mask)
        x = x + gate_reason.unsqueeze(1) * x_reason
        
        # 6. FFN
        x_norm = self.modulate(self.norm_ffn(x), shift_ffn, scale_ffn)
        x_ffn = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = x + gate_ffn.unsqueeze(1) * x_ffn
        
        return x


# =============================================================================
# Decoder-Only Transformer
# =============================================================================

class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-Only Transformer with sequential cross-attention to multiple condition sources.
    """
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        bev_dim: int = 512,
        state_dim: int = 128,
        vl_dim: int = 1536,
        reasoning_dim: int = 1536,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Projection layers
        self.bev_proj = nn.Sequential(nn.Linear(bev_dim, d_model), nn.LayerNorm(d_model))
        self.state_proj = nn.Sequential(nn.Linear(state_dim, d_model), nn.LayerNorm(d_model))
        self.vl_proj = nn.Sequential(nn.Linear(vl_dim, d_model), nn.LayerNorm(d_model))
        self.reasoning_proj = nn.Sequential(nn.Linear(reasoning_dim, d_model), nn.LayerNorm(d_model))
        
        # BEV position embedding
        self.bev_pos_emb = nn.Parameter(torch.zeros(1, 256, d_model))
        
        # Decoder blocks
        self.layers = nn.ModuleList([
            DecoderOnlyBlock(d_model, nhead, dim_feedforward, dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.bev_pos_emb, mean=0.0, std=0.02)
        for layer in self.layers:
            nn.init.zeros_(layer.adaLN_modulation[-1].weight)
            nn.init.zeros_(layer.adaLN_modulation[-1].bias)
    
    def forward(
        self,
        x: torch.Tensor,
        bev_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        vl_tokens: torch.Tensor,
        reasoning_tokens: torch.Tensor,
        conditioning: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        bev_padding_mask: Optional[torch.Tensor] = None,
        state_padding_mask: Optional[torch.Tensor] = None,
        vl_padding_mask: Optional[torch.Tensor] = None,
        reasoning_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Project conditions
        bev_proj = self.bev_proj(bev_tokens)
        bev_seq_len = bev_proj.shape[1]
        if bev_seq_len <= self.bev_pos_emb.shape[1]:
            bev_proj = bev_proj + self.bev_pos_emb[:, :bev_seq_len, :]
        
        state_proj = self.state_proj(state_tokens)
        vl_proj = self.vl_proj(vl_tokens)
        reasoning_proj = self.reasoning_proj(reasoning_tokens)
        
        # Decoder layers
        for layer in self.layers:
            x = layer(x, bev_proj, state_proj, vl_proj, reasoning_proj, conditioning,
                     self_attn_mask, bev_padding_mask, state_padding_mask, 
                     vl_padding_mask, reasoning_padding_mask)
        
        return self.final_norm(x)


# =============================================================================
# Route Head (Decoder-Only) - DEPRECATED, kept for reference
# =============================================================================

class AdaLNRouteHeadDecoderOnly(nn.Module):
    """Route prediction head using sequential cross-attention to conditions.
    
    DEPRECATED: Use unified decoder with heterogeneous queries instead.
    """
    def __init__(
        self,
        n_emb: int,
        num_waypoints: int = 20,
        status_dim: int = 12,
        n_layers: int = 2,
        n_head: int = 8,
        p_drop: float = 0.1,
        bev_dim: int = 512,
        state_dim: int = 128,
        vl_dim: int = 1536,
        reasoning_dim: int = 1536,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.n_emb = n_emb
        
        self.status_proj = nn.Sequential(
            nn.Linear(status_dim, n_emb), nn.SiLU(), nn.Linear(n_emb, n_emb)
        )
        
        self.route_queries = nn.Parameter(torch.randn(1, num_waypoints, n_emb))
        self.route_pos_emb = nn.Parameter(torch.randn(1, num_waypoints, n_emb))
        
        self.bev_proj = nn.Sequential(nn.Linear(bev_dim, n_emb), nn.LayerNorm(n_emb))
        self.state_proj = nn.Sequential(nn.Linear(state_dim, n_emb), nn.LayerNorm(n_emb))
        self.vl_proj = nn.Sequential(nn.Linear(vl_dim, n_emb), nn.LayerNorm(n_emb))
        self.reasoning_proj = nn.Sequential(nn.Linear(reasoning_dim, n_emb), nn.LayerNorm(n_emb))
        
        self.blocks = nn.ModuleList([
            DecoderOnlyBlock(n_emb, n_head, 4 * n_emb, p_drop, batch_first=True)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(n_emb)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(n_emb, 2 * n_emb, bias=True))
        self.output_proj = nn.Linear(n_emb, 2)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.route_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.route_pos_emb, mean=0.0, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
    
    def forward(
        self,
        bev_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        vl_tokens: torch.Tensor,
        reasoning_tokens: torch.Tensor,
        ego_status: torch.Tensor,
        conditioning: torch.Tensor,
        bev_padding_mask: Optional[torch.Tensor] = None,
        state_padding_mask: Optional[torch.Tensor] = None,
        vl_padding_mask: Optional[torch.Tensor] = None,
        reasoning_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = bev_tokens.shape[0]
        
        if ego_status.dim() == 3:
            ego_status = ego_status[:, -1, :]
        
        route_conditioning = self.status_proj(ego_status) + conditioning
        x = self.route_queries.expand(B, -1, -1) + self.route_pos_emb
        
        bev_proj = self.bev_proj(bev_tokens)
        state_proj = self.state_proj(state_tokens)
        vl_proj = self.vl_proj(vl_tokens)
        reasoning_proj = self.reasoning_proj(reasoning_tokens)
        
        for block in self.blocks:
            x = block(x, bev_proj, state_proj, vl_proj, reasoning_proj, route_conditioning,
                     None, bev_padding_mask, state_padding_mask, vl_padding_mask, reasoning_padding_mask)
        
        x = self.final_norm(x)
        shift, scale = self.final_adaLN(route_conditioning).chunk(2, dim=1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        return self.output_proj(x)


# =============================================================================
# Unified Decoder with Heterogeneous Queries
# =============================================================================

class UnifiedDecoderOnlyTransformer(nn.Module):
    """
    Unified Decoder-Only Transformer with heterogeneous queries.
    
    Combines trajectory prediction (horizon points) and route prediction (20 waypoints)
    into a single decoder, using:
    - Segment embeddings to distinguish query types (trajectory vs route)
    - History status as cross-attention source (contains complete low-dim state info)
    - Shared cross-attention to all condition sources
    - Separate output heads for trajectory and route
    
    Query structure: [trajectory (horizon) | route (num_waypoints)]
    Cross-attention sources: History -> BEV -> VL -> Reasoning
    
    Note: No separate State cross-attention since History already contains
    complete low-dim status (speed, theta, command, target_point, waypoints).
    """
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        status_dim: int = 13,
        bev_dim: int = 512,
        vl_dim: int = 1536,
        reasoning_dim: int = 1536,
        horizon: int = 8,
        num_waypoints: int = 20,
        n_obs_steps: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.horizon = horizon
        self.num_waypoints = num_waypoints
        self.n_obs_steps = n_obs_steps
        
        # Projection layers for conditions
        self.hist_proj = nn.Sequential(nn.Linear(status_dim, d_model), nn.LayerNorm(d_model))
        self.bev_proj = nn.Sequential(nn.Linear(bev_dim, d_model), nn.LayerNorm(d_model))
        self.vl_proj = nn.Sequential(nn.Linear(vl_dim, d_model), nn.LayerNorm(d_model))
        self.reasoning_proj = nn.Sequential(nn.Linear(reasoning_dim, d_model), nn.LayerNorm(d_model))
        
        # Position embeddings for cross-attention sources
        self.hist_pos_emb = nn.Parameter(torch.zeros(1, n_obs_steps, d_model))
        self.bev_pos_emb = nn.Parameter(torch.zeros(1, 256, d_model))
        
        # Route queries (learnable)
        self.route_queries = nn.Parameter(torch.randn(1, num_waypoints, d_model))
        
        # Position embeddings for queries
        self.route_pos_emb = nn.Parameter(torch.zeros(1, num_waypoints, d_model))
        
        # Segment embeddings to distinguish query types
        self.traj_segment_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.route_segment_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Decoder blocks
        self.layers = nn.ModuleList([
            DecoderOnlyBlock(d_model, nhead, dim_feedforward, dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.hist_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.bev_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.route_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.route_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.traj_segment_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.route_segment_emb, mean=0.0, std=0.02)
        for layer in self.layers:
            nn.init.zeros_(layer.adaLN_modulation[-1].weight)
            nn.init.zeros_(layer.adaLN_modulation[-1].bias)
    
    def forward(
        self,
        traj_emb: torch.Tensor,  # (B, horizon, d_model) - trajectory query embeddings
        hist_status: torch.Tensor,  # (B, To, status_dim) - raw history status for cross-attention
        bev_tokens: torch.Tensor,
        vl_tokens: torch.Tensor,
        reasoning_tokens: torch.Tensor,
        conditioning: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        hist_padding_mask: Optional[torch.Tensor] = None,
        bev_padding_mask: Optional[torch.Tensor] = None,
        vl_padding_mask: Optional[torch.Tensor] = None,
        reasoning_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with unified queries.
        
        Args:
            traj_emb: (B, horizon, d_model) - trajectory query embeddings (noisy traj embedded)
            hist_status: (B, To, status_dim) - raw history status sequence
            bev_tokens: (B, seq_len, bev_dim) - BEV spatial tokens
            vl_tokens: (B, T_vl, vl_dim) - VL tokens
            reasoning_tokens: (B, T_r, reasoning_dim) - reasoning tokens
            conditioning: (B, d_model) - timestep + current_status conditioning
            
        Returns:
            traj_out: (B, horizon, d_model) - trajectory output
            route_out: (B, num_waypoints, d_model) - route output
        """
        B = traj_emb.shape[0]
        T_traj = traj_emb.shape[1]
        To = hist_status.shape[1]
        
        # Add segment embedding to trajectory
        traj_emb = traj_emb + self.traj_segment_emb
        
        # Route queries with position and segment embeddings
        route_emb = self.route_queries.expand(B, -1, -1) + self.route_pos_emb + self.route_segment_emb
        
        # Concatenate queries: [trajectory | route]
        x = torch.cat([traj_emb, route_emb], dim=1)  # (B, horizon + num_waypoints, d_model)
        
        # Project history status to d_model and add position embedding
        hist_proj = self.hist_proj(hist_status)  # (B, To, d_model)
        hist_pos = self.hist_pos_emb[:, :To, :] if To <= self.hist_pos_emb.shape[1] else self.hist_pos_emb
        hist_proj = hist_proj + hist_pos
        
        # Project other conditions
        bev_proj = self.bev_proj(bev_tokens)
        bev_seq_len = bev_proj.shape[1]
        if bev_seq_len <= self.bev_pos_emb.shape[1]:
            bev_proj = bev_proj + self.bev_pos_emb[:, :bev_seq_len, :]
        
        vl_proj = self.vl_proj(vl_tokens)
        reasoning_proj = self.reasoning_proj(reasoning_tokens)
        
        # Decoder layers: History -> BEV -> VL -> Reasoning
        for layer in self.layers:
            x = layer(x, hist_proj, bev_proj, vl_proj, reasoning_proj, conditioning,
                     self_attn_mask, hist_padding_mask, bev_padding_mask,
                     vl_padding_mask, reasoning_padding_mask)
        
        x = self.final_norm(x)
        
        # Split outputs
        traj_out = x[:, :T_traj, :]
        route_out = x[:, T_traj:, :]
        
        return traj_out, route_out


# =============================================================================
# Main Model: TransformerForDiffusion (Decoder-Only with Unified Queries)
# =============================================================================

class TransformerForDiffusion(ModuleAttrMixin):
    """
    Decoder-Only Transformer for Diffusion-based Trajectory Prediction.
    
    Key features:
    - Unified decoder with heterogeneous queries for trajectory + route
    - Sequential cross-attention to: History -> BEV -> VL -> Reasoning
    - History status as cross-attention source (complete 4-frame sequence)
    - No separate State cross-attention (History contains complete low-dim state)
    - Segment embeddings to distinguish query types
    - MLP output heads (no GRU)
    - AdaLN modulation based on timestep + current_ego_status
    
    Query structure: [trajectory (horizon) | route (20)]
    Cross-attention sources: History (4 frames) -> BEV -> VL -> Reasoning
    
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
        n_layer: int = 12,
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
        
        # Conditioning: timestep + current_status + GRU-encoded history
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.ego_status_proj = nn.Linear(status_dim, n_emb)  # For current status only
        self.history_encoder = HistoryEncoder(status_dim, n_emb)  # GRU for global history encoding
        
        # Unified Decoder with heterogeneous queries
        # History status is cross-attention source (no separate State cross-attention)
        self.decoder = UnifiedDecoderOnlyTransformer(
            d_model=n_emb,
            nhead=n_head,
            num_layers=n_layer,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            status_dim=status_dim,  # For history projection inside decoder
            bev_dim=bev_dim,
            vl_dim=vl_emb_dim,
            reasoning_dim=reasoning_emb_dim,
            horizon=horizon,
            num_waypoints=num_waypoints,
            n_obs_steps=n_obs_steps,
        )
        
        # Causal mask for trajectory queries (route queries can attend to all)
        self.causal_attn = causal_attn
        # Note: We create mask dynamically in forward() to handle variable lengths
        
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
        
        logger.info("TransformerForDiffusion (Unified Decoder-Only) - parameters: %e", 
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
        elif isinstance(module, TransformerForDiffusion):
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
        for name in param_dict:
            if 'pos_emb' in name or '_dummy_variable' in name or 'segment_emb' in name:
                no_decay.add(name)
            elif 'route_queries' in name or 'pool_query' in name:
                no_decay.add(name)
        
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0
        
        return [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
    
    def configure_optimizers(self, learning_rate: float = 1e-4, weight_decay: float = 1e-3,
                            betas: Tuple[float, float] = (0.9, 0.95)):
        return torch.optim.AdamW(self.get_optim_groups(weight_decay), lr=learning_rate, betas=betas)
    
    def _create_unified_attn_mask(self, T_traj: int, T_route: int, device: torch.device, dtype: torch.dtype):
        """
        Create attention mask for unified queries.
        
        Structure: [trajectory (T_traj) | route (T_route)]
        
        Mask rules:
        - Trajectory tokens: causal within trajectory
        - Route tokens: can attend to everything (no causal constraint)
        """
        total_len = T_traj + T_route
        
        if not self.causal_attn:
            return None
        
        # Start with full attention allowed
        mask = torch.zeros(total_len, total_len, device=device, dtype=dtype)
        
        # Trajectory: causal within trajectory
        for i in range(T_traj):
            mask[i, i+1:T_traj] = float('-inf')  # Can't attend to future trajectory
        
        # Route: can attend to everything (no masking)
        # Already zeros, so nothing to do
        
        return mask
    
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
        Forward pass with unified decoder.
        
        Args:
            sample: (B, T, input_dim) - noisy trajectory
            timestep: diffusion timestep
            cond: (B, T_obs, cond_dim) - for API compatibility (not used)
            bev_tokens: (B, seq_len, bev_dim) - BEV spatial tokens (last frame only)
            gen_vit_tokens: (B, T_vl, vl_dim) - VL tokens
            reasoning_query_tokens: (B, T_r, reasoning_dim) - reasoning tokens
            ego_status: (B, T_obs, status_dim) - ego status history (complete 4 frames)
            
        Returns:
            trajectory: (B, T, output_dim) - for longitudinal control
            route_pred: (B, num_waypoints, 2) - for lateral control
            
        History status utilization:
            1. AdaLN conditioning: current_status (last frame) + GRU-encoded global history
            2. Cross-attention: complete history sequence as first cross-attention source
            
        Note: state_tokens from BEV encoder are NOT used since ego_status
        already contains complete low-dim state info (speed, theta, command, etc.)
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
        
        # ========== Create Attention Mask ==========
        self_attn_mask = self._create_unified_attn_mask(
            T_traj, self.num_waypoints, sample.device, model_dtype
        )
        
        # ========== Padding Masks ==========
        vl_padding_mask = ~vl_mask if vl_mask is not None else (torch.norm(vl_tokens, dim=-1) == 0)
        reasoning_padding_mask = ~reasoning_mask if reasoning_mask is not None else (torch.norm(reasoning_tokens, dim=-1) == 0)
        
        # ========== Unified Decoder ==========
        # Cross-attention order: History -> BEV -> VL -> Reasoning
        traj_out, route_out = self.decoder(
            traj_emb,
            ego_status,  # Complete history for cross-attention (B, T_obs, status_dim)
            bev_tokens, vl_tokens, reasoning_tokens, conditioning,
            self_attn_mask, None, bev_padding_mask, vl_padding_mask, reasoning_padding_mask
        )
        
        # ========== Output Heads ==========
        trajectory = self.trajectory_head(traj_out, conditioning)
        route_pred = self.route_head(route_out)
        
        return trajectory, route_pred


# =============================================================================
# Test
# =============================================================================

def test():
    """Test the unified decoder-only architecture with history as cross-attention."""
    print("=" * 60)
    print("Testing TransformerForDiffusion (Unified Decoder-Only)")
    print("History as Cross-Attention Source (No State Cross-Attn)")
    print("=" * 60)
    
    transformer = TransformerForDiffusion(
        input_dim=2,
        output_dim=2,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        n_layer=6,
        n_head=8,
        n_emb=512,
        causal_attn=True,
        vl_emb_dim=1536,
        reasoning_emb_dim=1536,
        status_dim=13,
        bev_dim=512,
        num_waypoints=20,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    B = 4
    timestep = torch.tensor(0)
    sample = torch.randn((B, 8, 2))
    cond = torch.zeros((B, 4, 10))
    bev_tokens = torch.randn((B, 197, 512))
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
    bev_tokens2 = torch.randn((B, 197, 512))
    traj2, _ = transformer(sample=sample, timestep=timestep, cond=cond,
                          bev_tokens=bev_tokens2,
                          gen_vit_tokens=vl_tokens, reasoning_query_tokens=reasoning_tokens,
                          ego_status=ego_status)
    diff = torch.abs(trajectory - traj2).mean()
    print(f"  Difference: {diff:.6f}")
    assert diff > 0, "BEV should affect output"
    
    print("\nTest 3: Different history affects output")
    ego_status2 = torch.randn((B, 4, 13))
    traj3, _ = transformer(sample=sample, timestep=timestep, cond=cond,
                          bev_tokens=bev_tokens,
                          gen_vit_tokens=vl_tokens, reasoning_query_tokens=reasoning_tokens,
                          ego_status=ego_status2)
    diff_hist = torch.abs(trajectory - traj3).mean()
    print(f"  Difference: {diff_hist:.6f}")
    assert diff_hist > 0, "History should affect output"
    
    print("\nTest 4: Optimizer")
    opt = transformer.configure_optimizers()
    print(f"  Optimizer created: {type(opt).__name__}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nArchitecture summary:")
    print("  - Queries: [trajectory (8) | route (20)]")
    print("  - Cross-attention order: History(4) -> BEV(197) -> VL -> Reasoning")
    print("  - AdaLN conditioning: timestep + current_status + GRU(history)")
    print("  - No State cross-attention (History contains complete low-dim state)")
    print("\nControl design:")
    print("  - Trajectory output: longitudinal control (speed/acceleration)")
    print("  - Route output: lateral control (steering direction)")
    print("=" * 60)


if __name__ == "__main__":
    test()
