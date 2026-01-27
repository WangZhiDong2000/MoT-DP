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


def apply_rope_single(x, cos, sin):
    """Apply RoPE to a single tensor (only K, not Q)."""
    return (x * cos) + (rotate_half(x) * sin)


# =============================================================================
# Sparse Instance Spatial Cross Attention for SparseDrive Features
# =============================================================================

class SparseInstanceSpatialAttention(nn.Module):
    """
    Simplified Spatial Cross Attention for SparseDrive sparse instance features.
    
    Key improvements over previous version:
    1. Use learned position embeddings instead of explicit distance computation
    2. Don't rely on coordinate system alignment (let model learn it)
    3. Simpler architecture with better gradient flow
    4. Position info is encoded and added to features, not used for explicit bias
    
    SparseDrive features structure:
    - det_instance_feature: (B, 50, 256) - 50 detection instances
    - det_prediction: (B, 50, 11) - positions (x, y, z, w, l, h, sin, cos, vx, vy, vz)
    
    Architecture:
    1. Encode detection positions into embeddings
    2. Encode trajectory positions (normalized) into embeddings  
    3. Standard cross-attention with position-enhanced features
    4. Gated residual connection
    """
    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        num_det_instances: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_det_instances = num_det_instances
        
        # Position encoders - let the model learn coordinate transformation
        # Detection: (x, y, z, w, l, h, sin, cos, vx, vy, vz) -> embed_dims
        self.det_pos_encoder = nn.Sequential(
            nn.Linear(11, embed_dims // 2),
            nn.GELU(),
            nn.Linear(embed_dims // 2, embed_dims),
        )
        
        # Trajectory: (x, y) normalized -> embed_dims
        self.traj_pos_encoder = nn.Sequential(
            nn.Linear(2, embed_dims // 2),
            nn.GELU(),
            nn.Linear(embed_dims // 2, embed_dims),
        )
        
        # Standard multi-head cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projection with gating
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.spatial_gate = nn.Parameter(torch.zeros(1))  # Initialize to 0 for stable training
        
        self.layer_norm = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.det_pos_encoder, self.traj_pos_encoder]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        traj_queries: torch.Tensor,      # (B, T, embed_dims) - trajectory query features
        traj_points: torch.Tensor,        # (B, T, 2) - trajectory positions (normalized)
        det_features: torch.Tensor,       # (B, 50, embed_dims) - detection instance features
        det_positions: torch.Tensor,      # (B, 50, 11) - detection predictions
        det_mask: Optional[torch.Tensor] = None,  # (B, 50) - mask for valid detections
    ) -> torch.Tensor:
        """
        Simplified spatial cross-attention.
        
        Key insight: Let the model learn coordinate transformation through position encodings
        instead of hard-coding coordinate alignment.
        
        Args:
            traj_queries: Trajectory query features (B, T, embed_dims)
            traj_points: Trajectory positions normalized to [-1, 1] (B, T, 2)
            det_features: Detection instance features (B, 50, embed_dims)
            det_positions: Detection predictions (B, 50, 11) - (x,y,z,w,l,h,sin,cos,vx,vy,vz)
            det_mask: Mask for valid detections, True = valid (B, 50)
            
        Returns:
            Enhanced trajectory features (B, T, embed_dims)
        """
        B, T, _ = traj_queries.shape
        
        # Encode positions - let model learn the coordinate transformation
        det_pos_emb = self.det_pos_encoder(det_positions)  # (B, 50, embed_dims)
        traj_pos_emb = self.traj_pos_encoder(traj_points)  # (B, T, embed_dims)
        
        # Add position embeddings to features
        det_features_with_pos = det_features + det_pos_emb  # (B, 50, embed_dims)
        traj_queries_with_pos = traj_queries + traj_pos_emb  # (B, T, embed_dims)
        
        # Prepare attention mask
        key_padding_mask = None
        if det_mask is not None:
            key_padding_mask = ~det_mask  # True = padding (to be masked)
        
        # Cross attention: trajectory queries attend to detection features
        attn_out, _ = self.cross_attn(
            query=traj_queries_with_pos,
            key=det_features_with_pos,
            value=det_features,  # Use original features without position for value
            key_padding_mask=key_padding_mask,
        )
        
        # Project and apply gating
        out = self.output_proj(attn_out)
        gate = torch.sigmoid(self.spatial_gate)
        out = gate * out
        
        # Residual connection with layer norm
        out = self.layer_norm(traj_queries + self.dropout(out))
        
        return out


class MultiSourceAttentionBlock(nn.Module):
    """
    Multi-Source Attention Block with separate projections for different sources.
    
    Key features:
    - Multi-head self-attention with RoPE
    - Cross-attention to BEV, Reasoning, and Action tokens
    - QK Normalization (RMSNorm) for stable training
    - Gating mechanism for Reasoning and Action tokens
    - Source-specific residual paths (main + branch) for better information flow
    - AdaLN modulation for timestep conditioning
    
    Note: Low-dim state (hist_tokens) is NOT used for cross-attention.
    It is only used for AdaLN conditioning through the conditioning vector.
    
    Architecture:
        1. Self-attention on main sequence with RoPE on both Q and K
        2. Cross-attention to BEV tokens with RoPE on K only
        3. Cross-attention to Reasoning + Action tokens with RoPE on K only + gating
        4. Source-specific residual paths (BEV branch, Reasoning branch)
        5. Residual + FFN
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, rope_theta: float = 10000.0, max_seq_len: int = 512,
                 norm_type: str = "rmsnorm"):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        # RoPE embedding
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        
        # ========== QK Normalization ==========
        if norm_type == "rmsnorm":
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        
        # ========== Q projections (source-specific for better modality adaptation) ==========
        self.q_proj = nn.Linear(d_model, d_model)  # Shared base Q
        # Small adaptation layers for cross-attention queries (low-rank for efficiency)
        self.q_adapter_bev = nn.Linear(d_model, d_model // 4)
        self.q_adapter_reason = nn.Linear(d_model, d_model // 4)
        self.q_adapter_action = nn.Linear(d_model, d_model // 4)  # For action tokens
        self.q_adapter_out = nn.Linear(d_model // 4, d_model)
        
        # ========== Self-attention K, V ==========
        self.k_self = nn.Linear(d_model, d_model)
        self.v_self = nn.Linear(d_model, d_model)
        
        # ========== BEV cross-attention K, V ==========
        self.k_bev = nn.Linear(d_model, d_model)
        self.v_bev = nn.Linear(d_model, d_model)
        
        # ========== Task cross-attention K, V (for reasoning and action) ==========
        self.k_reasoning = nn.Linear(d_model, d_model)
        self.v_reasoning = nn.Linear(d_model, d_model)
        self.k_action = nn.Linear(d_model, d_model)  # For action tokens
        self.v_action = nn.Linear(d_model, d_model)  # For action tokens
        
        # ========== Output projection ==========
        self.o_proj = nn.Linear(d_model, d_model)
        
        # ========== Gating factors for Reasoning and Action tokens ==========
        # Initialize to small positive value for stable training start
        self.gating_factor = nn.Parameter(torch.tensor(0.1))  # For reasoning
        self.gating_factor_action = nn.Parameter(torch.tensor(0.1))  # For action
        
        # ========== Per-source learnable temperature (log scale for stability) ==========
        self.temp_self = nn.Parameter(torch.zeros(1))
        self.temp_bev = nn.Parameter(torch.zeros(1))
        self.temp_reason = nn.Parameter(torch.zeros(1))
        self.temp_action = nn.Parameter(torch.zeros(1))
        
        # ========== Per-source learnable bias (attention prior) ==========
        self.bias_self = nn.Parameter(torch.zeros(1))
        self.bias_bev = nn.Parameter(torch.zeros(1))
        self.bias_reason = nn.Parameter(torch.zeros(1))
        self.bias_action = nn.Parameter(torch.zeros(1))
        
        # ========== Source-Specific Residual Paths (main + branch) ==========
        # BEV residual path: pooling + MLP projection
        self.bev_residual_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.bev_residual_gate = nn.Parameter(torch.zeros(1))
        
        # Reasoning residual path: attention pooling (query-based)
        self.reason_residual_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.reason_residual_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
        self.reason_residual_proj = nn.Linear(d_model, d_model)
        self.reason_residual_gate = nn.Parameter(torch.zeros(1))
        
        # ========== Dropout layers for regularization ==========
        self.dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)  # Dropout after output projection
        
        # ========== FFN ==========
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        
        # ========== Layer Norm for pre-attention ==========
        self.norm_pre = nn.LayerNorm(d_model)
        
        # ========== AdaLN modulation - 6 params: shift, scale, gate ==========
        # (shift_pre, scale_pre, gate_attn, shift_ffn, scale_ffn, gate_ffn)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply QK normalization to query and key tensors."""
        return self.q_norm(q), self.k_norm(k)
    
    def _get_rope_embed(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Generate RoPE cos/sin embeddings."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)
    
    def _reshape_heads(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        """Reshape (B, L, d_model) -> (B, nhead, L, head_dim)"""
        return x.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
    
    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
    def forward(
        self,
        x: torch.Tensor,  # (B, T, d_model) - main sequence
        bev_tokens: torch.Tensor,   # (B, T_b, d_model) - BEV tokens (already projected)
        reasoning_tokens: torch.Tensor,  # (B, T_r, d_model) - reasoning tokens (already projected)
        action_tokens: torch.Tensor,  # (B, T_a, d_model) - action tokens (already projected)
        conditioning: torch.Tensor,  # (B, d_model) - conditioning for AdaLN (contains low-dim state info)
        self_attn_mask: Optional[torch.Tensor] = None,
        bev_padding_mask: Optional[torch.Tensor] = None,
        reasoning_padding_mask: Optional[torch.Tensor] = None,
        action_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with multi-source attention.
        
        Note: Low-dim state is NOT used for cross-attention here.
        It is already incorporated into the conditioning vector for AdaLN modulation.
        
        Cross-attention sources:
        1. BEV tokens (SparseDrive sparse features)
        2. Reasoning tokens (VLM reasoning hidden states)
        3. Action tokens (VLM action hidden states)
        
        Residual paths (main + branch):
        - Main path: attention output
        - BEV branch: BEV pooling → MLP → gated add
        - Reasoning branch: attention pooling → MLP → gated add
        """
        B, T, C = x.shape
        T_b = bev_tokens.shape[1]
        T_r = reasoning_tokens.shape[1]
        T_a = action_tokens.shape[1]
        
        # ========== AdaLN modulation parameters ==========
        mod_params = self.adaLN_modulation(conditioning)
        shift_pre, scale_pre, gate_attn, shift_ffn, scale_ffn, gate_ffn = mod_params.chunk(6, dim=1)
        
        # ========== Pre-LayerNorm with modulation ==========
        x_norm = self.modulate(self.norm_pre(x), shift_pre, scale_pre)
        
        # ========== Gating factors ==========
        g = self.gating_factor
        ratio_g = torch.tanh(g)
        g_action = self.gating_factor_action
        ratio_g_action = torch.tanh(g_action)
        
        # ========== Temperature scaling factors ==========
        temp_self = 1.0 + torch.nn.functional.softplus(self.temp_self)
        temp_bev = 1.0 + torch.nn.functional.softplus(self.temp_bev)
        temp_reason = 1.0 + torch.nn.functional.softplus(self.temp_reason)
        temp_action = 1.0 + torch.nn.functional.softplus(self.temp_action)
        
        # ========== Q projection ==========
        q_base = self.q_proj(x_norm)  # (B, T, d_model)
        
        # Source-specific Q adaptations
        q_adapt_bev = self.q_adapter_out(torch.tanh(self.q_adapter_bev(x_norm)))
        q_adapt_reason = self.q_adapter_out(torch.tanh(self.q_adapter_reason(x_norm)))
        q_adapt_action = self.q_adapter_out(torch.tanh(self.q_adapter_action(x_norm)))
        
        # ========== Self-attention K, V ==========
        k_self = self.k_self(x_norm)
        v_self = self.v_self(x_norm)
        
        # ========== Cross-attention K, V ==========
        k_bev = self.k_bev(bev_tokens)
        v_bev = self.v_bev(bev_tokens)
        k_reason = self.k_reasoning(reasoning_tokens)
        v_reason = self.v_reasoning(reasoning_tokens)
        k_action = self.k_action(action_tokens)
        v_action = self.v_action(action_tokens)
        
        # ========== Reshape to multi-head ==========
        q_base = self._reshape_heads(q_base, B, T)
        q_bev = self._reshape_heads(q_base.transpose(1, 2).reshape(B, T, C) + q_adapt_bev, B, T)
        q_reason = self._reshape_heads(q_base.transpose(1, 2).reshape(B, T, C) + q_adapt_reason, B, T)
        q_action = self._reshape_heads(q_base.transpose(1, 2).reshape(B, T, C) + q_adapt_action, B, T)
        
        k_self = self._reshape_heads(k_self, B, T)
        v_self = self._reshape_heads(v_self, B, T)
        k_bev = self._reshape_heads(k_bev, B, T_b)
        v_bev = self._reshape_heads(v_bev, B, T_b)
        k_reason = self._reshape_heads(k_reason, B, T_r)
        v_reason = self._reshape_heads(v_reason, B, T_r)
        k_action = self._reshape_heads(k_action, B, T_a)
        v_action = self._reshape_heads(v_action, B, T_a)
        
        # ========== Apply QK Normalization ==========
        q_base, k_self = self._apply_qk_norm(q_base, k_self)
        q_bev, k_bev = self._apply_qk_norm(q_bev, k_bev)
        q_reason, k_reason = self._apply_qk_norm(q_reason, k_reason)
        q_action, k_action = self._apply_qk_norm(q_action, k_action)
        
        # ========== Apply RoPE ==========
        # Self-attention: both Q and K
        cos_main, sin_main = self._get_rope_embed(T, x.device, x.dtype)
        cos_main = cos_main.unsqueeze(0).unsqueeze(0)
        sin_main = sin_main.unsqueeze(0).unsqueeze(0)
        q_base = apply_rope_single(q_base, cos_main, sin_main)
        k_self = apply_rope_single(k_self, cos_main, sin_main)
        
        # Cross-attention Q: apply same RoPE as self-attention
        q_bev = apply_rope_single(q_bev, cos_main, sin_main)
        q_reason = apply_rope_single(q_reason, cos_main, sin_main)
        q_action = apply_rope_single(q_action, cos_main, sin_main)
        
        # Cross-attention K: only K gets RoPE
        cos_b, sin_b = self._get_rope_embed(T_b, x.device, x.dtype)
        cos_b = cos_b.unsqueeze(0).unsqueeze(0)
        sin_b = sin_b.unsqueeze(0).unsqueeze(0)
        k_bev = apply_rope_single(k_bev, cos_b, sin_b)
        
        cos_r, sin_r = self._get_rope_embed(T_r, x.device, x.dtype)
        cos_r = cos_r.unsqueeze(0).unsqueeze(0)
        sin_r = sin_r.unsqueeze(0).unsqueeze(0)
        k_reason = apply_rope_single(k_reason, cos_r, sin_r)
        
        cos_a, sin_a = self._get_rope_embed(T_a, x.device, x.dtype)
        cos_a = cos_a.unsqueeze(0).unsqueeze(0)
        sin_a = sin_a.unsqueeze(0).unsqueeze(0)
        k_action = apply_rope_single(k_action, cos_a, sin_a)
        
        # ========== Compute attention scores ==========
        scale = math.sqrt(self.head_dim)
        
        # Self-attention scores
        attn_self = torch.matmul(q_base, k_self.transpose(-2, -1)) * temp_self + self.bias_self
        
        # Cross-attention scores (with gating for Reasoning and Action)
        attn_bev = torch.matmul(q_bev, k_bev.transpose(-2, -1)) * temp_bev + self.bias_bev
        attn_reason = torch.matmul(q_reason, k_reason.transpose(-2, -1)) * ratio_g * temp_reason + self.bias_reason
        attn_action = torch.matmul(q_action, k_action.transpose(-2, -1)) * ratio_g_action * temp_action + self.bias_action
        
        # ========== Concatenate all attention scores ==========
        attn_scores = torch.cat([attn_self, attn_bev, attn_reason, attn_action], dim=-1)
        attn_scores = attn_scores / scale
        
        # Apply self-attention mask if provided
        if self_attn_mask is not None:
            total_kv_len = T + T_b + T_r + T_a
            full_mask = torch.zeros(T, total_kv_len, device=x.device, dtype=x.dtype)
            full_mask[:, :T] = self_attn_mask
            attn_scores = attn_scores + full_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply padding masks
        offset = T
        if bev_padding_mask is not None:
            bev_mask = bev_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores[:, :, :, offset:offset+T_b] = attn_scores[:, :, :, offset:offset+T_b].masked_fill(bev_mask, float('-inf'))
        offset += T_b
        
        if reasoning_padding_mask is not None:
            reason_mask = reasoning_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores[:, :, :, offset:offset+T_r] = attn_scores[:, :, :, offset:offset+T_r].masked_fill(reason_mask, float('-inf'))
        offset += T_r
        
        if action_padding_mask is not None:
            action_mask = action_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores[:, :, :, offset:] = attn_scores[:, :, :, offset:].masked_fill(action_mask, float('-inf'))
        
        # ========== Softmax and weighted sum (Main Path) ==========
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        v_combined = torch.cat([v_self, v_bev, v_reason, v_action], dim=2)
        output = torch.matmul(attn_weights, v_combined)
        
        # Reshape and output projection
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)
        output = self.proj_dropout(output)
        
        # ========== Source-Specific Residual Paths (Branch) ==========
        # BEV branch: mean pooling → MLP → gated add
        bev_gate = torch.sigmoid(self.bev_residual_gate)
        bev_pooled = bev_tokens.mean(dim=1, keepdim=True)  # (B, 1, d_model)
        bev_residual = self.bev_residual_proj(bev_pooled).expand(-1, T, -1)  # (B, T, d_model)
        
        # Reasoning branch: attention pooling → MLP → gated add
        reason_gate = torch.sigmoid(self.reason_residual_gate)
        reason_query = self.reason_residual_query.expand(B, -1, -1)  # (B, 1, d_model)
        reason_pooled, _ = self.reason_residual_attn(
            query=reason_query, key=reasoning_tokens, value=reasoning_tokens,
            key_padding_mask=reasoning_padding_mask
        )  # (B, 1, d_model)
        reason_residual = self.reason_residual_proj(reason_pooled).expand(-1, T, -1)  # (B, T, d_model)
        
        # Combine main path with branch paths
        output = output + bev_gate * bev_residual + reason_gate * reason_residual
        
        # ========== Residual with gate ==========
        x = x + gate_attn.unsqueeze(1) * output
        
        # ========== FFN with AdaLN ==========
        x_norm_ffn = self.modulate(nn.functional.layer_norm(x, [C]), shift_ffn, scale_ffn)
        x_ffn = self.ffn(x_norm_ffn)
        x = x + gate_ffn.unsqueeze(1) * x_ffn
        
        return x


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
# Unified Decoder with Heterogeneous Queries
# =============================================================================

class UnifiedDecoderOnlyTransformer(nn.Module):
    """
    Decoder-Only Transformer for trajectory prediction.
    
    Uses trajectory queries with multi-source cross-attention:
    - BEV tokens (SparseDrive sparse features)
    - Reasoning tokens (VLM reasoning hidden states)
    - Action tokens (VLM action hidden states)
    
    Note: Low-dim state (history status) is NOT used for cross-attention.
    It is only used for AdaLN conditioning through the conditioning vector,
    which is computed in TransformerForDiffusion.
    
    Query structure: [trajectory (horizon)]
    Cross-attention sources:
        - BEV tokens with RoPE on K only
        - Reasoning + Action tokens with RoPE on K only + gating
    
    Architecture also includes source-specific residual paths (main + branch)
    for better information flow from BEV and Reasoning features.
    """
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        status_dim: int = 13,  # Kept for compatibility but not used for cross-attention
        bev_dim: int = 512,
        reasoning_dim: int = 1536,
        horizon: int = 8,
        n_obs_steps: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        
        # Projection layers for conditions (no hist_proj - low-dim state only for AdaLN)
        self.bev_proj = nn.Sequential(nn.Linear(bev_dim, d_model), nn.LayerNorm(d_model))
        self.reasoning_proj = nn.Sequential(nn.Linear(reasoning_dim, d_model), nn.LayerNorm(d_model))
        self.action_proj = nn.Sequential(nn.Linear(reasoning_dim, d_model), nn.LayerNorm(d_model))  # Same dim as reasoning
        
        # Position embeddings for cross-attention sources
        self.bev_pos_emb = nn.Parameter(torch.zeros(1, 256, d_model))
        
        # Decoder blocks - using new MultiSourceAttentionBlock
        self.layers = nn.ModuleList([
            MultiSourceAttentionBlock(d_model, nhead, dim_feedforward, dropout)
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
        traj_emb: torch.Tensor,  # (B, horizon, d_model) - trajectory query embeddings
        bev_tokens: torch.Tensor,
        reasoning_tokens: torch.Tensor,
        action_tokens: torch.Tensor,  # (B, T_a, action_dim) - action hidden states
        conditioning: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        bev_padding_mask: Optional[torch.Tensor] = None,
        reasoning_padding_mask: Optional[torch.Tensor] = None,
        action_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with trajectory queries and multi-source attention.
        
        Args:
            traj_emb: (B, horizon, d_model) - trajectory query embeddings (noisy traj embedded)
            bev_tokens: (B, seq_len, bev_dim) - BEV spatial tokens (or SparseDrive sparse tokens)
            reasoning_tokens: (B, T_r, reasoning_dim) - reasoning tokens
            action_tokens: (B, T_a, action_dim) - action hidden states
            conditioning: (B, d_model) - timestep + current_status + history conditioning
            
        Returns:
            traj_out: (B, horizon, d_model) - trajectory output
            
        Note: Low-dim state info is already encoded in the conditioning vector.
        """
        B = traj_emb.shape[0]
        
        # Trajectory queries
        x = traj_emb  # (B, horizon, d_model)
        
        # Project conditions to d_model
        bev_proj = self.bev_proj(bev_tokens)
        bev_seq_len = bev_proj.shape[1]
        if bev_seq_len <= self.bev_pos_emb.shape[1]:
            bev_proj = bev_proj + self.bev_pos_emb[:, :bev_seq_len, :]
        
        reasoning_proj = self.reasoning_proj(reasoning_tokens)
        action_proj = self.action_proj(action_tokens)
        
        # Decoder layers with multi-source attention
        for layer in self.layers:
            x = layer(
                x, 
                bev_proj, 
                reasoning_proj,
                action_proj,
                conditioning,
                self_attn_mask, 
                bev_padding_mask,
                reasoning_padding_mask,
                action_padding_mask
            )
        
        x = self.final_norm(x)
        
        # Return trajectory output only
        return x


# =============================================================================
# Main Model: TransformerForDiffusion (Decoder-Only)
# =============================================================================

class TransformerForDiffusion(ModuleAttrMixin):
    """
    Decoder-Only Transformer for Diffusion-based Trajectory Prediction.
    
    Key features:
    - Decoder with trajectory queries
    - Multi-source cross-attention to: BEV (SparseDrive) -> Reasoning -> Action
    - Sparse Instance Spatial Attention for trajectory-aware detection feature aggregation
    - Low-dim state only used for AdaLN conditioning (not cross-attention)
    - MLP output head (no GRU)
    - AdaLN modulation based on timestep + current_ego_status + history
    
    Query structure: [trajectory (horizon)]
    Cross-attention sources: BEV (SparseDrive sparse tokens) -> Reasoning -> Action
    Spatial attention: Trajectory points query nearby detection instances
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
        reasoning_emb_dim: int = 1536,
        action_emb_dim: int = 1536,  # Same as reasoning
        status_dim: int = 15,
        ego_status_seq_len: int = 1,
        bev_dim: int = 512,
        state_token_dim: int = 128,
        # SparseDrive spatial attention parameters
        use_sparse_spatial_attn: bool = True,
        sparse_feature_dim: int = 256,  # SparseDrive feature dimension
        num_det_instances: int = 50,
        num_map_instances: int = 10,
    ) -> None:
        super().__init__()
        
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.status_dim = status_dim
        self.reasoning_emb_dim = reasoning_emb_dim
        self.action_emb_dim = action_emb_dim
        self.bev_dim = bev_dim
        self.T = horizon
        self.use_sparse_spatial_attn = use_sparse_spatial_attn
        
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
        
        # Decoder for trajectory prediction
        self.decoder = UnifiedDecoderOnlyTransformer(
            d_model=n_emb,
            nhead=n_head,
            num_layers=n_layer,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            status_dim=status_dim,
            bev_dim=bev_dim,
            reasoning_dim=reasoning_emb_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
        )
        
        # Sparse Instance Spatial Attention for SparseDrive features
        if use_sparse_spatial_attn:
            # Project SparseDrive detection features to n_emb
            self.det_feature_proj = nn.Sequential(
                nn.Linear(sparse_feature_dim, n_emb),
                nn.LayerNorm(n_emb),
            )
            
            self.sparse_spatial_attn = SparseInstanceSpatialAttention(
                embed_dims=n_emb,
                num_heads=n_head,
                num_det_instances=num_det_instances,
                dropout=p_drop_attn,
            )
        
        # Causal mask for trajectory queries
        self.causal_attn = causal_attn
        
        # Output head for trajectory only
        self.trajectory_head = TrajectoryMLPHead(n_emb, output_dim, p_drop_emb)
        
        self.apply(self._init_weights)
        
        logger.info("TransformerForDiffusion (Decoder-Only) - parameters: %e", 
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
                # Only check actual parameter name, not full path
                # pn == "bias" or pn contains "bias" as the actual param name
                if pn == "bias" or pn.endswith(".bias") or "_bias" in pn or pn.startswith("bias_"):
                    no_decay.add(fpn)
                elif "weight" in pn and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        for name in param_dict:
            if 'pos_emb' in name or '_dummy_variable' in name:
                no_decay.add(name)
            elif 'gating_factor' in name or 'gating_factor_action' in name:
                no_decay.add(name)
            elif 'temp_' in name:
                # Temperature parameters - no weight decay
                no_decay.add(name)
            elif '.bias' in name or 'bias_' in name:
                # All bias parameters - no weight decay
                no_decay.add(name)
            elif 'inv_freq' in name:
                # inv_freq is a buffer, should not be in param_dict, but handle if exists
                no_decay.add(name)
            elif 'residual_gate' in name or 'residual_query' in name:
                # Residual path gating and query parameters - no weight decay
                no_decay.add(name)
            elif 'spatial_gate' in name:
                # Spatial attention gating parameter - no weight decay
                no_decay.add(name)
        
        # Remove from decay any params that are in no_decay (no_decay takes priority)
        decay = decay - no_decay
        
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Params in both decay and no_decay: {inter_params}"
        assert len(param_dict.keys() - union_params) == 0, f"Missing params: {param_dict.keys() - union_params}"
        
        return [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
    
    def configure_optimizers(self, learning_rate: float = 1e-4, weight_decay: float = 1e-3,
                            betas: Tuple[float, float] = (0.9, 0.95)):
        return torch.optim.AdamW(self.get_optim_groups(weight_decay), lr=learning_rate, betas=betas)
    
    def _create_causal_attn_mask(self, T_traj: int, device: torch.device, dtype: torch.dtype):
        """
        Create causal attention mask for trajectory queries.
        
        Mask rules:
        - Trajectory tokens: causal within trajectory (can only attend to past)
        """
        
        if not self.causal_attn:
            return None
        
        # Start with full attention allowed
        mask = torch.zeros(T_traj, T_traj, device=device, dtype=dtype)
        
        # Trajectory: causal within trajectory
        for i in range(T_traj):
            mask[i, i+1:T_traj] = float('-inf')  # Can't attend to future trajectory
        
        return mask
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: torch.Tensor,
        bev_tokens: torch.Tensor,
        reasoning_query_tokens: torch.Tensor,
        action_tokens: torch.Tensor,  # action hidden states
        ego_status: torch.Tensor,
        bev_padding_mask: Optional[torch.Tensor] = None,
        reasoning_mask: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,  # action mask
        # SparseDrive spatial attention features (optional)
        det_instance_feature: Optional[torch.Tensor] = None,  # (B, 50, 256)
        det_prediction: Optional[torch.Tensor] = None,  # (B, 50, 11) with positions
        det_mask: Optional[torch.Tensor] = None,  # (B, 50) valid detections
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with decoder and optional sparse spatial attention.
        
        Args:
            sample: (B, T, input_dim) - noisy trajectory
            timestep: diffusion timestep
            cond: (B, T_obs, cond_dim) - for API compatibility (not used)
            bev_tokens: (B, seq_len, bev_dim) - SparseDrive sparse tokens (det + map + ego)
            reasoning_query_tokens: (B, T_r, reasoning_dim) - reasoning tokens
            action_tokens: (B, T_a, action_dim) - action hidden states
            ego_status: (B, T_obs, status_dim) - ego status history (complete 4 frames)
            
            # SparseDrive spatial attention (optional):
            det_instance_feature: (B, 50, 256) - detection instance features
            det_prediction: (B, 50, 11) - detection predictions with positions (x,y,z,w,l,h,sin,cos,vx,vy,vz)
            det_mask: (B, 50) - mask for valid detections (True = valid)
            
        Returns:
            trajectory: (B, T, output_dim) - predicted trajectory
            
        Low-dim state utilization:
            - AdaLN conditioning: current_status (last frame) + GRU-encoded global history
            - NO cross-attention to history status (only used for conditioning)
            
        Sparse Spatial Attention:
            - When det_instance_feature and det_prediction are provided, the model uses
              distance-based attention to aggregate detection features relevant to
              each trajectory point, improving spatial awareness.
        """
        model_dtype = next(self.parameters()).dtype
        
        sample = sample.contiguous().to(dtype=model_dtype)
        bev_tokens = bev_tokens.contiguous().to(dtype=model_dtype)
        reasoning_tokens = reasoning_query_tokens.contiguous().to(dtype=model_dtype)
        action_tokens = action_tokens.contiguous().to(dtype=model_dtype)
        ego_status = ego_status.to(dtype=model_dtype)
        
        B = sample.shape[0]
        T_traj = sample.shape[1]
        
        # Timestep handling
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timesteps = timestep.expand(B)
        
        # ========== Conditioning (includes low-dim state info) ==========
        # 1. Timestep embedding
        time_emb = self.time_emb(timesteps).to(dtype=model_dtype)
        
        # 2. Current status embedding (last frame only for AdaLN)
        current_status = ego_status[:, -1, :]  # (B, status_dim)
        status_emb = self.ego_status_proj(current_status)
        
        # 3. GRU-encoded global history for AdaLN
        hist_global_emb = self.history_encoder(ego_status)  # (B, n_emb)
        
        # Combined conditioning for AdaLN modulation (contains all low-dim state info)
        conditioning = time_emb + status_emb + hist_global_emb
        
        # ========== Trajectory Query Embedding ==========
        traj_emb = self.input_emb(sample)  # (B, T_traj, n_emb)
        pos_emb = self.pos_emb[:, :T_traj, :]
        traj_emb = traj_emb + pos_emb
        traj_emb = self.drop(traj_emb)
        traj_emb = self.pre_decoder_norm(traj_emb)
        
        # ========== Create Attention Mask ==========
        self_attn_mask = self._create_causal_attn_mask(T_traj, sample.device, model_dtype)
        
        # ========== Padding Masks ==========
        reasoning_padding_mask = ~reasoning_mask if reasoning_mask is not None else (torch.norm(reasoning_tokens, dim=-1) == 0)
        action_padding_mask = ~action_mask if action_mask is not None else (torch.norm(action_tokens, dim=-1) == 0)
        
        # ========== Decoder ==========
        # Cross-attention: BEV (SparseDrive) -> Reasoning -> Action
        traj_out = self.decoder(
            traj_emb,
            bev_tokens, reasoning_tokens, action_tokens, conditioning,
            self_attn_mask, bev_padding_mask, reasoning_padding_mask, action_padding_mask
        )
        
        # ========== Sparse Instance Spatial Attention ==========
        # Apply spatial attention to enhance trajectory features with nearby detection info
        if self.use_sparse_spatial_attn and det_instance_feature is not None and det_prediction is not None:
            # Project detection features to n_emb dimension
            det_features_proj = self.det_feature_proj(det_instance_feature.to(dtype=model_dtype))  # (B, 50, n_emb)
            det_pred = det_prediction.to(dtype=model_dtype)  # (B, 50, 11)
            
            # Use noisy trajectory points for spatial queries
            traj_points = sample[..., :2]  # (B, T, 2) - (x, y) positions
            
            # Apply sparse spatial attention
            traj_out = self.sparse_spatial_attn(
                traj_queries=traj_out,
                traj_points=traj_points,
                det_features=det_features_proj,
                det_positions=det_pred,
                det_mask=det_mask,
            )
        
        # ========== Output Head ==========
        trajectory = self.trajectory_head(traj_out, conditioning)
        
        return trajectory


# =============================================================================
# Test
# =============================================================================

def test():
    """Test the decoder-only architecture with source-specific residual paths."""
    print("=" * 60)
    print("Testing TransformerForDiffusion (Decoder-Only)")
    print("Low-dim State for AdaLN only (No Cross-Attention)")
    print("With Source-Specific Residual Paths (BEV + Reasoning branches)")
    print("With Sparse Instance Spatial Attention for SparseDrive")
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
        reasoning_emb_dim=1536,
        action_emb_dim=1536,
        status_dim=13,
        bev_dim=512,
        use_sparse_spatial_attn=True,
        sparse_feature_dim=256,
        num_det_instances=50,
        num_map_instances=10,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    B = 4
    timestep = torch.tensor(0)
    sample = torch.randn((B, 8, 2))
    cond = torch.zeros((B, 4, 10))
    # SparseDrive sparse tokens: 50 det + 10 map + 1 ego = 61 tokens
    bev_tokens = torch.randn((B, 61, 512))
    reasoning_tokens = torch.randn((B, 10, 1536))
    action_tokens = torch.randn((B, 4, 1536))  # Action hidden states
    ego_status = torch.randn((B, 4, 13))  # 4 frames of history
    
    # SparseDrive detection features for spatial attention
    det_instance_feature = torch.randn((B, 50, 256))  # Detection instance features
    det_prediction = torch.randn((B, 50, 11))  # Detection predictions (x,y,z,w,l,h,sin,cos,vx,vy,vz)
    det_mask = torch.ones((B, 50), dtype=torch.bool)  # All valid
    det_mask[:, 40:] = False  # Last 10 are invalid (padding)
    
    print("\nTest 1: Basic forward pass (without spatial attention)")
    trajectory = transformer(
        sample=sample, timestep=timestep, cond=cond,
        bev_tokens=bev_tokens,
        reasoning_query_tokens=reasoning_tokens,
        action_tokens=action_tokens,
        ego_status=ego_status,
    )
    print(f"  Trajectory: {trajectory.shape}")
    assert trajectory.shape == (B, 8, 2), f"Expected (B, 8, 2), got {trajectory.shape}"
    
    print("\nTest 2: Forward pass with Sparse Spatial Attention")
    trajectory_with_spatial = transformer(
        sample=sample, timestep=timestep, cond=cond,
        bev_tokens=bev_tokens,
        reasoning_query_tokens=reasoning_tokens,
        action_tokens=action_tokens,
        ego_status=ego_status,
        det_instance_feature=det_instance_feature,
        det_prediction=det_prediction,
        det_mask=det_mask,
    )
    print(f"  Trajectory: {trajectory_with_spatial.shape}")
    assert trajectory_with_spatial.shape == (B, 8, 2), f"Expected (B, 8, 2), got {trajectory_with_spatial.shape}"
    
    print("\nTest 3: Spatial attention affects output")
    diff_spatial = torch.abs(trajectory - trajectory_with_spatial).mean()
    print(f"  Difference (with vs without spatial): {diff_spatial:.6f}")
    assert diff_spatial > 0, "Spatial attention should affect output"
    
    print("\nTest 4: Different detection positions affect output")
    det_prediction2 = torch.randn((B, 50, 11))  # Different positions
    traj_diff_det = transformer(
        sample=sample, timestep=timestep, cond=cond,
        bev_tokens=bev_tokens,
        reasoning_query_tokens=reasoning_tokens,
        action_tokens=action_tokens,
        ego_status=ego_status,
        det_instance_feature=det_instance_feature,
        det_prediction=det_prediction2,
        det_mask=det_mask,
    )
    diff_det = torch.abs(trajectory_with_spatial - traj_diff_det).mean()
    print(f"  Difference: {diff_det:.6f}")
    assert diff_det > 0, "Detection positions should affect output"
    
    print("\nTest 5: Different history affects output")
    ego_status2 = torch.randn((B, 4, 13))
    traj3 = transformer(sample=sample, timestep=timestep, cond=cond,
                        bev_tokens=bev_tokens,
                        reasoning_query_tokens=reasoning_tokens,
                        action_tokens=action_tokens,
                        ego_status=ego_status2,
                        det_instance_feature=det_instance_feature,
                        det_prediction=det_prediction,
                        det_mask=det_mask)
    diff_hist = torch.abs(trajectory_with_spatial - traj3).mean()
    print(f"  Difference: {diff_hist:.6f}")
    assert diff_hist > 0, "History should affect output"
    
    print("\nTest 6: Optimizer")
    opt = transformer.configure_optimizers()
    print(f"  Optimizer created: {type(opt).__name__}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test()