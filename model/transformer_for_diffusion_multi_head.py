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


class MultiSourceAttentionBlock(nn.Module):
    """
    Multi-Source Attention Block with separate projections for different sources.
    
    Implements the architecture from MLPResNetBlock_Pro with enhancements:
    - Multi-head self-attention + cross-attention (adapter / task)
    - RoPE encoding for Q/K
    - QK Normalization (RMSNorm) for stable training
    - Gating mechanism for Reasoning tokens only
    - Separate linear layers for different feature sources
    - AdaLN modulation preserved
    
    Enhancements over basic fusion:
    1. Per-source learnable temperature scaling for attention calibration
    2. Source-specific Q projections for modality-aware queries
    3. Learned bias terms for attention score adjustment
    
    Architecture:
        1. Self-attention on main sequence with RoPE on both Q and K
        2. Cross-attention to adapter tokens (bev + vl) with RoPE on K only
        3. Cross-attention to task tokens (reasoning) with RoPE on K only + gating
        4. Residual + FFN
        
    The attention scores from all sources are concatenated and softmaxed together,
    but the gating factor only scales the Reasoning (task) attention scores.
    
    Note: hist_tokens removed - ego_status history is now only used for AdaLN conditioning.
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
        self.q_adapter_vl = nn.Linear(d_model, d_model // 4)
        self.q_adapter_reason = nn.Linear(d_model, d_model // 4)
        self.q_adapter_out = nn.Linear(d_model // 4, d_model)
        
        # ========== Self-attention K, V ==========
        self.k_self = nn.Linear(d_model, d_model)
        self.v_self = nn.Linear(d_model, d_model)
        
        # ========== Adapter cross-attention K, V (for bev, vl) ==========
        self.k_bev = nn.Linear(d_model, d_model)
        self.v_bev = nn.Linear(d_model, d_model)
        self.k_vl = nn.Linear(d_model, d_model)
        self.v_vl = nn.Linear(d_model, d_model)
        
        # ========== Task cross-attention K, V (for reasoning) ==========
        self.k_reasoning = nn.Linear(d_model, d_model)
        self.v_reasoning = nn.Linear(d_model, d_model)
        
        # ========== Output projection ==========
        self.o_proj = nn.Linear(d_model, d_model)
        
        # ========== Gating factor for Reasoning tokens ==========
        self.gating_factor = nn.Parameter(torch.zeros(1))
        
        # ========== Per-source learnable temperature (log scale for stability) ==========
        # Controls attention sharpness per source: exp(temp) * scores
        self.temp_self = nn.Parameter(torch.zeros(1))      # log(1) = 0
        self.temp_bev = nn.Parameter(torch.zeros(1))
        self.temp_vl = nn.Parameter(torch.zeros(1))
        self.temp_reason = nn.Parameter(torch.zeros(1))
        
        # ========== Per-source learnable bias (attention prior) ==========
        # Adds a learnable bias to attention scores before softmax
        self.bias_self = nn.Parameter(torch.zeros(1))
        self.bias_bev = nn.Parameter(torch.zeros(1))
        self.bias_vl = nn.Parameter(torch.zeros(1))
        self.bias_reason = nn.Parameter(torch.zeros(1))
        
        # ========== Source-Specific Residual Paths ==========
        # These provide direct information flow from each source, bypassing the competitive softmax
        # Each path: pool source -> project -> gate -> add to output
        # This ensures each modality can contribute even if softmax suppresses it
        
        # BEV residual path: spatial average pooling + projection
        self.bev_residual_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.bev_residual_gate = nn.Parameter(torch.zeros(1))
        
        # VL residual path: attention pooling (query-based)
        self.vl_residual_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.vl_residual_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
        self.vl_residual_proj = nn.Linear(d_model, d_model)
        self.vl_residual_gate = nn.Parameter(torch.zeros(1))
        
        # Reasoning residual path: attention pooling (query-based)
        self.reason_residual_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.reason_residual_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
        self.reason_residual_proj = nn.Linear(d_model, d_model)
        self.reason_residual_gate = nn.Parameter(torch.zeros(1))
        
        # ========== Route-Specific Components (Stability Enhancement) ==========
        # These components provide route queries with independent processing paths,
        # simulating the effect of an independent Route Decoder within the unified architecture.
        
        # Route-specific Q adapters (separate from trajectory adapters)
        # This allows route to learn different attention patterns for cross-attention
        self.route_q_adapter_bev = nn.Linear(d_model, d_model // 4)
        self.route_q_adapter_vl = nn.Linear(d_model, d_model // 4)
        self.route_q_adapter_reason = nn.Linear(d_model, d_model // 4)
        self.route_q_adapter_out = nn.Linear(d_model // 4, d_model)
        
        # Route-specific attention temperature and bias
        # Allows route to have different attention sharpness/bias than trajectory
        self.route_temp_bev = nn.Parameter(torch.zeros(1))
        self.route_temp_vl = nn.Parameter(torch.zeros(1))
        self.route_temp_reason = nn.Parameter(torch.zeros(1))
        self.route_bias_bev = nn.Parameter(torch.zeros(1))
        self.route_bias_vl = nn.Parameter(torch.zeros(1))
        self.route_bias_reason = nn.Parameter(torch.zeros(1))
        
        # Route-specific AdaLN modulation (separate from shared modulation)
        # This is key for route stability - independent conditioning pathway
        self.route_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )
        
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
        x: torch.Tensor,  # (B, T, d_model) - main sequence [trajectory | route]
        bev_tokens: torch.Tensor,   # (B, T_b, d_model) - BEV tokens (already projected)
        vl_tokens: torch.Tensor,    # (B, T_v, d_model) - VL tokens (already projected)
        reasoning_tokens: torch.Tensor,  # (B, T_r, d_model) - reasoning tokens (already projected)
        conditioning: torch.Tensor,  # (B, d_model) - conditioning for AdaLN
        self_attn_mask: Optional[torch.Tensor] = None,  # (T, T) - block diagonal mask for segment isolation
        bev_padding_mask: Optional[torch.Tensor] = None,
        vl_padding_mask: Optional[torch.Tensor] = None,
        reasoning_padding_mask: Optional[torch.Tensor] = None,
        route_conditioning: Optional[torch.Tensor] = None,  # (B, d_model) - route-specific conditioning
        T_traj: Optional[int] = None,  # Number of trajectory queries (for route-specific processing)
    ) -> torch.Tensor:
        """
        Forward pass with multi-source attention.
        
        Attention is computed as:
        1. Self-attention: Q(x) @ K(x)^T with RoPE on both Q and K
           - Uses block diagonal mask to isolate trajectory and route segments
        2. Adapter cross-attention: Q_adapted(x) @ [K(bev), K(vl)]^T with RoPE on K only
        3. Task cross-attention: Q_adapted(x) @ K(reasoning)^T with RoPE on K only, scaled by gating
        
        Route-Specific Processing (when T_traj is provided):
        - Route queries use independent Q adapters for cross-attention
        - Route queries use independent temperature/bias for attention
        - Route queries use independent AdaLN modulation
        
        All attention scores are concatenated, temperature-scaled, bias-adjusted, and softmaxed together.
        
        Note: 
        - hist_tokens removed - ego_status history is now only used for AdaLN conditioning.
        - self_attn_mask should be block diagonal to isolate trajectory and route segments
        """
        B, T, C = x.shape
        T_b = bev_tokens.shape[1]
        T_v = vl_tokens.shape[1]
        T_r = reasoning_tokens.shape[1]
        
        # Determine if we have route-specific processing
        if T_traj is None:
            T_traj = T  # No route separation, treat all as trajectory
        T_route = T - T_traj
        
        # ========== AdaLN modulation parameters ==========
        mod_params = self.adaLN_modulation(conditioning)
        shift_pre, scale_pre, gate_attn, shift_ffn, scale_ffn, gate_ffn = mod_params.chunk(6, dim=1)
        
        # Route-specific AdaLN (if route exists and route_conditioning provided)
        if T_route > 0 and route_conditioning is not None:
            route_mod_params = self.route_adaLN_modulation(route_conditioning)
            route_shift_pre, route_scale_pre, route_gate_attn, route_shift_ffn, route_scale_ffn, route_gate_ffn = route_mod_params.chunk(6, dim=1)
        else:
            # Use shared modulation for route
            route_shift_pre, route_scale_pre = shift_pre, scale_pre
            route_gate_attn, route_shift_ffn, route_scale_ffn, route_gate_ffn = gate_attn, shift_ffn, scale_ffn, gate_ffn
        
        # ========== Pre-LayerNorm with modulation (segment-specific) ==========
        x_norm_ln = self.norm_pre(x)
        
        # Apply different modulation to trajectory and route
        x_norm_traj = self.modulate(x_norm_ln[:, :T_traj, :], shift_pre, scale_pre)
        if T_route > 0:
            x_norm_route = self.modulate(x_norm_ln[:, T_traj:, :], route_shift_pre, route_scale_pre)
            x_norm = torch.cat([x_norm_traj, x_norm_route], dim=1)
        else:
            x_norm = x_norm_traj
        
        # ========== Gating factor for Reasoning ==========
        g = self.gating_factor
        ratio_g = torch.tanh(g)
        
        # ========== Temperature scaling factors (softplus for positive values) ==========
        temp_self = 1.0 + torch.nn.functional.softplus(self.temp_self)
        temp_bev = 1.0 + torch.nn.functional.softplus(self.temp_bev)
        temp_vl = 1.0 + torch.nn.functional.softplus(self.temp_vl)
        temp_reason = 1.0 + torch.nn.functional.softplus(self.temp_reason)
        
        # Route-specific temperatures
        route_temp_bev = 1.0 + torch.nn.functional.softplus(self.route_temp_bev)
        route_temp_vl = 1.0 + torch.nn.functional.softplus(self.route_temp_vl)
        route_temp_reason = 1.0 + torch.nn.functional.softplus(self.route_temp_reason)
        
        # ========== Q projection (base + source-specific adaptations) ==========
        q_base = self.q_proj(x_norm)  # (B, T, d_model)
        
        # Trajectory Q adaptations (original adapters)
        q_adapt_bev_traj = self.q_adapter_out(torch.tanh(self.q_adapter_bev(x_norm[:, :T_traj, :])))
        q_adapt_vl_traj = self.q_adapter_out(torch.tanh(self.q_adapter_vl(x_norm[:, :T_traj, :])))
        q_adapt_reason_traj = self.q_adapter_out(torch.tanh(self.q_adapter_reason(x_norm[:, :T_traj, :])))
        
        # Route Q adaptations (route-specific adapters)
        if T_route > 0:
            q_adapt_bev_route = self.route_q_adapter_out(torch.tanh(self.route_q_adapter_bev(x_norm[:, T_traj:, :])))
            q_adapt_vl_route = self.route_q_adapter_out(torch.tanh(self.route_q_adapter_vl(x_norm[:, T_traj:, :])))
            q_adapt_reason_route = self.route_q_adapter_out(torch.tanh(self.route_q_adapter_reason(x_norm[:, T_traj:, :])))
            
            # Concatenate trajectory and route adaptations
            q_adapt_bev = torch.cat([q_adapt_bev_traj, q_adapt_bev_route], dim=1)
            q_adapt_vl = torch.cat([q_adapt_vl_traj, q_adapt_vl_route], dim=1)
            q_adapt_reason = torch.cat([q_adapt_reason_traj, q_adapt_reason_route], dim=1)
        else:
            q_adapt_bev = q_adapt_bev_traj
            q_adapt_vl = q_adapt_vl_traj
            q_adapt_reason = q_adapt_reason_traj
        
        # ========== Self-attention K, V ==========
        k_self = self.k_self(x_norm)  # (B, T, d_model)
        v_self = self.v_self(x_norm)  # (B, T, d_model)
        
        # ========== Adapter K, V (bev, vl) ==========
        k_bev = self.k_bev(bev_tokens)      # (B, T_b, d_model)
        v_bev = self.v_bev(bev_tokens)
        k_vl = self.k_vl(vl_tokens)          # (B, T_v, d_model)
        v_vl = self.v_vl(vl_tokens)
        
        # ========== Task K, V (reasoning) ==========
        k_reason = self.k_reasoning(reasoning_tokens)  # (B, T_r, d_model)
        v_reason = self.v_reasoning(reasoning_tokens)
        
        # ========== Reshape to multi-head ==========
        q_base = self._reshape_heads(q_base, B, T)            # (B, nhead, T, head_dim)
        # Create adapted Q for each source by adding adaptation to base Q
        q_bev = self._reshape_heads(q_base.transpose(1, 2).reshape(B, T, C) + q_adapt_bev, B, T)
        q_vl = self._reshape_heads(q_base.transpose(1, 2).reshape(B, T, C) + q_adapt_vl, B, T)
        q_reason = self._reshape_heads(q_base.transpose(1, 2).reshape(B, T, C) + q_adapt_reason, B, T)
        
        k_self = self._reshape_heads(k_self, B, T)
        v_self = self._reshape_heads(v_self, B, T)
        k_bev = self._reshape_heads(k_bev, B, T_b)
        v_bev = self._reshape_heads(v_bev, B, T_b)
        k_vl = self._reshape_heads(k_vl, B, T_v)
        v_vl = self._reshape_heads(v_vl, B, T_v)
        k_reason = self._reshape_heads(k_reason, B, T_r)
        v_reason = self._reshape_heads(v_reason, B, T_r)
        
        # ========== Apply QK Normalization ==========
        q_base, k_self = self._apply_qk_norm(q_base, k_self)
        q_bev, k_bev = self._apply_qk_norm(q_bev, k_bev)
        q_vl, k_vl = self._apply_qk_norm(q_vl, k_vl)
        q_reason, k_reason = self._apply_qk_norm(q_reason, k_reason)
        
        # ========== Apply RoPE ==========
        # Self-attention: both Q and K
        cos_main, sin_main = self._get_rope_embed(T, x.device, x.dtype)
        cos_main = cos_main.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
        sin_main = sin_main.unsqueeze(0).unsqueeze(0)
        q_base = apply_rope_single(q_base, cos_main, sin_main)
        k_self = apply_rope_single(k_self, cos_main, sin_main)
        
        # Cross-attention Q: apply same RoPE as self-attention
        q_bev = apply_rope_single(q_bev, cos_main, sin_main)
        q_vl = apply_rope_single(q_vl, cos_main, sin_main)
        q_reason = apply_rope_single(q_reason, cos_main, sin_main)
        
        # Adapter K: only K gets RoPE
        cos_b, sin_b = self._get_rope_embed(T_b, x.device, x.dtype)
        cos_b = cos_b.unsqueeze(0).unsqueeze(0)
        sin_b = sin_b.unsqueeze(0).unsqueeze(0)
        k_bev = apply_rope_single(k_bev, cos_b, sin_b)
        
        cos_v, sin_v = self._get_rope_embed(T_v, x.device, x.dtype)
        cos_v = cos_v.unsqueeze(0).unsqueeze(0)
        sin_v = sin_v.unsqueeze(0).unsqueeze(0)
        k_vl = apply_rope_single(k_vl, cos_v, sin_v)
        
        # Task K (reasoning): only K gets RoPE
        cos_r, sin_r = self._get_rope_embed(T_r, x.device, x.dtype)
        cos_r = cos_r.unsqueeze(0).unsqueeze(0)
        sin_r = sin_r.unsqueeze(0).unsqueeze(0)
        k_reason = apply_rope_single(k_reason, cos_r, sin_r)
        
        # ========== Compute attention scores with temperature and bias ==========
        scale = math.sqrt(self.head_dim)
        
        # Self-attention scores (with temperature and bias)
        attn_self = torch.matmul(q_base, k_self.transpose(-2, -1)) * temp_self + self.bias_self  # (B, nhead, T, T)
        
        # ========== Cross-attention with segment-specific temperature/bias ==========
        # Compute raw attention scores first
        attn_bev_raw = torch.matmul(q_bev, k_bev.transpose(-2, -1))   # (B, nhead, T, T_b)
        attn_vl_raw = torch.matmul(q_vl, k_vl.transpose(-2, -1))     # (B, nhead, T, T_v)
        attn_reason_raw = torch.matmul(q_reason, k_reason.transpose(-2, -1)) * ratio_g  # (B, nhead, T, T_r)
        
        if T_route > 0:
            # Apply trajectory-specific temperature/bias to trajectory part
            attn_bev_traj = attn_bev_raw[:, :, :T_traj, :] * temp_bev + self.bias_bev
            attn_vl_traj = attn_vl_raw[:, :, :T_traj, :] * temp_vl + self.bias_vl
            attn_reason_traj = attn_reason_raw[:, :, :T_traj, :] * temp_reason + self.bias_reason
            
            # Apply route-specific temperature/bias to route part
            attn_bev_route = attn_bev_raw[:, :, T_traj:, :] * route_temp_bev + self.route_bias_bev
            attn_vl_route = attn_vl_raw[:, :, T_traj:, :] * route_temp_vl + self.route_bias_vl
            attn_reason_route = attn_reason_raw[:, :, T_traj:, :] * route_temp_reason + self.route_bias_reason
            
            # Concatenate trajectory and route attention scores
            attn_bev = torch.cat([attn_bev_traj, attn_bev_route], dim=2)
            attn_vl = torch.cat([attn_vl_traj, attn_vl_route], dim=2)
            attn_reason = torch.cat([attn_reason_traj, attn_reason_route], dim=2)
        else:
            # No route, use trajectory-only processing
            attn_bev = attn_bev_raw * temp_bev + self.bias_bev
            attn_vl = attn_vl_raw * temp_vl + self.bias_vl
            attn_reason = attn_reason_raw * temp_reason + self.bias_reason
        
        # ========== Build attention mask ==========
        # Concatenate all attention scores
        attn_scores = torch.cat([attn_self, attn_bev, attn_vl, attn_reason], dim=-1)
        attn_scores = attn_scores / scale  # (B, nhead, T, T + T_b + T_v + T_r)
        
        # Apply self-attention mask if provided
        if self_attn_mask is not None:
            # self_attn_mask is for self-attention part only, need to expand
            # Create full mask: (T, T + T_b + T_v + T_r)
            total_kv_len = T + T_b + T_v + T_r
            full_mask = torch.zeros(T, total_kv_len, device=x.device, dtype=x.dtype)
            full_mask[:, :T] = self_attn_mask  # Apply causal mask to self-attention part
            attn_scores = attn_scores + full_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply padding masks
        if bev_padding_mask is not None:
            bev_mask = bev_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores[:, :, :, T:T+T_b] = attn_scores[:, :, :, T:T+T_b].masked_fill(bev_mask, float('-inf'))
        
        if vl_padding_mask is not None:
            vl_mask = vl_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores[:, :, :, T+T_b:T+T_b+T_v] = attn_scores[:, :, :, T+T_b:T+T_b+T_v].masked_fill(vl_mask, float('-inf'))
        
        if reasoning_padding_mask is not None:
            reason_mask = reasoning_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores[:, :, :, T+T_b+T_v:] = attn_scores[:, :, :, T+T_b+T_v:].masked_fill(reason_mask, float('-inf'))
        
        # ========== Softmax and weighted sum ==========
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, nhead, T, total_kv_len)
        attn_weights = self.dropout(attn_weights)
        
        # Concatenate all values
        v_combined = torch.cat([v_self, v_bev, v_vl, v_reason], dim=2)  # (B, nhead, total_kv_len, head_dim)
        
        # Weighted sum
        output = torch.matmul(attn_weights, v_combined)  # (B, nhead, T, head_dim)
        
        # ========== Reshape and output projection ==========
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)
        
        # ========== Source-Specific Residual Paths ==========
        # These bypass the competitive softmax to ensure each modality contributes
        
        # BEV residual: spatial average pooling -> broadcast to all query positions
        bev_gate = torch.sigmoid(self.bev_residual_gate)
        bev_pooled = bev_tokens.mean(dim=1, keepdim=True)  # (B, 1, d_model)
        bev_residual = self.bev_residual_proj(bev_pooled)  # (B, 1, d_model)
        bev_residual = bev_residual.expand(-1, T, -1)  # (B, T, d_model)
        
        # VL residual: attention pooling with learnable query
        vl_gate = torch.sigmoid(self.vl_residual_gate)
        vl_query = self.vl_residual_query.expand(B, -1, -1)  # (B, 1, d_model)
        vl_pooled, _ = self.vl_residual_attn(
            query=vl_query, key=vl_tokens, value=vl_tokens,
            key_padding_mask=vl_padding_mask
        )  # (B, 1, d_model)
        vl_residual = self.vl_residual_proj(vl_pooled)  # (B, 1, d_model)
        vl_residual = vl_residual.expand(-1, T, -1)  # (B, T, d_model)
        
        # Reasoning residual: attention pooling with learnable query
        reason_gate = torch.sigmoid(self.reason_residual_gate)
        reason_query = self.reason_residual_query.expand(B, -1, -1)  # (B, 1, d_model)
        reason_pooled, _ = self.reason_residual_attn(
            query=reason_query, key=reasoning_tokens, value=reasoning_tokens,
            key_padding_mask=reasoning_padding_mask
        )  # (B, 1, d_model)
        reason_residual = self.reason_residual_proj(reason_pooled)  # (B, 1, d_model)
        reason_residual = reason_residual.expand(-1, T, -1)  # (B, T, d_model)
        
        # Combine main output with residual paths
        output = output + bev_gate * bev_residual + vl_gate * vl_residual + reason_gate * reason_residual
        
        # ========== Residual with segment-specific gate ==========
        if T_route > 0:
            # Apply different gates to trajectory and route
            x_traj = x[:, :T_traj, :] + gate_attn.unsqueeze(1) * output[:, :T_traj, :]
            x_route = x[:, T_traj:, :] + route_gate_attn.unsqueeze(1) * output[:, T_traj:, :]
            x = torch.cat([x_traj, x_route], dim=1)
        else:
            x = x + gate_attn.unsqueeze(1) * output
        
        # ========== FFN with segment-specific AdaLN ==========
        x_ln = nn.functional.layer_norm(x, [C])
        
        if T_route > 0:
            # Apply different modulation to trajectory and route
            x_norm_ffn_traj = self.modulate(x_ln[:, :T_traj, :], shift_ffn, scale_ffn)
            x_norm_ffn_route = self.modulate(x_ln[:, T_traj:, :], route_shift_ffn, route_scale_ffn)
            x_norm_ffn = torch.cat([x_norm_ffn_traj, x_norm_ffn_route], dim=1)
        else:
            x_norm_ffn = self.modulate(x_ln, shift_ffn, scale_ffn)
        
        x_ffn = self.ffn(x_norm_ffn)
        
        if T_route > 0:
            # Apply different gates to trajectory and route
            x_traj = x[:, :T_traj, :] + gate_ffn.unsqueeze(1) * x_ffn[:, :T_traj, :]
            x_route = x[:, T_traj:, :] + route_gate_ffn.unsqueeze(1) * x_ffn[:, T_traj:, :]
            x = torch.cat([x_traj, x_route], dim=1)
        else:
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
# History Encoder with Attention
# =============================================================================

class HistoryEncoder(nn.Module):
    """
    Encodes the history sequence of ego status with GRU + Temporal Attention.
    
    Improvements over simple GRU:
    1. GRU captures sequential dependencies
    2. Temporal attention allows focusing on important history frames
    3. Combines global (GRU hidden) and selective (attention) information
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GRU for sequential encoding
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        # Temporal attention: last frame queries all history
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # Learnable query for global summary (alternative to using last frame)
        self.summary_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Combine GRU hidden state and attention output
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) - history sequence of ego status
        Returns:
            (B, hidden_dim) - encoded history representation
        """
        B, T, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # (B, T, hidden_dim)
        
        # GRU encoding
        gru_out, h_n = self.gru(x)  # gru_out: (B, T, hidden_dim), h_n: (1, B, hidden_dim)
        gru_hidden = h_n[-1]  # (B, hidden_dim) - global sequential summary
        
        # Temporal attention: summary query attends to all GRU outputs
        query = self.summary_query.expand(B, -1, -1)  # (B, 1, hidden_dim)
        attn_out, _ = self.temporal_attn(
            query=query,
            key=gru_out,
            value=gru_out
        )  # (B, 1, hidden_dim)
        attn_out = self.attn_norm(attn_out).squeeze(1)  # (B, hidden_dim)
        
        # Fuse GRU hidden and attention output
        combined = torch.cat([gru_hidden, attn_out], dim=-1)  # (B, hidden_dim * 2)
        output = self.fusion(combined)  # (B, hidden_dim)
        
        return output


# =============================================================================
# Trajectory Head with Route Guidance
# =============================================================================

class TrajectoryMLPHead(nn.Module):
    """
    MLP-based Trajectory Head with Route-to-Trajectory Guidance.
    
    Key features:
    1. Cross-attention from trajectory to route features (Route Guidance)
    2. Conditioning injection from timestep + ego_status
    3. MLP processing for final output
    
    This allows trajectory prediction to explicitly attend to and be guided by
    the planned route, improving trajectory-route consistency.
    """
    def __init__(self, n_emb: int, output_dim: int, p_drop: float = 0.1, num_heads: int = 8):
        super().__init__()
        self.n_emb = n_emb
        self.ln_f = nn.LayerNorm(n_emb)
        
        # Route Guidance: trajectory attends to route
        self.route_guidance_attn = nn.MultiheadAttention(
            embed_dim=n_emb,
            num_heads=num_heads,
            dropout=p_drop,
            batch_first=True
        )
        self.route_guidance_norm = nn.LayerNorm(n_emb)
        self.route_guidance_gate = nn.Parameter(torch.zeros(1))  # Learnable gate
        
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

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor, 
                route_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T_traj, n_emb) - trajectory decoder output
            conditioning: (B, n_emb) - timestep + ego_status conditioning
            route_features: (B, T_route, n_emb) - route decoder output for guidance
        Returns:
            (B, T_traj, output_dim) - trajectory prediction
        """
        x = self.ln_f(x)
        
        # Route Guidance: trajectory attends to route features
        if route_features is not None:
            gate = torch.sigmoid(self.route_guidance_gate)
            route_guided, _ = self.route_guidance_attn(
                query=x,
                key=route_features,
                value=route_features
            )
            route_guided = self.route_guidance_norm(route_guided)
            x = x + gate * route_guided  # Gated residual connection
        
        # Add conditioning as bias
        cond = self.cond_proj(conditioning).unsqueeze(1)  # (B, 1, n_emb)
        x = x + cond
        
        # MLP processing
        x = x + self.mlp(x)
        
        # Output projection
        return self.output_head(x)


# =============================================================================
# Route Head with Conditioning
# =============================================================================

class RouteMLPHead(nn.Module):
    """
    MLP-based Route Head with independent conditioning support.
    
    Key design (inspired by AdaLNRouteHeadDecoderOnly for stability):
    - Route-specific status projection (separate from shared conditioning)
    - Final AdaLN modulation before output (like original stable version)
    - Combined conditioning from both shared and route-specific sources
    
    Unlike trajectory, route does not need to attend to trajectory (unidirectional).
    But it benefits from route-specific conditioning for closed-loop stability.
    """
    def __init__(self, n_emb: int, status_dim: int = 14, output_dim: int = 2, p_drop: float = 0.1):
        super().__init__()
        self.ln_f = nn.LayerNorm(n_emb)
        
        # ========== Route-Specific Status Projection (Key for Stability) ==========
        # This mirrors AdaLNRouteHeadDecoderOnly's independent status_proj
        # Provides route-specific conditioning separate from shared trajectory conditioning
        self.route_status_proj = nn.Sequential(
            nn.Linear(status_dim, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb),
        )
        
        # Shared conditioning projection (for timestep + history)
        self.cond_proj = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.SiLU(),
        )
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(n_emb, n_emb),
            nn.GELU(),
            nn.Dropout(p_drop),
        )
        
        # ========== Final AdaLN Modulation (Key for Stability) ==========
        # This mirrors AdaLNRouteHeadDecoderOnly's final_adaLN
        # Provides fine-grained control over route output based on current ego state
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_emb, 2 * n_emb, bias=True),
        )
        
        # Output projection
        self.output_head = nn.Linear(n_emb, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize final AdaLN to identity (shift=0, scale=0 -> x * 1 + 0)
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        conditioning: torch.Tensor,
        ego_status: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T_route, n_emb) - route decoder output
            conditioning: (B, n_emb) - timestep + history conditioning
            ego_status: (B, status_dim) - current ego status for route-specific conditioning
        Returns:
            (B, T_route, output_dim) - route prediction (waypoints)
        """
        x = self.ln_f(x)
        
        # Combine route-specific and shared conditioning
        route_cond = self.route_status_proj(ego_status) + self.cond_proj(conditioning)
        
        # Add conditioning to features
        x = x + route_cond.unsqueeze(1)  # (B, T_route, n_emb)
        
        # MLP processing
        x = x + self.mlp(x)
        
        # ========== Final AdaLN Modulation (Key for Stability) ==========
        # Apply route-specific modulation before output
        shift, scale = self.final_adaLN(route_cond).chunk(2, dim=1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        # Output projection
        return self.output_head(x)


# =============================================================================
# Unified Decoder with Heterogeneous Queries
# =============================================================================

class UnifiedDecoderOnlyTransformer(nn.Module):
    """
    Unified Decoder-Only Transformer with heterogeneous queries.
    
    Combines trajectory prediction (horizon points) and route prediction (20 waypoints)
    into a single decoder, using:
    - Unified sinusoidal position encoding for both trajectory and route
    - Segment embeddings to distinguish query types (trajectory vs route)
    - Multi-source attention with RoPE and gating
    - Separate output heads for trajectory and route
    
    Query structure: [trajectory (horizon) | route (num_waypoints)]
    
    Position Encoding Design:
    - Both trajectory and route share the same sinusoidal position encoding scheme
    - Trajectory positions: 0, 1, 2, ... (horizon-1) representing future time steps
    - Route positions: 0, 1, 2, ... (num_waypoints-1) representing spatial waypoints
    - Segment embeddings differentiate the two modalities
    
    Cross-attention sources: All fused in single attention operation with:
        - Self-attention with RoPE on Q and K
        - Adapter cross-attention (bev, vl) with RoPE on K only
        - Task cross-attention (reasoning) with RoPE on K only + gating
    """
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        bev_dim: int = 512,
        vl_dim: int = 1536,
        reasoning_dim: int = 1536,
        horizon: int = 8,
        num_waypoints: int = 20,
        max_seq_len: int = 64,  # Max length for unified position encoding
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.horizon = horizon
        self.num_waypoints = num_waypoints
        
        # Projection layers for conditions
        self.bev_proj = nn.Sequential(nn.Linear(bev_dim, d_model), nn.LayerNorm(d_model))
        self.vl_proj = nn.Sequential(nn.Linear(vl_dim, d_model), nn.LayerNorm(d_model))
        self.reasoning_proj = nn.Sequential(nn.Linear(reasoning_dim, d_model), nn.LayerNorm(d_model))
        
        # Position embeddings for cross-attention sources
        self.bev_pos_emb = nn.Parameter(torch.zeros(1, 256, d_model))
        
        # Route queries (learnable)
        self.route_queries = nn.Parameter(torch.randn(1, num_waypoints, d_model))
        
        # ========== Unified Position Encoding ==========
        # Sinusoidal position encoding shared by trajectory and route
        # This provides a consistent spatial/temporal representation
        self.register_buffer(
            "unified_pos_encoding",
            self._create_sinusoidal_pos_encoding(max_seq_len, d_model)
        )
        
        # Learnable position scaling for each modality
        # Allows the model to learn different position importance
        self.traj_pos_scale = nn.Parameter(torch.ones(1, 1, d_model))
        self.route_pos_scale = nn.Parameter(torch.ones(1, 1, d_model))
        
        # Segment embeddings to distinguish query types (trajectory vs route)
        self.traj_segment_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.route_segment_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Decoder blocks - using new MultiSourceAttentionBlock
        self.layers = nn.ModuleList([
            MultiSourceAttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # ========== Route Residual Path (Stability Enhancement) ==========
        # This bypasses the shared decoder to preserve route-specific information
        # Key insight: Original AdaLNRouteHeadDecoderOnly had independent decoder,
        # this residual path simulates that independence within unified architecture
        self.route_residual_path = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Learnable gate, initialized to 0 (residual starts inactive, model learns to use it)
        self.route_residual_gate = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
    
    def _create_sinusoidal_pos_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Create sinusoidal position encoding.
        
        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
            
        Returns:
            pos_encoding: (1, max_len, d_model) position encoding tensor
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def _init_weights(self):
        nn.init.normal_(self.bev_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.route_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.traj_segment_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.route_segment_emb, mean=0.0, std=0.02)
        nn.init.ones_(self.traj_pos_scale)
        nn.init.ones_(self.route_pos_scale)
        # Initialize route residual gate to 0 (starts inactive)
        nn.init.zeros_(self.route_residual_gate)
        for layer in self.layers:
            nn.init.zeros_(layer.adaLN_modulation[-1].weight)
            nn.init.zeros_(layer.adaLN_modulation[-1].bias)
            # Initialize route-specific AdaLN to identity
            nn.init.zeros_(layer.route_adaLN_modulation[-1].weight)
            nn.init.zeros_(layer.route_adaLN_modulation[-1].bias)
    
    def _create_block_diagonal_mask(self, T_traj: int, T_route: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Create a unidirectional block self-attention mask.
        
        This ensures:
        - Trajectory queries can attend to: trajectory + route (full visibility)
        - Route queries can only attend to: route (isolated from trajectory)
        
        Rationale:
        - Trajectory prediction benefits from knowing the planned route
        - Route planning should be independent of specific trajectory details
        
        Attention pattern (0 = allowed, -inf = blocked):
        
                    | Trajectory | Route |
        ------------------------------------
        Trajectory  |     0      |   0   |  <- can see both
        Route       |   -inf     |   0   |  <- can only see route
        
        Args:
            T_traj: Number of trajectory queries
            T_route: Number of route queries
            device: Device for the mask tensor
            dtype: Data type for the mask tensor
            
        Returns:
            mask: (T_total, T_total) mask where -inf blocks attention
        """
        T_total = T_traj + T_route
        # Start with all allowed
        mask = torch.zeros((T_total, T_total), device=device, dtype=dtype)
        
        # Block route-to-trajectory attention (lower-left block)
        # Route queries (rows T_traj:) cannot attend to trajectory keys (cols :T_traj)
        mask[T_traj:, :T_traj] = float('-inf')
        
        return mask
    
    def forward(
        self,
        traj_emb: torch.Tensor,  # (B, horizon, d_model) - trajectory query embeddings
        bev_tokens: torch.Tensor,
        vl_tokens: torch.Tensor,
        reasoning_tokens: torch.Tensor,
        conditioning: torch.Tensor,
        bev_padding_mask: Optional[torch.Tensor] = None,
        vl_padding_mask: Optional[torch.Tensor] = None,
        reasoning_padding_mask: Optional[torch.Tensor] = None,
        route_conditioning: Optional[torch.Tensor] = None,  # (B, d_model) - route-specific conditioning
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with unified queries and multi-source attention.
        
        Uses unidirectional self-attention mask:
        - Trajectory queries can see: trajectory + route (route guides trajectory)
        - Route queries can only see: route (independent planning)
        - Both can attend to all cross-attention sources (BEV, VL, Reasoning)
        
        Route-Specific Processing:
        - Route queries use independent Q adapters for cross-attention
        - Route queries use independent temperature/bias for attention
        - Route queries use independent AdaLN modulation (if route_conditioning provided)
        
        Position Encoding:
        - Both trajectory and route use unified sinusoidal position encoding
        - Learnable scaling factors allow modality-specific position importance
        - Segment embeddings differentiate trajectory from route
        
        Args:
            traj_emb: (B, horizon, d_model) - trajectory query embeddings (noisy traj embedded)
            bev_tokens: (B, seq_len, bev_dim) - BEV spatial tokens
            vl_tokens: (B, T_vl, vl_dim) - VL tokens
            reasoning_tokens: (B, T_r, reasoning_dim) - reasoning tokens
            conditioning: (B, d_model) - timestep + current_status conditioning
            route_conditioning: (B, d_model) - route-specific conditioning (optional)
            
        Returns:
            traj_out: (B, horizon, d_model) - trajectory output
            route_out: (B, num_waypoints, d_model) - route output
        """
        B = traj_emb.shape[0]
        T_traj = traj_emb.shape[1]
        T_route = self.num_waypoints
        
        # ========== Unified Position Encoding ==========
        # Get sinusoidal position encoding for trajectory
        traj_pos = self.unified_pos_encoding[:, :T_traj, :] * self.traj_pos_scale
        # Get sinusoidal position encoding for route  
        route_pos = self.unified_pos_encoding[:, :T_route, :] * self.route_pos_scale
        
        # Add position + segment embeddings to trajectory
        traj_emb = traj_emb + traj_pos + self.traj_segment_emb
        
        # Route queries with position and segment embeddings
        route_emb = self.route_queries.expand(B, -1, -1) + route_pos + self.route_segment_emb
        
        # Concatenate queries: [trajectory | route]
        x = torch.cat([traj_emb, route_emb], dim=1)  # (B, horizon + num_waypoints, d_model)
        
        # Create unidirectional self-attention mask
        self_attn_mask = self._create_block_diagonal_mask(
            T_traj, T_route, device=x.device, dtype=x.dtype
        )
        
        # Project conditions
        bev_proj = self.bev_proj(bev_tokens)
        bev_seq_len = bev_proj.shape[1]
        if bev_seq_len <= self.bev_pos_emb.shape[1]:
            bev_proj = bev_proj + self.bev_pos_emb[:, :bev_seq_len, :]
        
        vl_proj = self.vl_proj(vl_tokens)
        reasoning_proj = self.reasoning_proj(reasoning_tokens)
        
        # Decoder layers with multi-source attention
        # Pass T_traj and route_conditioning for route-specific processing
        for layer in self.layers:
            x = layer(
                x, 
                bev_proj, 
                vl_proj, 
                reasoning_proj, 
                conditioning,
                self_attn_mask, 
                bev_padding_mask,
                vl_padding_mask, 
                reasoning_padding_mask,
                route_conditioning=route_conditioning,
                T_traj=T_traj,
            )
        
        x = self.final_norm(x)
        
        # Split outputs
        traj_out = x[:, :T_traj, :]
        route_out = x[:, T_traj:, :]
        
        # ========== Route Residual Path (Stability Enhancement) ==========
        # Add route-specific residual from initial queries (bypasses shared decoder)
        # This provides a stable baseline that the shared decoder output modulates
        # Similar to how AdaLNRouteHeadDecoderOnly had independent route_queries
        route_residual = self.route_residual_path(route_emb)
        route_out = route_out + torch.sigmoid(self.route_residual_gate) * route_residual
        
        return traj_out, route_out


# =============================================================================
# Main Model: TransformerForDiffusion (Decoder-Only with Unified Queries)
# =============================================================================

class TransformerForDiffusion(ModuleAttrMixin):
    """
    Decoder-Only Transformer for Diffusion-based Trajectory Prediction.
    
    Key features:
    - Unified decoder with heterogeneous queries for trajectory + route
    - Cross-attention to: BEV -> VL -> Reasoning
    - ego_status history used for AdaLN conditioning only (not cross-attention)
    - Segment embeddings to distinguish query types
    - MLP output heads (no GRU)
    - AdaLN modulation based on timestep + current_ego_status + GRU(history)
    
    Query structure: [trajectory (horizon) | route (20)]
    Cross-attention sources: BEV -> VL -> Reasoning
    
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
        
        # Route-specific conditioning generator (key for stability)
        # This provides route queries with independent conditioning pathway
        self.route_status_proj = nn.Sequential(
            nn.Linear(status_dim, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb),
        )
        
        # Unified Decoder with heterogeneous queries
        # hist_status removed - ego_status history is now only used for AdaLN conditioning
        self.decoder = UnifiedDecoderOnlyTransformer(
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
        
        # Causal mask for trajectory queries (route queries can attend to all)
        self.causal_attn = causal_attn
        # Note: We create mask dynamically in forward() to handle variable lengths
        
        # Output heads with Route Guidance
        # TrajectoryHead now receives route features for guidance
        self.trajectory_head = TrajectoryMLPHead(n_emb, output_dim, p_drop_emb, num_heads=n_head)
        # RouteHead with independent conditioning support (key for closed-loop stability)
        self.route_head = RouteMLPHead(n_emb, status_dim=status_dim, output_dim=2, p_drop=p_drop_emb)
        
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
            elif 'gating_factor' in name:
                no_decay.add(name)
            elif 'temp_' in name or 'bias_' in name:
                # Temperature and bias parameters - no weight decay
                no_decay.add(name)
            elif 'inv_freq' in name:
                # inv_freq is a buffer, should not be in param_dict, but handle if exists
                no_decay.add(name)
            # ========== New parameters for stability enhancement ==========
            elif '_residual_gate' in name or '_residual_query' in name:
                # Residual gates and queries - no weight decay (like other gate params)
                no_decay.add(name)
            elif '_pos_scale' in name:
                # Position scaling parameters - no weight decay
                no_decay.add(name)
            elif 'summary_query' in name:
                # History encoder summary query - no weight decay
                no_decay.add(name)
            elif 'guidance_gate' in name:
                # Route guidance gate - no weight decay
                no_decay.add(name)
            elif 'route_temp_' in name or 'route_bias_' in name:
                # Route-specific temperature and bias parameters - no weight decay
                no_decay.add(name)
        
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0, f"Missing params: {param_dict.keys() - union_params}"
        
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
            AdaLN conditioning: current_status (last frame) + GRU-encoded global history
            
        Note: hist_status removed from cross-attention - ego_status is now only used for 
        AdaLN conditioning, not as a cross-attention source.
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
        
        # Combined conditioning for trajectory AdaLN modulation
        conditioning = time_emb + status_emb + hist_global_emb
        
        # 4. Route-specific conditioning (independent pathway for stability)
        # Uses separate status projection + shared time embedding
        route_status_emb = self.route_status_proj(current_status)
        route_conditioning = time_emb + route_status_emb + hist_global_emb
        
        # ========== Trajectory Query Embedding ==========
        traj_emb = self.input_emb(sample)  # (B, T_traj, n_emb)
        pos_emb = self.pos_emb[:, :T_traj, :]
        traj_emb = traj_emb + pos_emb
        traj_emb = self.drop(traj_emb)
        traj_emb = self.pre_decoder_norm(traj_emb)
        
        # ========== Padding Masks ==========
        vl_padding_mask = ~vl_mask if vl_mask is not None else (torch.norm(vl_tokens, dim=-1) == 0)
        reasoning_padding_mask = ~reasoning_mask if reasoning_mask is not None else (torch.norm(reasoning_tokens, dim=-1) == 0)
        
        # ========== Unified Decoder ==========
        # Note: Unidirectional self-attention mask is created internally by decoder
        # - Trajectory can see: trajectory + route (route guides trajectory)
        # - Route can only see: route (independent planning)
        # Cross-attention order: BEV -> VL -> Reasoning
        # Route-specific processing: independent Q adapters, temperature/bias, AdaLN
        traj_out, route_out = self.decoder(
            traj_emb,
            bev_tokens, vl_tokens, reasoning_tokens, conditioning,
            bev_padding_mask, vl_padding_mask, reasoning_padding_mask,
            route_conditioning=route_conditioning,
        )
        
        # ========== Output Heads ==========
        # Route head processes first (no dependency on trajectory)
        # Pass current_status for route-specific conditioning (key for stability)
        route_pred = self.route_head(route_out, conditioning, ego_status=current_status)
        
        # Trajectory head with Route Guidance (attends to route features)
        trajectory = self.trajectory_head(traj_out, conditioning, route_features=route_out)
        
        return trajectory, route_pred


# =============================================================================
# Test
# =============================================================================

def test():
    """Test the unified decoder-only architecture without history cross-attention."""
    print("=" * 60)
    print("Testing TransformerForDiffusion (Unified Decoder-Only)")
    print("No History Cross-Attention - Only BEV, VL, Reasoning")
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
        status_dim=14,  # Updated: speed(1) + theta(1) + command(6) + target_point(2) + target_point_next(2) + waypoints(2)
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
    ego_status = torch.randn((B, 4, 14))  # 4 frames of history with updated dim
    
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
    
    print("\nTest 3: Different ego_status affects conditioning")
    ego_status2 = torch.randn((B, 4, 14))
    traj3, _ = transformer(sample=sample, timestep=timestep, cond=cond,
                          bev_tokens=bev_tokens,
                          gen_vit_tokens=vl_tokens, reasoning_query_tokens=reasoning_tokens,
                          ego_status=ego_status2)
    diff_hist = torch.abs(trajectory - traj3).mean()
    print(f"  Difference: {diff_hist:.6f}")
    assert diff_hist > 0, "Ego status should affect conditioning"
    
    print("\nTest 4: Optimizer")
    opt = transformer.configure_optimizers()
    print(f"  Optimizer created: {type(opt).__name__}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nArchitecture Improvements:")
    print("  1. History Encoder with Attention:")
    print("     - GRU for sequential encoding")
    print("     - Temporal attention for selective focus")
    print("     - Fusion of global and selective information")
    print("\n  2. Route-to-Trajectory Guidance:")
    print("     - TrajectoryHead attends to route features")
    print("     - Gated residual connection for controlled influence")
    print("     - Improves trajectory-route consistency")
    print("\n  3. Unified Position Encoding:")
    print("     - Sinusoidal encoding shared by trajectory & route")
    print("     - Learnable scaling per modality")
    print("     - Segment embeddings for type differentiation")
    print("\n  4. Unidirectional Self-Attention:")
    print("     - Trajectory can see: trajectory + route")
    print("     - Route can only see: route (independent)")
    print("\n  5. Source-Specific Residual Paths:")
    print("     - BEV: spatial avg pooling + MLP projection")
    print("     - VL: attention pooling with learnable query")
    print("     - Reasoning: attention pooling with learnable query")
    print("     - Gated addition bypasses competitive softmax")
    print("     - Ensures each modality contributes information")
    print("\n  6. Route-Specific Components (NEW - Stability Enhancement):")
    print("     - Route-specific Q adapters for cross-attention")
    print("     - Route-specific temperature & bias for attention")
    print("     - Route-specific AdaLN modulation")
    print("     - Route-specific conditioning pathway")
    print("     - Route residual path (bypasses shared decoder)")
    print("     - Final AdaLN in RouteMLPHead")
    print("\nInformation Flow:")
    print("  Main path: fused softmax over all sources (competitive)")
    print("  Residual: direct paths from each source (guaranteed)")
    print("  Route: env context → route planning (independent)")
    print("  Trajectory: env context + route guidance → trajectory")
    print("\nRoute Stability Design:")
    print("  - Independent Q adapters: route learns different attention")
    print("  - Independent AdaLN: route has separate modulation")
    print("  - Independent conditioning: route_status_proj + time_emb")
    print("  - Residual path: stable baseline from initial queries")
    print("  - Final AdaLN: fine-grained output control")
    print("\nOutput Heads:")
    print("  - RouteMLPHead: with independent conditioning + final AdaLN")
    print("  - TrajectoryMLPHead: with route guidance + conditioning")
    print("=" * 60)


if __name__ == "__main__":
    test()
