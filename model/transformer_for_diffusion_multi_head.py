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
        vl_tokens: torch.Tensor,    # (B, T_v, d_model) - VL tokens (already projected)
        reasoning_tokens: torch.Tensor,  # (B, T_r, d_model) - reasoning tokens (already projected)
        conditioning: torch.Tensor,  # (B, d_model) - conditioning for AdaLN
        self_attn_mask: Optional[torch.Tensor] = None,  # (T, T) - block diagonal mask for segment isolation
        bev_padding_mask: Optional[torch.Tensor] = None,
        vl_padding_mask: Optional[torch.Tensor] = None,
        reasoning_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with multi-source attention.
        
        Attention is computed as:
        1. Self-attention: Q(x) @ K(x)^T with RoPE on both Q and K
           - Uses block diagonal mask to isolate trajectory and route segments
        2. Adapter cross-attention: Q_adapted(x) @ [K(bev), K(vl)]^T with RoPE on K only
        3. Task cross-attention: Q_adapted(x) @ K(reasoning)^T with RoPE on K only, scaled by gating
        
        All attention scores are concatenated, temperature-scaled, bias-adjusted, and softmaxed together.
        
        Note: 
        - hist_tokens removed - ego_status history is now only used for AdaLN conditioning.
        - self_attn_mask should be block diagonal to isolate trajectory and route segments
        """
        B, T, C = x.shape
        T_b = bev_tokens.shape[1]
        T_v = vl_tokens.shape[1]
        T_r = reasoning_tokens.shape[1]
        
        # ========== AdaLN modulation parameters ==========
        mod_params = self.adaLN_modulation(conditioning)
        shift_pre, scale_pre, gate_attn, shift_ffn, scale_ffn, gate_ffn = mod_params.chunk(6, dim=1)
        
        # ========== Pre-LayerNorm with modulation ==========
        x_norm = self.modulate(self.norm_pre(x), shift_pre, scale_pre)
        
        # ========== Gating factor for Reasoning ==========
        g = self.gating_factor
        ratio_g = torch.tanh(g)
        
        # ========== Temperature scaling factors (softplus for positive values) ==========
        temp_self = 1.0 + torch.nn.functional.softplus(self.temp_self)
        temp_bev = 1.0 + torch.nn.functional.softplus(self.temp_bev)
        temp_vl = 1.0 + torch.nn.functional.softplus(self.temp_vl)
        temp_reason = 1.0 + torch.nn.functional.softplus(self.temp_reason)
        
        # ========== Q projection (base + source-specific adaptations) ==========
        q_base = self.q_proj(x_norm)  # (B, T, d_model)
        
        # Source-specific Q adaptations (low-rank: d -> d/4 -> d)
        q_adapt_bev = self.q_adapter_out(torch.tanh(self.q_adapter_bev(x_norm)))
        q_adapt_vl = self.q_adapter_out(torch.tanh(self.q_adapter_vl(x_norm)))
        q_adapt_reason = self.q_adapter_out(torch.tanh(self.q_adapter_reason(x_norm)))
        
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
        
        # Adapter cross-attention scores (with source-specific Q, temperature and bias)
        attn_bev = torch.matmul(q_bev, k_bev.transpose(-2, -1)) * temp_bev + self.bias_bev       # (B, nhead, T, T_b)
        attn_vl = torch.matmul(q_vl, k_vl.transpose(-2, -1)) * temp_vl + self.bias_vl           # (B, nhead, T, T_v)
        
        # Task cross-attention scores (with gating for Reasoning, temperature and bias)
        attn_reason = torch.matmul(q_reason, k_reason.transpose(-2, -1)) * ratio_g * temp_reason + self.bias_reason  # (B, nhead, T, T_r)
        
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
    Unified Decoder-Only Transformer with heterogeneous queries.
    
    Combines trajectory prediction (horizon points) and route prediction (20 waypoints)
    into a single decoder, using:
    - Segment embeddings to distinguish query types (trajectory vs route)
    - Multi-source attention with RoPE and gating (MLPResNetBlock_Pro style)
    - Separate output heads for trajectory and route
    
    Query structure: [trajectory (horizon) | route (num_waypoints)]
    Cross-attention sources: All fused in single attention operation with:
        - Self-attention with RoPE on Q and K
        - Adapter cross-attention (bev, vl) with RoPE on K only
        - Task cross-attention (reasoning) with RoPE on K only + gating
    
    Note: hist_status removed - ego_status history is now only used for AdaLN conditioning.
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
        
        # Position embeddings for queries
        self.route_pos_emb = nn.Parameter(torch.zeros(1, num_waypoints, d_model))
        
        # Segment embeddings to distinguish query types
        self.traj_segment_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.route_segment_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Decoder blocks - using new MultiSourceAttentionBlock
        self.layers = nn.ModuleList([
            MultiSourceAttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.bev_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.route_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.route_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.traj_segment_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.route_segment_emb, mean=0.0, std=0.02)
        for layer in self.layers:
            nn.init.zeros_(layer.adaLN_modulation[-1].weight)
            nn.init.zeros_(layer.adaLN_modulation[-1].bias)
    
    def _create_block_diagonal_mask(self, T_traj: int, T_route: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Create a block diagonal self-attention mask.
        
        This ensures:
        - Trajectory queries (0:T_traj) can only attend to other trajectory queries
        - Route queries (T_traj:T_traj+T_route) can only attend to other route queries
        
        Args:
            T_traj: Number of trajectory queries
            T_route: Number of route queries
            device: Device for the mask tensor
            dtype: Data type for the mask tensor
            
        Returns:
            mask: (T_total, T_total) mask where -inf blocks cross-segment attention
        """
        T_total = T_traj + T_route
        # Start with all blocked (will use additive mask, so -inf means blocked)
        mask = torch.full((T_total, T_total), float('-inf'), device=device, dtype=dtype)
        
        # Allow trajectory-to-trajectory attention (upper-left block)
        mask[:T_traj, :T_traj] = 0.0
        
        # Allow route-to-route attention (lower-right block)
        mask[T_traj:, T_traj:] = 0.0
        
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with unified queries and multi-source attention.
        
        Uses block diagonal self-attention mask to ensure:
        - Trajectory queries only attend to other trajectory queries in self-attention
        - Route queries only attend to other route queries in self-attention
        - Both can attend to all cross-attention sources (BEV, VL, Reasoning)
        
        This design allows trajectory and route to be learned independently
        while sharing the same cross-attention context.
        
        Args:
            traj_emb: (B, horizon, d_model) - trajectory query embeddings (noisy traj embedded)
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
        T_route = self.num_waypoints
        
        # Add segment embedding to trajectory
        traj_emb = traj_emb + self.traj_segment_emb
        
        # Route queries with position and segment embeddings
        route_emb = self.route_queries.expand(B, -1, -1) + self.route_pos_emb + self.route_segment_emb
        
        # Concatenate queries: [trajectory | route]
        x = torch.cat([traj_emb, route_emb], dim=1)  # (B, horizon + num_waypoints, d_model)
        
        # Create block diagonal self-attention mask
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
                reasoning_padding_mask
            )
        
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
            elif 'gating_factor' in name:
                no_decay.add(name)
            elif 'temp_' in name or 'bias_' in name:
                # Temperature and bias parameters - no weight decay
                no_decay.add(name)
            elif 'inv_freq' in name:
                # inv_freq is a buffer, should not be in param_dict, but handle if exists
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
        
        # Combined conditioning for AdaLN modulation
        conditioning = time_emb + status_emb + hist_global_emb
        
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
        # Note: Block diagonal self-attention mask is created internally by decoder
        # This ensures trajectory and route queries only attend to their own segment in self-attention
        # Cross-attention order: BEV -> VL -> Reasoning (hist removed)
        traj_out, route_out = self.decoder(
            traj_emb,
            bev_tokens, vl_tokens, reasoning_tokens, conditioning,
            bev_padding_mask, vl_padding_mask, reasoning_padding_mask
        )
        
        # ========== Output Heads ==========
        trajectory = self.trajectory_head(traj_out, conditioning)
        route_pred = self.route_head(route_out)
        
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
    print("\nArchitecture summary:")
    print("  - Queries: [trajectory (8) | route (20)]")
    print("  - Self-attention: Block diagonal (trajectory & route isolated)")
    print("  - Cross-attention sources: BEV(197) -> VL -> Reasoning")
    print("  - AdaLN conditioning: timestep + current_status + GRU(history)")
    print("  - ego_status used only for conditioning, not cross-attention")
    print("\nBlock Diagonal Self-Attention:")
    print("  - Trajectory queries only attend to other trajectory queries")
    print("  - Route queries only attend to other route queries")
    print("  - Both share same cross-attention context (BEV, VL, Reasoning)")
    print("\nControl design:")
    print("  - Trajectory output: longitudinal control (speed/acceleration)")
    print("  - Route output: lateral control (steering direction)")
    print("=" * 60)


if __name__ == "__main__":
    test()
