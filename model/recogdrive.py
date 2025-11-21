# ------------------------------------------------------------------------
# Modified from LightningDiT(https://github.com/hustvl/LightningDiT)
# Copyright (c) Xiaomi Corporation. All rights reserved.
# ------------------------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np

from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
)

from .blocks.rmsnorm import RMSNorm
from .blocks.attention import Attention
from .blocks.rope import RotaryEmbedding
from .blocks.encoder import SwiGLUFFN

class TimestepEncoder(nn.Module):
    """Encodes scalar timesteps into a high-dimensional vector."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = self.timestep_embedder.linear_1.weight.dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        return timesteps_emb



class FinalLayer(nn.Module):
    """
    The final output layer of the DiT model.

    Encapsulates the final adaptive layer normalization and the projection to
    the output dimension.
    """
    def __init__(self, hidden_size: int, output_dim: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, output_dim)
        self.modulation_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size )
        )

    @torch.compile
    def modulate(self, x, shift, scale):
        if shift is None:
            return x * (1 + scale.unsqueeze(1))
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation_proj(conditioning).chunk(2, dim=1)
        x = self.modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LightningDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        norm_type: str = "layer_norm", 
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        if norm_type == "layer_norm":
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(dim)

        self.attn = Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
        )

        if norm_type == "layer_norm":
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm2 = RMSNorm(dim)

        self.ffn = SwiGLUFFN(
            dim,
            bias=True,
        )

        self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, 6 * dim, bias=True)
            )

    @torch.compile
    def modulate(self, x, shift, scale):
        if shift is None:
            return x * (1 + scale.unsqueeze(1))
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        rotary_embedder: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        mod_params = self.adaLN_modulation(conditioning)
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = \
            mod_params.chunk(6, dim=1)

        normed_states = self.norm1(hidden_states)
        modulated_states = self.modulate(normed_states, shift_attn, scale_attn)

        attn_output = self.attn(
            modulated_states,
            encoder_hidden_states=encoder_hidden_states,
            rotary_embedder=rotary_embedder
        )
        hidden_states = hidden_states + gate_attn.unsqueeze(1) * attn_output

        normed_states = self.norm2(hidden_states)
        modulated_states = self.modulate(normed_states, shift_ffn, scale_ffn)
        ffn_output = self.ffn(modulated_states)

        hidden_states = hidden_states + gate_ffn.unsqueeze(1) * ffn_output
        
        return hidden_states

class LightningDiT(nn.Module):

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 48,
        output_dim: int = 512,
        num_layers: int = 16,
        dropout: float = 0.0,
        attention_bias: bool = True,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-5,
        interleave_attention: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.output_dim = output_dim
        self.interleave_attention = interleave_attention

        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim
        )

        self.rotary_embedder = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=8, 
        )

        self.transformer_blocks = nn.ModuleList([
            LightningDiTBlock(
                dim=self.inner_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                attention_bias=attention_bias,
                norm_type=norm_type,
                norm_eps=norm_eps,
                cross_attention_dim=self.inner_dim if (idx % 2 != 0 or not interleave_attention) else None
            ) for idx in range(num_layers)
        ])
                                           
        self.final_layer = FinalLayer(self.inner_dim, output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initializes weights for stable training.
        - Initializes positional embeddings with sine/cosine values.
        - Zeroes out the weights of modulation and final output layers.
        """
        def zero_out_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        for block in self.transformer_blocks:
            block.adaLN_modulation.apply(zero_out_init)
        self.final_layer.modulation_proj.apply(zero_out_init)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        conditioning_features: torch.Tensor,
        timesteps: torch.LongTensor,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the UnifiedDiT model.

        Args:
            hidden_states (torch.Tensor): Input sequence. Shape: (B, N, D).
            encoder_hidden_states (torch.Tensor): Context sequence for cross-attention.
            conditioning_features (torch.Tensor): Additional conditioning features.
            timesteps (torch.LongTensor): Diffusion timesteps.
            return_hidden_states (bool): If True, returns the output and a list of
                all intermediate hidden states.

        Returns:
            The final output tensor, or a tuple of the output and all hidden states.
        """
        seq_len = hidden_states.shape[1]
        #hidden_states = hidden_states + self.pos_embed[:, :seq_len, :]

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        conditioning_features = conditioning_features.contiguous()

        time_embedding = self.timestep_encoder(timesteps)
        conditioning = time_embedding + conditioning_features

        all_hidden_states = [hidden_states]
        for idx, block in enumerate(self.transformer_blocks):
            use_cross_attention = not (idx % 2 == 0 and self.interleave_attention)
            current_encoder_states = encoder_hidden_states if use_cross_attention else None
            
            hidden_states = block(
                hidden_states,
                conditioning=conditioning,
                encoder_hidden_states=current_encoder_states,
                rotary_embedder=self.rotary_embedder,
            )
            all_hidden_states.append(hidden_states)

        output = self.final_layer(hidden_states, conditioning)
        
        return (output, all_hidden_states) if return_hidden_states else output



import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding, rotate_half

class Attention(nn.Module):
    """
    A versatile and highly configurable attention module.

    Supports self-attention, cross-attention, QK Normalization, and RoPE,
    while prioritizing fused attention backends like FlashAttention.
    """
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        qk_norm: bool = True,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.num_heads = heads
        self.head_dim = dim_head
        self.scale = dim_head ** -0.5

        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        context_dim = cross_attention_dim or query_dim

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=bias)
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim, bias=out_bias),
            nn.Dropout(dropout)
        )
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_embedder: Optional[nn.Module] = None
    ) -> torch.Tensor:
        B, N_q, _ = hidden_states.shape
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        is_self_attention = encoder_hidden_states is None

        q = self.to_q(hidden_states).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)
        
        if rotary_embedder is not None:
            position_ids = torch.arange(N_q, device=hidden_states.device).unsqueeze(0)
            cos, sin = rotary_embedder(hidden_states, position_ids)
            q = (q * cos) + (rotate_half(q) * sin)
            
            if is_self_attention:
                k = (k * cos) + (rotate_half(k) * sin)

        if hasattr(F, 'scaled_dot_product_attention'):
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.to_out[-1].p if self.training else 0.0,
            )
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
            
            attn_probs = attn_scores.softmax(dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.to_out[-1].p, training=self.training)
            x = torch.matmul(attn_probs, v)

        x = x.transpose(1, 2).reshape(B, N_q, -1)
        return self.to_out(x)

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)



class SinusoidalPositionalEncoding(nn.Module):
    """
    Generates a sinusoidal encoding of shape (B, T, D) given timesteps of
    shape (B, T).
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps (torch.Tensor): A tensor of shape (B, T).

        Returns:
            torch.Tensor: The positional encoding. Shape: (B, T, D).
        """
        timesteps = timesteps.float()
        device = timesteps.device
        half_dim = self.embedding_dim // 2

        exponent = -torch.log(torch.tensor(10000.0)) / half_dim
        exponent = torch.arange(half_dim, dtype=torch.float32, device=device) * exponent

        freqs = timesteps.unsqueeze(-1) * torch.exp(exponent).unsqueeze(0)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)

        encoding = torch.cat([sin, cos], dim=-1)
        return encoding


class ActionEncoder(nn.Module):
    """
    Encodes a sequence of actions and their corresponding timesteps into a
    fixed-size embedding sequence.
    """
    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

        self.fc1 = nn.Linear(action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions (torch.Tensor): Action sequence. Shape: (B, T, action_dim).
            timesteps (torch.Tensor): Timestep for each action. Shape: (B, T).

        Returns:
            torch.Tensor: The encoded action sequence. Shape: (B, T, hidden_size).
        """
        B, T, _ = actions.shape

        action_embedding = self.fc1(actions)

        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        time_embedding = self.pos_encoding(timesteps).to(dtype=action_embedding.dtype)

        x = torch.cat([action_embedding, time_embedding], dim=-1)
        x = self.act(self.fc2(x))
        x = self.fc3(x)

        return x


class StateAttentionEncoder(nn.Module):
    """
    Encodes a flat state vector into a fixed-size embedding using a
    multi-head attention pooling mechanism.
    """
    def __init__(
        self,
        state_dim: int,
        embed_dim: int,
        num_kinematic_states: int,
        state_dropout: float = 0.75,
        num_heads: int = 4
    ):
        super().__init__()
        assert state_dim > num_kinematic_states, \
            "state_dim must be greater than num_kinematic_states"

        self.state_dim = state_dim
        self.num_kinematic_states = num_kinematic_states
        self.state_dropout = state_dropout

        self.linears = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(state_dim)
        ])
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.pos_embed = nn.Parameter(torch.Tensor(1, state_dim, embed_dim))
        self.query = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes learnable embeddings."""
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The flat state vector. Shape: (B, state_dim).

        Returns:
            torch.Tensor: The encoded state embedding. Shape: (B, embed_dim).
        """
        x_embed_list = [
            linear(x[:, i, None]) for i, linear in enumerate(self.linears)
        ]
        x_embed = torch.stack(x_embed_list, dim=1)
        x_embed = x_embed + self.pos_embed

        key_padding_mask = None
        if self.training and self.state_dropout > 0:
            kinematic_mask = torch.rand(
                (x_embed.shape[0], self.num_kinematic_states), device=x.device
            ) < self.state_dropout

            num_command_states = self.state_dim - self.num_kinematic_states
            command_mask = torch.zeros(
                (x_embed.shape[0], num_command_states),
                device=x.device,
                dtype=torch.bool
            )
            key_padding_mask = torch.cat([kinematic_mask, command_mask], dim=1)

        query = self.query.expand(x_embed.shape[0], -1, -1)
        x_state, _ = self.attn(
            query=query,
            key=x_embed,
            value=x_embed,
            key_padding_mask=key_padding_mask,
        )

        return x_state.squeeze(1)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for
                                   numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class RotaryEmbedding(nn.Module):
    """
    The Rotary Position Embedding (RoPE) module.

    This implementation uses a pre-computed cache of sine and cosine values to
    efficiently apply rotary embeddings to query and key tensors. It can
    dynamically expand the cache if a sequence longer than the initial
    `max_position_embeddings` is encountered.

    Attributes:
        dim (int): The dimension of the head the RoPE is applied to.
        max_position_embeddings (int): The maximum sequence length for the pre-computed cache.
        theta (float): The base for the geometric progression of frequencies.
        inv_freq (torch.Tensor): A buffer holding the inverse frequencies.
        cos_cached (torch.Tensor): A buffer holding the cached cosine values.
        sin_cached (torch.Tensor): A buffer holding the cached sine values.
    """
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        theta: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta

        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device):
        """
        Updates the sine and cosine cache.

        Args:
            seq_len (int): The new maximum sequence length.
            device (torch.device): The device to store the cache on.
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates rotary embeddings for the given positions.

        Args:
            x (torch.Tensor): A dummy tensor used only to get the device and dtype.
            position_ids (torch.LongTensor): The positions of the tokens in the
                sequence. Shape: (batch_size, sequence_length).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the cosine and
                sine embeddings. Shape of each: (batch_size, 1, sequence_length, dim).
        """
        seq_len = position_ids.max().item() + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device)

        cos = self.cos_cached.gather(
            2, position_ids.unsqueeze(1).unsqueeze(3).expand(-1, -1, -1, self.dim)
        )
        sin = self.sin_cached.gather(
            2, position_ids.unsqueeze(1).unsqueeze(3).expand(-1, -1, -1, self.dim)
        )
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of the input tensor.

    Splits the last dimension of the tensor into two halves, negates the
    second half, and then concatenates them back together.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The tensor with half of its dimensions rotated.
    """

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


