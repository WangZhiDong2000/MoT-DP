from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
import math

logger = logging.getLogger(__name__)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(self,
        action_dim, obs_dim,
        # obs mask setup
        max_n_obs_steps=2, 
        fix_obs_steps=True, 
        # action mask
        action_visible=False
        ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape, seed=None):
        device = self.device
        B, T, D = shape
        assert D == (self.action_dim + self.obs_dim)

        # create all tensors on this device
        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)

        # generate dim mask
        dim_mask = torch.zeros(size=shape, 
            dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[...,:self.action_dim] = True
        is_obs_dim = ~is_action_dim

        # generate obs mask
        if self.fix_obs_steps:
            obs_steps = torch.full((B,), 
            fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1, high=self.max_n_obs_steps+1, 
                size=(B,), generator=rng, device=device)
            
        steps = torch.arange(0, T, device=device).reshape(1,T).expand(B,T)
        obs_mask = (steps.T < obs_steps).T.reshape(B,T,1).expand(B,T,D)
        obs_mask = obs_mask & is_obs_dim

        # generate action mask
        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1, 
                torch.tensor(0,
                    dtype=obs_steps.dtype, 
                    device=obs_steps.device))
            action_mask = (steps.T < action_steps).T.reshape(B,T,1).expand(B,T,D)
            action_mask = action_mask & is_action_dim

        mask = obs_mask
        if self.action_visible:
            mask = mask | action_mask
        
        return mask

class CustomEncoderBlock(nn.Module):
    """
    Encoder block that handles condition embedding, VL pooling, encoding, and memory projection
    """
    def __init__(self, n_emb, n_head, n_cond_layers, p_drop_emb, p_drop_attn, vl_emb_dim, 
                 obs_as_cond, cond_dim, T_cond):
        super().__init__()
        self.n_emb = n_emb
        self.obs_as_cond = obs_as_cond
        self.T_cond = T_cond
        
        # Time embedding
        self.time_emb = SinusoidalPosEmb(n_emb)
        
        # Observation condition embedding
        self.cond_obs_emb = None
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        
        # VL features processing
        self.vl_emb_proj = nn.Linear(vl_emb_dim, n_emb)
        self.vl_emb_norm = nn.LayerNorm(n_emb)
        self.vl_attention_pooling = nn.MultiheadAttention(
            embed_dim=n_emb,
            num_heads=n_head,
            dropout=p_drop_attn,
            batch_first=True
        )
        self.vl_pool_query = nn.Parameter(torch.randn(1, 1, n_emb))
        
        # Position embedding and preprocessing
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        self.pre_encoder_norm = nn.LayerNorm(n_emb)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_cond_layers
        )
        
        # Memory processing
        self.memory_norm = nn.LayerNorm(n_emb)
        self.memory_proj = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.GELU(),
            nn.Dropout(p_drop_attn),
            nn.Linear(n_emb, n_emb)
        )
    
    def _attention_pool_vl_features(self, vl_features: torch.Tensor, vl_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention pooling to vl features"""
        batch_size = vl_features.shape[0]
        query = self.vl_pool_query.expand(batch_size, -1, -1)  # (B, 1, n_emb)
        pooled_features, _ = self.vl_attention_pooling(
            query=query,                    # (B, 1, n_emb)
            key=vl_features,               # (B, T_vl, n_emb)
            value=vl_features,             # (B, T_vl, n_emb)
            key_padding_mask=vl_padding_mask  # (B, T_vl)
        )
        return pooled_features  # (B, 1, n_emb)
    
    def forward(self, timestep: torch.Tensor, vl_embeds: torch.Tensor, cond: Optional[torch.Tensor] = None, 
                vl_padding_mask: Optional[torch.Tensor] = None):
        """
        timestep: (B,) timestep tensor
        vl_embeds: (B, T_vl, D_vl) vision-language embeddings
        cond: (B, T_cond, cond_dim) condition tensor
        vl_padding_mask: (B, T_vl) padding mask for VL features
        """
        
        # 1. Time embedding
        time_emb = self.time_emb(timestep).unsqueeze(1)  # (B, 1, n_emb)
        
        # 2. Process VL features
        vl_features = self.vl_emb_proj(vl_embeds)  # (B, T_vl, n_emb)
        vl_features = self.vl_emb_norm(vl_features)
        vl_features_processed = self._attention_pool_vl_features(vl_features, vl_padding_mask)
        
        # 3. Combine condition embeddings
        cond_embeddings = time_emb
        if self.obs_as_cond and cond is not None:
            cond_obs_emb = self.cond_obs_emb(cond)
            cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
        cond_embeddings = torch.cat([cond_embeddings, vl_features_processed], dim=1)
        
        # 4. Add position embedding
        tc = cond_embeddings.shape[1]
        if tc <= self.cond_pos_emb.shape[1]:
            position_embeddings = self.cond_pos_emb[:, :tc, :]
        else:
            position_embeddings = torch.zeros(1, tc, self.cond_pos_emb.shape[2], 
                                            device=self.cond_pos_emb.device, dtype=self.cond_pos_emb.dtype)
            position_embeddings[:, :self.cond_pos_emb.shape[1], :] = self.cond_pos_emb
            if tc > self.cond_pos_emb.shape[1]:
                torch.nn.init.normal_(position_embeddings[:, self.cond_pos_emb.shape[1]:, :], mean=0.0, std=0.02)
        
        # 5. Apply dropout and pre-norm
        x = self.drop(cond_embeddings + position_embeddings)
        x = self.pre_encoder_norm(x)
        
        # 6. Transformer encoder
        x = self.encoder(x)
        
        # 7. Memory processing
        memory = self.memory_norm(x)
        memory = memory + self.memory_proj(memory)
        
        return memory, vl_features


class CustomDecoderLayer(nn.Module):
    """
    DP decoder. Memory first enhanced by VL feature, than guide trajectory generation
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.memory_vl_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.traj_memory_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        self.activation = torch.nn.functional.gelu
    
    def forward(self, tgt, memory, vl_features, tgt_mask=None, memory_mask=None, 
                vl_key_padding_mask=None):
        
        # three masks are used: tgt_mask, memory_mask, vl_key_padding_mask
        # 1. Self-attention on trajectory 
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask, key_padding_mask=None)
        tgt = tgt + self.dropout1(tgt2)
        
        # 2. Memory-VL cross attention 
        memory2 = self.norm2(memory)
        enhanced_memory_output, _ = self.memory_vl_cross_attn(memory2, vl_features, vl_features, 
                                                      key_padding_mask=vl_key_padding_mask)
        enhanced_memory = memory + self.dropout2(enhanced_memory_output)
        
        # 3. Trajectory-Memory cross attention 
        tgt2 = self.norm3(tgt)
        tgt2, _ = self.traj_memory_cross_attn(tgt2, enhanced_memory, enhanced_memory, 
                                             attn_mask=memory_mask, key_padding_mask=None)
        tgt = tgt + self.dropout3(tgt2)
        
        # 4. Feed forward
        tgt2 = self.norm4(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
            
        return tgt


class CustomTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, obs_as_cond=False, causal_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        
        # Mask generation settings
        self.obs_as_cond = obs_as_cond
        self.causal_attn = causal_attn
    
    def _generate_memory_mask(self, x, memory, cond, tgt_mask):
        """Generate dynamic memory mask for causal attention"""
        if not self.causal_attn or tgt_mask is None:
            return None
            
        actual_memory_length = memory.shape[1]
        T_actual = x.shape[1]  
        S_actual = actual_memory_length 
        time_pos = 0  
        obs_start = 1   
        obs_end = obs_start + (cond.shape[1] if cond is not None and self.obs_as_cond else 0)
        
        # VL features are pooled to 1 token
        vl_start = obs_end
        vl_end = vl_start + 1
        
        memory_mask_dynamic = torch.zeros((T_actual, S_actual), device=x.device, dtype=torch.bool)
        
        for t in range(T_actual):
            # Time embedding: visible to all positions
            memory_mask_dynamic[t, time_pos] = True
            
            # Vision-language features: visible to all positions (1 pooled token)
            if vl_start < S_actual and vl_end <= S_actual:
                memory_mask_dynamic[t, vl_start:vl_end] = True
            
            # Observation conditions: causal visibility
            if self.obs_as_cond and cond is not None and obs_start < obs_end:
                visible_obs_end = min(obs_start + t + 1, obs_end)
                memory_mask_dynamic[t, obs_start:visible_obs_end] = True
        
        memory_mask = memory_mask_dynamic.float().masked_fill(
            memory_mask_dynamic == 0, float('-inf')
        ).masked_fill(memory_mask_dynamic == 1, float(0.0))
        
        return memory_mask
        
    def forward(self, tgt, memory, vl_features, cond=None, tgt_mask=None, 
                vl_key_padding_mask=None):
        # Generate dynamic memory mask
        memory_mask = self._generate_memory_mask(tgt, memory, cond, tgt_mask)
        
        # Decoder layers
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, vl_features, tgt_mask=tgt_mask, memory_mask=memory_mask,
                          vl_key_padding_mask=vl_key_padding_mask)
        return output


class TrajectoryHead(nn.Module):
    def __init__(self, n_emb, output_dim, p_drop_emb):
        super().__init__()
        self.ln_f = nn.LayerNorm(n_emb)
        self.drop = nn.Dropout(p_drop_emb)
        self.head = nn.Linear(n_emb, output_dim)

    def forward(self, x):
        x = self.ln_f(x)
        x = self.drop(x)
        x = self.head(x)
        return x


class TransformerForDiffusion(ModuleAttrMixin):
    def __init__(self,
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
            causal_attn: bool=False,
            obs_as_cond: bool=False,
            n_cond_layers: int = 4,
            vl_emb_dim: int = 1536
        ) -> None:
        super().__init__()

        if n_obs_steps is None:
            n_obs_steps = horizon
        
        T = horizon
        T_cond = 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            T_cond += n_obs_steps
        
        # Compute VL tokens count for T_cond
        self.target_vl_tokens = None
        vl_tokens_count = 1 
        T_cond += vl_tokens_count   

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        self.pre_decoder_norm = nn.LayerNorm(n_emb)

        # Custom encoder block that handles all condition processing
        self.encoder_block = CustomEncoderBlock(
            n_emb=n_emb,
            n_head=n_head,
            n_cond_layers=n_cond_layers,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            vl_emb_dim=vl_emb_dim,
            obs_as_cond=obs_as_cond,
            cond_dim=cond_dim,
            T_cond=T_cond
        )
        # Custom decoder that integrates VL cross attention and pre-processing
        custom_decoder_layer = CustomDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=p_drop_attn,
            batch_first=True
        )
        self.decoder = CustomTransformerDecoder(
            decoder_layer=custom_decoder_layer,
            num_layers=n_layer,
            obs_as_cond=obs_as_cond,
            causal_attn=causal_attn
        )

        # attention mask
        if causal_attn:
            # self-attention causal mask is computed here
            # however, cross attention mask is moved to the forward pass
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
            self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # trajectory head block
        self.trajectory_head = TrajectoryHead(n_emb, output_dim, p_drop_emb)
        
        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.obs_as_cond = obs_as_cond
        self.vl_emb_dim = vl_emb_dim

        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.GELU,
            nn.Sequential,
            CustomEncoderBlock,
            CustomDecoderLayer,
            CustomTransformerDecoder,
            TrajectoryHead,
            nn.ModuleList
        )  #
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if hasattr(module, 'vl_pool_query'):
                torch.nn.init.normal_(module.vl_pool_query, mean=0.0, std=0.02)
            if hasattr(module, 'cond_pos_emb') and module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root module as not decayed
        param_dict = {pn: p for pn, p in self.named_parameters()}
        for name in param_dict:
            if 'pos_emb' in name or '_dummy_variable' in name:
                no_decay.add(name)
            elif 'vl_pool_query' in name or 'cond_pos_emb' in name:
                no_decay.add(name)

        # validate that we considered every parameter
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        vl_embeds: torch.Tensor,
        cond: torch.Tensor, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        vl_embeds: (B, T_vl, D_vl) 
        output: (B,T,input_dim)
        """
        sample = sample.contiguous()
        cond = cond.contiguous() 
        vl_embeds = vl_embeds.contiguous()
        
        # 1. Prepare timesteps
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        
        # 2. Check VL padding
        vl_padding_mask = None
        if 'vl_mask' in kwargs and kwargs['vl_mask'] is not None:
            vl_padding_mask = ~kwargs['vl_mask']  
        else:
            vl_norm = torch.norm(vl_embeds, dim=-1)  # (B, T_vl)
            vl_padding_mask = (vl_norm == 0) 
        
        # 3. Process conditions through encoder block to get memory
        memory, vl_features = self.encoder_block(
            timestep=timesteps, 
            vl_embeds=vl_embeds, 
            cond=cond, 
            vl_padding_mask=vl_padding_mask)
        
        # 4. Pre-decoder processing
        token_embeddings = self.input_emb(sample)
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.pre_decoder_norm(x)
        
        # 5. Decoder with integrated memory mask handling and VL cross attention
        x = self.decoder(
            tgt=x,
            memory=memory,
            vl_features=vl_features,
            cond=cond,
            tgt_mask=self.mask,
            vl_key_padding_mask=vl_padding_mask
        )        
        
        # 6. trajectory head 
        x = self.trajectory_head(x)
        return x



def test():
    # GPT with time embedding and obs cond and encoder
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        n_cond_layers=4,
        vl_emb_dim=1536
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    vl_embeds = torch.ones((4,36,1536))
    out = transformer(sample, timestep, vl_embeds, cond)
    print(out.shape)

    
if __name__ == "__main__":
    test()
