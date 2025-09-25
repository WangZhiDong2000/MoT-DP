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

class TransformerForDiffusion(ModuleAttrMixin):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int = None,
            cond_dim: int = 0,
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal_attn: bool=False,
            time_as_cond: bool=True,
            obs_as_cond: bool=False,
            n_cond_layers: int = 0,
            vl_emb_dim: int = 1536,
            max_vl_tokens: int = 512,
            use_vl_pooling: bool = False,  # pooling
            use_adaptive_vl_projection: bool = True, # not pooling, project to a suitable length
            target_vl_tokens: Optional[int] = None,  # a suitable length
            adaptive_projection_method: str = 'learned_pooling'  # project methods, 'learned_pooling', 'conv_downsample', 'linear_compress'
        ) -> None:
        super().__init__()

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps
        
        # Compute a suitable target VL length and adjust T_cond
        if use_adaptive_vl_projection:
            if target_vl_tokens is None:
                base_cond_tokens = 1 + (n_obs_steps if obs_as_cond else 0)  # time + obs_tokens
                self.target_vl_tokens = max(1, min(base_cond_tokens * 2, n_obs_steps)) if obs_as_cond else 1
            else:
                self.target_vl_tokens = target_vl_tokens
        else:
            self.target_vl_tokens = None
        if T_cond > 0:  
            if use_vl_pooling:
                vl_tokens_count = 1 
            elif use_adaptive_vl_projection and self.target_vl_tokens is not None:
                vl_tokens_count = self.target_vl_tokens
            else:
                vl_tokens_count = max_vl_tokens
            T_cond += vl_tokens_count   

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        self.vl_emb_proj = nn.Linear(vl_emb_dim, n_emb)
        self.vl_emb_norm = nn.LayerNorm(n_emb)

        # Adaptive VL projection modules
        self.adaptive_vl_projection = None
        self.adaptive_pooling_queries = None
        self.adaptive_conv_layer = None
        self.adaptive_linear_layer = None
        if use_adaptive_vl_projection and self.target_vl_tokens is not None:
            if adaptive_projection_method == 'learned_pooling':
                self.adaptive_pooling_queries = nn.Parameter(torch.randn(1, self.target_vl_tokens, n_emb))
                self.adaptive_vl_projection = nn.MultiheadAttention(
                    embed_dim=n_emb,
                    num_heads=n_head,
                    dropout=p_drop_attn,
                    batch_first=True
                )
            elif adaptive_projection_method == 'conv_downsample':
                downsample_ratio = max(1, max_vl_tokens // self.target_vl_tokens)
                self.adaptive_conv_layer = nn.Conv1d(n_emb, n_emb, kernel_size=downsample_ratio, 
                                                   stride=downsample_ratio, padding=0)
                self.adaptive_vl_projection = nn.LayerNorm(n_emb)
            elif adaptive_projection_method == 'linear_compress':
                self.adaptive_linear_layer = nn.Linear(max_vl_tokens * n_emb, self.target_vl_tokens * n_emb)
                self.adaptive_vl_projection = nn.LayerNorm(n_emb)
        
        # attention pooling for vl features
        self.vl_attention_pooling = nn.MultiheadAttention(
            embed_dim=n_emb,
            num_heads=n_head,
            dropout=p_drop_attn,
            batch_first=True
        )
        self.vl_pool_query = nn.Parameter(torch.randn(1, 1, n_emb))
        
        # cross attention for vl features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=n_emb,
            num_heads=n_head,
            dropout=p_drop_attn,
            batch_first=True
        )

        # FFN layer before trajectory head
        self.cross_attn_norm = nn.LayerNorm(n_emb)
        self.cross_attn_ffn = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.GELU(),
            nn.Dropout(p_drop_attn),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(p_drop_attn)
        )
        self.ffn_norm = nn.LayerNorm(n_emb)

        # memory norm and projection
        self.pre_encoder_norm = nn.LayerNorm(n_emb)
        self.memory_norm = nn.LayerNorm(n_emb)
        self.memory_proj = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.GELU(),
            nn.Dropout(p_drop_attn),
            nn.Linear(n_emb, n_emb)
        )
        self.pre_decoder_norm = nn.LayerNorm(n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
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
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
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

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
            
        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.vl_emb_dim = vl_emb_dim
        self.max_vl_tokens = max_vl_tokens
        self.use_vl_pooling = use_vl_pooling
        self.use_adaptive_vl_projection = use_adaptive_vl_projection
        self.adaptive_projection_method = adaptive_projection_method

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.GELU,
            nn.Sequential,
            nn.Conv1d)  # Add Conv1d to ignored types
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
            # Initialize adaptive pooling queries if present
            if hasattr(module, 'adaptive_pooling_queries') and module.adaptive_pooling_queries is not None:
                torch.nn.init.normal_(module.adaptive_pooling_queries, mean=0.0, std=0.02)
            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def _adaptive_project_vl_features(self, vl_features: torch.Tensor, vl_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply vlm feature projection to a suitable length.

        """
        if not self.use_adaptive_vl_projection or self.target_vl_tokens is None:
            return vl_features
        
        batch_size, seq_len, n_emb = vl_features.shape
        
        if self.adaptive_projection_method == 'learned_pooling':
            if self.adaptive_pooling_queries is None or self.adaptive_vl_projection is None:
                return vl_features
                
            queries = self.adaptive_pooling_queries.expand(batch_size, -1, -1)  # (B, target_vl_tokens, n_emb)
            
            projected_features, _ = self.adaptive_vl_projection(
                query=queries,                    # (B, target_vl_tokens, n_emb)
                key=vl_features,                 # (B, T_vl, n_emb)
                value=vl_features,               # (B, T_vl, n_emb)
                key_padding_mask=vl_padding_mask  # (B, T_vl)
            )
            
        elif self.adaptive_projection_method == 'conv_downsample':
            if self.adaptive_conv_layer is None or self.adaptive_vl_projection is None:
                return vl_features
            vl_transposed = vl_features.transpose(1, 2)  # (B, n_emb, T_vl)
            conv_output = self.adaptive_conv_layer(vl_transposed)  # (B, n_emb, output_len)
            projected_features = conv_output.transpose(1, 2)  # (B, output_len, n_emb)
            projected_features = self.adaptive_vl_projection(projected_features)
            current_len = projected_features.shape[1]
            if current_len > self.target_vl_tokens:
                projected_features = projected_features[:, :self.target_vl_tokens, :]
            elif current_len < self.target_vl_tokens:
                padding = torch.zeros(batch_size, self.target_vl_tokens - current_len, n_emb,
                                    device=vl_features.device, dtype=vl_features.dtype)
                projected_features = torch.cat([projected_features, padding], dim=1)
        
        elif self.adaptive_projection_method == 'linear_compress':
            if self.adaptive_linear_layer is None or self.adaptive_vl_projection is None:
                return vl_features
            vl_flattened = vl_features.view(batch_size, -1)  # (B, T_vl * n_emb)
            expected_input_size = self.adaptive_linear_layer.in_features
            if vl_flattened.shape[1] < expected_input_size:
                padding_size = expected_input_size - vl_flattened.shape[1]
                padding = torch.zeros(batch_size, padding_size, 
                                    device=vl_features.device, dtype=vl_features.dtype)
                vl_flattened = torch.cat([vl_flattened, padding], dim=1)
            elif vl_flattened.shape[1] > expected_input_size:
                vl_flattened = vl_flattened[:, :expected_input_size]
            
            compressed = self.adaptive_linear_layer(vl_flattened)  # (B, target_vl_tokens * n_emb)
            projected_features = compressed.view(batch_size, self.target_vl_tokens, n_emb)
            projected_features = self.adaptive_vl_projection(projected_features)
        
        else:
            raise ValueError(f"Unknown adaptive projection method: {self.adaptive_projection_method}")
        
        return projected_features
    
    
    def _attention_pool_vl_features(self, vl_features: torch.Tensor, vl_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention pooling to vl features
        """
        batch_size = vl_features.shape[0]
        query = self.vl_pool_query.expand(batch_size, -1, -1)  # (B, 1, n_emb)
        pooled_features, _ = self.vl_attention_pooling(
            query=query,                    # (B, 1, n_emb)
            key=vl_features,               # (B, T_vl, n_emb)
            value=vl_features,             # (B, T_vl, n_emb)
            key_padding_mask=vl_padding_mask  # (B, T_vl)
        )
        return pooled_features  # (B, 1, n_emb)
    
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

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        no_decay.add("vl_pool_query")  # Learnable query for vision-language attention pooling

        if self.adaptive_pooling_queries is not None:
            no_decay.add("adaptive_pooling_queries")  # Learnable queries for adaptive pooling
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
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
        cond: Optional[torch.Tensor]=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        vl_embeds: (B, T_vl, D_vl) 
        output: (B,T,input_dim)
        """
        sample = sample.contiguous()
        cond= cond.contiguous() if cond is not None else None
        vl_embeds = vl_embeds.contiguous()
        if len(vl_embeds.shape) != 3:
            raise ValueError(f"vl_embeds must be 3D tensor, got shape {vl_embeds.shape}")
        
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        # 2. condition input
        input_emb = self.input_emb(sample)
        vl_features = self.vl_emb_proj(vl_embeds)  # (B, T_vl, n_emb)
        vl_features = self.vl_emb_norm(vl_features)
        
        # 3. Check if vl_embeds has padding
        vl_padding_mask = None
        if 'vl_mask' in kwargs and kwargs['vl_mask'] is not None:
            vl_padding_mask = ~kwargs['vl_mask']  
        else:
            vl_norm = torch.norm(vl_embeds, dim=-1)  # (B, T_vl)
            vl_padding_mask = (vl_norm == 0) 
        vl_features_for_cross_attn = vl_features
        vl_padding_mask_for_cross_attn = vl_padding_mask  



        ## (1.1). encoder, handle time/observation/vision language as condition
        cond_embeddings = time_emb
        if self.obs_as_cond and cond is not None:
            cond_obs_emb = self.cond_obs_emb(cond)
            cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            
        # (1.2). Handle VL features - either pooled, adaptive projected, or full sequence
        if self.use_vl_pooling:
            vl_features_processed = self._attention_pool_vl_features(vl_features, vl_padding_mask)
        elif self.use_adaptive_vl_projection:
            vl_features_processed = self._adaptive_project_vl_features(vl_features, vl_padding_mask)
        else:
            vl_seq_len = vl_features.shape[1]
            if vl_seq_len > self.max_vl_tokens:
                vl_features_processed = vl_features[:, :self.max_vl_tokens, :]
                if vl_padding_mask is not None:
                    vl_padding_mask = vl_padding_mask[:, :self.max_vl_tokens]
            elif vl_seq_len < self.max_vl_tokens:
                padding_length = self.max_vl_tokens - vl_seq_len
                padding = torch.zeros(vl_features.shape[0], padding_length, vl_features.shape[2], 
                                    device=vl_features.device, dtype=vl_features.dtype)
                vl_features_processed = torch.cat([vl_features, padding], dim=1)
                if vl_padding_mask is not None:
                    padding_mask = torch.ones(vl_features.shape[0], padding_length, 
                                            device=vl_features.device, dtype=torch.bool)
                    vl_padding_mask = torch.cat([vl_padding_mask, padding_mask], dim=1)
            else:
                vl_features_processed = vl_features
        cond_embeddings = torch.cat([cond_embeddings, vl_features_processed], dim=1)

        # (1.3).Create encoder attention mask for padded tokens
        encoder_attn_mask = None
        if vl_padding_mask is not None and not self.use_vl_pooling:
            batch_size = cond_embeddings.shape[0]
            time_mask = torch.zeros(batch_size, 1, device=cond_embeddings.device, dtype=torch.bool)  # time is never masked
            
            obs_mask = torch.zeros(batch_size, cond.shape[1] if (self.obs_as_cond and cond is not None) else 0, 
                                 device=cond_embeddings.device, dtype=torch.bool) if self.obs_as_cond and cond is not None else torch.empty(batch_size, 0, device=cond_embeddings.device, dtype=torch.bool)
            

            vl_mask_for_encoder = vl_padding_mask
            if self.use_adaptive_vl_projection and vl_features_processed.shape[1] != vl_padding_mask.shape[1]:
                target_len = vl_features_processed.shape[1]
                if vl_padding_mask.shape[1] > target_len:
                    vl_mask_for_encoder = vl_padding_mask[:, :target_len]
                elif vl_padding_mask.shape[1] < target_len:
                    padding_mask_len = target_len - vl_padding_mask.shape[1]
                    padding_mask_ext = torch.ones(batch_size, padding_mask_len, device=cond_embeddings.device, dtype=torch.bool)
                    vl_mask_for_encoder = torch.cat([vl_padding_mask, padding_mask_ext], dim=1)
            
            # [time_mask, obs_mask, vl_mask]
            if obs_mask.shape[1] > 0:
                encoder_attn_mask = torch.cat([time_mask, obs_mask, vl_mask_for_encoder], dim=1)
            else:
                encoder_attn_mask = torch.cat([time_mask, vl_mask_for_encoder], dim=1)

            
        # (1.4). Handle position embedding
        tc = cond_embeddings.shape[1]
        if self.cond_pos_emb is not None:
            if tc <= self.cond_pos_emb.shape[1]:
                position_embeddings = self.cond_pos_emb[:, :tc, :]
            else:
                position_embeddings = torch.zeros(1, tc, self.cond_pos_emb.shape[2], 
                                                device=self.cond_pos_emb.device, dtype=self.cond_pos_emb.dtype)
                position_embeddings[:, :self.cond_pos_emb.shape[1], :] = self.cond_pos_emb
                if tc > self.cond_pos_emb.shape[1]:
                    torch.nn.init.normal_(position_embeddings[:, self.cond_pos_emb.shape[1]:, :], mean=0.0, std=0.02)
        else:
            position_embeddings = torch.zeros(1, tc, cond_embeddings.shape[2], 
                                            device=cond_embeddings.device, dtype=cond_embeddings.dtype)

        x = self.drop(cond_embeddings + position_embeddings)
        x = self.pre_encoder_norm(x)
        if self.encoder is not None:
            if encoder_attn_mask is not None and hasattr(self.encoder, 'layers'):
                x = self.encoder(x, src_key_padding_mask=encoder_attn_mask)
            else:
                x = self.encoder(x)
            
        # (1.5). Handle memory 
        memory = self.memory_norm(x) 
        memory = memory + self.memory_proj(memory)  
        
        # (2.1). Pre-decoder
        token_embeddings = input_emb
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  
        x = self.drop(token_embeddings + position_embeddings)
        x = self.pre_decoder_norm(x)  

        # (2.2).Handle memory_mask
        actual_memory_length = memory.shape[1]
        if self.mask is not None: 
            T_actual = x.shape[1]  
            S_actual = actual_memory_length 
            time_pos = 0  
            obs_start = 1   
            obs_end = obs_start + (cond.shape[1] if cond is not None and self.obs_as_cond else 0)
            
            if self.use_vl_pooling:
                vl_start = obs_end
                vl_end = vl_start + 1
            else:
                vl_start = obs_end
                vl_len = vl_features_processed.shape[1] if 'vl_features_processed' in locals() else self.max_vl_tokens
                vl_end = vl_start + vl_len
            
            memory_mask_dynamic = torch.zeros((T_actual, S_actual), device=x.device, dtype=torch.bool)
            
            for t in range(T_actual):
                # Time embedding: visible to all positions
                memory_mask_dynamic[t, time_pos] = True
                
                # Vision-language features: visible to all positions  
                if vl_start < S_actual and vl_end <= S_actual:
                    memory_mask_dynamic[t, vl_start:vl_end] = True
                
                # Observation conditions: causal visibility
                if self.obs_as_cond and cond is not None and obs_start < obs_end:
                    visible_obs_end = min(obs_start + t + 1, obs_end)
                    memory_mask_dynamic[t, obs_start:visible_obs_end] = True
            
            memory_mask = memory_mask_dynamic.float().masked_fill(
                memory_mask_dynamic == 0, float('-inf')
            ).masked_fill(memory_mask_dynamic == 1, float(0.0))
        else:
            memory_mask = None

        # (2.3).Decoder, with memory mask and causal mask
        if self.decoder is not None:
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=memory_mask
            )

        # (2.4). VL cross attention 
        x_norm_for_cross = self.cross_attn_norm(x)  
        x_cross, _ = self.cross_attention(
            query=x_norm_for_cross,  # (B, T, n_emb) 
            key=vl_features_for_cross_attn,         # (B, T_vl, n_emb) 
            value=vl_features_for_cross_attn,       # (B, T_vl, n_emb)
            key_padding_mask=vl_padding_mask_for_cross_attn  # (B, T_vl) 
        )
        x = x + x_cross 
        
        # (2.5). FFN 
        x_norm_for_ffn = self.ffn_norm(x)  
        x_ffn = self.cross_attn_ffn(x_norm_for_ffn)
        x = x + x_ffn        
        
        # 3. trajectory head
        x = self.ln_f(x)          
        x = self.drop(x)          
        x = self.head(x)         
        return x



# def test():
#     # GPT with time embedding and obs cond and encoder
#     transformer = TransformerForDiffusion(
#         input_dim=16,
#         output_dim=16,
#         horizon=8,
#         n_obs_steps=4,
#         cond_dim=10,
#         causal_attn=True,
#         time_as_cond=True,
#         n_cond_layers=4,
#         vl_emb_dim=1536
#     )
#     opt = transformer.configure_optimizers()
    
#     timestep = torch.tensor(0)
#     sample = torch.zeros((4,8,16))
#     cond = torch.zeros((4,4,10))
#     vl_embeds = torch.ones((4,36,1536))
#     out = transformer(sample, timestep, vl_embeds, cond)
#     print(out.shape)

    
# if __name__ == "__main__":
#     test()   

  

