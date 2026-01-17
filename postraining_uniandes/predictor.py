"""
Predictor architecture for post-training at Uniandes.
"""

from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from postraining_uniandes.logger_helper import get_logger


logger = get_logger()


class RoPEAttention(nn.Module):
    """
    Multi-head self-attention with Rotary Position Embeddings (RoPE).
    Supports 3D-RoPE for patches and temporal-only RoPE for action/state tokens.
    """
    
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        use_rope=True,
        grid_size=14,
        grid_depth=8,
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.use_rope = use_rope
        self.head_dim = head_dim
        if self.verbose:
            logger.info(f"[RoPEAttention] Initializing with {num_heads} heads, head_dim={head_dim}")
        # Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # RoPE frequencies
        if use_rope:
            self.grid_size = grid_size
            self.grid_depth = grid_depth
            
            self.d_t = int(2 * ((head_dim // 3) // 2))  # temporal
            self.d_h = int(2 * ((head_dim // 3) // 2))  # height
            self.d_w = int(2 * ((head_dim // 3) // 2))  # width

            # Frequency bands for spatial (h, w) and temporal (t) dimensions
            self.register_buffer("freqs_h", self._get_freqs(self.d_h))
            self.register_buffer("freqs_w", self._get_freqs(self.d_w))
            self.register_buffer("freqs_t", self._get_freqs(self.d_t))
            if self.verbose:
                logger.info(f"[RoPEAttention] RoPE enabled with grid_size={grid_size}, grid_depth={grid_depth}")
                logger.info(f"[RoPEAttention] RoPE dims - h:{self.d_h}, w:{self.d_w}, t:{self.d_t}... (total={self.d_t+self.d_h+self.d_w}/{head_dim})")

    def _get_freqs(self, dim):
        """Generate frequency bands for RoPE."""
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        return freqs
    
    def apply_rope_3d(self, x, positions_h, positions_w, positions_t):
        """
        Apply 3D Rotary Position Embeddings.
        Args:
            x: [B, num_heads, N, head_dim]
            positions_h, positions_w, positions_t: [N] position indices
        """
        B, num_heads, N, head_dim = x.shape
        
        # Split using pre-computed dimensions
        s = 0
        x_t = x[..., s:s + self.d_t]
        s += self.d_t
        x_h = x[..., s:s + self.d_h]
        s += self.d_h
        x_w = x[..., s:s + self.d_w]
        s += self.d_w
        
        # Apply rotations
        x_t = self._apply_rope_1d(x_t, positions_t, self.freqs_t)
        x_h = self._apply_rope_1d(x_h, positions_h, self.freqs_h)
        x_w = self._apply_rope_1d(x_w, positions_w, self.freqs_w)
        
        # Handle remainder dimensions if they exist (not used for RoPE)
        if s < head_dim:
            x_remainder = x[..., s:]
            return torch.cat([x_t, x_h, x_w, x_remainder], dim=-1)
        else:
            return torch.cat([x_t, x_h, x_w], dim=-1)
    
    def _apply_rope_1d(self, x, positions, freqs):
        """Apply 1D rotation."""
        # x: [B, num_heads, N, d]
        # positions: [N]
        # freqs: [d//2]
        
        d = x.shape[-1]
        # Reshape for rotation: split into pairs
        x = x.reshape(*x.shape[:-1], d // 2, 2)  # [B, num_heads, N, d//2, 2]
        
        # Compute angles
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)  # [N, d//2]
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # [1, 1, N, d//2]
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        
        # Rotate
        x_0, x_1 = x[..., 0], x[..., 1]
        x_rot_0 = x_0 * cos - x_1 * sin
        x_rot_1 = x_0 * sin + x_1 * cos
        
        x_rot = torch.stack([x_rot_0, x_rot_1], dim=-1)
        return x_rot.reshape(*x.shape[:-2], d)
    
    def forward(self, x, attn_mask=None, rope_positions=None):
        """
        Args:
            x: [B, N, dim]
            attn_mask: [N, N] attention mask
            rope_positions: dict with 'h', 'w', 't' keys for position indices [N]
        """
        if self.verbose:
            logger.info("[RoPEAttention] Forward pass")
            #logger.info(f"Rope positions provided: type is {type(rope_positions)}")
            #logger.info(f"Rope positions keys: {rope_positions['h']}")
            #logger.info(f"Rope positions keys: {rope_positions['w']}")
            #logger.info(f"Rope positions keys: {rope_positions['t']}")
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE to Q and K
        if self.use_rope and rope_positions is not None:
            q = self.apply_rope_3d(q, rope_positions['h'], rope_positions['w'], rope_positions['t'])
            k = self.apply_rope_3d(k, rope_positions['h'], rope_positions['w'], rope_positions['t'])
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        
        # Apply causal mask if provided
        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """Simple MLP with GELU activation."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, use_silu=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU() if use_silu else nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        #logger.info(f"[MLP] in={in_features}, hidden={hidden_features}, out={out_features}, act={'SiLU' if use_silu else 'GELU'}")
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Simple Transformer block with:
    - RoPE attention (3D for patches, temporal for actions/states)
    - MLP
    - Layer norms
    - Residual connections
    """
    
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        use_rope=True,
        grid_size=14,
        grid_depth=8,
        norm_layer=nn.LayerNorm,
        use_silu=False,
        block_idx=0,
        verbose=False,
    ):
        super().__init__()
        self.block_idx = block_idx
        self.verbose = verbose
        if self.verbose:
            logger.info(f"[TransformerBlock-{block_idx}] Initializing with dim={dim}, heads={num_heads}, mlp_ratio={mlp_ratio}")
        
        self.norm1 = norm_layer(dim)
        self.attn = RoPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_rope=use_rope,
            grid_size=grid_size,
            grid_depth=grid_depth,
        )
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
            use_silu=use_silu,
        )
        
        # Stochastic depth (drop path)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        if drop_path > 0.0:
            if self.verbose:
                logger.info(f"[TransformerBlock-{block_idx}] Using DropPath with rate={drop_path}")
    
    def forward(self, x, attn_mask=None, rope_positions=None):
        # Attention block with residual
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask, rope_positions))
        
        # MLP block with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class VisionTransformerPredictorAC(nn.Module):
    """Action Conditioned Vision Transformer Predictor"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=1408,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=True,
        use_silu=False,
        wide_silu=True,
        is_frame_causal=True,
        use_activation_checkpointing=False,
        use_rope=True,
        action_embed_dim=7,
        use_extrinsics=False,
        use_sdpa=True,
        verbose = False,
        **kwargs
    ):
        super().__init__()
        
        self.verbose = verbose
        if self.verbose:
            logger.info("="*80)
            logger.info("[VisionTransformerPredictorAC] Initializing predictor...")
            logger.info("="*80)
        
        self.is_frame_causal = is_frame_causal
        self.use_rope = use_rope
        self.predictor_embed_dim = predictor_embed_dim
        
        # Calculate grid dimensions
        if isinstance(img_size, (list, tuple)):
            self.grid_height = img_size[0] // patch_size
            self.grid_width = img_size[1] // patch_size
        else:
            self.grid_height = img_size // patch_size
            self.grid_width = img_size // patch_size
        
        self.grid_depth = num_frames // tubelet_size
        self.tokens_per_frame = self.grid_height * self.grid_width
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        
        if self.verbose:
                
            logger.info(f"[Config] Image size: {img_size}, Patch size: {patch_size}")
            logger.info(f"[Config] Num frames: {num_frames}, Tubelet size: {tubelet_size}")
            logger.info(f"[Config] Grid dimensions: {self.grid_height}x{self.grid_width}x{self.grid_depth}")
            logger.info(f"[Config] Tokens per frame: {self.tokens_per_frame}")
            logger.info(f"[Config] Frame causal: {is_frame_causal}, RoPE: {use_rope}")
            
            # Input projections
            logger.info("-"*80)
            logger.info("[Input Projections] Setting up embedding layers...")
            logger.info(f"  Encoder output dim: {embed_dim}")
            logger.info(f"  Predictor hidden dim: {predictor_embed_dim}")
            logger.info(f"  Action/State dim: {action_embed_dim}")
            
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        if self.verbose:
            logger.info(f"  Predictor embed: {embed_dim} -> {predictor_embed_dim}")
        
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        if self.verbose:
            logger.info(f"  Action encoder: {action_embed_dim} -> {predictor_embed_dim}")
        
        self.state_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        if self.verbose:
            logger.info(f"  State encoder: {action_embed_dim} -> {predictor_embed_dim}")
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if self.verbose:
            logger.info(f"[Stochastic Depth] Drop path rate range: 0.0 -> {drop_path_rate}")
        
        # Transformer blocks
        if self.verbose:
            logger.info("-"*80)
            logger.info(f"[Transformer Blocks] Building {depth} transformer blocks...")
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                use_rope=use_rope,
                grid_size=self.grid_height,
                grid_depth=self.grid_depth,
                norm_layer=norm_layer,
                use_silu=use_silu,
                block_idx=i,
                verbose = self.verbose,
            )
            for i in range(depth)
        ])
        if self.verbose:
            logger.info(f"  Created {depth} transformer blocks")
        
        # Output projection
        if self.verbose:
            logger.info("-"*80)
            logger.info("[Output Projection] Setting up output layers...")
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        if self.verbose:
            logger.info(f"  Output projection: {predictor_embed_dim} -> {embed_dim}")
        
        self.norm = norm_layer(predictor_embed_dim)
        if self.verbose:
            logger.info(f"  Final LayerNorm with dim={predictor_embed_dim}")
        
        # Initialize weights
        if self.verbose:
            logger.info("-"*80)
            logger.info(f"[Weight Init] Applying weight initialization (std={init_std})...")
        self.init_std = init_std
        self.apply(self._init_weights)
        if self.verbose:
            logger.info("  Weight initialization complete")
        
        # Summary
        if self.verbose:
            logger.info("="*80)
            logger.info("[Summary] Predictor architecture:")
            logger.info(f"  - Depth: {depth} blocks")
            logger.info(f"  - Num heads: {num_heads}")
            logger.info(f"  - Hidden dim: {predictor_embed_dim}")
            logger.info(f"  - MLP ratio: {mlp_ratio}")
            logger.info(f"  - Grid: {self.grid_height}x{self.grid_width}x{self.grid_depth}")
            logger.info(f"  - Tokens/frame: {self.tokens_per_frame}")
            logger.info("="*80)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_block_causal_mask(self, B, T, tokens_per_frame, device):
        """
        Block-causal mask: tokens at time t can attend to all tokens at t and before.
        Each timestep has: 2 tokens (action + state) + tokens_per_frame (patches)
        """
        tokens_per_timestep = 2 + tokens_per_frame
        total_tokens = T * tokens_per_timestep
        if self.verbose:
            logger.info(f"[Causal Mask] Creating block-causal mask...")
            logger.info(f"  Tokens per timestep: {tokens_per_timestep} (2 control + {tokens_per_frame} patches)")
            logger.info(f"  Total timesteps: {T}")
            logger.info(f"  Total tokens: {total_tokens}")
            
        # Create causal mask
        mask = torch.triu(torch.ones(total_tokens, total_tokens, device=device), diagonal=1)
        
        # Allow attention within blocks
        for t in range(T):
            start = t * tokens_per_timestep
            end = (t + 1) * tokens_per_timestep
            mask[start:end, start:end] = 0
        
        # Convert to attention mask format
        attn_mask = mask.masked_fill(mask.bool(), float('-inf'))
        if self.verbose:
            logger.info(f"  Causal mask created: {attn_mask.shape}")
            
        return attn_mask

    def create_rope_positions(self, T, device):
        """
        Create RoPE position indices for all tokens.
        - Patches get 3D positions (h, w, t)
        - Actions/states get temporal-only positions (0, 0, t)
        """
        tokens_per_timestep = 2 + self.tokens_per_frame
        total_tokens = T * tokens_per_timestep
        
        if self.verbose:
            logger.info(f"[RoPE Positions] Creating position indices...")
        
        pos_h = torch.zeros(total_tokens, device=device, dtype=torch.long)
        pos_w = torch.zeros(total_tokens, device=device, dtype=torch.long)
        pos_t = torch.zeros(total_tokens, device=device, dtype=torch.long)
        
        # Generate spatial positions for patches
        h_coords = torch.arange(self.grid_height, device=device)
        w_coords = torch.arange(self.grid_width, device=device)
        
        for t in range(T):
            start_idx = t * tokens_per_timestep
            
            # Action and state tokens (positions 0, 0, t)
            pos_t[start_idx:start_idx+2] = t
            
            # Patch tokens (positions h, w, t)
            patch_start = start_idx + 2
            for i, (h, w) in enumerate([(h, w) for h in h_coords for w in w_coords]):
                pos_h[patch_start + i] = h
                pos_w[patch_start + i] = w
                pos_t[patch_start + i] = t
        
        if self.verbose:
            logger.info(f"  Position indices created: h/w/t each with shape {pos_h.shape}")
            logger.info(f"  Action/State tokens: temporal only (h=0, w=0)")
            logger.info(f"  Patch tokens: 3D positions (h={self.grid_height}, w={self.grid_width}, t={T})")
        
        return {'h': pos_h, 'w': pos_w, 't': pos_t}

    def forward(self, z, actions, states):
        """
        Args:
            z: Encoder features [B, T*H*W, D_enc]
            actions: [B, T-1, 7]
            states: [B, T, 7]
        Returns:
            predictions: [B, T*H*W, D_enc]
        """
        B, seq_len, D_enc = z.shape
        T = states.shape[1]

        self.verbose = True  # Enable verbose logging for this forward pass
        
        if self.verbose:
            logger.info("="*80)
            logger.info("[Forward Pass] Starting predictor forward pass...")
            logger.info(f"  Input shapes - z: {z.shape}, actions: {actions.shape}, states: {states.shape}")
            logger.info(f"  Batch size: {B}, Timesteps: {T}")
        
        # Step 1: Project to predictor dimension
        if self.verbose:
            logger.info("[Step 1] Projecting inputs to predictor dimension...")
        x = self.predictor_embed(z)  # [B, T*H*W, D_pred]
        if self.verbose:
            logger.info(f"  Patches projected: {z.shape} -> {x.shape}")
        
        a = self.action_encoder(actions)  # [B, T-1, D_pred]
        if self.verbose:
            logger.info(f"  Actions encoded: {actions.shape} -> {a.shape}")
        
        s = self.state_encoder(states)  # [B, T, D_pred]
        if self.verbose:
            logger.info(f"  States encoded: {states.shape} -> {s.shape}")
        
        # Step 2: Reshape patches into frames
        if self.verbose:
            logger.info("[Step 2] Reshaping patches into frames...")
        x = x.view(B, T, self.tokens_per_frame, self.predictor_embed_dim)
        if self.verbose:
            logger.info(f"  Patches reshaped: {x.shape} (B, T, H*W, D)")
        
        # Step 3: Pack tokens
        if self.verbose:
            logger.info("[Step 3] Packing tokens (action, state, patches) per frame...")
        tokens_list = []
        
        # First frame: dummy action + state + patches
        dummy_action = torch.zeros_like(s[:, 0:1])
        frame_tokens = torch.cat([dummy_action, s[:, 0:1], x[:, 0]], dim=1)
        tokens_list.append(frame_tokens)
        if self.verbose:
            logger.info(f"  Frame 0: dummy_action + state + patches = {frame_tokens.shape}")
        
        # Subsequent frames: action + state + patches
        for t in range(1, T):
            frame_tokens = torch.cat([a[:, t-1:t], s[:, t:t+1], x[:, t]], dim=1)
            tokens_list.append(frame_tokens)
        
        x = torch.cat(tokens_list, dim=1)
        if self.verbose:
            logger.info(f"  All frames packed: {x.shape} (B, T*(2+H*W), D)")
        
        # Step 4: Create masks and positions
        if self.verbose:
            logger.info("[Step 4] Creating attention mask and RoPE positions...")
        attn_mask = self.create_block_causal_mask(B, T, self.tokens_per_frame, z.device) if self.is_frame_causal else None
        if not self.is_frame_causal:
            logger.info("  No causal mask (frame_causal=False)")
        
        rope_positions = self.create_rope_positions(T, z.device) if self.use_rope else None
        if not self.use_rope:
            logger.info("  No RoPE positions (use_rope=False)")
        
        # Step 5: Pass through transformer blocks
        if self.verbose:
            logger.info(f"[Step 5] Passing through {len(self.blocks)} transformer blocks...")
        for i, blk in enumerate(self.blocks):
            if i % 6 == 0:  # Log every 6 blocks
                if self.verbose:
                    logger.info(f"  Processing block {i}/{len(self.blocks)}...")
            x = blk(x, attn_mask=attn_mask, rope_positions=rope_positions)
        if self.verbose:
            logger.info(f"  All blocks processed, output shape: {x.shape}")
        
        # Step 6: Final norm and projection
        if self.verbose:
            logger.info("[Step 6] Applying final normalization and projection...")
        x = self.norm(x)
        if self.verbose:
            logger.info(f"  Normalized: {x.shape}")
        
        x = self.predictor_proj(x)
        if self.verbose:
            logger.info(f"  Projected back to encoder dim: {x.shape}")
        
        # Step 7: Extract patch predictions
        if self.verbose:
            logger.info("[Step 7] Extracting patch predictions (removing control tokens)...")
        predictions = []
        for t in range(T):
            start_idx = t * (2 + self.tokens_per_frame) + 2
            end_idx = start_idx + self.tokens_per_frame
            predictions.append(x[:, start_idx:end_idx])
        
        predictions = torch.cat(predictions, dim=1)
        if self.verbose:
            logger.info(f"  Final predictions: {predictions.shape} (B, T*H*W, D_enc)")
        
        if self.verbose:
            logger.info("="*80)
            logger.info("[Forward Pass] Complete!")
            logger.info("="*80)
        
        return predictions


def my_predictor(*args, **kwargs):
    logger.info("\n" + "="*80)
    logger.info("[my_predictor] Creating VisionTransformerPredictorAC...")
    logger.info("="*80)
    
    model = VisionTransformerPredictorAC(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("="*80)
    logger.info("[my_predictor] Model created successfully!")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: ~{total_params * 4 / 1024**2:.2f} MB (fp32)")
    logger.info("="*80 + "\n")
    
    return model