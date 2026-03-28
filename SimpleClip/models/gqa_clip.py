import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        weight = self.weight.float()

        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms * weight
        x = x.to(dtype)

        return x


class SwiGLUFFN(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0, align_to=64):
        super(SwiGLUFFN, self).__init__()
        d = int(hidden_dim * 2 / 3)
        hidden_dim = ((d + align_to - 1) // align_to) * align_to

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        x = self.dropout(x)

        return x


class LayerScale(nn.Module):

    def __init__(self, dim, init_values=1e-5):
        super(LayerScale, self).__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        x = x * self.gamma

        return x


class RopePositionEmbedding1D(nn.Module):

    def __init__(self, dim, num_heads, max_seq_length=2048, base=10000.0):
        super(RopePositionEmbedding1D, self).__init__()
        head_dim = dim // num_heads

        inv_freq = 1.0 / (base**(
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        t = torch.arange(max_seq_length, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        freqs = freqs.repeat_interleave(2, dim=1)
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)

    def forward(self, seq_length):
        return self.sin_cached[:seq_length], self.cos_cached[:seq_length]


class RopePositionEmbedding2D(nn.Module):

    def __init__(self, dim, num_heads, base=10000.0):
        super(RopePositionEmbedding2D, self).__init__()

        head_dim = dim // num_heads
        assert head_dim % 2 == 0

        half_head_dim = head_dim // 2

        freqs_h = 1.0 / (base**(torch.arange(
            0, half_head_dim, 2, dtype=torch.float32) / half_head_dim))
        freqs_w = 1.0 / (base**(torch.arange(
            0, half_head_dim, 2, dtype=torch.float32) / half_head_dim))

        self.register_buffer('freqs_h', freqs_h, persistent=False)
        self.register_buffer('freqs_w', freqs_w, persistent=False)

    def forward(self, H, W):
        device = self.freqs_h.device

        coords_h = torch.arange(H, dtype=torch.float32, device=device)
        coords_w = torch.arange(W, dtype=torch.float32, device=device)

        grid_h, grid_w = torch.meshgrid(coords_h, coords_w, indexing="ij")

        grid_h = (grid_h + 0.5) / H
        grid_w = (grid_w + 0.5) / W

        coords_h = grid_h.flatten()
        coords_w = grid_w.flatten()

        angles_h = coords_h[:, None] * self.freqs_h[None, :]  # [HW, D//4]
        angles_w = coords_w[:, None] * self.freqs_w[None, :]  # [HW, D//4]

        angles_h = angles_h.repeat_interleave(2, dim=1)  # [HW, head_dim//2]
        angles_w = angles_w.repeat_interleave(2, dim=1)  # [HW, head_dim//2]
        angles = torch.cat([angles_h, angles_w], dim=1)  # [HW, head_dim]

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return (sin, cos)


def apply_rope(x, sin, cos):
    """
    Standard pair-wise RoPE rotation for 1D position embeddings/2D position embeddings.
    
    Args:
        x: [..., D] - Input tensor
        sin: [..., D] - Sine values in repeated format [s0, s0, s1, s1, ...]
        cos: [..., D] - Cosine values in repeated format [c0, c0, c1, c1, ...]
    
    Returns:
        Rotated tensor with same shape as input
    """
    # Reshape to pairs: [..., D//2, 2]
    x_pairs = x.reshape(*x.shape[:-1], -1, 2)

    # sin/cos are already repeated, reshape and take first of each pair
    sin = sin.reshape(*sin.shape[:-1], -1, 2)[..., 0]  # [..., D//2]
    cos = cos.reshape(*cos.shape[:-1], -1, 2)[..., 0]  # [..., D//2]

    # Extract pair elements
    x1 = x_pairs[..., 0]  # [..., D//2]
    x2 = x_pairs[..., 1]  # [..., D//2]

    # Standard RoPE rotation:
    # [cos  -sin] [x1]
    # [sin   cos] [x2]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # Stack and flatten back to original shape
    rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)

    return rotated.flatten(-2)


class GroupedQueryAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 num_kv_heads,
                 use_rope_2d=False,
                 dropout=0.0):
        super(GroupedQueryAttention, self).__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = self.head_dim**-0.5
        self.use_rope_2d = use_rope_2d
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope=None, attn_mask=None):
        B, N, C = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads,
                                self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads,
                                self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_kv_heads,
                                self.head_dim).transpose(1, 2)

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 应用RoPE
        if rope is not None:
            sin, cos = rope
            if self.use_rope_2d:
                assert N - 1 == sin.shape[
                    0], f"Patch count {N-1} != RoPE length {sin.shape[0]}"
                sin = sin.unsqueeze(0).unsqueeze(0)
                cos = cos.unsqueeze(0).unsqueeze(0)

                # 分离cls_token和patch_tokens
                q_cls, q_patches = q[:, :, :1, :], q[:, :, 1:, :]
                k_cls, k_patches = k[:, :, :1, :], k[:, :, 1:, :]

                # 只对patch_tokens应用RoPE
                q_patches = apply_rope(q_patches, sin, cos)
                k_patches = apply_rope(k_patches, sin, cos)

                # 合并回去
                q = torch.cat([q_cls, q_patches], dim=2)
                k = torch.cat([k_cls, k_patches], dim=2)
            else:
                # 1D RoPE for text
                # [1, 1, N, D]
                assert N <= sin.shape[
                    0], f"Sequence length {N} exceeds RoPE max length {sin.shape[0]}"
                sin = sin[:N].unsqueeze(0).unsqueeze(0)
                cos = cos[:N].unsqueeze(0).unsqueeze(0)
                q = apply_rope(q, sin, cos)
                k = apply_rope(k, sin, cos)

        # GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Flash Attention
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale)

        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.o_proj(out)

        return out


class TransformerBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 num_kv_heads,
                 mlp_ratio=4.0,
                 use_rope_2d=False,
                 dropout=0.0,
                 layer_scale_init_values=1e-5):
        super(TransformerBlock, self).__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim=dim,
                                          num_heads=num_heads,
                                          num_kv_heads=num_kv_heads,
                                          use_rope_2d=use_rope_2d,
                                          dropout=dropout)

        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLUFFN(dim, int(dim * mlp_ratio), dropout=dropout)

        self.ls1 = LayerScale(
            dim=dim, init_values=layer_scale_init_values
        ) if layer_scale_init_values is not None else nn.Identity()
        self.ls2 = LayerScale(
            dim=dim, init_values=layer_scale_init_values
        ) if layer_scale_init_values is not None else nn.Identity()

    def forward(self, x, rope=None, attn_mask=None):
        x = x + self.ls1(
            self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
        x = x + self.ls2(self.mlp(self.norm2(x)))

        return x


class VisionEncoder(nn.Module):

    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 num_kv_heads=4,
                 mlp_ratio=4.0,
                 output_dim=512,
                 dropout=0.0,
                 layer_scale_init_values=1e-5,
                 rope_base=10000.0,
                 use_gradient_checkpoint=False):
        super(VisionEncoder, self).__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.patch_embed = nn.Conv2d(in_channels,
                                     embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.rope = RopePositionEmbedding2D(dim=embed_dim,
                                            num_heads=num_heads,
                                            base=rope_base)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim,
                             num_heads=num_heads,
                             num_kv_heads=num_kv_heads,
                             mlp_ratio=mlp_ratio,
                             use_rope_2d=True,
                             dropout=dropout,
                             layer_scale_init_values=layer_scale_init_values)
            for _ in range(depth)
        ])

        self.norm = RMSNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, output_dim, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        patches = self.patch_embed(x)
        B, C, H, W = patches.shape
        x = patches.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        rope_sincos = self.rope(H, W)

        for _, block in enumerate(self.blocks):
            if self.use_gradient_checkpoint:
                x = checkpoint(block,
                               x,
                               rope_sincos,
                               None,
                               use_reentrant=False)
            else:
                x = block(x, rope_sincos)

        x = self.norm(x[:, 0])
        x = self.proj(x)

        return x


class TextEncoder(nn.Module):

    def __init__(self,
                 vocab_size=49408,
                 context_length=77,
                 embed_dim=512,
                 depth=12,
                 num_heads=8,
                 num_kv_heads=4,
                 mlp_ratio=4.0,
                 output_dim=512,
                 dropout=0.0,
                 layer_scale_init_values=1e-5,
                 pool_type='argmax',
                 pad_id=0,
                 no_causal_mask=False,
                 use_gradient_checkpoint=False):
        super(TextEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.pool_type = pool_type
        self.pad_id = pad_id
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        self.rope = RopePositionEmbedding1D(dim=embed_dim,
                                            num_heads=num_heads,
                                            max_seq_length=context_length)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim,
                             num_heads=num_heads,
                             num_kv_heads=num_kv_heads,
                             mlp_ratio=mlp_ratio,
                             use_rope_2d=False,
                             dropout=dropout,
                             layer_scale_init_values=layer_scale_init_values)
            for _ in range(depth)
        ])

        self.norm = RMSNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, output_dim, bias=False)

        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer('attn_mask',
                                 self.build_causal_mask(),
                                 persistent=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.token_embed.weight, std=0.02)

    def build_causal_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)

        return mask

    def text_global_pool(self, x, text=None, pool_type='argmax'):
        if pool_type == 'first':
            pooled, tokens = x[:, 0], x[:, 1:]
        elif pool_type == 'last':
            pooled, tokens = x[:, -1], x[:, :-1]
        elif pool_type == 'argmax':
            assert text is not None
            pooled, tokens = x[torch.arange(x.shape[0]),
                               text.argmax(dim=-1)], x
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, text):
        B, L = text.shape

        x = self.token_embed(text)

        rope_sincos = self.rope(L)

        attn_mask = self.attn_mask
        if attn_mask is not None and L < self.context_length:
            attn_mask = attn_mask[:L, :L]

        for _, block in enumerate(self.blocks):
            if self.use_gradient_checkpoint:
                x = checkpoint(block,
                               x,
                               rope_sincos,
                               attn_mask,
                               use_reentrant=False)
            else:
                x = block(x, rope_sincos, attn_mask)

        x = self.norm(x)

        x, _ = self.text_global_pool(x, text, pool_type=self.pool_type)
        x = self.proj(x)

        return x


class GQACLIP(nn.Module):

    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 vision_embed_dim=768,
                 vision_depth=12,
                 vision_num_heads=12,
                 vision_num_kv_heads=4,
                 vision_mlp_ratio=4.0,
                 vision_layer_scale_init_values=1e-5,
                 vocab_size=49408,
                 context_length=77,
                 text_embed_dim=512,
                 text_depth=12,
                 text_num_heads=8,
                 text_num_kv_heads=4,
                 text_mlp_ratio=4.0,
                 text_layer_scale_init_values=1e-5,
                 text_pad_id=0,
                 text_no_causal_mask=False,
                 text_pool_type='argmax',
                 output_dim=512,
                 dropout=0.0,
                 rope_base=10000.0,
                 init_logit_scale=np.log(1 / 0.07),
                 init_logit_bias=None,
                 use_gradient_checkpoint=False):
        super(GQACLIP, self).__init__()

        self.visual = VisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=vision_embed_dim,
            depth=vision_depth,
            num_heads=vision_num_heads,
            num_kv_heads=vision_num_kv_heads,
            mlp_ratio=vision_mlp_ratio,
            output_dim=output_dim,
            dropout=dropout,
            layer_scale_init_values=vision_layer_scale_init_values,
            rope_base=rope_base,
            use_gradient_checkpoint=use_gradient_checkpoint)

        self.text = TextEncoder(
            vocab_size=vocab_size,
            context_length=context_length,
            embed_dim=text_embed_dim,
            depth=text_depth,
            num_heads=text_num_heads,
            num_kv_heads=text_num_kv_heads,
            mlp_ratio=text_mlp_ratio,
            output_dim=output_dim,
            dropout=dropout,
            layer_scale_init_values=text_layer_scale_init_values,
            pool_type=text_pool_type,
            pad_id=text_pad_id,
            no_causal_mask=text_no_causal_mask,
            use_gradient_checkpoint=use_gradient_checkpoint)

        self.logit_scale = nn.Parameter(
            torch.tensor(init_logit_scale, dtype=torch.float32))

        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(
                torch.tensor(init_logit_bias, dtype=torch.float32))
        else:
            self.logit_bias = None

        self.context_length = context_length
        self.vocab_size = vocab_size

    def encode_image(self, image):
        image_features = self.visual(image)

        return image_features

    def encode_text(self, text):
        text_features = self.text(text)

        return text_features

    def forward(self, images, tokens):
        image_features = self.encode_image(images)
        text_features = self.encode_text(tokens)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        outputs = {
            'image_features': image_features,
            'text_features': text_features,
            'logit_scale': self.logit_scale.exp()
        }

        if self.logit_bias is not None:
            outputs['logit_bias'] = self.logit_bias

        return outputs


####################################################################################
# CLIP Models
####################################################################################


def vit_base_patch16_gqa_clip(image_size=224,
                              patch_size=16,
                              context_length=77,
                              vocab_size=49408,
                              **kwargs):
    return GQACLIP(image_size=image_size,
                   patch_size=patch_size,
                   vision_embed_dim=768,
                   vision_depth=12,
                   vision_num_heads=12,
                   vision_num_kv_heads=4,
                   vision_mlp_ratio=4.0,
                   vision_layer_scale_init_values=1e-5,
                   vocab_size=vocab_size,
                   context_length=context_length,
                   text_embed_dim=512,
                   text_depth=12,
                   text_num_heads=8,
                   text_num_kv_heads=4,
                   text_mlp_ratio=4.0,
                   text_layer_scale_init_values=1e-5,
                   output_dim=512,
                   **kwargs)


def vit_large_patch16_gqa_clip(image_size=224,
                               patch_size=16,
                               context_length=77,
                               vocab_size=49408,
                               **kwargs):
    return GQACLIP(image_size=image_size,
                   patch_size=patch_size,
                   vision_embed_dim=1024,
                   vision_depth=24,
                   vision_num_heads=16,
                   vision_num_kv_heads=4,
                   vision_mlp_ratio=4.0,
                   vision_layer_scale_init_values=1e-5,
                   vocab_size=vocab_size,
                   context_length=context_length,
                   text_embed_dim=768,
                   text_depth=12,
                   text_num_heads=12,
                   text_num_kv_heads=4,
                   text_mlp_ratio=4.0,
                   text_layer_scale_init_values=1e-5,
                   output_dim=768,
                   **kwargs)


def vit_huge_patch16_gqa_clip(image_size=224,
                              patch_size=16,
                              context_length=77,
                              vocab_size=49408,
                              **kwargs):
    return GQACLIP(image_size=image_size,
                   patch_size=patch_size,
                   vision_embed_dim=1280,
                   vision_depth=32,
                   vision_num_heads=16,
                   vision_num_kv_heads=4,
                   vision_mlp_ratio=4.0,
                   vision_layer_scale_init_values=1e-5,
                   vocab_size=vocab_size,
                   context_length=context_length,
                   text_embed_dim=1024,
                   text_depth=16,
                   text_num_heads=16,
                   text_num_kv_heads=4,
                   text_mlp_ratio=4.0,
                   text_layer_scale_init_values=1e-5,
                   output_dim=1024,
                   **kwargs)


def vit_1B_patch16_gqa_clip(image_size=224,
                            patch_size=16,
                            context_length=77,
                            vocab_size=49408,
                            **kwargs):
    return GQACLIP(image_size=image_size,
                   patch_size=patch_size,
                   vision_embed_dim=1536,
                   vision_depth=36,
                   vision_num_heads=24,
                   vision_num_kv_heads=6,
                   vision_mlp_ratio=4.0,
                   vision_layer_scale_init_values=1e-5,
                   vocab_size=vocab_size,
                   context_length=context_length,
                   text_embed_dim=1280,
                   text_depth=20,
                   text_num_heads=16,
                   text_num_kv_heads=4,
                   text_mlp_ratio=4.0,
                   text_layer_scale_init_values=1e-5,
                   output_dim=1280,
                   **kwargs)


####################################################################################
# SigLIP Models
####################################################################################


def vit_base_patch16_gqa_siglip(image_size=224,
                                patch_size=16,
                                context_length=64,
                                vocab_size=256000,
                                **kwargs):
    return GQACLIP(image_size=image_size,
                   patch_size=patch_size,
                   vision_embed_dim=768,
                   vision_depth=12,
                   vision_num_heads=12,
                   vision_num_kv_heads=4,
                   vision_mlp_ratio=4.0,
                   vision_layer_scale_init_values=1e-5,
                   vocab_size=vocab_size,
                   context_length=context_length,
                   text_embed_dim=512,
                   text_depth=12,
                   text_num_heads=8,
                   text_num_kv_heads=4,
                   text_mlp_ratio=4.0,
                   text_layer_scale_init_values=1e-5,
                   text_no_causal_mask=True,
                   text_pool_type='last',
                   output_dim=512,
                   init_logit_scale=np.log(10),
                   init_logit_bias=-10,
                   **kwargs)


def vit_large_patch16_gqa_siglip(image_size=224,
                                 patch_size=16,
                                 context_length=64,
                                 vocab_size=256000,
                                 **kwargs):
    return GQACLIP(image_size=image_size,
                   patch_size=patch_size,
                   vision_embed_dim=1024,
                   vision_depth=24,
                   vision_num_heads=16,
                   vision_num_kv_heads=4,
                   vision_mlp_ratio=4.0,
                   vision_layer_scale_init_values=1e-5,
                   vocab_size=vocab_size,
                   context_length=context_length,
                   text_embed_dim=768,
                   text_depth=12,
                   text_num_heads=12,
                   text_num_kv_heads=4,
                   text_mlp_ratio=4.0,
                   text_layer_scale_init_values=1e-5,
                   text_no_causal_mask=True,
                   text_pool_type='last',
                   output_dim=768,
                   init_logit_scale=np.log(10),
                   init_logit_bias=-10,
                   **kwargs)


def vit_huge_patch16_gqa_siglip(image_size=224,
                                patch_size=16,
                                context_length=64,
                                vocab_size=256000,
                                **kwargs):
    return GQACLIP(image_size=image_size,
                   patch_size=patch_size,
                   vision_embed_dim=1280,
                   vision_depth=32,
                   vision_num_heads=16,
                   vision_num_kv_heads=4,
                   vision_mlp_ratio=4.0,
                   vision_layer_scale_init_values=1e-5,
                   vocab_size=vocab_size,
                   context_length=context_length,
                   text_embed_dim=1024,
                   text_depth=16,
                   text_num_heads=16,
                   text_num_kv_heads=4,
                   text_mlp_ratio=4.0,
                   text_layer_scale_init_values=1e-5,
                   text_no_causal_mask=True,
                   text_pool_type='last',
                   output_dim=1024,
                   init_logit_scale=np.log(10),
                   init_logit_bias=-10,
                   **kwargs)


def vit_1B_patch16_gqa_siglip(image_size=224,
                              patch_size=16,
                              context_length=64,
                              vocab_size=256000,
                              **kwargs):
    return GQACLIP(image_size=image_size,
                   patch_size=patch_size,
                   vision_embed_dim=1536,
                   vision_depth=36,
                   vision_num_heads=24,
                   vision_num_kv_heads=6,
                   vision_mlp_ratio=4.0,
                   vision_layer_scale_init_values=1e-5,
                   vocab_size=vocab_size,
                   context_length=context_length,
                   text_embed_dim=1280,
                   text_depth=20,
                   text_num_heads=16,
                   text_num_kv_heads=4,
                   text_mlp_ratio=4.0,
                   text_layer_scale_init_values=1e-5,
                   text_no_causal_mask=True,
                   text_pool_type='last',
                   output_dim=1280,
                   init_logit_scale=np.log(10),
                   init_logit_bias=-10,
                   **kwargs)


if __name__ == '__main__':
    import os
    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)

    from tokenizer import SimpleTokenizer, SigLipTokenizer

    net = vit_base_patch16_gqa_clip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SimpleTokenizer(context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'images': images,
                                              'tokens': tokens,
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'2222, flops: {flops}, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'])

    net = vit_base_patch16_gqa_clip(use_gradient_checkpoint=True)
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SimpleTokenizer(context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'images': images,
                                              'tokens': tokens,
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'2222, flops: {flops}, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'])

    net = vit_large_patch16_gqa_clip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SimpleTokenizer(context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'images': images,
                                              'tokens': tokens,
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'2222, flops: {flops}, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'])

    net = vit_huge_patch16_gqa_clip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SimpleTokenizer(context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'images': images,
                                              'tokens': tokens,
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'2222, flops: {flops}, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'])

    net = vit_1B_patch16_gqa_clip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SimpleTokenizer(context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'images': images,
                                              'tokens': tokens,
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'2222, flops: {flops}, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'])

    ####################################################################################
    net = vit_base_patch16_gqa_siglip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SigLipTokenizer('gemma', context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'images': images,
                                              'tokens': tokens,
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'2222, flops: {flops}, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'],
          outputs['logit_bias'])

    net = vit_large_patch16_gqa_siglip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SigLipTokenizer('gemma', context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'images': images,
                                              'tokens': tokens,
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'2222, flops: {flops}, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'],
          outputs['logit_bias'])

    net = vit_huge_patch16_gqa_siglip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SigLipTokenizer('gemma', context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'images': images,
                                              'tokens': tokens,
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'2222, flops: {flops}, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'],
          outputs['logit_bias'])

    net = vit_1B_patch16_gqa_siglip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SigLipTokenizer('gemma', context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'images': images,
                                              'tokens': tokens,
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'2222, flops: {flops}, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'],
          outputs['logit_bias'])
