import math
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
        # [max_seq_length, head_dim//2],每个旋转对只需一个角度
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)

    def forward(self, seq_length):
        return self.sin_cached[:seq_length], self.cos_cached[:seq_length]


class RopePositionEmbedding2D(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 min_period=0.5,
                 max_period=100.0,
                 jitter_coord_scale=2.0,
                 rescale_coord_scale=2.0):
        super(RopePositionEmbedding2D, self).__init__()

        head_dim = dim // num_heads
        # head_dim需要被4整除:一半维度给h一半给w,每半内部再两两配对做旋转
        assert head_dim % 4 == 0

        self.min_period = min_period
        self.max_period = max_period
        self.jitter_coord_scale = jitter_coord_scale
        self.rescale_coord_scale = rescale_coord_scale

        # 每个方向(h/w)的period数量 = head_dim//4
        num_periods = head_dim // 4
        if num_periods > 1:
            exponents = torch.arange(num_periods,
                                     dtype=torch.float32) / (num_periods - 1)
        else:
            exponents = torch.zeros(num_periods, dtype=torch.float32)
        periods = min_period * (max_period / min_period)**exponents

        self.register_buffer('periods', periods, persistent=False)

    def forward(self, H, W, batch_size):
        device = self.periods.device

        # 坐标归一化到[-1,1],与H/W无关
        coords_h = 2.0 * (torch.arange(H, dtype=torch.float32, device=device) +
                          0.5) / H - 1.0
        coords_w = 2.0 * (torch.arange(W, dtype=torch.float32, device=device) +
                          0.5) / W - 1.0

        grid_h, grid_w = torch.meshgrid(coords_h, coords_w, indexing="ij")

        # [H, W] -> [batch_size, H, W],每个样本独立采样坐标增强(与DINOv3一致)。
        # 若整个batch共享同一份增强,对比学习中所有正负样本被施加相同畸变,
        # 增强的有效样本数会从batch_size退化为1。
        grid_h = grid_h[None].expand(batch_size, -1, -1)
        grid_w = grid_w[None].expand(batch_size, -1, -1)

        if self.training:
            if self.jitter_coord_scale > 1.0:
                log_jitter = math.log(self.jitter_coord_scale)
                jitter = torch.empty(batch_size, 2, device=device).uniform_(
                    -log_jitter, log_jitter).exp()
                grid_h = grid_h * jitter[:, 0, None, None]
                grid_w = grid_w * jitter[:, 1, None, None]

            if self.rescale_coord_scale > 1.0:
                log_rescale = math.log(self.rescale_coord_scale)
                rescale = torch.empty(batch_size, 1, 1,
                                      device=device).uniform_(
                                          -log_rescale, log_rescale).exp()
                grid_h = grid_h * rescale
                grid_w = grid_w * rescale

        # [batch_size, HW]
        coords_h = grid_h.reshape(batch_size, -1)
        coords_w = grid_w.reshape(batch_size, -1)

        # [batch_size, HW, head_dim//4]
        angles_h = 2.0 * math.pi * coords_h[:, :, None] / self.periods[None,
                                                                       None, :]
        angles_w = 2.0 * math.pi * coords_w[:, :, None] / self.periods[None,
                                                                       None, :]

        # [batch_size, HW, head_dim//2],每个旋转对只需一个角度
        angles = torch.cat([angles_h, angles_w], dim=-1)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return (sin, cos)


def apply_rope(x, sin, cos):
    # Reshape to pairs: [..., D//2, 2]
    x_pairs = x.reshape(*x.shape[:-1], -1, 2)

    # sin/cos为紧凑形式,补一个尾轴广播到旋转对的2个分量上。
    # [..., D//2, 1]
    sin = sin.to(x.dtype).unsqueeze(-1)
    # [..., D//2, 1]
    cos = cos.to(x.dtype).unsqueeze(-1)

    # Extract pair elements
    # [..., D//2, 1]
    x1 = x_pairs[..., 0:1]
    # [..., D//2, 1]
    x2 = x_pairs[..., 1:2]

    # Standard RoPE rotation:
    # [cos  -sin] [x1]
    rotated_x1 = x1 * cos - x2 * sin
    # [sin   cos] [x2]
    rotated_x2 = x1 * sin + x2 * cos

    # Concat and flatten back to original shape
    rotated = torch.cat([rotated_x1, rotated_x2], dim=-1)
    rotated = rotated.flatten(-2)

    return rotated


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

    def forward(self, x, rope=None, attn_mask=None, is_causal=False):
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
                # 视觉塔无cls_token,全部token都是patch,直接整体应用2D RoPE
                # sin/cos为[B, N, head_dim//2],每个样本的坐标增强相互独立
                assert N == sin.shape[
                    1], f"Patch count {N} != RoPE length {sin.shape[1]}"
                # [B, N, D//2] -> [B, 1, N, D//2]
                sin = sin.unsqueeze(1)
                cos = cos.unsqueeze(1)
            else:
                # 1D RoPE for text
                # [1, 1, N, D//2]
                assert N <= sin.shape[
                    0], f"Sequence length {N} exceeds RoPE max length {sin.shape[0]}"
                sin = sin[:N].unsqueeze(0).unsqueeze(0)
                cos = cos[:N].unsqueeze(0).unsqueeze(0)
            q = apply_rope(q, sin, cos)
            k = apply_rope(k, sin, cos)

        # Flash Attention
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
            is_causal=is_causal,
            enable_gqa=True)

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

    def forward(self, x, rope=None, attn_mask=None, is_causal=False):
        x = x + self.ls1(
            self.attn(self.norm1(x),
                      rope=rope,
                      attn_mask=attn_mask,
                      is_causal=is_causal))
        x = x + self.ls2(self.mlp(self.norm2(x)))

        return x


class AttentionPool(nn.Module):

    def __init__(self, dim, num_heads, num_kv_heads):
        super(AttentionPool, self).__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.query = nn.Parameter(torch.zeros(dim))

        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(self.query).view(1, self.num_heads, 1,
                                         self.head_dim).expand(B, -1, -1, -1)
        k = self.k_proj(x).view(B, N, self.num_kv_heads,
                                self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_kv_heads,
                                self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        out = F.scaled_dot_product_attention(q,
                                             k,
                                             v,
                                             scale=self.scale,
                                             enable_gqa=True)

        out = self.o_proj(out.transpose(1, 2).reshape(B, C))

        return out


class VisionEncoder(nn.Module):

    def __init__(self,
                 image_size=256,
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
                 rope_min_period=0.5,
                 rope_max_period=100.0,
                 rope_jitter_coord_scale=2.0,
                 rope_rescale_coord_scale=2.0,
                 use_gradient_checkpoint=False):
        super(VisionEncoder, self).__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.patch_embed = nn.Conv2d(in_channels,
                                     embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)

        self.rope = RopePositionEmbedding2D(
            dim=embed_dim,
            num_heads=num_heads,
            min_period=rope_min_period,
            max_period=rope_max_period,
            jitter_coord_scale=rope_jitter_coord_scale,
            rescale_coord_scale=rope_rescale_coord_scale)

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
        self.attn_pool = AttentionPool(dim=embed_dim,
                                       num_heads=num_heads,
                                       num_kv_heads=num_kv_heads)
        self.proj = nn.Linear(embed_dim, output_dim, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.trunc_normal_(self.attn_pool.query, std=0.02)

    def forward(self, x):
        patches = self.patch_embed(x)
        B, C, H, W = patches.shape
        x = patches.flatten(2).transpose(1, 2)

        rope_sincos = self.rope(H, W, B)

        for _, block in enumerate(self.blocks):
            if self.use_gradient_checkpoint:
                x = checkpoint(block,
                               x,
                               rope_sincos,
                               None,
                               False,
                               use_reentrant=False)
            else:
                x = block(x, rope_sincos)

        x = self.attn_pool(self.norm(x))
        x = self.proj(x)

        return x


class TextEncoder(nn.Module):

    def __init__(self,
                 vocab_size=49408,
                 context_length=96,
                 embed_dim=512,
                 depth=12,
                 num_heads=8,
                 num_kv_heads=4,
                 mlp_ratio=4.0,
                 output_dim=512,
                 dropout=0.0,
                 layer_scale_init_values=1e-5,
                 pad_id=0,
                 no_causal_mask=False,
                 use_key_padding_mask=True,
                 use_gradient_checkpoint=False):
        super(TextEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.pad_id = pad_id
        self.no_causal_mask = no_causal_mask
        self.use_key_padding_mask = use_key_padding_mask
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

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.token_embed.weight, std=0.02)

    def build_key_padding_mask(self, text):
        keep = (text != self.pad_id)
        keep = keep.clone()
        keep[:, 0] = True

        return keep[:, None, None, :]

    def build_attn_mask(self, text):
        if self.no_causal_mask:
            if self.use_key_padding_mask:
                return self.build_key_padding_mask(text), False

            return None, False

        return None, True

    def text_global_pool(self, x, text):
        eos_indices = ((text != self.pad_id).sum(dim=-1) - 1).clamp(min=0)
        pooled, tokens = x[torch.arange(x.shape[0]), eos_indices], x

        return pooled, tokens

    def forward(self, text):
        B, L = text.shape

        x = self.token_embed(text)

        rope_sincos = self.rope(L)

        attn_mask, is_causal = self.build_attn_mask(text)

        for _, block in enumerate(self.blocks):
            if self.use_gradient_checkpoint:
                x = checkpoint(block,
                               x,
                               rope_sincos,
                               attn_mask,
                               is_causal,
                               use_reentrant=False)
            else:
                x = block(x, rope_sincos, attn_mask, is_causal)

        x = self.norm(x)

        x, _ = self.text_global_pool(x, text)
        x = self.proj(x)

        return x


class GQACLIP(nn.Module):

    def __init__(self,
                 image_size=256,
                 patch_size=16,
                 vision_embed_dim=768,
                 vision_depth=12,
                 vision_num_heads=12,
                 vision_num_kv_heads=4,
                 vision_mlp_ratio=4.0,
                 vision_layer_scale_init_values=1e-5,
                 vocab_size=248070,
                 context_length=96,
                 text_embed_dim=512,
                 text_depth=12,
                 text_num_heads=8,
                 text_num_kv_heads=4,
                 text_mlp_ratio=4.0,
                 text_layer_scale_init_values=1e-5,
                 text_pad_id=248055,
                 text_no_causal_mask=False,
                 text_use_key_padding_mask=True,
                 output_dim=512,
                 dropout=0.0,
                 vision_rope_min_period=0.5,
                 vision_rope_max_period=100.0,
                 vision_rope_jitter_coord_scale=2.0,
                 vision_rope_rescale_coord_scale=2.0,
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
            rope_min_period=vision_rope_min_period,
            rope_max_period=vision_rope_max_period,
            rope_jitter_coord_scale=vision_rope_jitter_coord_scale,
            rope_rescale_coord_scale=vision_rope_rescale_coord_scale,
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
            pad_id=text_pad_id,
            no_causal_mask=text_no_causal_mask,
            use_key_padding_mask=text_use_key_padding_mask,
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

        return F.normalize(image_features, dim=-1)

    def encode_text(self, text):
        text_features = self.text(text)

        return F.normalize(text_features, dim=-1)

    def forward(self, images, tokens):
        image_features = self.encode_image(images)
        text_features = self.encode_text(tokens)

        outputs = {
            'image_features': image_features,
            'text_features': text_features,
            'logit_scale': self.logit_scale.exp()
        }

        if self.logit_bias is not None:
            # for deepspeed=0.19.3 zero stage3, self.logit_bias must +0.
            outputs['logit_bias'] = self.logit_bias + 0.

        return outputs


####################################################################################
# CLIP Models
####################################################################################


def vit_base_patch16_gqa_clip(image_size=256,
                              patch_size=16,
                              vision_embed_dim=768,
                              vision_depth=12,
                              vision_num_heads=12,
                              vision_num_kv_heads=6,
                              vision_mlp_ratio=4.5,
                              vision_layer_scale_init_values=1e-5,
                              vocab_size=248070,
                              context_length=96,
                              text_embed_dim=512,
                              text_depth=12,
                              text_num_heads=8,
                              text_num_kv_heads=4,
                              text_mlp_ratio=4.5,
                              text_layer_scale_init_values=1e-5,
                              text_pad_id=248055,
                              text_no_causal_mask=False,
                              text_use_key_padding_mask=True,
                              output_dim=512,
                              init_logit_scale=np.log(1 / 0.07),
                              init_logit_bias=None,
                              **kwargs):
    return GQACLIP(
        image_size=image_size,
        patch_size=patch_size,
        vision_embed_dim=vision_embed_dim,
        vision_depth=vision_depth,
        vision_num_heads=vision_num_heads,
        vision_num_kv_heads=vision_num_kv_heads,
        vision_mlp_ratio=vision_mlp_ratio,
        vision_layer_scale_init_values=vision_layer_scale_init_values,
        vocab_size=vocab_size,
        context_length=context_length,
        text_embed_dim=text_embed_dim,
        text_depth=text_depth,
        text_num_heads=text_num_heads,
        text_num_kv_heads=text_num_kv_heads,
        text_mlp_ratio=text_mlp_ratio,
        text_layer_scale_init_values=text_layer_scale_init_values,
        text_pad_id=text_pad_id,
        text_no_causal_mask=text_no_causal_mask,
        text_use_key_padding_mask=text_use_key_padding_mask,
        output_dim=output_dim,
        init_logit_scale=init_logit_scale,
        init_logit_bias=init_logit_bias,
        **kwargs)


def vit_large_patch16_gqa_clip(image_size=256,
                               patch_size=16,
                               vision_embed_dim=1024,
                               vision_depth=24,
                               vision_num_heads=16,
                               vision_num_kv_heads=8,
                               vision_mlp_ratio=4.5,
                               vision_layer_scale_init_values=1e-5,
                               vocab_size=248070,
                               context_length=96,
                               text_embed_dim=768,
                               text_depth=12,
                               text_num_heads=12,
                               text_num_kv_heads=6,
                               text_mlp_ratio=4.5,
                               text_layer_scale_init_values=1e-5,
                               text_pad_id=248055,
                               text_no_causal_mask=False,
                               text_use_key_padding_mask=True,
                               output_dim=768,
                               init_logit_scale=np.log(1 / 0.07),
                               init_logit_bias=None,
                               **kwargs):
    return GQACLIP(
        image_size=image_size,
        patch_size=patch_size,
        vision_embed_dim=vision_embed_dim,
        vision_depth=vision_depth,
        vision_num_heads=vision_num_heads,
        vision_num_kv_heads=vision_num_kv_heads,
        vision_mlp_ratio=vision_mlp_ratio,
        vision_layer_scale_init_values=vision_layer_scale_init_values,
        vocab_size=vocab_size,
        context_length=context_length,
        text_embed_dim=text_embed_dim,
        text_depth=text_depth,
        text_num_heads=text_num_heads,
        text_num_kv_heads=text_num_kv_heads,
        text_mlp_ratio=text_mlp_ratio,
        text_layer_scale_init_values=text_layer_scale_init_values,
        text_pad_id=text_pad_id,
        text_no_causal_mask=text_no_causal_mask,
        text_use_key_padding_mask=text_use_key_padding_mask,
        output_dim=output_dim,
        init_logit_scale=init_logit_scale,
        init_logit_bias=init_logit_bias,
        **kwargs)


def vit_huge_patch16_gqa_clip(image_size=256,
                              patch_size=16,
                              vision_embed_dim=1280,
                              vision_depth=32,
                              vision_num_heads=20,
                              vision_num_kv_heads=10,
                              vision_mlp_ratio=4.5,
                              vision_layer_scale_init_values=1e-5,
                              vocab_size=248070,
                              context_length=96,
                              text_embed_dim=1024,
                              text_depth=24,
                              text_num_heads=16,
                              text_num_kv_heads=8,
                              text_mlp_ratio=4.5,
                              text_layer_scale_init_values=1e-5,
                              text_pad_id=248055,
                              text_no_causal_mask=False,
                              text_use_key_padding_mask=True,
                              output_dim=1024,
                              init_logit_scale=np.log(1 / 0.07),
                              init_logit_bias=None,
                              **kwargs):
    return GQACLIP(
        image_size=image_size,
        patch_size=patch_size,
        vision_embed_dim=vision_embed_dim,
        vision_depth=vision_depth,
        vision_num_heads=vision_num_heads,
        vision_num_kv_heads=vision_num_kv_heads,
        vision_mlp_ratio=vision_mlp_ratio,
        vision_layer_scale_init_values=vision_layer_scale_init_values,
        vocab_size=vocab_size,
        context_length=context_length,
        text_embed_dim=text_embed_dim,
        text_depth=text_depth,
        text_num_heads=text_num_heads,
        text_num_kv_heads=text_num_kv_heads,
        text_mlp_ratio=text_mlp_ratio,
        text_layer_scale_init_values=text_layer_scale_init_values,
        text_pad_id=text_pad_id,
        text_no_causal_mask=text_no_causal_mask,
        text_use_key_padding_mask=text_use_key_padding_mask,
        output_dim=output_dim,
        init_logit_scale=init_logit_scale,
        init_logit_bias=init_logit_bias,
        **kwargs)


def vit_1B_patch16_gqa_clip(image_size=256,
                            patch_size=16,
                            vision_embed_dim=1536,
                            vision_depth=36,
                            vision_num_heads=24,
                            vision_num_kv_heads=12,
                            vision_mlp_ratio=4.5,
                            vision_layer_scale_init_values=1e-5,
                            vocab_size=248070,
                            context_length=96,
                            text_embed_dim=1024,
                            text_depth=24,
                            text_num_heads=16,
                            text_num_kv_heads=8,
                            text_mlp_ratio=4.5,
                            text_layer_scale_init_values=1e-5,
                            text_pad_id=248055,
                            text_no_causal_mask=False,
                            text_use_key_padding_mask=True,
                            output_dim=1024,
                            init_logit_scale=np.log(1 / 0.07),
                            init_logit_bias=None,
                            **kwargs):
    return GQACLIP(
        image_size=image_size,
        patch_size=patch_size,
        vision_embed_dim=vision_embed_dim,
        vision_depth=vision_depth,
        vision_num_heads=vision_num_heads,
        vision_num_kv_heads=vision_num_kv_heads,
        vision_mlp_ratio=vision_mlp_ratio,
        vision_layer_scale_init_values=vision_layer_scale_init_values,
        vocab_size=vocab_size,
        context_length=context_length,
        text_embed_dim=text_embed_dim,
        text_depth=text_depth,
        text_num_heads=text_num_heads,
        text_num_kv_heads=text_num_kv_heads,
        text_mlp_ratio=text_mlp_ratio,
        text_layer_scale_init_values=text_layer_scale_init_values,
        text_pad_id=text_pad_id,
        text_no_causal_mask=text_no_causal_mask,
        text_use_key_padding_mask=text_use_key_padding_mask,
        output_dim=output_dim,
        init_logit_scale=init_logit_scale,
        init_logit_bias=init_logit_bias,
        **kwargs)


####################################################################################
# SigLIP Models
####################################################################################


def vit_base_patch16_gqa_siglip(image_size=256,
                                patch_size=16,
                                vision_embed_dim=768,
                                vision_depth=12,
                                vision_num_heads=12,
                                vision_num_kv_heads=6,
                                vision_mlp_ratio=4.5,
                                vision_layer_scale_init_values=1e-5,
                                vocab_size=248070,
                                context_length=96,
                                text_embed_dim=512,
                                text_depth=12,
                                text_num_heads=8,
                                text_num_kv_heads=4,
                                text_mlp_ratio=4.5,
                                text_layer_scale_init_values=1e-5,
                                text_pad_id=248055,
                                text_no_causal_mask=False,
                                text_use_key_padding_mask=False,
                                output_dim=512,
                                init_logit_scale=np.log(10),
                                init_logit_bias=-10,
                                **kwargs):
    return GQACLIP(
        image_size=image_size,
        patch_size=patch_size,
        vision_embed_dim=vision_embed_dim,
        vision_depth=vision_depth,
        vision_num_heads=vision_num_heads,
        vision_num_kv_heads=vision_num_kv_heads,
        vision_mlp_ratio=vision_mlp_ratio,
        vision_layer_scale_init_values=vision_layer_scale_init_values,
        vocab_size=vocab_size,
        context_length=context_length,
        text_embed_dim=text_embed_dim,
        text_depth=text_depth,
        text_num_heads=text_num_heads,
        text_num_kv_heads=text_num_kv_heads,
        text_mlp_ratio=text_mlp_ratio,
        text_layer_scale_init_values=text_layer_scale_init_values,
        text_pad_id=text_pad_id,
        text_no_causal_mask=text_no_causal_mask,
        text_use_key_padding_mask=text_use_key_padding_mask,
        output_dim=output_dim,
        init_logit_scale=init_logit_scale,
        init_logit_bias=init_logit_bias,
        **kwargs)


def vit_large_patch16_gqa_siglip(image_size=256,
                                 patch_size=16,
                                 vision_embed_dim=1024,
                                 vision_depth=24,
                                 vision_num_heads=16,
                                 vision_num_kv_heads=8,
                                 vision_mlp_ratio=4.5,
                                 vision_layer_scale_init_values=1e-5,
                                 vocab_size=248070,
                                 context_length=96,
                                 text_embed_dim=768,
                                 text_depth=12,
                                 text_num_heads=12,
                                 text_num_kv_heads=6,
                                 text_mlp_ratio=4.5,
                                 text_layer_scale_init_values=1e-5,
                                 text_pad_id=248055,
                                 text_no_causal_mask=False,
                                 text_use_key_padding_mask=False,
                                 output_dim=768,
                                 init_logit_scale=np.log(10),
                                 init_logit_bias=-10,
                                 **kwargs):
    return GQACLIP(
        image_size=image_size,
        patch_size=patch_size,
        vision_embed_dim=vision_embed_dim,
        vision_depth=vision_depth,
        vision_num_heads=vision_num_heads,
        vision_num_kv_heads=vision_num_kv_heads,
        vision_mlp_ratio=vision_mlp_ratio,
        vision_layer_scale_init_values=vision_layer_scale_init_values,
        vocab_size=vocab_size,
        context_length=context_length,
        text_embed_dim=text_embed_dim,
        text_depth=text_depth,
        text_num_heads=text_num_heads,
        text_num_kv_heads=text_num_kv_heads,
        text_mlp_ratio=text_mlp_ratio,
        text_layer_scale_init_values=text_layer_scale_init_values,
        text_pad_id=text_pad_id,
        text_no_causal_mask=text_no_causal_mask,
        text_use_key_padding_mask=text_use_key_padding_mask,
        output_dim=output_dim,
        init_logit_scale=init_logit_scale,
        init_logit_bias=init_logit_bias,
        **kwargs)


def vit_huge_patch16_gqa_siglip(image_size=256,
                                patch_size=16,
                                vision_embed_dim=1280,
                                vision_depth=32,
                                vision_num_heads=20,
                                vision_num_kv_heads=10,
                                vision_mlp_ratio=4.5,
                                vision_layer_scale_init_values=1e-5,
                                vocab_size=248070,
                                context_length=96,
                                text_embed_dim=1024,
                                text_depth=24,
                                text_num_heads=16,
                                text_num_kv_heads=8,
                                text_mlp_ratio=4.5,
                                text_layer_scale_init_values=1e-5,
                                text_pad_id=248055,
                                text_no_causal_mask=False,
                                text_use_key_padding_mask=False,
                                output_dim=1024,
                                init_logit_scale=np.log(10),
                                init_logit_bias=-10,
                                **kwargs):
    return GQACLIP(
        image_size=image_size,
        patch_size=patch_size,
        vision_embed_dim=vision_embed_dim,
        vision_depth=vision_depth,
        vision_num_heads=vision_num_heads,
        vision_num_kv_heads=vision_num_kv_heads,
        vision_mlp_ratio=vision_mlp_ratio,
        vision_layer_scale_init_values=vision_layer_scale_init_values,
        vocab_size=vocab_size,
        context_length=context_length,
        text_embed_dim=text_embed_dim,
        text_depth=text_depth,
        text_num_heads=text_num_heads,
        text_num_kv_heads=text_num_kv_heads,
        text_mlp_ratio=text_mlp_ratio,
        text_layer_scale_init_values=text_layer_scale_init_values,
        text_pad_id=text_pad_id,
        text_no_causal_mask=text_no_causal_mask,
        text_use_key_padding_mask=text_use_key_padding_mask,
        output_dim=output_dim,
        init_logit_scale=init_logit_scale,
        init_logit_bias=init_logit_bias,
        **kwargs)


def vit_1B_patch16_gqa_siglip(image_size=256,
                              patch_size=16,
                              vision_embed_dim=1536,
                              vision_depth=36,
                              vision_num_heads=24,
                              vision_num_kv_heads=12,
                              vision_mlp_ratio=4.5,
                              vision_layer_scale_init_values=1e-5,
                              vocab_size=248070,
                              context_length=96,
                              text_embed_dim=1024,
                              text_depth=24,
                              text_num_heads=16,
                              text_num_kv_heads=8,
                              text_mlp_ratio=4.5,
                              text_layer_scale_init_values=1e-5,
                              text_pad_id=248055,
                              text_no_causal_mask=False,
                              text_use_key_padding_mask=False,
                              output_dim=1024,
                              init_logit_scale=np.log(10),
                              init_logit_bias=-10,
                              **kwargs):
    return GQACLIP(
        image_size=image_size,
        patch_size=patch_size,
        vision_embed_dim=vision_embed_dim,
        vision_depth=vision_depth,
        vision_num_heads=vision_num_heads,
        vision_num_kv_heads=vision_num_kv_heads,
        vision_mlp_ratio=vision_mlp_ratio,
        vision_layer_scale_init_values=vision_layer_scale_init_values,
        vocab_size=vocab_size,
        context_length=context_length,
        text_embed_dim=text_embed_dim,
        text_depth=text_depth,
        text_num_heads=text_num_heads,
        text_num_kv_heads=text_num_kv_heads,
        text_mlp_ratio=text_mlp_ratio,
        text_layer_scale_init_values=text_layer_scale_init_values,
        text_pad_id=text_pad_id,
        text_no_causal_mask=text_no_causal_mask,
        text_use_key_padding_mask=text_use_key_padding_mask,
        output_dim=output_dim,
        init_logit_scale=init_logit_scale,
        init_logit_bias=init_logit_bias,
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

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(BASE_DIR)

    from SimpleClip.models.tokenizer import Qwen35Tokenizer

    net = vit_base_patch16_gqa_clip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = Qwen35Tokenizer(context_length=net.context_length)
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
    tokenizer = Qwen35Tokenizer(context_length=net.context_length)
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
    tokenizer = Qwen35Tokenizer(context_length=net.context_length)
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
    tokenizer = Qwen35Tokenizer(context_length=net.context_length)
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
    tokenizer = Qwen35Tokenizer(context_length=net.context_length)
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
    tokenizer = Qwen35Tokenizer(context_length=net.context_length)
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
    tokenizer = Qwen35Tokenizer(context_length=net.context_length)
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
    tokenizer = Qwen35Tokenizer(context_length=net.context_length)
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
    tokenizer = Qwen35Tokenizer(context_length=net.context_length)
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
