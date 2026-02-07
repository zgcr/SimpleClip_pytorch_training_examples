import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint


class PatchEmbeddingBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 embed_dim,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 has_norm=False):
        super(PatchEmbeddingBlock, self).__init__()
        bias = False if has_norm else True

        self.proj = nn.Conv2d(in_channels,
                              embed_dim,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=bias)
        self.norm = nn.LayerNorm(embed_dim,
                                 eps=1e-6) if has_norm else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)

        return x


class FeedForward(nn.Module):

    def __init__(self, dim, dim_feedforward, dropout=0.):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, dim_feedforward)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(dim_feedforward, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class LayerScale(nn.Module):

    def __init__(self, dim, init_values=1e-5):
        super(LayerScale, self).__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        x = x * self.gamma

        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim, num_heads=8, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads)**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.dropout_p = dropout

    def forward(self, x, attn_mask=None):
        b, n, c = x.shape

        # [b,n,c] -> [b,n,3,head_num,c//head_num] -> [3,b,head_num,n,c//head_num]
        x = self.qkv(x).view(b, n, 3, self.num_heads,
                             c // self.num_heads).permute(2, 0, 3, 1, 4)
        # [3,b,head_num,n,c//head_num] -> 3个 [b,head_num,n,c//head_num]
        q, k, v = torch.unbind(x, dim=0)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            scale=self.scale)

        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 feedforward_ratio=4,
                 dropout=0.,
                 layer_scale_init_values=1e-5):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadAttention(dim=dim,
                                       num_heads=num_heads,
                                       dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = FeedForward(dim=dim,
                               dim_feedforward=int(dim * feedforward_ratio),
                               dropout=dropout)

        self.ls1 = LayerScale(
            dim=dim, init_values=layer_scale_init_values
        ) if layer_scale_init_values is not None else nn.Identity()
        self.ls2 = LayerScale(
            dim=dim, init_values=layer_scale_init_values
        ) if layer_scale_init_values is not None else nn.Identity()

    def forward(self, x, attn_mask=None):
        x = x + self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask))
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
                 mlp_ratio=4.0,
                 output_dim=512,
                 dropout=0.0,
                 layer_scale_init_values=1e-5,
                 use_gradient_checkpoint=False):
        super(VisionEncoder, self).__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.patch_embed = PatchEmbeddingBlock(in_channels,
                                               embed_dim,
                                               kernel_size=patch_size,
                                               stride=patch_size,
                                               padding=0,
                                               groups=1,
                                               has_norm=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        num_patches = (image_size // patch_size)**2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                dim=embed_dim,
                num_heads=num_heads,
                feedforward_ratio=mlp_ratio,
                dropout=dropout,
                layer_scale_init_values=layer_scale_init_values)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.proj = nn.Linear(embed_dim, output_dim, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed

        for block in self.blocks:
            if self.use_gradient_checkpoint:
                x = checkpoint(block, x, None, use_reentrant=False)
            else:
                x = block(x, attn_mask=None)

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
        self.pos_embed = nn.Parameter(torch.zeros(1, context_length,
                                                  embed_dim))

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                dim=embed_dim,
                num_heads=num_heads,
                feedforward_ratio=mlp_ratio,
                dropout=dropout,
                layer_scale_init_values=layer_scale_init_values)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
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
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

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
        x = x + self.pos_embed[:, :L, :]

        attn_mask = self.attn_mask
        if attn_mask is not None and L < self.context_length:
            attn_mask = attn_mask[:L, :L]

        for block in self.blocks:
            if self.use_gradient_checkpoint:
                x = checkpoint(block, x, attn_mask, use_reentrant=False)
            else:
                x = block(x, attn_mask=attn_mask)

        x = self.norm(x)

        x, _ = self.text_global_pool(x, text, pool_type=self.pool_type)
        x = self.proj(x)

        return x


class CLIP(nn.Module):

    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 vision_embed_dim=768,
                 vision_depth=12,
                 vision_num_heads=12,
                 vision_mlp_ratio=4.0,
                 vision_layer_scale_init_values=1e-5,
                 vocab_size=49408,
                 context_length=77,
                 text_embed_dim=512,
                 text_depth=12,
                 text_num_heads=8,
                 text_mlp_ratio=4.0,
                 text_layer_scale_init_values=1e-5,
                 text_pad_id=0,
                 text_no_causal_mask=False,
                 text_pool_type='argmax',
                 output_dim=512,
                 dropout=0.0,
                 init_logit_scale=np.log(1 / 0.07),
                 init_logit_bias=None,
                 use_gradient_checkpoint=False):
        super(CLIP, self).__init__()

        self.visual = VisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=vision_embed_dim,
            depth=vision_depth,
            num_heads=vision_num_heads,
            mlp_ratio=vision_mlp_ratio,
            output_dim=output_dim,
            dropout=dropout,
            layer_scale_init_values=vision_layer_scale_init_values,
            use_gradient_checkpoint=use_gradient_checkpoint)

        self.text = TextEncoder(
            vocab_size=vocab_size,
            context_length=context_length,
            embed_dim=text_embed_dim,
            depth=text_depth,
            num_heads=text_num_heads,
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


def vit_base_patch16_clip(image_size=224,
                          patch_size=16,
                          context_length=77,
                          vocab_size=49408,
                          **kwargs):
    return CLIP(image_size=image_size,
                patch_size=patch_size,
                vision_embed_dim=768,
                vision_depth=12,
                vision_num_heads=12,
                vision_mlp_ratio=4.0,
                vision_layer_scale_init_values=1e-5,
                vocab_size=vocab_size,
                context_length=context_length,
                text_embed_dim=512,
                text_depth=12,
                text_num_heads=8,
                text_mlp_ratio=4.0,
                text_layer_scale_init_values=1e-5,
                output_dim=512,
                **kwargs)


def vit_large_patch16_clip(image_size=224,
                           patch_size=16,
                           context_length=77,
                           vocab_size=49408,
                           **kwargs):
    return CLIP(image_size=image_size,
                patch_size=patch_size,
                vision_embed_dim=1024,
                vision_depth=24,
                vision_num_heads=16,
                vision_mlp_ratio=4.0,
                vision_layer_scale_init_values=1e-5,
                vocab_size=vocab_size,
                context_length=context_length,
                text_embed_dim=768,
                text_depth=12,
                text_num_heads=12,
                text_mlp_ratio=4.0,
                text_layer_scale_init_values=1e-5,
                output_dim=768,
                **kwargs)


def vit_huge_patch16_clip(image_size=224,
                          patch_size=16,
                          context_length=77,
                          vocab_size=49408,
                          **kwargs):
    return CLIP(image_size=image_size,
                patch_size=patch_size,
                vision_embed_dim=1280,
                vision_depth=32,
                vision_num_heads=16,
                vision_mlp_ratio=4.0,
                vision_layer_scale_init_values=1e-5,
                vocab_size=vocab_size,
                context_length=context_length,
                text_embed_dim=1024,
                text_depth=16,
                text_num_heads=16,
                text_mlp_ratio=4.0,
                text_layer_scale_init_values=1e-5,
                output_dim=1024,
                **kwargs)


def vit_1B_patch16_clip(image_size=224,
                        patch_size=16,
                        context_length=77,
                        vocab_size=49408,
                        **kwargs):
    return CLIP(image_size=image_size,
                patch_size=patch_size,
                vision_embed_dim=1536,
                vision_depth=36,
                vision_num_heads=24,
                vision_mlp_ratio=4.0,
                vision_layer_scale_init_values=1e-5,
                vocab_size=vocab_size,
                context_length=context_length,
                text_embed_dim=1280,
                text_depth=20,
                text_num_heads=16,
                text_mlp_ratio=4.0,
                text_layer_scale_init_values=1e-5,
                output_dim=1280,
                **kwargs)


####################################################################################
# SigLIP Models
####################################################################################


def vit_base_patch16_siglip(image_size=224,
                            patch_size=16,
                            context_length=64,
                            vocab_size=256000,
                            **kwargs):
    return CLIP(image_size=image_size,
                patch_size=patch_size,
                vision_embed_dim=768,
                vision_depth=12,
                vision_num_heads=12,
                vision_mlp_ratio=4.0,
                vision_layer_scale_init_values=1e-5,
                vocab_size=vocab_size,
                context_length=context_length,
                text_embed_dim=512,
                text_depth=12,
                text_num_heads=8,
                text_mlp_ratio=4.0,
                text_layer_scale_init_values=1e-5,
                text_no_causal_mask=True,
                text_pool_type='last',
                output_dim=512,
                init_logit_scale=np.log(10),
                init_logit_bias=-10,
                **kwargs)


def vit_large_patch16_siglip(image_size=224,
                             patch_size=16,
                             context_length=64,
                             vocab_size=256000,
                             **kwargs):
    return CLIP(image_size=image_size,
                patch_size=patch_size,
                vision_embed_dim=1024,
                vision_depth=24,
                vision_num_heads=16,
                vision_mlp_ratio=4.0,
                vision_layer_scale_init_values=1e-5,
                vocab_size=vocab_size,
                context_length=context_length,
                text_embed_dim=768,
                text_depth=12,
                text_num_heads=12,
                text_mlp_ratio=4.0,
                text_layer_scale_init_values=1e-5,
                text_no_causal_mask=True,
                text_pool_type='last',
                output_dim=768,
                init_logit_scale=np.log(10),
                init_logit_bias=-10,
                **kwargs)


def vit_huge_patch16_siglip(image_size=224,
                            patch_size=16,
                            context_length=64,
                            vocab_size=256000,
                            **kwargs):
    return CLIP(image_size=image_size,
                patch_size=patch_size,
                vision_embed_dim=1280,
                vision_depth=32,
                vision_num_heads=16,
                vision_mlp_ratio=4.0,
                vision_layer_scale_init_values=1e-5,
                vocab_size=vocab_size,
                context_length=context_length,
                text_embed_dim=1024,
                text_depth=16,
                text_num_heads=16,
                text_mlp_ratio=4.0,
                text_layer_scale_init_values=1e-5,
                text_no_causal_mask=True,
                text_pool_type='last',
                output_dim=1024,
                init_logit_scale=np.log(10),
                init_logit_bias=-10,
                **kwargs)


def vit_1B_patch16_siglip(image_size=224,
                          patch_size=16,
                          context_length=64,
                          vocab_size=256000,
                          **kwargs):
    return CLIP(image_size=image_size,
                patch_size=patch_size,
                vision_embed_dim=1536,
                vision_depth=36,
                vision_num_heads=24,
                vision_mlp_ratio=4.0,
                vision_layer_scale_init_values=1e-5,
                vocab_size=vocab_size,
                context_length=context_length,
                text_embed_dim=1280,
                text_depth=20,
                text_num_heads=16,
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

    net = vit_base_patch16_clip()
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

    net = vit_base_patch16_clip(use_gradient_checkpoint=True)
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

    net = vit_large_patch16_clip()
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

    net = vit_huge_patch16_clip()
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

    net = vit_1B_patch16_clip()
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
    net = vit_base_patch16_siglip()
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

    net = vit_large_patch16_siglip()
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

    net = vit_huge_patch16_siglip()
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

    net = vit_1B_patch16_siglip()
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
