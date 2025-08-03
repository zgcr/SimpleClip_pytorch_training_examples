""" 
https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py
"""
import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from SimpleClip.models.modified_resnet import ModifiedResNet
from SimpleClip.models.transformer import VisionTransformer, TextTransformer

# no weight decay layer:'positional_embedding', 'class_embedding', 'logit_scale', 'logit_bias'

__all__ = [
    'resnet50_clip',
    'resnet101_clip',
    'ViT_B_16_clip',
    'ViT_B_32_clip',
    'ViT_L_14_clip',
    'ViT_L_16_clip',
    'ViT_H_14_clip',
    'ViT_H_16_clip',
    'ViT_G_14_clip',
    'ViT_bigG_14_clip',
    'ViT_B_16_siglip',
    'ViT_B_16_siglip2',
]


class ResNetCLIP(nn.Module):

    def __init__(self,
                 image_size=224,
                 output_planes=1024,
                 image_encoder_inplanes=64,
                 image_encoder_layer_nums=[3, 4, 6, 3],
                 text_encoder_context_length=77,
                 text_encoder_vocab_size=49408,
                 text_encoder_inplanes=512,
                 text_encoder_layer_nums=12,
                 text_encoder_head_nums=8,
                 text_encoder_mlp_ratio=4.0,
                 text_encoder_init_value=None,
                 text_encoder_no_causal_mask=False,
                 text_encoder_pad_id=0,
                 text_encoder_pool_type='argmax',
                 init_logit_scale=np.log(1 / 0.07),
                 init_logit_bias=None,
                 use_gradient_checkpoint=False):
        super(ResNetCLIP, self).__init__()
        image_encoder_head_nums = image_encoder_inplanes // 2
        self.visual = ModifiedResNet(
            image_size=image_size,
            planes=image_encoder_inplanes,
            layer_nums=image_encoder_layer_nums,
            head_nums=image_encoder_head_nums,
            outplanes=output_planes,
            use_gradient_checkpoint=use_gradient_checkpoint)

        self.text = TextTransformer(
            context_length=text_encoder_context_length,
            vocab_size=text_encoder_vocab_size,
            inplanes=text_encoder_inplanes,
            layer_nums=text_encoder_layer_nums,
            head_nums=text_encoder_head_nums,
            mlp_ratio=text_encoder_mlp_ratio,
            output_planes=output_planes,
            init_value=text_encoder_init_value,
            no_causal_mask=text_encoder_no_causal_mask,
            pad_id=text_encoder_pad_id,
            pool_type=text_encoder_pool_type,
            use_gradient_checkpoint=use_gradient_checkpoint)

        self.logit_scale = nn.Parameter(
            torch.tensor(init_logit_scale, dtype=torch.float32))

        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(
                torch.tensor(init_logit_bias, dtype=torch.float32))
        else:
            self.logit_bias = None

        self.context_length = text_encoder_context_length
        self.vocab_size = text_encoder_vocab_size

    def text_global_pool(self, x, text=None, pool_type='argmax'):
        if pool_type == 'first':
            pooled, tokens = x[:, 0], x[:, 1:]
        elif pool_type == 'last':
            pooled, tokens = x[:, -1], x[:, :-1]
        elif pool_type == 'argmax':
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            assert text is not None
            pooled, tokens = x[torch.arange(x.shape[0]),
                               text.argmax(dim=-1)], x
        else:
            pooled = tokens = x

        return pooled, tokens

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


def resnet50_clip(**kwargs):
    return ResNetCLIP(output_planes=1024,
                      image_encoder_inplanes=64,
                      image_encoder_layer_nums=[3, 4, 6, 3],
                      text_encoder_context_length=77,
                      text_encoder_vocab_size=49408,
                      text_encoder_inplanes=512,
                      text_encoder_layer_nums=12,
                      text_encoder_head_nums=8,
                      **kwargs)


def resnet101_clip(**kwargs):
    return ResNetCLIP(output_planes=512,
                      image_encoder_inplanes=64,
                      image_encoder_layer_nums=[3, 4, 23, 3],
                      text_encoder_context_length=77,
                      text_encoder_vocab_size=49408,
                      text_encoder_inplanes=512,
                      text_encoder_layer_nums=12,
                      text_encoder_head_nums=8,
                      **kwargs)


class CLIP(nn.Module):

    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 output_planes=512,
                 image_encoder_inplanes=768,
                 image_encoder_layer_nums=12,
                 image_encoder_head_planes=64,
                 image_encoder_mlp_ratio=4.0,
                 image_encoder_init_value=None,
                 image_encoder_patch_dropout=0.,
                 image_encoder_pos_embed_type='learnable',
                 image_encoder_pool_type='tok',
                 text_encoder_context_length=77,
                 text_encoder_vocab_size=49408,
                 text_encoder_inplanes=512,
                 text_encoder_layer_nums=12,
                 text_encoder_head_nums=8,
                 text_encoder_mlp_ratio=4.0,
                 text_encoder_init_value=None,
                 text_encoder_no_causal_mask=False,
                 text_encoder_pad_id=0,
                 text_encoder_pool_type='argmax',
                 init_logit_scale=np.log(1 / 0.07),
                 init_logit_bias=None,
                 use_gradient_checkpoint=False):
        super(CLIP, self).__init__()
        image_encoder_head_nums = image_encoder_inplanes // image_encoder_head_planes
        self.visual = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            inplanes=image_encoder_inplanes,
            layer_nums=image_encoder_layer_nums,
            head_nums=image_encoder_head_nums,
            mlp_ratio=image_encoder_mlp_ratio,
            output_planes=output_planes,
            init_value=image_encoder_init_value,
            patch_dropout=image_encoder_patch_dropout,
            pos_embed_type=image_encoder_pos_embed_type,
            pool_type=image_encoder_pool_type,
            use_gradient_checkpoint=use_gradient_checkpoint)

        self.text = TextTransformer(
            context_length=text_encoder_context_length,
            vocab_size=text_encoder_vocab_size,
            inplanes=text_encoder_inplanes,
            layer_nums=text_encoder_layer_nums,
            head_nums=text_encoder_head_nums,
            mlp_ratio=text_encoder_mlp_ratio,
            output_planes=output_planes,
            init_value=text_encoder_init_value,
            no_causal_mask=text_encoder_no_causal_mask,
            pad_id=text_encoder_pad_id,
            pool_type=text_encoder_pool_type,
            use_gradient_checkpoint=use_gradient_checkpoint)

        self.logit_scale = nn.Parameter(
            torch.tensor(init_logit_scale, dtype=torch.float32))

        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(
                torch.tensor(init_logit_bias, dtype=torch.float32))
        else:
            self.logit_bias = None

        self.context_length = text_encoder_context_length
        self.vocab_size = text_encoder_vocab_size

    def text_global_pool(self, x, text=None, pool_type='argmax'):
        if pool_type == 'first':
            pooled, tokens = x[:, 0], x[:, 1:]
        elif pool_type == 'last':
            pooled, tokens = x[:, -1], x[:, :-1]
        elif pool_type == 'argmax':
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            assert text is not None
            pooled, tokens = x[torch.arange(x.shape[0]),
                               text.argmax(dim=-1)], x
        else:
            pooled = tokens = x

        return pooled, tokens

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


def ViT_B_16_clip(image_size=224,
                  patch_size=16,
                  text_encoder_context_length=77,
                  text_encoder_vocab_size=49408,
                  **kwargs):
    return CLIP(image_size=image_size,
                patch_size=patch_size,
                output_planes=512,
                image_encoder_inplanes=768,
                image_encoder_layer_nums=12,
                text_encoder_context_length=text_encoder_context_length,
                text_encoder_vocab_size=text_encoder_vocab_size,
                text_encoder_inplanes=512,
                text_encoder_layer_nums=12,
                text_encoder_head_nums=8,
                **kwargs)


def ViT_B_32_clip(image_size=224,
                  patch_size=32,
                  text_encoder_context_length=77,
                  text_encoder_vocab_size=49408,
                  **kwargs):
    return CLIP(image_size=image_size,
                patch_size=patch_size,
                output_planes=512,
                image_encoder_inplanes=768,
                image_encoder_layer_nums=12,
                text_encoder_context_length=text_encoder_context_length,
                text_encoder_vocab_size=text_encoder_vocab_size,
                text_encoder_inplanes=512,
                text_encoder_layer_nums=12,
                text_encoder_head_nums=8,
                **kwargs)


def ViT_L_14_clip(image_size=224,
                  patch_size=14,
                  text_encoder_context_length=77,
                  text_encoder_vocab_size=49408,
                  **kwargs):
    return CLIP(image_size=image_size,
                patch_size=patch_size,
                output_planes=768,
                image_encoder_inplanes=1024,
                image_encoder_layer_nums=24,
                text_encoder_context_length=text_encoder_context_length,
                text_encoder_vocab_size=text_encoder_vocab_size,
                text_encoder_inplanes=768,
                text_encoder_layer_nums=12,
                text_encoder_head_nums=12,
                **kwargs)


def ViT_L_16_clip(image_size=224,
                  patch_size=16,
                  text_encoder_context_length=77,
                  text_encoder_vocab_size=49408,
                  **kwargs):
    return CLIP(image_size=image_size,
                patch_size=patch_size,
                output_planes=768,
                image_encoder_inplanes=1024,
                image_encoder_layer_nums=24,
                text_encoder_context_length=text_encoder_context_length,
                text_encoder_vocab_size=text_encoder_vocab_size,
                text_encoder_inplanes=768,
                text_encoder_layer_nums=12,
                text_encoder_head_nums=12,
                **kwargs)


def ViT_H_14_clip(image_size=224,
                  patch_size=14,
                  text_encoder_context_length=77,
                  text_encoder_vocab_size=49408,
                  **kwargs):
    return CLIP(image_size=image_size,
                patch_size=patch_size,
                output_planes=1024,
                image_encoder_inplanes=1280,
                image_encoder_layer_nums=32,
                image_encoder_head_planes=80,
                text_encoder_context_length=text_encoder_context_length,
                text_encoder_vocab_size=text_encoder_vocab_size,
                text_encoder_inplanes=1024,
                text_encoder_layer_nums=24,
                text_encoder_head_nums=16,
                **kwargs)


def ViT_H_16_clip(image_size=224,
                  patch_size=16,
                  text_encoder_context_length=77,
                  text_encoder_vocab_size=49408,
                  **kwargs):
    return CLIP(image_size=image_size,
                patch_size=patch_size,
                output_planes=1024,
                image_encoder_inplanes=1280,
                image_encoder_layer_nums=32,
                image_encoder_head_planes=80,
                text_encoder_context_length=text_encoder_context_length,
                text_encoder_vocab_size=text_encoder_vocab_size,
                text_encoder_inplanes=1024,
                text_encoder_layer_nums=24,
                text_encoder_head_nums=16,
                **kwargs)


def ViT_G_14_clip(image_size=224,
                  patch_size=14,
                  text_encoder_context_length=77,
                  text_encoder_vocab_size=49408,
                  **kwargs):
    return CLIP(image_size=image_size,
                patch_size=patch_size,
                output_planes=1024,
                image_encoder_inplanes=1408,
                image_encoder_layer_nums=40,
                image_encoder_head_planes=88,
                image_encoder_mlp_ratio=4.3637,
                text_encoder_context_length=text_encoder_context_length,
                text_encoder_vocab_size=text_encoder_vocab_size,
                text_encoder_inplanes=1024,
                text_encoder_layer_nums=24,
                text_encoder_head_nums=16,
                **kwargs)


def ViT_bigG_14_clip(image_size=224,
                     patch_size=14,
                     text_encoder_context_length=77,
                     text_encoder_vocab_size=49408,
                     **kwargs):
    return CLIP(image_size=image_size,
                patch_size=patch_size,
                output_planes=1280,
                image_encoder_inplanes=1664,
                image_encoder_layer_nums=48,
                image_encoder_head_planes=104,
                image_encoder_mlp_ratio=4.9231,
                text_encoder_context_length=text_encoder_context_length,
                text_encoder_vocab_size=text_encoder_vocab_size,
                text_encoder_inplanes=1280,
                text_encoder_layer_nums=32,
                text_encoder_head_nums=20,
                **kwargs)


def ViT_B_16_siglip(image_size=224,
                    patch_size=16,
                    text_encoder_context_length=64,
                    text_encoder_vocab_size=32000,
                    **kwargs):
    return CLIP(image_size=image_size,
                patch_size=patch_size,
                output_planes=768,
                image_encoder_inplanes=768,
                image_encoder_layer_nums=12,
                text_encoder_context_length=text_encoder_context_length,
                text_encoder_vocab_size=text_encoder_vocab_size,
                text_encoder_inplanes=768,
                text_encoder_layer_nums=12,
                text_encoder_head_nums=12,
                text_encoder_no_causal_mask=True,
                text_encoder_pool_type='last',
                init_logit_scale=np.log(10),
                init_logit_bias=-10,
                **kwargs)


def ViT_B_16_siglip2(image_size=224,
                     patch_size=16,
                     text_encoder_context_length=64,
                     text_encoder_vocab_size=256000,
                     **kwargs):
    return CLIP(image_size=image_size,
                patch_size=patch_size,
                output_planes=768,
                image_encoder_inplanes=768,
                image_encoder_layer_nums=12,
                text_encoder_context_length=text_encoder_context_length,
                text_encoder_vocab_size=text_encoder_vocab_size,
                text_encoder_inplanes=768,
                text_encoder_layer_nums=12,
                text_encoder_head_nums=12,
                text_encoder_no_causal_mask=True,
                text_encoder_pool_type='last',
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

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(BASE_DIR)

    from SimpleClip.models.tokenizer import SimpleTokenizer, SigLipTokenizer

    net = resnet50_clip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SimpleTokenizer(context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(images, tokens), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'2222, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'])

    net = resnet50_clip(use_gradient_checkpoint=True)
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SimpleTokenizer(context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(images, tokens), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'2222, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'])

    net = ViT_B_16_clip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SimpleTokenizer(context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(images, tokens), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'2222, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'])

    net = ViT_B_16_clip(use_gradient_checkpoint=True)
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SimpleTokenizer(context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(images, tokens), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'2222, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'])

    net = ViT_B_16_siglip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SigLipTokenizer('c4-en', context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(images, tokens), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'2222, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'],
          outputs['logit_bias'])

    net = ViT_B_16_siglip(text_encoder_vocab_size=250000)
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SigLipTokenizer('mc4', context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(images, tokens), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'2222, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'],
          outputs['logit_bias'])

    net = ViT_B_16_siglip2()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SigLipTokenizer('gemma', context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(images, tokens), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'2222, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'],
          outputs['logit_bias'])

    net = ViT_B_32_clip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SimpleTokenizer(context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(images, tokens), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'2222, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'])

    net = ViT_L_14_clip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SimpleTokenizer(context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(images, tokens), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'2222, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'])

    net = ViT_L_16_clip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SimpleTokenizer(context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(images, tokens), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'2222, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'])

    net = ViT_H_14_clip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SimpleTokenizer(context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(images, tokens), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'2222, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'])

    net = ViT_H_16_clip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SimpleTokenizer(context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(images, tokens), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'2222, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'])

    net = ViT_G_14_clip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SimpleTokenizer(context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(images, tokens), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'2222, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'])

    net = ViT_bigG_14_clip()
    net = net.cuda()
    image_h, image_w = 224, 224
    images = torch.randn(1, 3, image_h, image_w).cuda()
    tokenizer = SimpleTokenizer(context_length=net.context_length)
    texts = ['a dog']
    tokens = [tokenizer([str(per_image_text)])[0] for per_image_text in texts]
    tokens = torch.stack(tokens, dim=0).long()
    tokens = tokens.cuda()
    print('1111', images.shape, tokens.shape)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(images, tokens), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'2222, macs: {macs}, params: {params}')
    outputs = net(images, tokens)
    print('3333', outputs['image_features'].shape,
          outputs['text_features'].shape, outputs['logit_scale'])
