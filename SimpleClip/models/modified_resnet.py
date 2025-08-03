import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.expansion = 4

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes,
                               planes * self.expansion,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = nn.ReLU(inplace=True)

        # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
        self.downsample = None
        if stride > 1 or inplanes != planes * self.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                collections.OrderedDict([
                    ("-1", nn.AvgPool2d(stride)),
                    ("0",
                     nn.Conv2d(inplanes,
                               planes * self.expansion,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)),
                    ("1", nn.BatchNorm2d(planes * self.expansion))
                ]))

    def forward(self, x):
        identity = x

        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = self.bn3(self.conv3(x))

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.act3(x)

        return x


class AttentionPool2d(nn.Module):

    def __init__(self, inplanes, embedding_planes, outplanes, head_nums):
        super(AttentionPool2d, self).__init__()
        self.head_nums = head_nums

        self.positional_embedding = nn.Parameter(
            torch.randn(inplanes**2 + 1, embedding_planes) /
            embedding_planes**0.5)

        self.k_proj = nn.Linear(embedding_planes, embedding_planes)
        self.q_proj = nn.Linear(embedding_planes, embedding_planes)
        self.v_proj = nn.Linear(embedding_planes, embedding_planes)
        self.c_proj = nn.Linear(embedding_planes, outplanes)

    def forward(self, x):
        # NCHW -> (HW)NC
        x = x.reshape(x.shape[0], x.shape[1],
                      x.shape[2] * x.shape[3]).permute(2, 0, 1)
        # (HW+1)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        # (HW+1)NC
        x = x + self.positional_embedding[:, None, :]

        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.head_nums,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)

        x = x[0]

        return x


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self,
                 image_size,
                 planes,
                 layer_nums,
                 head_nums,
                 outplanes,
                 use_gradient_checkpoint=False):
        super(ModifiedResNet, self).__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3,
                               planes // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes // 2)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes // 2,
                               planes // 2,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes // 2)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes // 2,
                               planes,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.act3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        # this is a *mutable* variable used during construction
        self.inplanes = planes
        self.layer1 = self._make_layer(planes, layer_nums[0])
        self.layer2 = self._make_layer(planes * 2, layer_nums[1], stride=2)
        self.layer3 = self._make_layer(planes * 4, layer_nums[2], stride=2)
        self.layer4 = self._make_layer(planes * 8, layer_nums[3], stride=2)

        # the ResNet feature dimension
        embedding_planes = planes * 32
        self.attnpool = AttentionPool2d(image_size // 32, embedding_planes,
                                        outplanes, head_nums)

        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features**-0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4,
        ]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def _make_layer(self, planes, block_nums, stride=1):
        layers = [Bottleneck(self.inplanes, planes, stride)]

        self.inplanes = planes * 4
        for _ in range(1, block_nums):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)

        if self.use_gradient_checkpoint:
            x = checkpoint(self.layer1, x, use_reentrant=False)
            x = checkpoint(self.layer2, x, use_reentrant=False)
            x = checkpoint(self.layer3, x, use_reentrant=False)
            x = checkpoint(self.layer4, x, use_reentrant=False)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        x = self.attnpool(x)

        return x


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

    head_nums = 64 // 2
    net = ModifiedResNet(image_size=224,
                         planes=64,
                         layer_nums=[3, 4, 6, 3],
                         head_nums=head_nums,
                         outplanes=1024)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params}, out_shape: {out.shape}')

    head_nums = 64 // 2
    net = ModifiedResNet(image_size=224,
                         planes=64,
                         layer_nums=[3, 4, 6, 3],
                         head_nums=head_nums,
                         outplanes=1024,
                         use_gradient_checkpoint=True)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params}, out_shape: {out.shape}')
