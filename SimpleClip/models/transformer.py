'''
https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
'''
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint


def get_1d_sincos_pos_embed_from_grid(embedding_planes, position):
    """
    embedding_planes: output dimension for each position
    position: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embedding_planes % 2 == 0
    omega = np.arange(embedding_planes // 2, dtype=float)
    omega /= embedding_planes / 2.
    # (D/2,)
    omega = 1. / 10000**omega

    # (M,)
    position = position.reshape(-1)
    # (M, D/2), outer product
    out = np.einsum('m,d->md', position, omega)

    # (M, D/2)
    emb_sin = np.sin(out)
    # (M, D/2)
    emb_cos = np.cos(out)
    # (M, D)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)

    return emb


def get_2d_sincos_pos_embed_from_grid(embedding_planes, grid):
    assert embedding_planes % 2 == 0

    # use half of dimensions to encode grid_h
    # (H*W, D/2)
    emb_h = get_1d_sincos_pos_embed_from_grid(embedding_planes // 2, grid[0])
    # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embedding_planes // 2, grid[1])
    # (H*W, D)
    emb = np.concatenate([emb_h, emb_w], axis=1)

    return emb


def get_2d_sincos_pos_embed(embedding_planes, grid_size, cls_token=False):
    """
    grid_size of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embedding_planes] or [1+grid_size*grid_size, embedding_planes] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    # here w goes first
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embedding_planes, grid)

    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embedding_planes]), pos_embed], axis=0)

    return pos_embed


class LayerScale(nn.Module):

    def __init__(self, inplanes, init_values=1e-5, inplace=False):
        super(LayerScale, self).__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(inplanes))

    def forward(self, x):
        if self.inplace:
            x = x.mul_(self.gamma)
        else:
            x = x * self.gamma

        return x


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super(PatchDropout, self).__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        # exclude CLS token
        self.exclude_first_token = exclude_first_token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch, num_tokens = x.shape[0], x.shape[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class ResidualAttentionBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 head_nums,
                 mlp_ratio=4.0,
                 init_value=None,
                 is_cross_attention=False,
                 batch_first=True):
        super(ResidualAttentionBlock, self).__init__()

        self.ln_1 = nn.LayerNorm(inplanes)
        self.attn = nn.MultiheadAttention(inplanes,
                                          head_nums,
                                          batch_first=batch_first)
        self.ls_1 = LayerScale(
            inplanes, init_value) if init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = nn.LayerNorm(inplanes)

        self.ln_2 = nn.LayerNorm(inplanes)
        mlp_planes = int(inplanes * mlp_ratio)
        self.mlp = nn.Sequential(
            collections.OrderedDict([("c_fc", nn.Linear(inplanes, mlp_planes)),
                                     ("gelu", nn.GELU()),
                                     ("c_proj",
                                      nn.Linear(mlp_planes, inplanes))]))
        self.ls_2 = LayerScale(
            inplanes, init_value) if init_value is not None else nn.Identity()

    def attention(self, q_x, k_x=None, v_x=None, attn_mask=None):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x,
                         k_x,
                         v_x,
                         need_weights=False,
                         attn_mask=attn_mask)[0]

    def forward(self, q_x, k_x=None, v_x=None, attn_mask=None):
        k_x = self.ln_1_kv(k_x) if hasattr(
            self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(
            self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.ls_1(
            self.attention(
                q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))

        return x


class Transformer(nn.Module):

    def __init__(self,
                 inplanes,
                 layer_nums,
                 head_nums,
                 mlp_ratio=4.0,
                 init_value=None,
                 batch_first=True,
                 use_gradient_checkpoint=False):
        super(Transformer, self).__init__()
        self.batch_first = batch_first
        self.use_gradient_checkpoint = use_gradient_checkpoint

        resblocks = []
        for _ in range(layer_nums):
            resblocks.append(
                ResidualAttentionBlock(inplanes,
                                       head_nums,
                                       mlp_ratio,
                                       init_value=init_value,
                                       batch_first=batch_first))
        self.resblocks = nn.ModuleList(resblocks)

    def forward(self, x, attn_mask=None):
        if not self.batch_first:
            # NLD -> LND
            x = x.transpose(0, 1).contiguous()

        for r in self.resblocks:
            if self.use_gradient_checkpoint:
                x = checkpoint(r,
                               x,
                               None,
                               None,
                               attn_mask,
                               use_reentrant=False)
            else:
                x = r(x, attn_mask=attn_mask)

        if not self.batch_first:
            # LND -> NLD
            x = x.transpose(0, 1)

        return x


class VisionTransformer(nn.Module):

    def __init__(self,
                 image_size,
                 patch_size,
                 inplanes,
                 layer_nums,
                 head_nums,
                 mlp_ratio,
                 output_planes,
                 init_value=None,
                 patch_dropout=0.,
                 pos_embed_type='learnable',
                 pool_type='tok',
                 use_gradient_checkpoint=False):
        super(VisionTransformer, self).__init__()
        assert pos_embed_type in ('learnable', 'sin_cos_2d')
        assert pool_type in ('tok', 'avg', 'none')

        self.pool_type = pool_type
        self.grid_size = [image_size // patch_size, image_size // patch_size]
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.conv1 = nn.Conv2d(3,
                               inplanes,
                               kernel_size=patch_size,
                               stride=patch_size,
                               padding=0,
                               bias=False)

        # class embeddings and positional embeddings
        scale = inplanes**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(inplanes))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(scale * torch.randn(
                self.grid_size[0] * self.grid_size[1] + 1, inplanes))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1],\
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(torch.zeros(
                self.grid_size[0] * self.grid_size[1] + 1, inplanes),
                                                     requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(inplanes,
                                                     self.grid_size[0],
                                                     cls_token=True)
            self.positional_embedding.data.copy_(
                torch.from_numpy(pos_embed_type).float())

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(
            patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = nn.LayerNorm(inplanes)

        self.transformer = Transformer(
            inplanes,
            layer_nums,
            head_nums,
            mlp_ratio,
            init_value=init_value,
            use_gradient_checkpoint=use_gradient_checkpoint)

        self.ln_post = nn.LayerNorm(inplanes)

        self.proj = nn.Parameter(scale * torch.randn(inplanes, output_planes))

    def global_pool(self, x):
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, x):
        # [batch, inplanes, grid, grid]
        x = self.conv1(x)
        # [batch, inplanes, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # [batch, grid ** 2, inplanes]
        x = x.permute(0, 2, 1)

        # class embeddings and positional embeddings
        x = torch.cat([
            self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1).to(
                x.dtype), x
        ],
                      dim=1)

        # [batch, grid ** 2 + 1, inplanes]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)

        x = self.ln_pre(x)

        x = self.transformer(x)

        x = self.ln_post(x)

        pooled, tokens = self.global_pool(x)

        pooled = pooled @ self.proj

        return pooled


class TextTransformer(nn.Module):

    def __init__(self,
                 context_length=77,
                 vocab_size=49408,
                 inplanes=512,
                 layer_nums=12,
                 head_nums=8,
                 mlp_ratio=4.0,
                 output_planes=512,
                 init_value=None,
                 no_causal_mask=False,
                 pad_id=0,
                 pool_type='argmax',
                 use_gradient_checkpoint=False):
        super(TextTransformer, self).__init__()
        assert pool_type in ('first', 'last', 'argmax', 'none')
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.num_pos = context_length
        self.head_nums = head_nums
        self.pad_id = pad_id
        self.pool_type = pool_type
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.token_embedding = nn.Embedding(vocab_size, inplanes)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.num_pos, inplanes))

        self.transformer = Transformer(
            inplanes,
            layer_nums,
            head_nums,
            mlp_ratio=mlp_ratio,
            init_value=init_value,
            use_gradient_checkpoint=use_gradient_checkpoint)

        self.ln_final = nn.LayerNorm(inplanes)

        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer('attn_mask',
                                 self.build_causal_mask(),
                                 persistent=False)

        self.text_projection = nn.Linear(inplanes, output_planes)

        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (inplanes**-0.5) * ((2 * layer_nums)**-0.5)
        attn_std = inplanes**-0.5
        fc_std = (2 * inplanes)**-0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if isinstance(self.text_projection, nn.Linear):
            nn.init.normal_(self.text_projection.weight, std=inplanes**-0.5)
            if self.text_projection.bias is not None:
                nn.init.zeros_(self.text_projection.bias)
        else:
            nn.init.normal_(self.text_projection, std=inplanes**-0.5)

    def build_causal_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        # zero out the lower diagonal
        mask.triu_(1)

        return mask

    def build_cls_mask(self, text):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
        additive_mask = torch.empty(cls_mask.shape, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)

        return additive_mask

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

    def forward(self, text):
        seq_len = text.shape[1]

        # [batch_size, n_ctx, d_model]
        x = self.token_embedding(text)
        attn_mask = self.attn_mask

        x = x + self.positional_embedding[:seq_len]
        x = self.transformer(x, attn_mask=attn_mask)

        # x.shape = [batch_size, n_ctx, inplanes]
        x = self.ln_final(x)
        pooled, tokens = self.text_global_pool(x,
                                               text,
                                               pool_type=self.pool_type)

        if isinstance(self.text_projection, nn.Linear):
            pooled = self.text_projection(pooled)
        else:
            pooled = pooled @ self.text_projection

        return pooled
