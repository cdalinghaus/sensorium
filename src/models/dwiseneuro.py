import math
import functools
from typing import Callable

import torch
from torch import nn

from functools import reduce
import operator

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer




class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

from functools import partial
import numpy as np
import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed



class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=8, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.img_size = img_size
        self.patch_size = patch_size

        self.tubelet_size = 5, 4, 8, 8
        self.num_patches = 256
        self.tubelet_projection = torch.nn.Linear(reduce(operator.mul, self.tubelet_size, 1), embed_dim)
        self.final_inverse_projection = torch.nn.Linear(192, 16*8*8)

        #--self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        #--num_patches = self.patch_embed.num_patches


        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        #self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        #self.decoder_blocks = nn.ModuleList([
        #    Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #    for i in range(decoder_depth)])

        #self.decoder_norm = norm_layer(decoder_embed_dim)
        #self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        self.encoder_pred = nn.Linear(embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        #self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        #pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.pos_embed.shape[-1]**.5), cls_token=True)
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed.requires_grad = True

        #decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        #self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        #print("forward", x.shape, noise.shape)
        # get patches form the original noise

        # Kernel sizes for channel, time, width, height
        kc, kt, kw, kh = self.tubelet_size  # Example kernel sizes

        # Strides for channel, time, width, height
        dc, dt, dw, dh = self.tubelet_size  # Example strides

        # Unfolding the tensor
        patches = x.unfold(1, kc, dc).unfold(2, kt, dt).unfold(3, kw, dw).unfold(4, kh, dh)

        # Reshape the patches to merge all patch dimensions into one dimension
        patches = patches.contiguous().view(patches.size(0), -1, kc, kt, kw, kh)

        # Flatten
        N, P = patches.shape[:2]
        x = patches.reshape(N, P, -1)

        x = self.tubelet_projection(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = self.encoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        N, C = x.shape[:2]
        #print(x.shape)
        x = self.final_inverse_projection(x.reshape(N, C, -1))
        x = x.reshape(N, C, 16, 8, 8)

        return x#, patches

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=8, embed_dim=768, depth=12, num_heads=12, img_size=128,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



class BatchNormAct(nn.Module):
    def __init__(self,
                 num_features: int,
                 bn_layer: Callable = nn.BatchNorm3d,
                 act_layer: Callable = nn.ReLU,
                 apply_act: bool = True):
        super().__init__()
        self.bn = bn_layer(num_features)
        self.act = act_layer() if apply_act else nn.Identity()

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        return x


class SqueezeExcite3d(nn.Module):
    def __init__(self,
                 in_features: int,
                 reduce_ratio: int = 16,
                 act_layer: Callable = nn.ReLU,
                 gate_layer: Callable = nn.Sigmoid):
        super().__init__()
        rd_channels = in_features // reduce_ratio
        self.conv_reduce = nn.Conv3d(in_features, rd_channels, (1, 1, 1), bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv3d(rd_channels, in_features, (1, 1, 1), bias=True)
        self.gate = gate_layer()

    def forward(self, x):
        x_se = x.mean((2, 3, 4), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0. and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class InvertedResidual3d(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 spatial_kernel: int = 3,
                 temporal_kernel: int = 3,
                 spatial_stride: int = 1,
                 expansion_ratio: int = 3,
                 se_reduce_ratio: int = 16,
                 act_layer: Callable = nn.ReLU,
                 bn_layer: Callable = nn.BatchNorm3d,
                 drop_path_rate: float = 0.,
                 bias: bool = False):
        super().__init__()
        self.spatial_stride = spatial_stride
        self.out_features = out_features
        mid_features = in_features * expansion_ratio
        stride = (1, spatial_stride, spatial_stride)

        # Point-wise expansion
        self.conv_pw = nn.Sequential(
            nn.Conv3d(in_features, mid_features, (1, 1, 1), bias=bias),
            BatchNormAct(mid_features, bn_layer=bn_layer, act_layer=act_layer),
        )

        # Spatial depth-wise convolution
        spatial_padding = spatial_kernel // 2
        self.spat_covn_dw = nn.Sequential(
            nn.Conv3d(mid_features, mid_features, (1, spatial_kernel, spatial_kernel),
                      stride=stride, padding=(0, spatial_padding, spatial_padding),
                      groups=mid_features, bias=bias),
            BatchNormAct(mid_features, bn_layer=bn_layer, act_layer=act_layer),
        )

        # Temporal depth-wise convolution
        temporal_padding = temporal_kernel // 2
        self.temp_covn_dw = nn.Sequential(
            nn.Conv3d(mid_features, mid_features, (temporal_kernel, 1, 1),
                      stride=(1, 1, 1), padding=(temporal_padding, 0, 0),
                      groups=mid_features, bias=bias),
            BatchNormAct(mid_features, bn_layer=bn_layer, act_layer=act_layer),
        )

        # Squeeze-and-excitation
        self.se = SqueezeExcite3d(mid_features, act_layer=act_layer, reduce_ratio=se_reduce_ratio)

        # Point-wise linear projection
        self.conv_pwl = nn.Sequential(
            nn.Conv3d(mid_features, out_features, (1, 1, 1), bias=bias),
            BatchNormAct(out_features, bn_layer=bn_layer, apply_act=False),
        )

        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.bn_sc = BatchNormAct(out_features, bn_layer=bn_layer, apply_act=False)

    def interpolate_shortcut(self, shortcut):
        _, c, t, h, w = shortcut.shape
        if self.spatial_stride > 1:
            size = (t, math.ceil(h / self.spatial_stride), math.ceil(w / self.spatial_stride))
            shortcut = nn.functional.interpolate(shortcut, size=size, mode="nearest")
        if c != self.out_features:
            tile_dims = (1, math.ceil(self.out_features / c), 1, 1, 1)
            shortcut = torch.tile(shortcut, tile_dims)[:, :self.out_features]
        shortcut = self.bn_sc(shortcut)
        return shortcut

    def forward(self, x):
        shortcut = x
        x = self.conv_pw(x)
        x = self.spat_covn_dw(x)
        x = self.temp_covn_dw(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.drop_path(x) + self.interpolate_shortcut(shortcut)
        return x


class PositionalEncoding3d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.orig_channels = channels
        channels = math.ceil(channels / 6) * 2
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_encoding", None, persistent=False)

    def get_emb(self, sin_inp):
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=0)
        return torch.flatten(emb, 0, 1)

    def create_cached_encoding(self, tensor):
        _, orig_ch, x, y, z = tensor.shape
        assert orig_ch == self.orig_channels
        self.cached_encoding = None
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", self.inv_freq, pos_x)
        sin_inp_y = torch.einsum("i,j->ij", self.inv_freq, pos_y)
        sin_inp_z = torch.einsum("i,j->ij", self.inv_freq, pos_z)
        emb_x = self.get_emb(sin_inp_x).unsqueeze(-1).unsqueeze(-1)
        emb_y = self.get_emb(sin_inp_y).unsqueeze(1).unsqueeze(-1)
        emb_z = self.get_emb(sin_inp_z).unsqueeze(1).unsqueeze(1)
        emb = torch.zeros((self.channels * 3, x, y, z), dtype=tensor.dtype, device=tensor.device)
        emb[:self.channels] = emb_x
        emb[self.channels: 2 * self.channels] = emb_y
        emb[2 * self.channels:] = emb_z
        emb = emb[None, :self.orig_channels].contiguous()
        self.cached_encoding = emb
        return emb

    def forward(self, x):
        if len(x.shape) != 5:
            raise RuntimeError("The input tensor has to be 5D")

        cached_encoding = self.cached_encoding
        if cached_encoding is None or cached_encoding.shape[1:] != x.shape[1:]:
            cached_encoding = self.create_cached_encoding(x)

        return x + cached_encoding.expand_as(x)


class ShuffleLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 groups: int = 1,
                 act_layer: Callable = nn.ReLU,
                 bn_layer: Callable = nn.BatchNorm1d,
                 drop_path_rate: float = 0.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.conv = nn.Conv1d(in_features, out_features, (1,), groups=groups, bias=False)
        self.bn = BatchNormAct(out_features, bn_layer=bn_layer, act_layer=act_layer)
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.bn_sc = BatchNormAct(out_features, bn_layer=bn_layer, apply_act=False)

    def shuffle_channels(self, x):
        if self.groups > 1:
            # Shuffle channels between groups
            b, c, t = x.shape
            x = x.view(b, self.groups, -1, t)
            x = torch.transpose(x, 1, 2)
            x = x.reshape(b, -1, t)
        return x

    def tile_shortcut(self, shortcut):
        if self.in_features != self.out_features:
            tile_dims = (1, math.ceil(self.out_features / self.in_features), 1)
            shortcut = torch.tile(shortcut, tile_dims)[:, :self.out_features]
        shortcut = self.bn_sc(shortcut)
        return shortcut

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.shuffle_channels(x)
        x = self.drop_path(x) + self.tile_shortcut(shortcut)
        return x


class Cortex(nn.Module):
    def __init__(self,
                 in_features: int,
                 features: tuple[int, ...],
                 groups: int = 1,
                 act_layer: Callable = nn.ReLU,
                 bn_layer: Callable = nn.BatchNorm1d,
                 drop_path_rate: float = 0.):
        super().__init__()
        self.layers = nn.Sequential()
        prev_num_features = in_features
        for num_features in features:
            self.layers.append(
                ShuffleLayer(
                    in_features=prev_num_features,
                    out_features=num_features,
                    groups=groups,
                    act_layer=act_layer,
                    bn_layer=bn_layer,
                    drop_path_rate=drop_path_rate,
                )
            )
            prev_num_features = num_features

    def forward(self, x):
        x = self.layers(x)
        return x


class Readout(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 groups: int = 1,
                 softplus_beta: float = 1.0,
                 drop_rate: float = 0.):
        super().__init__()
        self.out_features = out_features
        self.layer = nn.Sequential(
            nn.Dropout1d(p=drop_rate),
            nn.Conv1d(in_features,
                      math.ceil(out_features / groups) * groups, (1,),
                      groups=groups, bias=True),
        )
        self.gate = nn.Softplus(beta=softplus_beta)  # type: ignore

    def forward(self, x):
        x = self.layer(x)
        x = x[:, :self.out_features]
        x = self.gate(x)
        return x


class DepthwiseCore(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 features: tuple[int, ...] = (64, 128, 256, 512),
                 spatial_strides: tuple[int, ...] = (2, 2, 2, 2),
                 spatial_kernel: int = 3,
                 temporal_kernel: int = 3,
                 expansion_ratio: int = 3,
                 se_reduce_ratio: int = 16,
                 act_layer: Callable = nn.ReLU,
                 bn_layer: Callable = nn.BatchNorm3d,
                 drop_path_rate: float = 0.):
        super().__init__()
        num_blocks = len(features)
        assert num_blocks and num_blocks == len(spatial_strides)
        next_num_features = features[0]
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, next_num_features, (1, 1, 1), bias=False),
            BatchNormAct(next_num_features, bn_layer=bn_layer, apply_act=False),
        )

        blocks = []
        for block_index in range(num_blocks):
            num_features = features[block_index]
            spatial_stride = spatial_strides[block_index]
            if block_index < num_blocks - 1:
                next_num_features = features[block_index + 1]
            block_drop_path_rate = drop_path_rate * block_index / num_blocks

            blocks += [
                PositionalEncoding3d(num_features),
                InvertedResidual3d(
                    num_features,
                    next_num_features,
                    spatial_kernel=spatial_kernel,
                    temporal_kernel=temporal_kernel,
                    spatial_stride=spatial_stride,
                    expansion_ratio=expansion_ratio,
                    se_reduce_ratio=se_reduce_ratio,
                    act_layer=act_layer,
                    bn_layer=bn_layer,
                    drop_path_rate=block_drop_path_rate,
                    bias=False,
                )
            ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x


class DwiseNeuro(nn.Module):
    def __init__(self,
                 readout_outputs: tuple[int, ...],
                 in_channels: int = 5,
                 core_features: tuple[int, ...] = (64, 64, 64, 64, 128, 128, 128, 256, 256),
                 spatial_strides: tuple[int, ...] = (2, 1, 1, 1, 2, 1, 1, 2, 1),
                 spatial_kernel: int = 3,
                 temporal_kernel: int = 5,
                 expansion_ratio: int = 6,
                 se_reduce_ratio: int = 32,
                 cortex_features: tuple[int, ...] = (1024, 2048, 4096),
                 groups: int = 2,
                 softplus_beta: float = 0.07,
                 drop_rate: float = 0.4,
                 drop_path_rate: float = 0.1):
        super().__init__()
        act_layer = functools.partial(nn.SiLU, inplace=True)

        self.core = DepthwiseCore(
            in_channels=in_channels,
            features=core_features,
            spatial_strides=spatial_strides,
            spatial_kernel=spatial_kernel,
            temporal_kernel=temporal_kernel,
            expansion_ratio=expansion_ratio,
            se_reduce_ratio=se_reduce_ratio,
            act_layer=act_layer,
            bn_layer=nn.BatchNorm3d,
            drop_path_rate=drop_path_rate,
        )

        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        self.cortex = Cortex(
            in_features=core_features[-1],
            features=cortex_features,
            groups=groups,
            act_layer=act_layer,
            bn_layer=nn.BatchNorm1d,
            drop_path_rate=drop_path_rate,
        )

        self.readouts = nn.ModuleList()
        for readout_output in readout_outputs:
            self.readouts.append(
                Readout(
                    in_features=cortex_features[-1],
                    out_features=readout_output,
                    groups=groups,
                    softplus_beta=softplus_beta,
                    drop_rate=drop_rate,
                )
            )

    def forward(self, x: torch.Tensor, index: int | None = None) -> list[torch.Tensor] | torch.Tensor:
        # Input shape: (batch, channel, time, height, width), e.g. (32, 5, 16, 64, 64)
        x = self.core(x)  # (32, 256, 16, 8, 8)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (32, 256, 16)
        x = self.cortex(x)  # (32, 4096, 16)
        if index is None:
            return [readout(x) for readout in self.readouts]
        else:
            return self.readouts[index](x)  # (32, neurons, 16)
