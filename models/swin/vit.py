from functools import partial

import numpy
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Vit(nn.Module):
    def __init__(self, embed_dim=1024, depth=12, num_patches=21, mlp_ratio=4., num_heads=8, qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, dims=None,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_cfg=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        #self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches * self.num_patches, self.embed_dim))
        self.num_stages = self.depth
        self.out_indices = tuple(range(self.num_stages))
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.layer = nn.Sequential(*[
            Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i])
            for i in range(self.depth)])
        #trunc_normal_(self.pos_embed, std=.02)
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        #x = x + self.pos_embed
        output = self.layer(x)
        output = output.reshape(B, C, H, W)
        return output
if __name__ == '__main__':
    input = torch.randn(1, 441, 1024)
    device = torch.device('cuda:0')
    # print(model)
    model = Vit()
    model.to(device)
    input = input.to(device)
    output = model(input)
    print(output.shape)


