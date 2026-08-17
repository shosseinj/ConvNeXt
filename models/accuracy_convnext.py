"""Compact, fully dense ConvNeXt used by the accuracy-oriented experiments."""

from __future__ import annotations

import torch
from torch import nn
from timm.layers import DropPath, trunc_normal_


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * (x - mean) / torch.sqrt(variance + self.eps) + self.bias[:, None, None]


class DenseConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, drop_path: float = 0.0,
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        x = (self.gamma * x).permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class AccuracyConvNeXt(nn.Module):
    def __init__(self, num_classes: int, depths=(2, 2, 6, 2),
                 dims=(96, 192, 384, 768), kernel_size: int = 3,
                 drop_path_rate: float = 0.1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.depths = tuple(depths)
        self.dims = tuple(dims)
        self.kernel_size = int(kernel_size)
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, dims[0], 3, stride=1, padding=1),
                          LayerNorm(dims[0], data_format="channels_first"))
        ])
        for index in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[index], data_format="channels_first"),
                nn.Conv2d(dims[index], dims[index + 1], 3, stride=2, padding=1),
            ))
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        offset = 0
        self.stages = nn.ModuleList()
        for stage, depth in enumerate(depths):
            self.stages.append(nn.Sequential(*[
                DenseConvNeXtBlock(dims[stage], kernel_size, rates[offset + block])
                for block in range(depth)
            ]))
            offset += depth
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
        return self.norm(x.mean((-2, -1)))

    def forward(self, x):
        return self.head(self.forward_features(x))


def architecture_metadata(model: AccuracyConvNeXt):
    return {
        "model_type": "fully_dense_ann",
        "num_classes": model.num_classes,
        "depths": list(model.depths),
        "dims": list(model.dims),
        "kernel_size": model.kernel_size,
        "residual_operator": "sum",
        "pw1_mode": "dense",
        "pw2_mode": "dense",
        "dwconv_mode": "dense",
        "downsample_mode": "dense",
        "normalization": "layernorm",
        "activation": "gelu",
    }
