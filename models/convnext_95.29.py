# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

# ---------- 1.  helper (put right after your imports) ----------
def channel_wta(t, k):               # t: (N,C,H,W) spike times
    _, top_idx = torch.topk(t, k, dim=1, largest=False)   # indices of earliest spikes
    mask = torch.zeros_like(t).scatter_(1, top_idx, 1.)
    return t * mask + (1-mask) * t.max()   # silence losers by pushing to t_max


def call_spiking_torch(tj, W, D_i, t_min_prev, t_min, t_max):
    """
    PyTorch version of the TF `call_spiking`.
    Inputs:
      - tj: input spike times tensor (shape [..., C_in])  # last dim is channels/features
      - W: weight matrix (C_in, C_out) or (C_out, C_in) depending on matmul orientation
      - D_i: per-output delay (broadcastable to output shape)
      - t_min_prev: previous layer min spike time (unused for simplified form but kept for API)
      - t_min, t_max: scalars or tensors
    Returns:
      - ti: output spike times clamped by t_max
    Notes:
      - Ensure matmul dimension ordering matches your usage (here we use `... @ W` with W shape (C_in, C_out))
    """

    # threshold Eq. 18
    threshold = t_max - t_min - D_i

    # ensure floating dtype
    tj = tj.to(W.dtype)
    threshold = threshold.to(W.dtype)

    # compute ti = (tj - t_min) @ W + threshold + t_min
    # if last dim of tj is C_in and W is (C_in, C_out)
    delta = tj - t_min
    ti = torch.matmul(delta, W) + threshold + t_min

    # clamp to t_max
    ti = torch.where(ti < t_max, ti, t_max)
    return ti


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        # self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        # x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # if self.gamma is not None:
        #     x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x



class SpikingBlock(nn.Module):
    """
    Spiking variant of ConvNeXt Block using TTFS (time-to-first-spike) analytic mapping.
    Inputs/outputs are spike-time tensors with shape (N, C, H, W), values in [t_min, t_max].
    """
    def __init__(self, orig_block: Block, t_min=0.0, t_max=1.0):
        super().__init__()
        # reuse the depthwise conv weights and pointwise weights from the original block
        # orig_block.dwconv is nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # orig_block.pwconv1 / pwconv2 are nn.Linear layers applied on channels-last
        self.dwconv = orig_block.dwconv  # use same module (weights)
        # we'll not use orig_block.act (GELU), instead compute spike-time mapping
        self.pw1 = orig_block.pwconv1
        self.pw2 = orig_block.pwconv2
        self.gamma = getattr(orig_block, 'gamma', None)
        self.drop_path = orig_block.drop_path if hasattr(orig_block, 'drop_path') else nn.Identity()
        self.t_min = float(t_min)
        self.t_max = float(t_max)

    def forward(self, tj):
        """
        tj: input spike times shape (N, C_in, H, W)
        returns ti: output spike times shape (N, C_out, H, W) (C_out==C_in for ConvNeXt)
        """
        # 1) Depthwise conv: we approximate by applying the conv on the spike-time map directly.
        # This treats the conv as filtering the spike-time field (heuristic).
        x = tj
        # apply depthwise conv on times (float)
        x_dw = self.dwconv(x)  # shape (N, C, H, W)





        # 2) pointwise layers: convert to (N*H*W, C) to apply linear mapping per spatial location
        N, C, H, W = x_dw.shape
        x_flat = x_dw.permute(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C_in)

        # Prepare W matrices for matmul
        # pw1: Linear(in=C, out=C*4) with weight shape (out, in)
        W1 = self.pw1.weight.t().contiguous()  # shape (C_in, C_mid)
        # No per-output delays provided -> use zeros
        D_mid = torch.zeros(W1.shape[1], device=W1.device, dtype=W1.dtype)
        # call_spiking_torch expects t_min as scalar or tensor broadcastable
        t_min = torch.tensor(self.t_min, device=x_flat.device, dtype=x_flat.dtype)
        t_max = torch.tensor(self.t_max, device=x_flat.device, dtype=x_flat.dtype)
        # compute times after first linear layer
        t_mid = call_spiking_torch(x_flat, W1, D_mid, t_min_prev=None, t_min=t_min, t_max=t_max)

        # second linear
        W2 = self.pw2.weight.t().contiguous()  # shape (C_mid, C_out)
        D_out = torch.zeros(W2.shape[1], device=W2.device, dtype=W2.dtype)
        t_out = call_spiking_torch(t_mid, W2, D_out, t_min_prev=None, t_min=t_min, t_max=t_max)

        # reshape back to (N, C, H, W)
        t_out = t_out.view(N, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # residual
        out = tj + self.drop_path(t_out)
        return out
    
    

    
class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    # def __init__(self, in_chans=3, num_classes=1000, 
    #              depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
    #              layer_scale_init_value=1e-6, head_init_scale=1.,
    #              ):
    def __init__(self, in_chans=3, num_classes=1000, depths=(3, 3, 9, 3),
                 dims=(96, 192, 384, 768), drop_path_rate=0.,
                 head_init_scale=1., layer_scale_init_value=1e-6, **kwargs):
        # ----- timm ≥ 0.9 compatibility -----
        # ----- timm ≥ 0.9 passes this key; ignore it -----
        kwargs.pop('pretrained_cfg', None)
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            # LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1])
            # x = self.norm(x)
        return x # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class SpikingBlock(nn.Module):
    """
    Spiking variant of ConvNeXt Block using TTFS analytic mapping.
    Reuses weights from an existing Block instance.
    """
    def __init__(self, orig_block: Block, t_min=0.0, t_max=1.0):
        super().__init__()
        # reuse modules (share weights)
        self.dwconv = orig_block.dwconv
        self.pw1 = orig_block.pwconv1
        self.pw2 = orig_block.pwconv2
        self.gamma = getattr(orig_block, 'gamma', None)
        self.drop_path = orig_block.drop_path if hasattr(orig_block, 'drop_path') else nn.Identity()
        self.t_min = float(t_min)
        self.t_max = float(t_max)

    def forward(self, tj):
        # tj: spike times tensor (N, C, H, W)
        x = tj
        # depthwise conv: apply to the spike-time map (heuristic)
        x = self.dwconv(x)

        # prepare for pointwise linear mapping per spatial location
        N, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C_in)

        # pw1
        W1 = self.pw1.weight.t().contiguous()  # (C_in, C_mid)
        device = x_flat.device
        dtype = x_flat.dtype
        D_mid = torch.zeros(W1.shape[1], device=device, dtype=dtype)
        t_min = torch.tensor(self.t_min, device=device, dtype=dtype)
        t_max = torch.tensor(self.t_max, device=device, dtype=dtype)
        t_mid = call_spiking_torch(x_flat, W1, D_mid, None, t_min, t_max)

        # pw2
        W2 = self.pw2.weight.t().contiguous()  # (C_mid, C_out)
        D_out = torch.zeros(W2.shape[1], device=device, dtype=dtype)
        t_out = call_spiking_torch(t_mid, W2, D_out, None, t_min, t_max)

        # reshape back
        t_out = t_out.view(N, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = tj + self.drop_path(t_out)
        return out


class ConvNeXtSpiking(ConvNeXt):
    """ConvNeXt variant that operates on TTFS spike-time inputs.
    Expects inputs of shape (N, C, H, W) containing spike times in [t_min, t_max].
    """
    def __init__(self, *args, t_min=0.0, t_max=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        # replace stage blocks with spiking blocks that share weights
        new_stages = nn.ModuleList()
        for stage in self.stages:
            sp_blocks = []
            for blk in stage:
                sp_blocks.append(SpikingBlock(blk, t_min=self.t_min, t_max=self.t_max))
            new_stages.append(nn.Sequential(*sp_blocks))
        self.stages = new_stages

    def forward_features(self, x_t):
        # x_t: spike times (N, C, H, W)
        for i in range(4):
            x_t = self.downsample_layers[i](x_t)
            x_t = self.stages[i](x_t)
        x_t = x_t.mean([-2, -1])
        return x_t

    def forward(self, x_t):
        x_pool = self.forward_features(x_t)
        # convert spike times to scores (earlier spike -> higher score)
        logits = self.head(-x_pool)
        return logits


class ConvNeXtSpiking(ConvNeXt):
    def __init__(self, *args, t_min=0.0, t_max=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # replace blocks in stages with SpikingBlock wrappers preserving weights
        for si, stage in enumerate(self.stages):
            new_blocks = []
            for b in stage:
                spb = SpikingBlock(b, t_min=t_min, t_max=t_max)
                new_blocks.append(spb)
            self.stages[si] = nn.Sequential(*new_blocks)
        # head: we need to map spike times to logits; we'll treat head as linear on features,
        # but input into head should be scores, so we convert times -> negative times as activation.
        # Keep self.head as-is but forward will convert spike-times to scores before head.
        self.t_min = float(t_min)
        self.t_max = float(t_max)

    def forward_features(self, x_t):
        # x_t: spike times tensor (N, C_in, H, W)
        # propagate
        for i in range(4):
            x_t = self.downsample_layers[i](x_t)   # these are convs on time maps
            x_t = self.stages[i](x_t)
        # global pool: average over spatial dims of spike times
        x_pool = x_t.mean([-2, -1])  # shape (N, C)
        return x_pool

    def forward(self, x_t):
        # x_t: spike times in [t_min, t_max]
        x_pool = self.forward_features(x_t)  # spike times per channel
        # convert times to scores: earlier spike -> higher score; simple mapping s = -t
        logits = self.head(-x_pool)
        return logits
    


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
