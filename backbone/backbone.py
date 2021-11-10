
from collections import OrderedDict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from torch import nn, einsum
from einops import rearrange

class InterpolateModule(nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__()
		self.args = args
		self.kwargs = kwargs

	def forward(self, x):
		return F.interpolate(x, *self.args, **self.kwargs)

class ShortCut(nn.Module):
    def __init__(self, scopes):
        super().__init__()
        self.scopes = scopes

    def prepare(self, features={}):
        self.to_cat = [features[scope] for scope in self.scopes]

    def forward(self, x):
        return torch.cat([x]+self.to_cat, dim=1)

    def __repr__(self):
        fmtstr = self._get_name() + '('
        fmtstr += 'ShortCut Scopes {}'.format(self.scopes)
        fmtstr += ')'
        return fmtstr

class DoNothing(nn.Module):
    def __init__(self, *args):
        super().__init__()
    def forward(self, x):
        return x

# -----------------------------------------------------------------------------------------

class NonLocal(nn.Module):
    """ Nonlocal networks. """
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.to_q = nn.Conv2d(dim_in, dim_out, 1, bias=False)
        self.to_k = nn.Conv2d(dim_in, dim_out, 1, bias=False) 
        self.to_v = nn.Conv2d(dim_in, dim_out, 1, bias=False)
 
    def forward(self, x):
        b, c, hh, ww = x.shape

        q = self.to_q(x).flatten(2)
        k = self.to_k(x).flatten(2)
        v = self.to_v(x).flatten(2)

        att = einsum('b c k, b c q -> b k q', k, q)
        att = att.softmax(dim=-1)

        Y = einsum('b k q, b c k -> b c q', att, v)
        out = rearrange(Y, 'b c (hh ww) -> b c hh ww', hh=hh, ww=ww)
        out = torch.add(out, x)
        return out

class DSAGPredictor(nn.Module):
    """ Distance-Specific Attention-Guided Predictor. """
    def __init__(self, dim_in, dim_out, maxl=8):
        super().__init__()

        self.to_q = nn.Conv2d(dim_in, dim_out, 1, bias=False)
        self.to_k = nn.Conv2d(dim_in, dim_out, 1, bias=False) 
        self.to_v = nn.Conv2d(dim_in, dim_out, 1, bias=False)

        self.maxl = maxl
        self.embd = torch.nn.Parameter(torch.zeros(2, 2*self.maxl+1, dim_in).normal_(0,1))
        self.proj = nn.Conv2d(dim_in+1, dim_out, 1, bias=False)

    def forward(self, x, dist, isWithin=None):
        b, c, hh, ww = x.shape

        q = self.to_q(x).flatten(2)
        k = self.to_k(x).flatten(2)
        v = self.to_v(x).flatten(2)

        att = einsum('b c k, b c q -> b k q', k, q)
        att = att.softmax(dim=-1)
        Y = einsum('b k q, b c k -> b c q', att, v)
        out = rearrange(Y, 'b c (hh ww) -> b c hh ww', hh=hh, ww=ww)
        out = torch.add(out, x)

        # Get the relative distance embedding according to 
        # the relative distance and isWithin flag.
        rel_emb = self.embd[isWithin, dist+self.maxl].unsqueeze(1)
        pos = torch.einsum('z u c, b c n -> b z u n', rel_emb, x.flatten(2))
        pos = rearrange(pos, 'b z u (hh ww) -> (b z) u hh ww', hh=hh, ww=ww)
        out = self.proj(torch.cat([out.repeat(pos.size(0), 1,1,1), pos], 1))        

        return out

class MLPPredictor(nn.Module):
    """ Multi-layer Perception Predictor. """
    def __init__(self, in_dim, hidden_dim, out_dim): 
        # bottleneck structure
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Conv2d(hidden_dim, out_dim, 1, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 
        
# ---------------------------------------------------------------------------------------------------------------
""" Adapted from torchvision.models.resnet """
from typing import Type, Any, Callable, Union, List, Optional

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# -----------------------------------------------------------------------------------------

""" Function to build the network according to the config. 
    Each layer is a sequence of conv layers with bn and relu.
    Adapted from torchvision.models.vgg.make_layers. 
    
    - args:
        in_channels: input channels;
        conf: config (see config.py);
        actv_func: activation function set in config.py;
        norm_func: normalization function set in config.py;
    """
    
def make_net(in_channels, conf, actv_func, norm_func):
    if norm_func == None: 
        norm_func = DoNothing

    def _make_layer(layer_cfg):
        nonlocal in_channels

        if isinstance(layer_cfg[0], str):
            layer_mode = layer_cfg[0]

            if layer_mode == 'sct':
                layer = ShortCut(layer_cfg[1])
                num_channels = in_channels + sum(layer_cfg[2])
            
            elif layer_mode == 'mxp':
                if 'stride' not in layer_cfg[2]:
                    _stride = layer_cfg[1]
                else:
                    _stride = layer_cfg[2]['stride']
                    del layer_cfg[2]['stride']
                layer = nn.MaxPool2d(kernel_size=layer_cfg[1], stride=_stride, **layer_cfg[2])
                num_channels = in_channels

            elif layer_mode == 'avp':
                layer = nn.AvgPool2d(kernel_size=layer_cfg[1], stride=layer_cfg[2], **layer_cfg[3])
                num_channels = in_channels

            elif layer_mode == 'dro':
                layer = nn.Dropout(p=layer_cfg[1], **layer_cfg[2])
                num_channels = in_channels

            elif layer_mode == 'lin':
                layer = nn.Linear(in_channels, layer_cfg[1], **layer_cfg[2])
                num_channels = layer_cfg[1]
                layer = [layer, norm_func(num_channels), actv_func]

            elif layer_mode == 'res':
                downsample = None
                if 'stride' in layer_cfg[2]:
                    if layer_cfg[2]['stride'] != 1:
                        downsample = nn.Sequential(
                                conv1x1(in_channels, layer_cfg[1], layer_cfg[2]['stride']),
                                norm_func(layer_cfg[1]),
                                )
                
                layer = [BasicBlock(in_channels, layer_cfg[1], downsample=downsample, **layer_cfg[2])]
                num_channels = layer_cfg[1]    

        else:
            num_channels = layer_cfg[0]
            kernel_size  = layer_cfg[1]
            
            if 'padding' in layer_cfg[2]:
                padding = layer_cfg[2]['padding']
                del layer_cfg[2]['padding']
            else:
                padding = (kernel_size-1) // 2 

            if kernel_size > 0:
                layer = nn.Conv2d(in_channels, num_channels, kernel_size, 
                                  padding=padding, bias=True, **layer_cfg[2])
                layer = [layer, norm_func(num_channels), actv_func]

            else:
                if num_channels is None:
                    layer = InterpolateModule(scale_factor=-kernel_size, mode='nearest',
                                              **layer_cfg[2])
                else:
                    layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, 
                                               padding=padding, output_padding=1, **layer_cfg[2])
                    layer = [layer, norm_func(num_channels), actv_func]

        in_channels = num_channels if num_channels is not None else in_channels

        return layer if isinstance(layer, list) else [layer]

    net = OrderedDict()
    for scope, layer_cfg in conf.items():
        layer = sum([_make_layer(module_cfg) for module_cfg in layer_cfg], [])

        net.update({scope: nn.Sequential(*layer)})
    return nn.ModuleDict(net), in_channels