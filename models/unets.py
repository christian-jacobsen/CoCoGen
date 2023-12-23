''' 
Models for noise prediction in score-matching generative models
Author: Christian Jacobsen, University of Michigan 2023
'''

from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import sys
import pytorch_lightning as pl
import os
import os.path as osp
import argparse

from einops import rearrange, repeat

sys.path.append("/home/csjacobs/git/diffusionPDE")

from utils import instantiate_from_config, zero_module

class GroupNorm(torch.nn.Module):
    #https://github.com/Newbeeer/pfgmpp/blob/d57c1ee4488d8e4064a5b9c8548792f00395aa8b/training/networks.py#L114
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

class ResConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding =1, time_emb_dim=None, cond_emb_dim=None, 
                 dropout=0, skip_scale=1, adaptive_scale=True, groups=8, normalization=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_dim = time_emb_dim
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        if normalization:
            self.norm1 = GroupNorm(num_channels=in_channels, num_groups=groups) 
            self.norm2 = GroupNorm(num_channels=out_channels, num_groups=groups)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding) 
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.map_cond = nn.Linear(int(time_emb_dim)+int(cond_emb_dim), out_channels*(2 if adaptive_scale else 1))
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond_emb):
        orig = x
        x = self.conv1(nn.functional.silu(self.norm1(x)))
        params = self.map_cond(cond_emb).to(x.dtype)
        params = rearrange(params, 'b c -> b c 1 1')
        if self.adaptive_scale:
            scale, shift = params.chunk(2, dim=1)
            x = nn.functional.silu(torch.addcmul(shift, self.norm2(x), scale+1))
        else:
            x = nn.functional.silu(self.norm2(x.add_(params)))

        x = self.conv2(nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.res_conv(orig))
        x = x * self.skip_scale

        return x

def Upsample(in_channels, out_channels, kernel_size=2, stride=2, padding=0):
    '''
    Input: (N, C_in, H_in, W_in)
    H_out = ((H_in - 1)*stride[0] - 2*padding[0] + dilation[0]*(kernel_size[0]-1) + output_padding[0] + 1)
    W_out = ((W_in - 1)*stride[1] - 2*padding[1] + dilation[1]*(kernel_size[1]-1) + output_padding[1] + 1)
    '''
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)

def Downsample(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    '''
    Input: (N, C_in, H_in, W_in)
    H_out = ((H_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1)/stride[0] + 1) 
    W_out = ((W_in + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1)/stride[1] + 1) 
    '''
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

class CFGUNet(nn.Module):
    #https://github.com/Newbeeer/pfgmpp/blob/d57c1ee4488d8e4064a5b9c8548792f00395aa8b/training/networks.py#L250
    def __init__(self,
        data_size,                                # Dimensionality of the data. 
        in_channels         = 3,            # Number of color channels at input.
        out_channels        = 3,            # Number of color channels at output.
        kernel_size         = 3,            # Kernel size for convolution block.
        padding             = 1,            # Padding for convolution block.
        cond_size           = 0,            # Number of condition, 0 = unconditional.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        #attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        cond_drop_prob      = 0,            # Dropout probability of conditions for classifier-free guidance.
        #emb_type            = 'sinusoidal', # Type of positional embedding to use for the image.
        dim_mult_time       = 1,            # Time embedding multiplier for the noise level.
        adaptive_scale      = True,         # Scale shift
        skip_scale          = 1,            # Scale of the residual connection.
        groups              = 8            # Number of groups for group normalization.
        ):
        super().__init__()

        self.cond_drop_prob = cond_drop_prob
        emb_channels = model_channels * channel_mult_emb
        time_dim = model_channels * dim_mult_time
        self.data_size = data_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.null_emb = nn.Parameter(torch.randn(time_dim))
        self.map_time = EmbedFC(1, time_dim)
        self.map_cond = EmbedFC(cond_size, time_dim) if cond_size else None
        self.map_time_layer = nn.Linear(time_dim, emb_channels)
        self.map_cond_layer = nn.Linear(time_dim, emb_channels) if cond_size else None

        block_kwargs = dict(kernel_size=kernel_size, padding=padding, time_emb_dim = emb_channels, cond_emb_dim = emb_channels, dropout = dropout, skip_scale=skip_scale,
                        adaptive_scale=adaptive_scale, groups=groups, normalization=True)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        #caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = self.data_size >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}_init_conv'] = nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=3, padding=1)
            else:
                self.enc[f'{res}_conv_down'] = Downsample(in_channels=cout, out_channels=cout)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                #attn = (res in attn_resolutions)
                self.enc[f'{res}_conv_block{idx}'] = ResConv2DBlock(in_channels=cin, out_channels=cout, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = self.data_size >> level
            if level == len(channel_mult) - 1:
                #self.dec[f'{res}_in0'] = ResConv2DBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}_in0'] = ResConv2DBlock(in_channels=cout, out_channels=cout, **block_kwargs)
                self.dec[f'{res}_in1'] = ResConv2DBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = Upsample(in_channels=cout, out_channels=cout)
            for idx in range(num_blocks+1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                #attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}_block{idx}'] = ResConv2DBlock(in_channels=cin, out_channels=cout, **block_kwargs)
        self.final_norm = GroupNorm(num_channels=cout, num_groups=groups, eps=1e-5)
        self.final_conv = nn.Conv2d(in_channels=cout, out_channels=out_channels, kernel_size=3, padding=1)

    def prob_mask_like(self, shape, prob, device):
        if prob == 1:
            return torch.ones(shape, device = device, dtype = torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device = device, dtype = torch.bool)
        else:
            return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

    def forward(self, x, cond, time, context_mask, cond_drop_prob = None):
        # Mapping.
        batch_size = x.shape[0]
        # Mapping
        time_emb = self.map_time(time)
        #emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # why swap emb (sin/cos)?
        time_emb = nn.functional.silu(self.map_time_layer(time_emb))
        if self.map_cond is not None:
            cond_emb = self.map_cond(cond)
            cond_emb = nn.functional.silu(self.map_cond_layer(cond_emb.reshape(cond_emb.shape[0], -1)))
            emb = torch.cat((time_emb, cond_emb), dim=-1)
            if cond_drop_prob == None:
                cond_drop_prob = self.cond_drop_prob
            if cond_drop_prob > 0:
                keep_mask = self.prob_mask_like((batch_size,), 1 - cond_drop_prob, device = x.device)
                null_cond_emb = repeat(self.null_emb, 'd -> b d', b = batch_size) 

                cond_emb = torch.where(
                    rearrange(keep_mask, 'b -> b 1'),
                    cond_emb,
                    null_cond_emb
                )
        else:
            emb = time_emb

        # Encoder.
        skips = []
        for name, block in self.enc.items():
            x = block(x, emb) if isinstance(block, ResConv2DBlock) else block(x)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if x.shape[1] != block.in_channels:
                #print('x shape: ', x.shape)
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb) if isinstance(block, ResConv2DBlock) else block(x)

        x = self.final_conv(self.final_norm(x))

        return x

class PFGMPPUNet(torch.nn.Module):
    def __init__(self,
        dim,                                # Dimensionality of the data. 
        in_channels         = 3,            # Number of color channels at input.
        out_channels        = 3,            # Number of color channels at output.
        kernel_size         = 3,            # Kernel size for convolution block.
        padding             = 1,            # Padding for convolution block.
        cond_size           = 0,            # Number of condition, 0 = unconditional.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        #attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        cond_drop_prob      = 0,            # Dropout probability of conditions for classifier-free guidance.
        #emb_type            = 'sinusoidal', # Type of positional embedding to use for the image.
        dim_mult_time       = 1,            # Time embedding multiplier for the noise level.
        adaptive_scale      = True,         # Scale shift
        skip_scale          = 1,            # Scale of the residual connection.
        groups              = 8,            # Number of groups for group normalization.
        pfgmpp=False,
        D = 128,
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'CFGUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.D = D
        self.N = in_channels*dim*dim
        self.label_dim = cond_size
        self.pfgmpp = pfgmpp
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        ###########
        self.model = globals()[model_type](dim, in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=kernel_size, padding=padding, cond_size=cond_size,
                                           model_channels=model_channels, channel_mult=channel_mult, channel_mult_emb=channel_mult_emb, num_blocks=num_blocks,
                                           dropout=dropout, cond_drop_prob=cond_drop_prob, dim_mult_time=dim_mult_time, adaptive_scale=adaptive_scale, skip_scale=skip_scale, groups=groups,
                                           **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False,  **model_kwargs):

        x = x.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim],
                                                                    device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.model((x_in).to(dtype), class_labels, c_noise.flatten(), context_mask=None, **model_kwargs)

        # disabled due to mixed precision training
        #assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class CondVec2Img(nn.Module):
    def __init__(self, cond_size, data_size, channels):
        super(CondVec2Img, self).__init__()
        '''
        encode vector of size cond_size to data_size x data_size
        '''
        self.cond_size = cond_size
        self.data_size = data_size
        self.channels = channels

        # try to keep trainable parameter count low
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(cond_size, cond_size)
        self.fc2 = nn.Linear(cond_size, cond_size)
        self.fc3 = nn.Linear(cond_size, data_size**2)
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 1, self.data_size, self.data_size)
        return self.conv1(x)


class UNET1(nn.Module):
    def __init__(self, in_channels, n_feat = 256, pool_size=4, data_size=64, cond_size=16, controlnet=None):
        super(UNET1, self).__init__()

        if controlnet is None:
            self.controlnet = False
        else:
            self.controlnet = controlnet.use # True or False for using a ControlNet

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.data_size = data_size
        self.cond_size = cond_size

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(pool_size), nn.GELU())

        if self.controlnet:
            self.control_condition_encoder = instantiate_from_config(controlnet.condition_encoder) #CondVec2Img(cond_size, data_size, in_channels)
            self.zero_conv0 = zero_module(nn.Conv2d(in_channels, in_channels, 1))
            self.control_init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
            self.zero_conv1 = zero_module(nn.Conv2d(n_feat, n_feat, 1))
            self.control_down1 = UnetDown(n_feat, n_feat)
            self.zero_conv2 = zero_module(nn.Conv2d(n_feat, n_feat, 1))
            self.control_down2 = UnetDown(n_feat, 2*n_feat)
            self.zero_conv3 = zero_module(nn.Conv2d(2*n_feat, 2*n_feat, 1))

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(self.cond_size, 2*n_feat)
        self.contextembed2 = EmbedFC(self.cond_size, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, data_size // (4*pool_size), data_size // (4*pool_size)), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on

        #print('data shape: ', x)

        initconv = self.init_conv(x)
        down1 = self.down1(initconv)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        if self.controlnet:
            c_img = self.zero_conv0(self.control_condition_encoder(c)) + x
            control_initconv = self.zero_conv1(self.control_init_conv(c_img))
            control_down1 = self.zero_conv2(self.control_down1(control_initconv))
            control_down2 = self.zero_conv3(self.control_down2(control_down1))

        # convert context to one hot embedding
        #c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        # mask out context if context_mask == 1
        #context_mask = context_mask[:, None]
        #context_mask = context_mask.repeat(1,self.n_classes)
        #context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        c = c * 0#context_mask (unconditional model dummy input)
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        if self.controlnet:
            up1 = self.up0(hiddenvec) # no unet input
            up2 = self.up1(cemb1*up1 + temb1, down2 + control_down2)
            up3 = self.up2(cemb2*up2 + temb2, down1 + control_down1)
            out = self.out(torch.cat((up3, initconv + control_initconv), 1))

        else:
            up1 = self.up0(hiddenvec)
            # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
            up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
            up3 = self.up2(cemb2*up2+ temb2, down1)
            out = self.out(torch.cat((up3, initconv), 1))
        return out

