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

sys.path.append("/home/csjacobs/git/diffusionPDE")

from utils import instantiate_from_config

def zero_module(module):
    '''
    zero the parameters of a module and return it
    '''
    for p in module.parameters():
        p.detach().zero_()
    return module

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

