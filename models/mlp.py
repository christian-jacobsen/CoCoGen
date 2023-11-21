import torch
from torch import nn
from .embeddings import PositionalEmbedding
from einops import rearrange, repeat
from functools import partial

from utils import zero_module

class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_channels, dropout=0,
                 skip_scale=1, adaptive_scale=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = time_emb_channels
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.linear = nn.Linear(out_channels, out_channels)
        self.affine = nn.Linear(time_emb_channels, out_channels*(2 if adaptive_scale else 1))

    def forward(self, x, time_emb=None):
        #print(x.shape, emb.shape)
        orig = x
        params = nn.functional.silu(self.affine(time_emb).to(x.dtype))
        if self.adaptive_scale:
            scale, shift = params.chunk(2, dim=-1)
            x = nn.functional.silu(torch.addcmul(shift, x, scale+1))
        else:
            x = nn.functional.silu(x.add_(params))

        x = self.linear(nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(orig)
        x = x * self.skip_scale

        return x

class CondResNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_channels, cond_emb_channels, dropout=0,
                 skip_scale=1, adaptive_scale=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = time_emb_channels
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.linear = nn.Linear(out_channels, out_channels)
        self.affine = nn.Linear(time_emb_channels+cond_emb_channels, out_channels*(2 if adaptive_scale else 1))

    def forward(self, x, time_emb=None, cond_emb=None):
        #print(x.shape, emb.shape)
        orig = x
        emb = torch.cat((time_emb, cond_emb), dim = -1)
        params = nn.functional.silu(self.affine(emb).to(x.dtype))
        if self.adaptive_scale:
            scale, shift = params.chunk(2, dim=-1)
            x = nn.functional.silu(torch.addcmul(shift, x, scale+1))
        else:
            x = nn.functional.silu(x.add_(params))

        x = self.linear(nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(orig)
        x = x * self.skip_scale

        return x

class ResNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                label_dim           = 0,        # Number of class labels, 0 = unconditional.
                augment_dim         = 0,        # Augmentation label dimensionality, 0 = no augmentation.
                model_channels      = 128,      # Channel multiplier.
                channel_mult        = [1,1,1,1],# Channel multiplier for each resblock layer.
                channel_mult_emb    = 4,
                num_blocks          = 4,        # Number of resblocks(mid) per level.
                dropout             = 0.,      # Dropout rate.
                emb_type            = "sinusoidal",# Timestep embedding type
                channel_mult_noise  = 1,        # Time embedding size
                ):

        super().__init__()

        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        block_kwargs = dict(dropout = dropout, skip_scale=1.0, adaptive_scale=True)

        self.map_noise = PositionalEmbedding(size=noise_channels, type=emb_type)
        self.map_layer = nn.Linear(noise_channels, emb_channels)
        #self.map_layer1 = nn.Linear(emb_channels, emb_channels)

        self.first_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList()
        cout = model_channels
        for level, mult in enumerate(channel_mult):
            for _ in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.blocks.append(ResNetBlock(cin, cout, emb_channels, **block_kwargs))
        self.final_layer = nn.Linear(cout, out_channels)

    def forward(self, x, noise_labels, class_labels=None, augment_labels=None):
        # Mapping
        emb = self.map_noise(noise_labels)
        #emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # why swap emb (sin/cos)?
        emb = nn.functional.silu(self.map_layer(emb))
        #emb = nn.functional.silu(self.map_layer1(emb))
        x = self.first_layer(x)
        for block in self.blocks:
            x = block(x, emb)
        x = self.final_layer(nn.functional.silu(x))
        return x

class CFGResNet(torch.nn.Module):
    # https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py
    def __init__(self, in_channels, out_channels, cond_size,
                model_channels      = 128,      # Channel multiplier.
                channel_mult        = [1,1,1,1],# Channel multiplier for each resblock layer.
                channel_mult_emb    = 4,
                num_blocks          = 4,        # Number of resblocks(mid) per level.
                dropout             = 0.,      # Dropout rate.
                emb_type            = "sinusoidal",# Timestep embedding type
                channel_mult_time  = 1,        # Time embedding size
                channel_mult_cond   = 1,        # Conditional embedding size
                cond_drop_prob      = 0.0      # Probability of using null emb
                ):

        super().__init__()

        emb_channels = model_channels * channel_mult_emb
        time_channels = model_channels * channel_mult_time
        cond_channels = model_channels * channel_mult_cond
        block_kwargs = dict(dropout = dropout, skip_scale=1.0, adaptive_scale=True)

        self.null_emb = nn.Parameter(torch.randn(emb_channels))
        self.cond_size = cond_size
        self.cond_drop_prob = cond_drop_prob

        self.map_time = PositionalEmbedding(size=time_channels, type=emb_type)
        self.map_cond = PositionalEmbedding(size=cond_channels, type=emb_type)
        self.map_time_layer = nn.Linear(time_channels, emb_channels)
        self.map_cond_layer = nn.Linear(cond_channels, emb_channels)

        self.first_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList()
        cout = model_channels
        for level, mult in enumerate(channel_mult):
            for _ in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.blocks.append(CondResNetBlock(cin, cout, emb_channels, cond_channels, **block_kwargs))
        self.final_layer = nn.Linear(cout, out_channels)

    def prob_mask_like(self, shape, prob, device):
        if prob == 1:
            return torch.ones(shape, device = device, dtype = torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device = device, dtype = torch.bool)
        else:
            return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

    def forward(self, x, cond, time, context_mask=None, cond_drop_prob=0, cond_scale = 1., rescaled_phi = 0., sampling=False):
        if sampling:
            logits =  self._forward(x, cond, time, context_mask=None, cond_drop_prob=0.)
            if cond_scale == 1:
                return logits
            null_logits =  self._forward(x, cond, time, context_mask=None, cond_drop_prob=1.)
            scaled_logits = null_logits + (logits - null_logits) * cond_scale

            if rescaled_phi == 0.:
                return scaled_logits

            std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
            rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

            return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)
        else:
            return self._forward(x, cond, time, context_mask, cond_drop_prob)

    def _forward(self, x, cond, time, context_mask=None, cond_drop_prob=None):
        # context_mask dummy var
        batch_size = x.shape[0]
        # Mapping
        time_emb = self.map_time(time)
        cond_emb = self.map_cond(cond)
        #emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # why swap emb (sin/cos)?
        time_emb = nn.functional.silu(self.map_time_layer(time_emb))
        cond_emb = nn.functional.silu(self.map_cond_layer(cond_emb))
        #emb = nn.functional.silu(self.map_layer1(emb))
        if cond_drop_prob == None:
            cond_drop_prob = self.cond_drop_prob
        if cond_drop_prob > 0:
            keep_mask = self.prob_mask_like((batch_size,), 1 - self.cond_drop_prob, device = x.device)
            null_cond_emb = repeat(self.null_emb, 'd -> b d', b = batch_size) 

            c_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                c_emb,
                null_cond_emb
            )
        x = self.first_layer(x)
        for block in self.blocks:
            x = block(x, time_emb, cond_emb)
        x = self.final_layer(nn.functional.silu(x))
        return x

class ResidualConv1DBlock(nn.Module):
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
            nn.Conv1d(in_channels, out_channels, 1, 1),
            #nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, 1),
            #nn.BatchNorm2d(out_channels),
            nn.SiLU(),
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
            return out 
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class CtrlResNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, cond_size,
                conv_channels       = 256,      # Channel multiplier.
                model_channels      = 128,      # Channel multiplier.
                channel_mult        = [1,1,1,1],# Channel multiplier for each resblock layer.
                channel_mult_emb    = 4,
                num_blocks          = 4,        # Number of resblocks(mid) per level.
                dropout             = 0.,      # Dropout rate.
                emb_type            = "sinusoidal",# Timestep embedding type
                channel_mult_time  = 1,        # Time embedding size
                channel_mult_cond   = 1,        # Conditional embedding size
                ):

        super().__init__()

        emb_channels = model_channels * channel_mult_emb
        time_channels = model_channels * channel_mult_time
        cond_channels = model_channels * channel_mult_cond
        block_kwargs = dict(dropout = dropout, skip_scale=1.0, adaptive_scale=True)

        self.null_emb = nn.Parameter(torch.randn(emb_channels))
        self.cond_size = cond_size

        self.map_time = PositionalEmbedding(size=time_channels, type=emb_type)
        self.map_cond = PositionalEmbedding(size=cond_channels, type=emb_type)
        self.map_time_layer = nn.Linear(time_channels, emb_channels)
        self.map_cond_layer = nn.Linear(cond_channels, emb_channels) # Can also be used for the controlNet (replace Vec2Img)

        self.first_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList()
        cout = model_channels

        self.ctrlBlocks = nn.ModuleList()
        self.ctrlOutBlocks = nn.ModuleList()
        self.ctrl_init_conv = zero_module(nn.Conv1d(1, conv_channels, 1))

        for level, mult in enumerate(channel_mult):
            for _ in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.blocks.append(ResNetBlock(cin, cout, emb_channels, **block_kwargs))
                self.ctrlBlocks.append(zero_module(nn.Conv1d(conv_channels, conv_channels, 1)))
                self.ctrlOutBlocks.append(nn.Sequential(zero_module(nn.Conv1d(conv_channels, 1, 1)),
                                                        nn.Flatten(),
                                                        nn.SiLU()))
        self.final_layer = nn.Linear(cout, out_channels)

    def forward(self, x, cond, time, context_mask=None, controlnet=False):
        if context_mask is not None:
            cond = cond*context_mask
        # Mapping
        time_emb = self.map_time(time)
        cond_emb = self.map_cond(cond)
        #emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # why swap emb (sin/cos)?
        time_emb = nn.functional.silu(self.map_time_layer(time_emb))
        cond_emb = nn.functional.silu(self.map_cond_layer(cond_emb))
        #emb = nn.functional.silu(self.map_layer1(emb))

        x = self.first_layer(x)
        if controlnet:
            ctrl_cond = self.ctrl_init_conv(rearrange(cond_emb, 'b d -> b 1 d'))
            for block, ctrlBlock, ctrlOutBlock in zip(self.blocks, self.ctrlBlocks, self.ctrlOutBlocks):
                x = block(x, time_emb)
                ctrl_cond = ctrlBlock(ctrl_cond)
                ctrl_out =  ctrlOutBlock(ctrl_cond)
                x += ctrl_out
        else:
            for block in self.blocks:
                x = block(x, time_emb)
        x = self.final_layer(nn.functional.silu(x))
        return x