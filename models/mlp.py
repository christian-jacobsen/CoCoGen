import torch
from torch import nn
from .embeddings import PositionalEmbedding
from einops import rearrange, repeat
from functools import partial

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.ff = nn.Linear(size, size)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))

class Block_SS(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.ff = nn.Linear(size, size)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift=None):
        if scale_shift is not None:
            scale, shift = scale_shift
            return (scale + 1) * x + shift + self.act(self.ff(x))
        return x + self.act(self.ff(x))

class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb, scale=25.0)#GaussianFourierProjection(emb_size)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x

class MLP_SS(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.time_mlp = nn.Sequential(PositionalEmbedding(emb_size, time_emb, scale=25.0),
                                      nn.SiLU(), nn.Linear(emb_size, 2*emb_size))
        self.input_mlp = nn.Linear(2, hidden_size)
        #self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        #concat_size = 3
        #layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        self.blocks = nn.ModuleList([Block_SS(hidden_size) for _ in range(hidden_layers)])
        self.final_layer = nn.Linear(hidden_size, 2)

    def forward(self, x, t):
        x = self.input_mlp(x)
        x = nn.SiLU()(x)
        #x2_emb = self.input_mlp2(x[:, 1])
        temb = self.time_mlp(t)
        scale_shift = temb.chunk(2, dim=-1)
        for block in self.blocks:
            x = block(x, scale_shift)
        x = self.final_layer(x)
        return x

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
    def __init__(self, in_channels, out_channels,
                label_dim           = 0,        # Number of class labels, 0 = unconditional.
                augment_dim         = 0,        # Augmentation label dimensionality, 0 = no augmentation.
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

    def forward_with_cond_scale(self, *args, cond_scale = 1., rescaled_phi = 0., **kwargs):
        logits =  self.forward(*args, cond_drop_prob=0., **kwargs)
        if cond_scale == 1:
            return logits
        null_logits =  self.forward(*args, cond_drop_prob=1., **kwargs)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if rescaled_phi == 0.:
            return scaled_logits

        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

    def forward(self, x, time, cond, context_mask=None, cond_drop_prob=None, sampling=False, class_labels=None, augment_labels=None):
        batch_size = x.shape[0]
        if context_mask is not None:
            cond = cond*context_mask
        # Mapping
        time_emb = self.map_time(time)
        cond_emb = self.map_cond(cond)
        #emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # why swap emb (sin/cos)?
        time_emb = nn.functional.silu(self.map_time_layer(time_emb))
        cond_emb = nn.functional.silu(self.map_cond_layer(cond_emb))
        #emb = nn.functional.silu(self.map_layer1(emb))
        if not sampling:
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