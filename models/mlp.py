import torch
from torch import nn
from .embeddings import PositionalEmbedding

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
    def __init__(self, in_channels, out_channels, emb_channels, dropout=0,
                 skip_scale=1, adaptive_scale=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(out_channels, out_channels)
        self.affine = nn.Linear(emb_channels, out_channels*(2 if adaptive_scale else 1))

    def forward(self, x, emb):
        #print(x.shape, emb.shape)
        orig = x
        params = self.affine(emb).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(2, dim=-1)
            x = nn.functional.silu(torch.addcmul(shift, x, scale+1))
        else:
            x = nn.functional.silu(x.add_(params))

        x = self.linear1(nn.functional.dropout(x, p=self.dropout, training=self.training))
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
        self.map_layer0 = nn.Linear(noise_channels, emb_channels)
        self.map_layer1 = nn.Linear(emb_channels, emb_channels)

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
        # is swap sin/cos needed?
        emb = nn.functional.silu(self.map_layer0(emb))
        emb = nn.functional.silu(self.map_layer1(emb))
        #print('ini shape', x.shape)
        x = self.first_layer(x)
        #print('first layer', x.shape)
        for block in self.blocks:
            x = block(x, emb)
        #    print(x.shape)
        x = self.final_layer(x)
        #print('final layer', x.shape)
        return x