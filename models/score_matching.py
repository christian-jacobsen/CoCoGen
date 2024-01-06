'''
Define the pytorch lighting score-matching generative models using SDEs
Author: Christian Jacobsen, University of Michigan 2023
'''

import torch
import pytorch_lightning as pl
import time
import sys
from einops import rearrange, repeat
from torchvision.utils import make_grid

from .pfgmpp import StackedRandomGenerator, edm_sampler

sys.path.append("/home/csjacobs/git/diffusionPDE")

from utils import instantiate_from_config

class DiffusionSDE(pl.LightningModule):
    # base class for training a score-matching objective
    def __init__(self,
                 unet_config,
                 sde_config,
                 sampler_config,
                 residual_config,
                 eps=1e-5,
                 elbo_weight=0.,
                 log_every_t=50,
                 lr=1e-4,
                 ignore_keys=[],
                 ckpt_path=None):
        super().__init__()

        self.unet = instantiate_from_config(unet_config)
        self.sde = instantiate_from_config(sde_config)
        #self.sampler = instantiate_from_config(sampler_config)
        self.residual = instantiate_from_config(residual_config)
        self.eps = eps
        self.lr = lr
        self.elbo_weight = elbo_weight
        self.log_every_t = log_every_t

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        params = torch.load(path, map_location='cpu')
        if "state_dict" in list(params.keys()):
            params = params["state_dict"]
        keys = list(params.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del params[k]

        missing, unexpected = self.load_state_dict(params, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")

    def q_sample(self, x0, t):
        # sample from the SDE at time t (assume Gaussian)
        return self.sde.forward(x0, t)

    def p_losses(self, x0, c, t):
        # compute losses (including ELBO, score-matching loss)
        x_perturbed, std, z = self.q_sample(x0=x0, t=t)
        context_mask = torch.zeros_like(c)
        score_pred = self.unet(x_perturbed, c, t, context_mask)

        loss_dict = {}

        log_prefix = 'train' if self.training else 'val'

        # score-mathcing objective function
        score_loss = torch.sum((score_pred*std + z)**2, dim=(1,2,3))

        loss_dict.update({f'{log_prefix}/loss_score': score_loss.mean()})

        lamb = self.sde.g(t)**2
        loss_vlb = lamb*score_loss
        loss_vlb = loss_vlb.mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = score_loss.mean() + self.elbo_weight*loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, c):
        # select random time t \in (eps, 1]
        t = torch.rand(x.shape[0], device=x.device) * (1. - self.eps) + self.eps
        return self.p_losses(x, c, t)

    def training_step(self, batch):
        x, c = self.get_input(batch)
        loss, loss_dict = self.forward(x, c)
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def get_input(self, batch):
        x, c = batch
        return x, c

    @torch.no_grad()
    def sample(self, batch_size=8, c=None):
        return self.sampler.forward(self.unet, self.sde, 
                                    (batch_size, self.unet.in_channels, self.unet.data_size, self.unet.data_size), 
                                    c=c,
                                    device=self.device)


    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True):
        log = dict()

        # log samples
        x0, c = self.get_input(batch)
        N = min(x0.shape[0], N)
        x0 = x0.to(self.device)[:N]
        log["inputs"] = x0

        # forward sampling (adding noise)
        forward_row = list()
        for t in range(self.sampler.num_time_steps):
            if t % self.log_every_t == 0 or t == self.sampler.num_time_steps-1:
                t = repeat(torch.tensor([t]), '1 -> b', b=N)
                t = t.to(self.device).long()
                x_perturbed, _, _ = self.q_sample(x0=x0, t=t)
                forward_row.append(x_perturbed)

        #log["diffusion_row"] = self._get_rows_from_list(forward_row)

        # backward sampling (denoising)
        start_time = time.time()
        samples, intermediates = self.sample(batch_size=N, c=c)
        end_time = time.time()
        log["samples"] = samples
        
        p = samples[:, 0, :, :]*self.residual.sigma_p + self.residual.mu_p
        k = samples[:, 1, :, :]*self.residual.sigma_k + self.residual.mu_k
        residual, _, _, _, _, _, _ = self.residual.r_diff(p, k)

        self.log("physics_residual_l2_norm", torch.mean(residual**2),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("sample time", end_time-start_time, logger=True, prog_bar=True,
                 on_step=False, on_epoch=True)

        return log

    def sample_inpaint(self, values, indices):
        # samples with inpainting
        # inputs: values -> (b, c, n, n) defines values for batch size which have measurements [non-normalized]
        #         indices -> (s, 4) # defines the indices for each of the s measurements in the batch
        #                       where last dim corresponds to indices (b, c, n, n) of the data
        batch_size = values.shape[0]
        samples = self.sampler.inpaint(self.unet, self.sde, 
                                       (batch_size, self.unet.in_channels, self.unet.data_size, self.unet.data_size),
                                       values, indices, device=self.device)
        return samples


    def _get_rows_from_list(self, samples):
        n_per_row = len(samples)
        grid = rearrange(samples, 'n b c h w -> b n c h w')
        grid = rearrange(grid, 'b n c h w -> (b n) c h w')
        grid = make_grid(grid, nrow=n_per_row)
        return grid

    def configure_optimizers(self):
        lr = self.lr
        params = list(self.unet.parameters())
        return torch.optim.AdamW(params, lr=lr)

class DiffusionPFGMPP(pl.LightningModule):
    # base class for training a score-matching objective
    def __init__(self,
                 unet_config,
                 loss_fn_config,
                 sampler_config,
                 residual_config,
                 eps=1e-5,
                 elbo_weight=0.,
                 log_every_t=50,
                 lr=1e-4,
                 ignore_keys=[],
                 ckpt_path=None):
        super().__init__()

        self.unet = instantiate_from_config(unet_config)
        self.loss_fn = instantiate_from_config(loss_fn_config)
        self.sampler = instantiate_from_config(sampler_config)
        self.residual = instantiate_from_config(residual_config)
        self.eps = eps
        self.lr = lr
        self.elbo_weight = elbo_weight
        self.log_every_t = log_every_t

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        params = torch.load(path, map_location='cpu')
        if "state_dict" in list(params.keys()):
            params = params["state_dict"]
        keys = list(params.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del params[k]

        missing, unexpected = self.load_state_dict(params, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")

    def forward(self, x, c):
        loss_dict = {}
        log_prefix = 'train' if self.training else 'val'
        # score-mathcing objective function
        score_loss = self.loss_fn(self.unet, x, labels=c, augment_pipe=None, stf=False, pfgmpp=self.unet.pfgmpp)
        loss_dict.update({f'{log_prefix}/loss_score': score_loss.mean()})
        return score_loss.mean(), loss_dict

    def training_step(self, batch):
        x, c = self.get_input(batch)
        loss, loss_dict = self.forward(x, c)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def get_input(self, batch):
        x, c = batch
        return x, c

    @torch.no_grad()
    def sample(self, batch_size=8, c=None):
        rnd = StackedRandomGenerator(self.device, range(batch_size))
        if self.unet.pfgmpp:
            latents = rnd.rand_beta_prime([batch_size, self.unet.in_channels, self.unet.dim, self.unet.dim],
                                          N = self.unet.N, D = self.unet.D, pfgmpp=self.unet.pfgmpp, device=self.device,
                                          sigma_max=self.sampler_config['sigma_max'] if (self.unet.N > 256 * 256 * 3) else 80)
                                          #sigma_max=80)
        else:
            latents = rnd.randn([batch_size, self.unet.in_channels, self.unet.dim, self.unet.dim], device=self.device)
        return self.sampler(self.unet, latents=latents, class_labels=c, randn_like=rnd.randn_like)

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True):
        log = dict()

        # log samples
        x0, c = self.get_input(batch)
        N = min(x0.shape[0], N)
        x0 = x0.to(self.device)[:N]
        c = c.to(self.device)[:N]
        log["inputs"] = x0

        # backward sampling (denoising)
        start_time = time.time()
        samples, intermediates = self.sample(batch_size=N, c=c)
        end_time = time.time()
        log["samples"] = samples
        
        p = samples[:, 0, :, :]*self.residual.sigma_p + self.residual.mu_p
        k = samples[:, 1, :, :]*self.residual.sigma_k + self.residual.mu_k
        residual, _, _, _, _, _, _ = self.residual.r_diff(p, k)

        self.log("physics_residual_l2_norm", torch.mean(residual**2),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("sample time", end_time-start_time, logger=True, prog_bar=True,
                 on_step=False, on_epoch=True)

        return log

    def _get_rows_from_list(self, samples):
        n_per_row = len(samples)
        grid = rearrange(samples, 'n b c h w -> b n c h w')
        grid = rearrange(grid, 'b n c h w -> (b n) c h w')
        grid = make_grid(grid, nrow=n_per_row)
        return grid

    def configure_optimizers(self):
        lr = self.lr
        params = list(self.unet.parameters())
        return torch.optim.AdamW(params, lr=lr)

class Diffusion2D_NS(pl.LightningModule):
    # base class for training a score-matching objective
    def __init__(self,
                 unet_config,
                 loss_fn_config,
                 sampler_config,
                 residual_config,
                 eps=1e-5,
                 elbo_weight=0.,
                 log_every_t=50,
                 lr=1e-4,
                 ignore_keys=[],
                 ckpt_path=None):
        super().__init__()

        self.unet = instantiate_from_config(unet_config)
        self.loss_fn = instantiate_from_config(loss_fn_config)
        self.sampler = instantiate_from_config(sampler_config)
        self.residual = instantiate_from_config(residual_config)
        self.eps = eps
        self.lr = lr
        self.elbo_weight = elbo_weight
        self.log_every_t = log_every_t

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        params = torch.load(path, map_location='cpu')
        if "state_dict" in list(params.keys()):
            params = params["state_dict"]
        keys = list(params.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del params[k]

        missing, unexpected = self.load_state_dict(params, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")

    def forward(self, x, c):
        loss_dict = {}
        log_prefix = 'train' if self.training else 'val'
        # score-mathcing objective function
        score_loss = self.loss_fn(self.unet, x, labels=c, augment_pipe=None, stf=False, pfgmpp=self.unet.pfgmpp)
        loss_dict.update({f'{log_prefix}/loss_score': score_loss.mean()})
        return score_loss.mean(), loss_dict

    def training_step(self, batch):
        x, y = self.get_input(batch)
        loss, loss_dict = self.forward(y, c=None)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def get_input(self, batch):
        x, y = batch
        return x, y

    @torch.no_grad()
    def sample(self, batch_size=8, c=None):
        rnd = StackedRandomGenerator(self.device, range(batch_size))
        if self.unet.pfgmpp:
            latents = rnd.rand_beta_prime([batch_size, self.unet.in_channels, self.unet.dim, self.unet.dim],
                                          N = self.unet.N, D = self.unet.D, pfgmpp=self.unet.pfgmpp, device=self.device,
                                          sigma_max=self.sampler_config['sigma_max'] if (self.unet.N > 256 * 256 * 3) else 80)
                                          #sigma_max=80)
        else:
            latents = rnd.randn([batch_size, self.unet.in_channels, self.unet.dim, self.unet.dim], device=self.device)
        return self.sampler(self.unet, latents=latents, class_labels=c, randn_like=rnd.randn_like)

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True):
        log = dict()

        # log samples
        x, y = self.get_input(batch)
        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        y = y.to(self.device)[:N]
        log["inputs"] = y

        # backward sampling (denoising)
        start_time = time.time()
        samples, intermediates = self.sample(batch_size=N, c=None)
        end_time = time.time()
        log["samples"] = samples
        
        density_flux, Mx, My, energy_flux = torch.vmap(self.residual)(samples.view(-1, 4, self.residual.Nx, self.residual.Ny, self.residual.Nz))
        density_residual = torch.abs(torch.sum((density_flux), dim=(1,2,3)))
        Mx_residual = torch.abs(torch.sum((Mx), dim=(1,2,3)))
        My_residual = torch.abs(torch.sum((My), dim=(1,2,3)))
        energy_residual = torch.abs(torch.sum((energy_flux), dim=(1,2,3)))
        self.log("density_flux_residual_MAE", torch.mean(density_residual),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("Mx_residual_MAE", torch.mean(Mx_residual),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("My_residual_MAE", torch.mean(My_residual),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("energy_flux_residual_MAE", torch.mean(energy_residual),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("sample time", end_time-start_time, logger=True, prog_bar=True,
                 on_step=False, on_epoch=True)

        return log

    def _get_rows_from_list(self, samples):
        n_per_row = len(samples)
        grid = rearrange(samples, 'n b c h w -> b n c h w')
        grid = rearrange(grid, 'b n c h w -> (b n) c h w')
        grid = make_grid(grid, nrow=n_per_row)
        return grid

    def configure_optimizers(self):
        lr = self.lr
        params = list(self.unet.parameters())
        return torch.optim.AdamW(params, lr=lr)