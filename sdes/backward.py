'''
SDE samplers (backward)
Author: Christian Jacobsen, University of Michigan 2023
'''

import torch
import torch.nn as nn
import sys

class EulerMaruyama(nn.Module):
    def __init__(self, 
                 num_time_steps=500, 
                 eps=1e-7,
                 intermediate_steps=100000):
        super().__init__()

        self.num_time_steps = num_time_steps
        self.eps = eps
        self.intermediate_steps = intermediate_steps

    @torch.no_grad()
    def predictor_step(self, x, c, t, context_mask, step_size, unet, sde, device):
        mean_x = x - (sde.f(x, t) - sde.g(t)**2*unet(x, c, t, context_mask))*step_size
        x = mean_x + torch.sqrt(step_size)*sde.g(t)*torch.randn_like(x)
        return x, mean_x

    @torch.no_grad()
    def forward(self, unet, sde, data_size, device, return_intermediates=False):
        batch_size = data_size[0]
        noise = sde.sample_prior(data_size, device=device)
        time_steps = torch.linspace(1., self.eps, self.num_time_steps, device=device)
        step_size = time_steps[0]-time_steps[1]
        c = torch.zeros((batch_size, unet.cond_size), device=device)
        context_mask = torch.zeros_like(c)

        x = noise + 0.
        intermediates = []
        i = 1
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            x, mean_x = self.predictor_step(x, c, batch_time_step, context_mask, step_size, unet, sde, device)
            if i % self.intermediate_steps == 0:
                intermediates.append(x)
            i += 1

        # no noise in last step
        return mean_x, intermediates

class EulerPhysics(nn.Module):
    def __init__(self, 
                 physics_config,
                 num_time_steps=500, 
                 eps=1e-7,
                 intermediate_steps=100000):
        super().__init__()

        self.residual = instantiate_from_config(physics_config)
        self.num_time_steps = num_time_steps
        self.eps = eps
        self.intermediate_steps = intermediate_steps

    @torch.no_grad()
    def predictor_step(self, x, c, t, context_mask, step_size, unet, sde, device):
        mean_x = x - (sde.f(x, t) - sde.g(t)**2*unet(x, c, t, context_mask))*step_size
        x = mean_x + torch.sqrt(step_size)*sde.g(t)*torch.randn_like(x)
        return x, mean_x

    @torch.no_grad()
    def forward(self, unet, sde, data_size, device, return_intermediates=False):
        batch_size = data_size[0]
        noise = sde.sample_prior(data_size, device=device)
        time_steps = torch.linspace(1., self.eps, self.num_time_steps, device=device)
        step_size = time_steps[0]-time_steps[1]
        c = torch.zeros((batch_size, unet.cond_size), device=device)
        context_mask = torch.zeros_like(c)

        x = noise + 0.
        intermediates = []
        i = 1
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            x, mean_x = self.predictor_step(x, c, batch_time_step, context_mask, step_size, unet, sde, device)
            if i % self.intermediate_steps == 0:
                intermediates.append(x)
            i += 1

        # no noise in last step
        return mean_x, intermediates

class PredictorCorrector(EulerMaruyama):
    def __init__(self, 
                 num_time_steps=500, 
                 corrector_steps=1,
                 eps=1e-7,
                 r=1e-5,
                 intermediate_steps=100000):
        super().__init__(num_time_steps, eps, intermediate_steps)

        self.num_time_steps = num_time_steps # number of predictor steps
        self.corrector_steps = corrector_steps # number of corrector steps per predictor step
        self.eps = eps
        self.intermediate_steps = intermediate_steps
        self.r = r # signal to noise ratio

    @torch.no_grad()
    def corrector_step(self, x, c, t, context_mask, unet, sde, device):
        noise = torch.randn_like(x)
        score = unet(x, c, t, context_mask)
        eps = sde.eps(noise, score, t, self.r)
        x_mean = x + eps*score
        x = x_mean + torch.sqrt(2*eps)*noise
        return x, x_mean

    @torch.no_grad()
    def forward(self, unet, sde, data_size, device, return_intermediates=False):
        batch_size = data_size[0]
        noise = sde.sample_prior(data_size, device=device)
        time_steps = torch.linspace(1., self.eps, self.num_time_steps, device=device)
        step_size = time_steps[0]-time_steps[1]
        c = torch.zeros((batch_size, unet.cond_size), device=device)
        context_mask = torch.zeros_like(c)

        x = noise + 0.
        intermediates = []
        i = 1
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            x, mean_x = self.predictor_step(x, c, batch_time_step, context_mask, step_size, unet, sde, device)
            for corrector_step in range(self.corrector_steps):
                x, mean_x = self.corrector_step(x, c, batch_time_step, context_mask, unet, sde, device)

            if i % self.intermediate_steps == 0:
                intermediates.append(x)
            i += 1

        # no noise in last step
        return mean_x, intermediates

