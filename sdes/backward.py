'''
SDE samplers (backward)
Author: Christian Jacobsen, University of Michigan 2023
'''

import numpy as np
import torch
import torch.nn as nn
import sys
from scipy import integrate

from models.utils import from_flattened_numpy, to_flattened_numpy

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
    def predictor_step(self, x, c, t, context_mask, step_size, unet, sde, device, **kwargs):
        mean_x = x - (sde.f(x, t) - sde.g(t)**2*unet.forward_with_cond_scale(x, c, t, context_mask, **kwargs))*step_size
        x = mean_x + torch.sqrt(step_size)*sde.g(t)*torch.randn_like(x)
        return x, mean_x

    @torch.no_grad()
    def forward(self, unet, sde, data_size, context_mask, device, return_intermediates=False, **kwargs):
        #TODO: context_mask
        batch_size = data_size[0]
        noise = sde.sample_prior(data_size, device=device)
        time_steps = torch.linspace(1., self.eps, self.num_time_steps, device=device)
        step_size = time_steps[0]-time_steps[1]
        c = torch.zeros((batch_size, unet.cond_size), device=device)

        x = noise + 0.
        intermediates = []
        i = 1
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            x, mean_x = self.predictor_step(x, c, batch_time_step, context_mask, step_size, unet, sde, device, **kwargs)
            if i % self.intermediate_steps == 0:
                intermediates.append(x)
            i += 1

        # no noise in last step
        return mean_x, intermediates

class uncon_EulerMaruyama(nn.Module):
    def __init__(self, 
                 num_time_steps=500, 
                 eps=1e-7,
                 intermediate_steps=100000):
        super().__init__()

        self.num_time_steps = num_time_steps
        self.eps = eps
        self.intermediate_steps = intermediate_steps

    @torch.no_grad()
    def predictor_step(self, x, t, step_size, unet, sde, device):
        mean_x = x - (sde.f(x, t) - sde.g(t)**2*unet(x, t))*step_size
        x = mean_x + torch.sqrt(step_size)*sde.g(t)*torch.randn_like(x)
        return x, mean_x

    @torch.no_grad()
    def forward(self, unet, sde, data_size, device, return_intermediates=False):
        batch_size = data_size[0]
        noise = sde.sample_prior(data_size, device=device)
        time_steps = torch.linspace(1., self.eps, self.num_time_steps, device=device)
        step_size = time_steps[0]-time_steps[1]

        x = noise + 0.
        intermediates = []
        i = 1
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            x, mean_x = self.predictor_step(x, batch_time_step, step_size, unet, sde, device)
            if i % self.intermediate_steps == 0:
                intermediates.append(x)
            i += 1

        # no noise in last step
        return mean_x, intermediates

class ODE_uncon_EulerMaruyama(nn.Module):
    def __init__(self,
                 num_time_steps=500, 
                 eps=1e-7,
                 intermediate_steps=100000,
                 rtol = 1e-5,
                 atol = 1e-5,
                 method = 'RK45'):
        super().__init__()

        self.num_time_steps = num_time_steps
        self.eps = eps
        self.intermediate_steps = intermediate_steps
        self.rtol = rtol
        self.atol = atol
        self.method = method

    @torch.no_grad()
    def forward(self, score_model, sde, data_size, device, return_intermediates=False):
        
        def ode_func(t, x):        
            """The ODE function for use by the ODE solver."""
            x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=x.device) * t  
            #print(x.shape, vec_t.shape)
            g = sde.g(vec_t)
            f = sde.f(x, vec_t)
            #print(g.shape, f.shape)
            return to_flattened_numpy(f -0.5 * (g**2) * score_model(x, vec_t))

        #batch_size = data_size[0]
        noise = sde.sample_prior(data_size, device=device)
        #time_steps = torch.linspace(1., self.eps, self.num_time_steps, device=device)
        #step_size = time_steps[0]-time_steps[1]
        #c = torch.zeros((batch_size, score_model.cond_size), device=device)
        #context_mask = torch.zeros_like(c)

        x = noise + 0.
        shape = x.shape
        intermediates = []
        
        # Run the black-box ODE solver.
        res = integrate.solve_ivp(ode_func, (1., self.eps), to_flattened_numpy(x), rtol=self.rtol, atol=self.atol, method=self.method)  
        print(f"Number of function evaluations: {res.nfev}")
        x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

        # no noise in last step
        return x, intermediates

# Proposed EDM sampler (Algorithm 2 in EDM paper).
@torch.no_grad()
def edm_sampler(
    net, latents, class_labels=None, 
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=0,
    pfgmpp=False,
):
    net.eval()
    # Time step discretization.
    # orig implementation float64, change to float32
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    if pfgmpp:
        x_next = latents.to(torch.float32)
    else:
        x_next = latents.to(torch.float32) * t_steps[0]

    #whole_trajectory = torch.zeros((num_steps, *x_next.shape), dtype=torch.float32)
    # Main sampling loop.
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1

        x_cur = x_next
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = torch.as_tensor(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        # Euler step.
        denoised = net(x_hat, t_hat.repeat(x_hat.shape[0]), class_labels).to(torch.float32)
        #print('denoised', denoised.shape)
        d_cur = (x_hat - denoised) / t_hat
        #print('x_hat', x_hat.shape)
        #print('d_cur', d_cur.shape)
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float32)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        #whole_trajectory[i] = x_next

    return x_next#, whole_trajectory

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

