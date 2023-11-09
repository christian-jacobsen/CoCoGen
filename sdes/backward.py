'''
SDE samplers (backward)
Author: Christian Jacobsen, University of Michigan 2023
'''

import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from utils import instantiate_from_config

class EulerMaruyama(nn.Module):
    def __init__(self, 
                 num_time_steps=500, 
                 eps=1e-7,
                 resample=1,
                 intermediate_steps=100000):
        super().__init__()

        self.num_time_steps = num_time_steps
        self.eps = eps
        self.intermediate_steps = intermediate_steps
        self.resample = resample

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
        for time_step in tqdm(time_steps, desc="Sampling", unit="iteration"):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            x, mean_x = self.predictor_step(x, c, batch_time_step, context_mask, step_size, unet, sde, device)
            if i % self.intermediate_steps == 0:
                intermediates.append(x)
            i += 1

        # no noise in last step
        return mean_x, intermediates

    @torch.no_grad()
    def inpaint(self, unet, sde, data_size, values, inds, device):
        # if sensor values exist
        if len(inds) > 0:
            # sample with inpainting
            batch_size = data_size[0]
            noise = sde.sample_prior(data_size, device=device)
            time_steps = torch.linspace(1., self.eps, self.num_time_steps, device=device)
            step_size = time_steps[0]-time_steps[1]
            c = torch.zeros((batch_size, unet.cond_size), device=device)
            context_mask = torch.zeros_like(c)
            x = noise + 0.
            intermediates = []
            i = 0

            # normalize the values if needed
            normalized_values = values + 0. #values*self.residual.sigma_p + self.residual.mu_p

            # initial replacement of data values with noisy known values
            noisy_values, _, _ = sde.forward(normalized_values, torch.ones(batch_size, device=device))
            #noisy_values = normalized_values + 0.
            x[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]] = noisy_values[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]]

            for time_step in tqdm(time_steps, desc="Sampling", unit="iteration"):
                batch_time_step = torch.ones(batch_size, device=device) * time_step


                for i in range(self.resample):
                    # predict backward in time for one step
                    x, mean_x = self.predictor_step(x, c, batch_time_step, context_mask, step_size, unet, sde, device)

                    # replace x with noisy versions of the known dimensions
                    noisy_values, _, _ = sde.forward(normalized_values, batch_time_step-step_size)
                    #noisy_values = normalized_values + 0.
                    x[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]] = noisy_values[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]]

                    if i < (self.resample-1):
                        # simulate forward in time for one step
                        x = x + sde.f(x, batch_time_step)*step_size + self.g(batch_time_step)*torch.sqrt(step_size)*torch.randn_like(x)

                if i % self.intermediate_steps == 0:
                    intermediates.append(x)
                i += 1

            mean_x[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]] = normalized_values[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]]

            return mean_x

        else:
            mean_x, _ = self.forward(unet, sde, data_size, device)
            return mean_x


class ProbabilityFlowODE(EulerMaruyama):
    def __init__(self,
                 num_time_steps=500, 
                 eps=1e-7,
                 intermediate_steps=100000):
        super().__init__(num_time_steps, eps, intermediate_steps)

    @torch.no_grad()
    def predictor_step(self, x, c, t, context_mask, step_size, unet, sde, device):
        # predictor step has a slightly modified form and no noise
        mean_x = x - (sde.f(x, t) - 0.5*sde.g(t)**2*unet(x, c, t, context_mask))*step_size
        x = mean_x + 0.
        return x, mean_x


class EulerPhysics(nn.Module):
    def __init__(self, 
                 residual_config,
                 residual_step_size=1e-3,
                 num_time_steps=500, 
                 resample=1,
                 eps=1e-7,
                 intermediate_steps=100000):
        super().__init__()

        self.residual = instantiate_from_config(residual_config)
        self.residual_step_size = residual_step_size
        self.num_time_steps = num_time_steps
        self.eps = eps
        self.resample = resample
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
        i = 0
        for time_step in tqdm(time_steps, desc="Sampling", unit="iteration"):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            x, mean_x = self.predictor_step(x, c, batch_time_step, context_mask, step_size, unet, sde, device)
            
            if i >= (self.num_time_steps-0):#50):
                drdp = self.residual(x)
                eps_step = 2e-3 / torch.amax(drdp, dim=(1,2,3))
                x = (x*self.residual.sigma_p + self.residual.mu_p) - eps_step[:, None, None, None]*drdp
                x = (x - self.residual.mu_p) / self.residual.sigma_p

                mean_x = x + 0.

            if i % self.intermediate_steps == 0:
                intermediates.append(x)
            i += 1

        # additional physical consistency
        for i in range(0):#50):
            drdp = self.residual(x)
            eps_step = 2e-4 / torch.amax(drdp, dim=(1,2,3))
            x = (x*self.residual.sigma_p + self.residual.mu_p) - eps_step[:, None, None, None]*drdp
            x = (x - self.residual.mu_p) / self.residual.sigma_p
            mean_x = x + 0.

        # no noise in last step
        return mean_x, intermediates

    @torch.no_grad()
    def inpaint(self, unet, sde, data_size, values, inds, device):
        # if sensor values exist
        if len(inds) > 0:
            # sample with inpainting
            batch_size = data_size[0]
            noise = sde.sample_prior(data_size, device=device)
            time_steps = torch.linspace(1., self.eps, self.num_time_steps, device=device)
            step_size = time_steps[0]-time_steps[1]
            c = torch.zeros((batch_size, unet.cond_size), device=device)
            context_mask = torch.zeros_like(c)
            x = noise + 0.
            intermediates = []
            i = 0

            # normailize values if needed
            normalized_values = values + 0. #values*self.residual.sigma_p + self.residual.mu_p

            # replace x with noisy versions of the known dimensions (initial)
            noisy_values, _, _ = sde.forward(normalized_values, torch.ones(batch_size, device=device))
            #noisy_values = normalized_values + 0.
            x[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]] = noisy_values[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]]

            for time_step in tqdm(time_steps, desc="Sampling", unit="iteration"):
                batch_time_step = torch.ones(batch_size, device=device) * time_step

                #if ((i > 1900) and (i < 1990)):

                for j in range(self.resample):
                    # predict backward in time
                    x, mean_x = self.predictor_step(x, c, batch_time_step, context_mask, step_size, unet, sde, device)

                    # replace x with noisy versions of the known dimensions
                    noisy_values, _, _ = sde.forward(normalized_values, batch_time_step-step_size)
                    #noisy_values = normalized_values + 0.
                    x[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]] = noisy_values[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]]

                    if (j < (self.resample-1)) and (time_step > time_steps[-1]):
                        # simulate forward in time for one step
                        x = x + sde.f(x, batch_time_step-step_size)*step_size + sde.g(batch_time_step-step_size)*torch.sqrt(step_size)*torch.randn_like(x)
                    
                    if i >= (self.num_time_steps-50):
                        drdp = self.residual(x)
                        drdp[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]] = 0.0
                        eps_step = 2e-3 / torch.amax(drdp, dim=(1,2,3))
                        x = (x*self.residual.sigma_p + self.residual.mu_p) - eps_step[:, None, None, None]*drdp
                        x = (x - self.residual.mu_p) / self.residual.sigma_p

                        mean_x = x + 0.

                '''
                else:
                        # predict backward in time
                        x, mean_x = self.predictor_step(x, c, batch_time_step, context_mask, step_size, unet, sde, device)
                        
                        if i >= (self.num_time_steps-50):
                            drdp = self.residual(x)
                            drdp[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]] = 0.0
                            eps_step = 2e-3 / torch.amax(drdp, dim=(1,2,3))
                            x = (x*self.residual.sigma_p + self.residual.mu_p) - eps_step[:, None, None, None]*drdp
                            x = (x - self.residual.mu_p) / self.residual.sigma_p

                            mean_x = x + 0.
                '''

                if i % self.intermediate_steps == 0:
                    intermediates.append(x)
                i += 1

            # additional physical consistency
            # replace x with the known values
            x[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]] = normalized_values[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]]
            for i in range(10):#50):
                drdp = self.residual(x)
                drdp[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]] = 0.0
                eps_step = 2e-4 / torch.amax(drdp, dim=(1,2,3))
                x = (x*self.residual.sigma_p + self.residual.mu_p) - eps_step[:, None, None, None]*drdp
                x = (x - self.residual.mu_p) / self.residual.sigma_p
                mean_x = x + 0.

            # replace x with the known values
            #mean_x[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]] = normalized_values[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]]
            # no noise in last step
            return mean_x

        else:
            mean_x, _ = self.forward(unet, sde, data_size, device)
            return mean_x

class ProbabilityFlowODEPhysics(EulerPhysics):
    def __init__(self,
                 residual_config,
                 residual_step_size=1e-3,
                 num_time_steps=500, 
                 resample=1,
                 eps=1e-7,
                 intermediate_steps=100000):
        super().__init__(residual_config, residual_step_size, num_time_steps, resample, eps, intermediate_steps)

        self.residual = instantiate_from_config(residual_config)
        self.residual_step_size = residual_step_size
        self.num_time_steps = num_time_steps
        self.eps = eps
        self.intermediate_steps = intermediate_steps

    @torch.no_grad()
    def predictor_step(self, x, c, t, context_mask, step_size, unet, sde, device):
        # predictor step has a slightly modified form and no noise
        mean_x = x - (sde.f(x, t) - 0.5*sde.g(t)**2*unet(x, c, t, context_mask))*step_size
        x = mean_x + 0.
        return x, mean_x


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

