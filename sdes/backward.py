'''
SDE samplers (backward)
Author: Christian Jacobsen, University of Michigan 2023
'''

import torch
import torch.nn as nn

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
            batch_step = torch.ones(batch_size, device=device) * time_step
            mean_x = x - (sde.f(x, batch_step) - sde.g(batch_step)**2*unet(x, c, batch_step, context_mask))*step_size
            x = mean_x + torch.sqrt(step_size)*sde.g(batch_step)*torch.randn_like(x)
            if i % self.intermediate_steps == 0:
                intermediates.append(x)
            i += 1

        # no noise in last step
        return mean_x, intermediates


