'''
Implementation of different SDE forms [VP], [VE]
Author: Christian Jacobsen, University of Michigan 2023
'''
import numpy as np
import torch
import torch.nn as nn

class VP(nn.Module):
    def __init__(self, 
                 beta_min=1e-4, 
                 beta_max=1.0):
        super().__init__()

        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t):
        return self.beta_min + t*(self.beta_max - self.beta_min)

    def eps(self, noise, score, t, r):
        # used for corrector step in predictor-corrector methods
        alpha = 1 - self.beta(t)
        noise_norm = torch.linalg.norm(noise.reshape((noise.shape[0], -1)), dim=-1).mean()
        score_norm = torch.linalg.norm(score.reshape((score.shape[0], -1)), dim=-1).mean()
        eps = 2*alpha*(r*noise_norm/score_norm)**2
        return eps[:, None, None, None]

    def f(self, x, t):
        return -0.5*self.beta(t)[:, None, None, None]*x

    def g(self, t):
        return torch.sqrt(self.beta(t))[:, None, None, None]

    def q_mu(self, x0, t):
        return x0*torch.exp(-0.25*t**2*(self.beta_max-self.beta_min)-0.5*t*self.beta_min)[:, None, None, None]

    def q_std(self, x0, t):
        return torch.sqrt(1-torch.exp(-0.5*t**2*(self.beta_max-self.beta_min)-t*self.beta_min))[:, None, None, None]

    def forward(self, x0, t):
        # forward SDE transition kernel. Return noisy sample, noise, and standard deviation
        z = torch.randn_like(x0)
        std = self.q_std(x0, t)
        mu = self.q_mu(x0, t)
        return mu + z*std, std, z

    def sample_prior(self, data_size, device):
        return torch.randn(data_size, device=device)

class uncon_VP_1D(nn.Module):
    def __init__(self, 
                 beta_min=1e-4, 
                 beta_max=1.0):
        super().__init__()

        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t):
        return self.beta_min + t*(self.beta_max - self.beta_min)

    def eps(self, noise, score, t, r):
        # used for corrector step in predictor-corrector methods
        alpha = 1 - self.beta(t)
        noise_norm = torch.linalg.norm(noise.reshape((noise.shape[0], -1)), dim=-1).mean()
        score_norm = torch.linalg.norm(score.reshape((score.shape[0], -1)), dim=-1).mean()
        eps = 2*alpha*(r*noise_norm/score_norm)**2
        return eps[:, None]

    def f(self, x, t):
        return -0.5*self.beta(t)[:, None]*x

    def g(self, t):
        return torch.sqrt(self.beta(t))[:, None]

    def q_mu(self, x0, t):
        return x0*torch.exp(-0.25*t**2*(self.beta_max-self.beta_min)-0.5*t*self.beta_min)[:, None]

    def q_std(self, x0, t):
        return torch.sqrt(1-torch.exp(-0.5*t**2*(self.beta_max-self.beta_min)-t*self.beta_min))[:, None]

    def forward(self, x0, t):
        # forward SDE transition kernel. Return noisy sample, noise, and standard deviation
        z = torch.randn_like(x0)
        std = self.q_std(x0, t)
        mu = self.q_mu(x0, t)
        return mu + z*std, std, z

    def sample_prior(self, data_size, device):
        return torch.randn(data_size, device=device)
    
    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1)) / 2.
        return logps