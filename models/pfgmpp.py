# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
from torch.distributions import Beta
import numpy as np
from scipy.stats import betaprime
from einops import repeat
#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, D=128, N=3072, gamma=5, opts=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.D = D
        self.N = N
        self.gamma = gamma
        #self.opts = opts
        print(f"In EDM loss: D:{self.D}, N:{self.N}")

    def __call__(self, net, images, labels=None, augment_pipe=None, stf=False, pfgmpp=False, ref_images=None):

        if pfgmpp:
            # ln(sigma) ~ N(P_mean, P_std^2), Appendix C
            rnd_normal = torch.randn(images.shape[0], device=images.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            #if self.opts.lsun:
                # use larger sigma for high-resolution datasets
            #    sigma *= 380. / 80

            r = sigma.double() * np.sqrt(self.D).astype(np.float64) #Th 4.1
            # Sampling form inverse-beta distribution, Appendix B & Prop 3.2
            samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2.,
                                          size=images.shape[0]).astype(np.double) #R1

            samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)

            inverse_beta = samples_norm / (1 - samples_norm + 1e-8) #R2
            inverse_beta = torch.from_numpy(inverse_beta).to(images.device).double()
            # Sampling from p_r(R) by change-of-variable
            samples_norm = r * torch.sqrt(inverse_beta + 1e-8) #R3
            samples_norm = samples_norm.view(len(samples_norm), -1)
            # Uniformly sample the angle direction
            gaussian = torch.randn(images.shape[0], self.N).to(samples_norm.device) #Appendix B
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
            # Construct the perturbation for x
            perturbation_x = unit_gaussian * samples_norm
            perturbation_x = perturbation_x.float()

            sigma = sigma.reshape((len(sigma), 1, 1, 1))

            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
            y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
            n = perturbation_x.view_as(y)
            D_yn = net(y + n, sigma, labels)
        else:
            rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            #if self.opts.lsun:
                # use larger sigma for high-resolution datasets
            #    sigma *= 380. / 80.
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
            y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
            n = torch.randn_like(y) * sigma
            D_yn = net(y + n, sigma, labels)

        if stf:
            ref_images[len(y):], augment_labels_2 = augment_pipe(ref_images[len(y):]) \
                if augment_pipe is not None else (images, None)
            # update augmented original images
            ref_images[:len(y)] = y
        if pfgmpp:
            if stf:
                target = self.pfgmpp_target(r.squeeze(), y+n, ref_images)
                target = target.view_as(y)
            else:
                target = y
        elif stf:
            # Diffusion (D-> \inf)
            target = self.stf_scores(sigma.squeeze(), y+n, ref_images)
            target = target.view_as(y)
        else:
            target = y

        loss = weight * ((D_yn - target) ** 2)
        return loss

    def stf_scores(self, sigmas, perturbed_samples, samples_full):

        with torch.no_grad():
            #print("perturbed shape:", perturbed_samples.shape, "full shape:", samples_full.shape)
            perturbed_samples_vec = perturbed_samples.reshape((len(perturbed_samples), -1))
            samples_full_vec = samples_full.reshape((len(samples_full), -1))

            gt_distance = torch.sum((perturbed_samples_vec.unsqueeze(1) - samples_full_vec) ** 2,
                                    dim=[-1])
            gt_distance = - gt_distance / (2 * sigmas.unsqueeze(1) ** 2)
            distance = - torch.max(gt_distance, dim=1, keepdim=True)[0] + gt_distance
            distance = torch.exp(distance)
            distance = distance[:, :, None]
            weights = distance / (torch.sum(distance, dim=1, keepdim=True))
            target = samples_full_vec.unsqueeze(0).repeat(len(perturbed_samples), 1, 1)

            gt_direction = torch.sum(weights * target, dim=1)

            return gt_direction

    def pfgm_target(self, perturbed_samples_vec, samples_full):
        real_samples_vec = torch.cat(
            (samples_full.reshape(len(samples_full), -1), torch.zeros((len(samples_full), 1)).to(samples_full.device)),
            dim=1)

        data_dim = self.N + self.D
        gt_distance = torch.sum((perturbed_samples_vec.unsqueeze(1) - real_samples_vec) ** 2,
                                dim=[-1]).sqrt()

        # For numerical stability, timing each row by its minimum value
        distance = torch.min(gt_distance, dim=1, keepdim=True)[0] / (gt_distance + 1e-7)
        distance = distance ** data_dim
        distance = distance[:, :, None]
        # Normalize the coefficients (effectively multiply by c(\tilde{x}) in the paper)
        coeff = distance / (torch.sum(distance, dim=1, keepdim=True) + 1e-7)
        diff = - (perturbed_samples_vec.unsqueeze(1) - real_samples_vec)

        # Calculate empirical Poisson field (N+D dimension in the augmented space)
        gt_direction = torch.sum(coeff * diff, dim=1)
        gt_direction = gt_direction.view(gt_direction.size(0), -1)

        # Normalizing the N+D-dimensional Poisson field
        gt_norm = gt_direction.norm(p=2, dim=1)
        gt_direction /= (gt_norm.view(-1, 1) + self.gamma)
        gt_direction *= np.sqrt(data_dim)

        target = gt_direction
        target[:, -1] = target[:, -1] / np.sqrt(self.D)

        return target

    def pfgmpp_target(self, r, perturbed_samples, samples_full):
        # # Augment the data with extra dimension z
        perturbed_samples_vec = torch.cat((perturbed_samples.reshape(len(perturbed_samples), -1),
                                           r[:, None]), dim=1).double()
        real_samples_vec = torch.cat(
            (samples_full.reshape(len(samples_full), -1), torch.zeros((len(samples_full), 1)).to(samples_full.device)),
            dim=1).double()

        data_dim = self.N + self.D
        gt_distance = torch.sum((perturbed_samples_vec.unsqueeze(1) - real_samples_vec) ** 2,
                                dim=[-1]).sqrt()

        # For numerical stability, timing each row by its minimum value
        distance = torch.min(gt_distance, dim=1, keepdim=True)[0] / (gt_distance + 1e-7)
        distance = distance ** data_dim
        distance = distance[:, :, None]
        # Normalize the coefficients (effectively multiply by c(\tilde{x}) in the paper)
        coeff = distance / (torch.sum(distance, dim=1, keepdim=True) + 1e-7)

        target = real_samples_vec.unsqueeze(0).repeat(len(perturbed_samples), 1, 1)
        # Calculate empirical Poisson field (N+D dimension in the augmented space)
        gt_direction = torch.sum(coeff * target, dim=1)
        gt_direction = gt_direction.view(gt_direction.size(0), -1)
        gt_direction = gt_direction[:, :-1].float()

        return gt_direction

    def pfgm_perturation(self, samples, r):

        # Sampling form inverse-beta distribution
        samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2.,
                                      size=samples.shape[0]).astype(np.double)
        inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
        inverse_beta = torch.from_numpy(inverse_beta).to(samples.device).double()
        # Sampling from p_r(R) by change-of-variable
        samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
        samples_norm = samples_norm.view(len(samples_norm), -1)
        # Uniformly sample the angle direction
        gaussian = torch.randn(samples.shape[0], self.N).to(samples_norm.device)
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
        # Construct the perturbation for x
        perturbation_x = unit_gaussian * samples_norm
        perturbation_x = perturbation_x.float()

        return samples + perturbation_x.view_as(samples)

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=0,
    pfgmpp=False, deterministic=False,
):

    N = net.N
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    if pfgmpp:
        x_next = latents.to(torch.float64)
    else:
        x_next = latents.to(torch.float64) * t_steps[0]

    whole_trajectory = torch.zeros((num_steps, *x_next.shape), dtype=torch.float64)
    # Main sampling loop.
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        if not deterministic:
            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        denoised = net(x_hat, repeat(t_hat.reshape(-1), 'w -> h w', h=x_hat.shape[0]), class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, repeat(t_next.reshape(-1), 'w -> h w', h=x_next.shape[0]), class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        whole_trajectory[i] = x_next

    return x_next, whole_trajectory

class EDM_Sampler:
    def __init__(self, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
                 S_churn=0, S_min=0, S_max=float('inf'), S_noise=0, pfgmpp=False):
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.pfgmpp = pfgmpp

    def __call__(self, net, latents, class_labels=None, randn_like=torch.randn_like):
        return edm_sampler(
            net, latents, class_labels, randn_like,
            num_steps=self.num_steps, sigma_min=self.sigma_min, 
            sigma_max=self.sigma_max, rho=self.rho, S_churn=self.S_churn,
            S_min=self.S_min, S_max=self.S_max, S_noise=self.S_noise, pfgmpp=self.pfgmpp
        )

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]
        self.seeds = seeds
        self.device = device

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def rand_beta_prime(self, size, N=3072, D=128, **kwargs):
        # sample from beta_prime (N/2, D/2)
        # print(f"N:{N}, D:{D}")
        assert size[0] == len(self.seeds)
        latent_list = []
        beta_gen = Beta(torch.FloatTensor([N / 2.]), torch.FloatTensor([D / 2.]))
        for seed in self.seeds:
            torch.manual_seed(seed)
            sample_norm = beta_gen.sample().to(kwargs['device']).double()
            # inverse beta distribution
            inverse_beta = sample_norm / (1-sample_norm)

            if N < 256 * 256 * 3:
                sigma_max = 80
            else:
                sigma_max = kwargs['sigma_max']

            sample_norm = torch.sqrt(inverse_beta) * sigma_max * np.sqrt(D)
            gaussian = torch.randn(N).to(sample_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2)
            init_sample = unit_gaussian * sample_norm
            latent_list.append(init_sample.reshape((1, *size[1:])))

        latent = torch.cat(latent_list, dim=0)
        return latent

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

def karras_ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    deterministic=False, epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, beta_d=19.9, beta_min=0.1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']
    assert deterministic in [True, False]

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=beta_d, beta_min=beta_min)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=beta_d, beta_min=beta_min)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    whole_trajectory = torch.zeros((num_steps, *x_next.shape), dtype=torch.float64)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        if not deterministic:
            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
            t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
            x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        h = t_next - t_hat
        #print(sigma(t_hat).reshape(-1).shape)
        denoised = net(x_hat / s(t_hat), repeat(sigma(t_hat).reshape(-1), 'w -> h w', h=x_hat.shape[0]), class_labels).to(torch.float64)
        # For edm, d_cur = (1/t_hat)*x_hat - (1/t_hat)*denoised) 
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), repeat(sigma(t_hat).reshape(-1), 'w -> h w', h=x_hat.shape[0]), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

        whole_trajectory[i] = x_next

    return x_next, whole_trajectory

class KarrasAblationSampler:
    def __init__(self, num_steps=18, sigma_min=None, sigma_max=None, rho=7,
                 solver='heun', discretization='edm', schedule='linear', scaling='none',
                 deterministic=False, epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
                 S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, beta_d=19.9, beta_min=0.1):
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.solver = solver
        self.discretization = discretization
        self.schedule = schedule
        self.scaling = scaling
        self.deterministic = deterministic
        self.epsilon_s = epsilon_s
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.alpha = alpha
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.beta_d = beta_d
        self.beta_min = beta_min

    def __call__(self, net, latents, class_labels=None, randn_like=torch.randn_like):
        return karras_ablation_sampler(
            net, latents, class_labels, randn_like,
            num_steps=self.num_steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max, rho=self.rho,
            solver=self.solver, discretization=self.discretization, schedule=self.schedule, scaling=self.scaling,
            deterministic=self.deterministic, epsilon_s=self.epsilon_s, C_1=self.C_1, C_2=self.C_2, M=self.M, alpha=self.alpha,
            S_churn=self.S_churn, S_min=self.S_min, S_max=self.S_max, S_noise=self.S_noise, beta_d=self.beta_d, beta_min=self.beta_min
        )
