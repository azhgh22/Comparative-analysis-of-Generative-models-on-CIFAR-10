# NCSN Sampler: Annealed Langevin Dynamics

import torch

def annealed_langevin_dynamics(score_net, x_init, sigmas, step_sizes, num_steps=100):
    x = x_init
    for sigma, step_size in zip(sigmas, step_sizes):
        for _ in range(num_steps):
            score = score_net(x, sigma)
            noise = torch.randn_like(x)
            x = x + 0.5 * step_size * score + torch.sqrt(step_size) * noise
    return x