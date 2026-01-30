import torch
import math
import numpy as np

from torch import nn


from .ncsn import NCSN, ScoreNet, DoubleScoreNet
from .sde_diffusion import SDEDiffusion, VESDE, VPSDE, subVPSDE, create_sde_diffusion

def get_sigmas(config):
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                               config.model.num_classes))).float().to(config.DEVICE)
    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas