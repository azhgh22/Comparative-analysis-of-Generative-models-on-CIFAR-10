# NCSN Loss: Denoising Score Matching

import torch

def ncsn_loss(score_net, x, sigma):
    # sigma is noise level
    noise = torch.randn_like(x)
    noisy_x = x + sigma * noise
    score = score_net(noisy_x, sigma)
    
    # Denoising matching loss
    loss = torch.mean(torch.sum((score + noise / sigma)**2, dim=[1,2,3]))
    return loss