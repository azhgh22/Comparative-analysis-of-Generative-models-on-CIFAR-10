# VAE Sampler: Sample from latent space and decode

import torch

def vae_sample(decoder, num_samples=16,device='cuda',latent_dim=128):
    z = torch.randn(num_samples, latent_dim)
    samples = decoder(z.to(device))
    return samples