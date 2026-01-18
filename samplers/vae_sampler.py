# VAE Sampler: Sample from latent space and decode

import torch
from models.vae import Decoder

def vae_sample(decoder, latent_dim=128, num_samples=16):
    z = torch.randn(num_samples, latent_dim)
    samples = decoder(z)
    return samples