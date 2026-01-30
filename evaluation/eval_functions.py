# FID calculation (optional)

# Note: Requires pytorch-fid or similar library
# This is a placeholder

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import math

def compute_fid(
    real_loader,
    samples,
    device,
    num_gen=10000
):
    """
    real_loader : DataLoader with real images (e.g. CIFAR-10 test)
    sample_fn   : function(batch_size) -> images in [0,1], shape (B,3,H,W)
    device      : torch device
    """

    fid_metric = FrechetInceptionDistance(
        feature=2048,
        normalize=True
    ).to(device)

    # --------------------------------------------------
    # 1) Real images
    # --------------------------------------------------
    with torch.no_grad():
        for x, _ in real_loader:
            x = x.to(device)

            # If real images are in [-1, 1], uncomment:
            # x = (x + 1) / 2

            fid_metric.update(x, real=True)

    # --------------------------------------------------
    # 2) Generated images
    # --------------------------------------------------
    batch_size = real_loader.batch_size or 128
    steps = num_gen // batch_size

    # i = 1
    for sample in samples:
        sample = sample.to(device)
        samples_c = sample.clamp(0, 1)
        fid_metric.update(samples_c, real=False)


    # with torch.no_grad():
    #     for _ in range(steps):
    #         samples = model.sample(batch_size).to(device)
    #         print("FID: ",i)
    #         i+=1
    #         # If samples are in [-1, 1], uncomment:
    #         # samples = (samples + 1) / 2

    #         samples = samples.clamp(0, 1)
    #         fid_metric.update(samples, real=False)

    return fid_metric.compute().item()


def compute_is(
    samples,
    device,
    num_gen=10_000,
    batch_size=256
):
    is_metric = InceptionScore(
        splits=10,      # standard
        normalize=True  # expects images in [0,1]
    ).to(device)

    for sample in samples:
        sample = sample.to(device)
        sample_c = sample.clamp(0, 1)
        is_metric.update(sample_c)


    # with torch.no_grad():
    #     for _ in range(num_gen // batch_size):
    #         samples = model.sample(batch_size).to(device)

    #         # If samples in [-1,1], uncomment:
    #         # samples = (samples + 1) / 2

    #         samples = samples.clamp(0, 1)
    #         is_metric.update(samples)

    is_mean, is_std = is_metric.compute()
    return is_mean.item(), is_std.item()


def compute_mi(
    model,
    data_loader,
    device,
    num_batches=100
):
    """
    Estimates I(X; Z) for a Gaussian VAE
    Uses Monte Carlo approximation
    """

    model.eval()

    log_qz_given_x = []
    log_qz = []

    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            if i >= num_batches:
                break

            x = x.to(device)

            # Encode
            _, z, mu, logvar = model(x)
            # std = torch.exp(0.5 * logvar)

            # # Sample z ~ q(z|x)
            # eps = torch.randn_like(std)
            # z = mu + eps * std

            # log q(z|x)
            log_qz_x = (
                -0.5 * (
                    math.log(2 * math.pi)
                    + logvar
                    + ((z - mu) ** 2) / torch.exp(logvar)
                )
            ).sum(dim=1)

            log_qz_given_x.append(log_qz_x)

            # Estimate log q(z)
            # using minibatch aggregation trick
            batch_size, z_dim = z.shape
            mu_all = mu.unsqueeze(1)
            logvar_all = logvar.unsqueeze(1)

            z_all = z.unsqueeze(0)

            log_qz_all = (
                -0.5 * (
                    math.log(2 * math.pi)
                    + logvar_all
                    + ((z_all - mu_all) ** 2) / torch.exp(logvar_all)
                )
            ).sum(dim=2)

            # log-sum-exp over batch
            log_qz_est = torch.logsumexp(log_qz_all, dim=1) - math.log(batch_size)
            log_qz.append(log_qz_est)

    log_qz_given_x = torch.cat(log_qz_given_x)
    log_qz = torch.cat(log_qz)

    mi = (log_qz_given_x - log_qz).mean()
    return mi.item()
    

def compute_mi_vqvae(
    model,
    data_loader,
    device,
    num_batches=None
):
    """
    Computes I(X;Z) for a VQ-VAE using codebook entropy.
    """

    model.eval()
    num_embeddings = model.vq.num_embeddings

    # Count code usage
    counts = torch.zeros(num_embeddings, device=device)

    total_latents = 0

    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            if num_batches is not None and i >= num_batches:
                break

            x = x.to(device)

            # Encode + quantize
            z_e = model.encode(x)
            _, _, indices = model.quantize(z_e)
            # indices: [B, H*W]

            indices = indices.view(-1)  # [B*H*W]

            counts += torch.bincount(
                indices,
                minlength=num_embeddings
            )

            total_latents += indices.numel()

    # Empirical distribution q(z)
    probs = counts / total_latents
    probs = probs[probs > 0]  # avoid log(0)

    # Entropy = Mutual Information
    mi = -torch.sum(probs * torch.log(probs))

    return mi.item()

