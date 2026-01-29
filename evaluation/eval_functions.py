# FID calculation (optional)

# Note: Requires pytorch-fid or similar library
# This is a placeholder

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

def compute_fid(
    real_loader,
    model,
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

    with torch.no_grad():
        for _ in range(steps):
            samples = model.sample(batch_size).to(device)

            # If samples are in [-1, 1], uncomment:
            # samples = (samples + 1) / 2

            samples = samples.clamp(0, 1)
            fid_metric.update(samples, real=False)

    return fid_metric.compute().item()


def compute_is(
    model,
    device,
    num_gen=10_000,
    batch_size=128
):
    is_metric = InceptionScore(
        splits=10,      # standard
        normalize=True  # expects images in [0,1]
    ).to(device)

    with torch.no_grad():
        for _ in range(num_gen // batch_size):
            samples = model.sample(batch_size).to(device)

            # If samples in [-1,1], uncomment:
            # samples = (samples + 1) / 2

            samples = samples.clamp(0, 1)
            is_metric.update(samples)

    is_mean, is_std = is_metric.compute()
    return is_mean.item(), is_std.item()



