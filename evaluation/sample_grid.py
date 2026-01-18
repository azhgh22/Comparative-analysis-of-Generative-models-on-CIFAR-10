# Sample grid visualization

import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def save_sample_grid(samples, filename='samples.png'):
    grid = vutils.make_grid(samples, nrow=4, normalize=True)
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.axis('off')
    plt.savefig(filename)
    plt.close()