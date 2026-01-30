import torch
from matplotlib import pyplot as plt
from torch.utils.data import Subset, DataLoader

from data.cifar10 import load_cifar10
from utils.get_device import get_device


def load_test_batch(num_samples = 16, device=get_device()) -> torch.Tensor:
    # Load CIFAR-10 dataset
    data_dir = '/Users/nika.matcharadze/Documents/uni/generative models/Comparative-analysis-of-Generative-models-on-CIFAR-10/data'
    full_dataset, _ = load_cifar10(batch_size=32, data_dir=data_dir, normalize_inputs=True)

    indices = list(range(num_samples))
    tiny_dataset = Subset(full_dataset.dataset, indices)

    # Create dataloader with batch size = num_samples (single batch)
    tiny_loader = DataLoader(tiny_dataset, batch_size=num_samples, shuffle=False)

    test_batch = None
    # Get the single batch for visualization
    for batch_images, batch_labels in tiny_loader:
        test_batch = batch_images.to(device)
        break

    print(f"Test batch shape: {test_batch.shape}")
    print(f"Test batch range: [{test_batch.min().item():.3f}, {test_batch.max().item():.3f}]")

    return test_batch


def show_images(images, title="Images", n_row=4):
    # Display a grid of images
    images = images.cpu()
    # Denormalize from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)

    n = len(images)
    n_col = n_row
    n_row = (n + n_col - 1) // n_col

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 2, n_row * 2))
    axes = axes.flatten() if n > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < n:
            img = images[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()