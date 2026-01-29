import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from utils.get_device import get_device

from models.scorebased_models import SDEDiffusion
from data.cifar10 import load_cifar10

DEVICE = get_device()
NUM_EPOCHS = 500
# Select only 16 images for overfitting test
NUM_SAMPLES = 16

def load_test_batch() -> torch.Tensor:
    # Load CIFAR-10 dataset
    full_dataset, _ = load_cifar10(batch_size=32, data_dir='./../data')

    indices = list(range(NUM_SAMPLES))
    tiny_dataset = Subset(full_dataset.dataset, indices)

    # Create dataloader with batch size = NUM_SAMPLES (single batch)
    tiny_loader = DataLoader(tiny_dataset, batch_size=NUM_SAMPLES, shuffle=False)

    test_batch = None
    # Get the single batch for visualization
    for batch_images, batch_labels in tiny_loader:
        test_batch = batch_images.to(DEVICE)
        break

    print(f"Test batch shape: {test_batch.shape}")
    print(f"Test batch range: [{test_batch.min().item():.3f}, {test_batch.max().item():.3f}]")

    return test_batch

def show_images(images, title="Images", n_row=4):
    # Display a grid of images
    images = images.cpu()
    # Denormalize from [-1, 1] to [0, 1]
    # images = (images + 1) / 2
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

def main():
    losses = []

    # Create VP-SDE model (DDPM-style)
    model = SDEDiffusion(sde_type='VPSDE', lr=1e-3)
    model.train()
    test_batch = load_test_batch()

    for epoch in range(NUM_EPOCHS):
        # Train on the single batch repeatedly
        loss_dict = model.train_step(test_batch, epoch)
        loss_value = loss_dict['total_loss']
        losses.append(loss_value)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {loss_value:.6f}")

    print("Training completed!")

    # Sampling
    samples = model.sample(batch_size=64)
    
    show_images(samples)

if __name__ == "__main__":
    main()