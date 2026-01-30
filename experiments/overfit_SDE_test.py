from models.scorebased_models.ddpm import create_ddpm
from models.scorebased_models.sde import SDEModel
from utils.get_device import get_device
from utils.helper_for_overfitting import load_test_batch, show_images

DEVICE = get_device()
NUM_EPOCHS = 500
NUM_SAMPLES = 16

def main():
    losses = []

    # Create VP-SDE model (DDPM-style)
    # score_network_type = {'double_scorenet', 'scorenet', 'sde'}
    # sde_type={'VESDE', 'VPSDE', 'subVPSDE'}
    # model = SDEDiffusion(score_network_type='sde', channels=64, sde_type='VESDE', lr=1e-3)

    model = create_ddpm(image_size=32, image_channels=3, timesteps=1000)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)

    model.train()
    test_batch = load_test_batch(NUM_SAMPLES)

    log = {'epoch': [], 'mean': [], 'std': [], 'absolute_mean': []}
    for epoch in range(NUM_EPOCHS):
        # Train on the single batch repeatedly
        loss_dict = model.train_step(test_batch)
        loss_value = loss_dict['total_loss']
        losses.append(loss_value)

        if (epoch + 1) % 5 == 0:
            log['epoch'].append(epoch + 1)
            log['mean'].append(model.model.conv_in.weight.grad.mean().cpu().item())
            log['std'].append(model.model.conv_in.weight.grad.std().cpu().item())
            log['absolute_mean'].append(model.model.conv_in.weight.grad.abs().mean().cpu().item())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {loss_value:.6f}")

    print("Training completed!")

    # plot log grad values as a function of epochs
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(log['epoch'], log['mean'], label='Mean of Gradients')
    plt.xlabel('Epoch')
    plt.ylabel('Mean')
    plt.title('Mean of Gradients over Epochs')
    plt.grid()
    plt.subplot(1, 3, 2)
    plt.plot(log['epoch'], log['std'], label='Std of Gradients', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Standard Deviation')
    plt.title('Std of Gradients over Epochs')
    plt.grid()
    plt.subplot(1, 3, 3)
    plt.plot(log['epoch'], log['absolute_mean'], label='Absolute Mean of Gradients', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Absolute Mean')
    plt.title('Absolute Mean of Gradients over Epochs')
    plt.grid()
    plt.tight_layout()
    plt.show()

    num_samples = 16
    # Sampling
    samples = model.sample(batch_size=num_samples)

    show_images(test_batch, title="Samples", n_row=int(num_samples/4))
    show_images(samples, title="Overfitted", n_row=int(num_samples/4))

if __name__ == "__main__":
    main()