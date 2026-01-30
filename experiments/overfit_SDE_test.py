from models.scorebased_models.sde import SDEModel
from utils.get_device import get_device
from utils.helper_for_overfitting import load_test_batch, show_images

DEVICE = get_device()
NUM_EPOCHS = 2000
NUM_SAMPLES = 16

def main():
    losses = []

    # Create VP-SDE model (DDPM-style)
    # score_network_type = {'double_scorenet', 'scorenet', 'sde'}
    # sde_type={'VESDE', 'VPSDE', 'subVPSDE'}
    # model = SDEDiffusion(score_network_type='sde', channels=64, sde_type='VESDE', lr=1e-3)

    model = SDEModel(lr=1e-3, device=get_device())

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)

    model.train()
    test_batch = load_test_batch(NUM_SAMPLES)

    for epoch in range(NUM_EPOCHS):
        # Train on the single batch repeatedly
        loss_dict = model.train_step(test_batch, epoch)
        loss_value = loss_dict['total_loss']
        losses.append(loss_value)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {loss_value:.6f}")

    print("Training completed!")

    num_samples = 16
    # Sampling
    samples = model.sample(batch_size=num_samples)

    show_images(test_batch, title="Samples", n_row=int(num_samples/4))
    show_images(samples, title="Overfitted", n_row=int(num_samples/4))

if __name__ == "__main__":
    main()