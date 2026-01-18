# Train VAE

import torch
import torch.optim as optim
from data.cifar10 import load_cifar10
from models.vae import VAE
from losses.vae_loss import vae_loss

def train_vae(epochs=10, latent_dim=128, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, _ = load_cifar10()
    model = VAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

if __name__ == '__main__':
    train_vae()