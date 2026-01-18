# Train NCSN

import torch
import torch.optim as optim
from data.cifar10 import load_cifar10
from models.ncsn import ScoreNet
from losses.ncsn_loss import ncsn_loss

def train_ncsn(epochs=10, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, _ = load_cifar10()
    model = ScoreNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    sigmas = torch.tensor([0.01, 0.1, 1.0])  # example noise levels
    
    for epoch in range(epochs):
        for x, _ in train_loader:
            x = x.to(device)
            sigma = sigmas[torch.randint(0, len(sigmas), (1,))].to(device)
            optimizer.zero_grad()
            loss = ncsn_loss(model, x, sigma)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

if __name__ == '__main__':
    train_ncsn()