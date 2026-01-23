import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),   # 32 → 16
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1), # 16 → 8
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1), # 8 → 4
            nn.ReLU(inplace=True),
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, z_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()

        self.fc = nn.Linear(z_dim, 256 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 4 → 8
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8 → 16
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),    # 16 → 32
            nn.Sigmoid()  # assumes inputs in [0,1]
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), 256, 4, 4)
        return self.deconv(h)

class ConvVAE(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()

        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

        lr = 2e-4
        gamma = 0.95

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=gamma
        )

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    # -------- Training step --------
    def train_step(self, x,epoch):
        """
        Performs a single training step.
        - x: input batch
        - optimizer: optional, if not set use self.optimizer
        """
        optimizer = self.optimizer

        # Forward
        recon_x, mu, logvar = self.forward(x)

        warmup_epochs = 20
        beta = min(1.0, epoch / warmup_epochs)

        # Loss with beta
        total_loss, recon_loss, kl_loss = self.vae_loss(recon_x, x, mu, logvar,beta)

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item()
        }

    def vae_loss(self,recon_x, x, mu, logvar, beta=1.0):
        """
        Computes VAE loss: reconstruction + KL divergence
        - recon_x: reconstructed images [B, C, H, W]
        - x: original images
        - mu, logvar: latent parameters
        - beta: KL weight
        """
        # -------- Reconstruction Loss --------
        # Mean over batch and pixels (stabilizes scale)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / 128

        # -------- KL Divergence --------
        # Sum over latent dim, mean over batch
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # [B]
        kl_div = kl_div.mean()  # average over batch

        # -------- Total Loss --------
        total_loss = recon_loss + beta * kl_div

        return total_loss, recon_loss, kl_div

    def epoch_step(self):
        self.scheduler.step()

    def get_init_loss_dict(self):
        return {"total_loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}

    def get_model_state(self, epoch):
        return {
          "epoch": epoch,
          "weights": self.state_dict(),
          "scheduler_info" : self.scheduler.state_dict() 
        }

    def load_state(self, checkpoint):
      self.load_state_dict(checkpoint["weights"])
      self.scheduler.load_state_dict(checkpoint["scheduler_info"])

