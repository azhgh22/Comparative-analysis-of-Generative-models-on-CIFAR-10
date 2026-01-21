import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.vae_loss import vae_loss  # assumes vae_loss returns (total_loss, recon_loss, kl_loss)

class ConvVAE(nn.Module):
    def __init__(self, z_dim=128, beta=1):
        super().__init__()
        self.beta = beta

        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(256*4*4, z_dim)
        self.fc_logvar = nn.Linear(256*4*4, z_dim)

        # ---------- Decoder ----------
        self.fc_dec = nn.Linear(z_dim, 256*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid()
        )

        # ---------- Optimizer & Scheduler placeholders ----------
        self.optimizer = None
        self.scheduler = None

        self.set_optimizer()

    # -------- Forward and latent functions --------
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), 256, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    # -------- Training step --------
    def train_step(self, x, optimizer=None):
        """
        Performs a single training step.
        - x: input batch
        - optimizer: optional, if not set use self.optimizer
        """
        if optimizer is None:
            if self.optimizer is None:
                raise ValueError("Optimizer not set. Pass optimizer or call set_optimizer().")
            optimizer = self.optimizer

        self.train()
        x = x.to(next(self.parameters()).device)

        # Forward
        recon_x, mu, logvar = self.forward(x)

        # Loss with beta
        total_loss, recon_loss, kl_loss = vae_loss(recon_x, x, mu, logvar)
        total_loss = recon_loss + self.beta * kl_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item()
        }

    # -------- Optional helpers --------
    def set_optimizer(self, lr=2e-4, weight_decay=0, scheduler_step=None, gamma=0.95):
        """
        Creates optimizer and optional LR scheduler
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        if scheduler_step is not None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=gamma)

    def step_epoch(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def get_init_loss_dict(self):
        return {"total_loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}

    def get_model_state(self, epoch):
        return {
          "epoch": epoch,
          "weights": self.state_dict(),
          "scheduler_info" : self.scheduler.state_dict()  
        }
