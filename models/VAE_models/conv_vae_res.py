import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False, upsample=False):
        super().__init__()
        self.downsample = downsample
        self.upsample = upsample

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

        if downsample:
            self.pool = nn.AvgPool2d(2)
        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        h = x
        if self.upsample:
            h = self.up(h)
            x = self.up(x)

        h = self.conv1(h)
        h = self.bn1(h)
        h = F.relu(h)

        h = self.conv2(h)
        h = self.bn2(h)

        if self.downsample:
            h = self.pool(h)
            x = self.pool(x)

        return F.relu(h + self.skip(x))



# ---------- Encoder ----------
class Encoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, 3, padding=1)

        self.block1 = ResBlock(64, 128, downsample=True)
        self.block2 = ResBlock(128, 256, downsample=True)
        self.block3 = ResBlock(256, 512, downsample=True)

        self.fc_mu = nn.Linear(512*4*4, z_dim)
        self.fc_logvar = nn.Linear(512*4*4, z_dim)

    def forward(self, x):
        h = self.conv_in(x)
        h = self.block1(h)   # 16x16
        h = self.block2(h)   # 8x8
        h = self.block3(h)   # 4x4
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)



class Decoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.fc = nn.Linear(z_dim, 512*4*4)

        self.block1 = ResBlock(512, 256, upsample=True)
        self.block2 = ResBlock(256, 128, upsample=True)
        self.block3 = ResBlock(128, 64, upsample=True)

        self.conv_out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 512, 4, 4)
        h = self.block1(h)   # 8x8
        h = self.block2(h)   # 16x16
        h = self.block3(h)   # 32x32
        return torch.sigmoid(self.conv_out(h))


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

        # warmup_epochs = 20
        # beta = min(1.0, epoch / warmup_epochs)
        beta = 0.1

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
        # recon_loss = F.l1_loss(recon_x, x, reduction='sum') / 128

        # -------- KL Divergence --------
        # Sum over latent dim, mean over batch
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # [B]
        kl_div = kl_div.mean()  # average over batch

        # -------- Total Loss --------
        total_loss = recon_loss + beta * kl_div

        return total_loss, recon_loss, kl_div

    @torch.no_grad()
    def sample(self, n_samples=16,temperature=1.0):
        device = next(self.parameters()).device
        z = torch.randn(
            n_samples,
            self.encoder.fc_mu.out_features,
            device=device
        ) * temperature
        return self.decode(z)

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

