import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# ResNet Blocks
# ============================================================
class ResBlockDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        x = self.skip(x)
        h = self.pool(h)
        x = self.pool(x)
        return F.relu(h + x)


class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        x = self.skip(x)
        return F.relu(h + x)


# ============================================================
# Encoder (ResNet CIFAR-10)
# ============================================================
class Encoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, 3, 1, 1)

        self.block1 = ResBlockDown(64, 128)   # 32 -> 16
        self.block2 = ResBlockDown(128, 256)  # 16 -> 8
        self.block3 = ResBlockDown(256, 512)  # 8 -> 4

        self.fc_mu = nn.Linear(512 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, z_dim)

    def forward(self, x):
        h = F.relu(self.conv_in(x))
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# ============================================================
# Decoder (ResNet CIFAR-10)
# ============================================================
class Decoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.fc = nn.Linear(z_dim, 512 * 4 * 4)

        self.block1 = ResBlockUp(512, 256)   # 4 -> 8
        self.block2 = ResBlockUp(256, 128)   # 8 -> 16
        self.block3 = ResBlockUp(128, 64)    # 16 -> 32

        self.conv_out = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), 512, 4, 4)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        x = torch.sigmoid(self.conv_out(h))
        return x


# ============================================================
# VampVAE
# ============================================================
class VampVae(nn.Module):
    def __init__(self, z_dim=128, n_pseudo=500, lr=2e-4):
        super().__init__()
        self.z_dim = z_dim
        self.n_pseudo = n_pseudo

        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

        # learnable pseudo-inputs
        self.pseudo_inputs = nn.Parameter(torch.randn(n_pseudo, 3, 32, 32))

        # optimizer inside model
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999))

    # --------------------------------------------------------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def log_normal(self, z, mu, logvar):
        return -0.5 * (logvar + (z - mu) ** 2 / torch.exp(logvar) + math.log(2 * math.pi)).sum(dim=1)

    def vamp_prior_logprob(self, z):
        mu_p, logvar_p = self.encoder(self.pseudo_inputs)

        z = z.unsqueeze(1)              # [B,1,D]
        mu_p = mu_p.unsqueeze(0)        # [1,K,D]
        logvar_p = logvar_p.unsqueeze(0)

        log_probs = -0.5 * (
            logvar_p + (z - mu_p) ** 2 / torch.exp(logvar_p) + math.log(2 * math.pi)
        ).sum(dim=2)

        log_prior = torch.logsumexp(log_probs, dim=1) - math.log(self.n_pseudo)
        return log_prior

    # --------------------------------------------------------
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decoder(z)
        return x_rec, mu, logvar, z

    # --------------------------------------------------------
    def train_step(self, x, epoch=None):
        self.optimizer.zero_grad()

        x_rec, mu, logvar, z = self.forward(x)

        # ----------------------------------
        # Reconstruction (Laplace likelihood → L1)
        # ----------------------------------
        recon_loss = F.l1_loss(x_rec, x, reduction="none")
        recon_loss = recon_loss.view(x.size(0), -1).sum(dim=1).mean()

        # ----------------------------------
        # KL(q(z|x) || VampPrior)
        # ----------------------------------
        log_qzx = self.log_normal(z, mu, logvar)
        log_pz = self.vamp_prior_logprob(z)
        kl = (log_qzx - log_pz).mean()

        # ----------------------------------
        # Total loss
        # ----------------------------------
        kl_weight = min(1.0, epoch / 20)
        loss = recon_loss + kl_weight * kl
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "recon": recon_loss.item(),
            "kl": kl.item()
        }

    # --------------------------------------------------------
    @torch.no_grad()
    def sample(self, n_samples):
        self.eval()
        idx = torch.randint(0, self.n_pseudo, (n_samples,), device=self.pseudo_inputs.device)
        u = self.pseudo_inputs[idx]
        mu, logvar = self.encoder(u)
        z = self.reparameterize(mu, logvar)
        x = self.decoder(z)
        return x

    # --------------------------------------------------------
    def get_init_loss_dict(self):
        return {"loss": 0.0, "recon": 0.0, "kl": 0.0}

    def get_model_state(self, epoch):
        return {"epoch": epoch, "weights": self.state_dict()}

    def load_state(self, checkpoint):
        self.load_state_dict(checkpoint["weights"])

    def epoch_step(self):
        pass
