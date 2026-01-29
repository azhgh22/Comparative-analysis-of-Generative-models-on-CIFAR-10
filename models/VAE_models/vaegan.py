import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()

        # 32x32 → 16x16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 16x16 → 8x8
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 8x8 → 4x4
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 256 * 4 * 4 = 4096 → 2048
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True)
        )

        # VAE heads
        self.fc_mu = nn.Linear(2048, z_dim)
        self.fc_logvar = nn.Linear(2048, z_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        h = self.fc(x)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()

        # z → 8*8*256
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 8 * 8 * 256),
            nn.BatchNorm1d(8 * 8 * 256),
            nn.ReLU(inplace=True)
        )

        # 8x8 → 16x16
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 256, kernel_size=5, stride=2,
                padding=2, output_padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 16x16 → 32x32
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=5, stride=2,
                padding=2, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # keep 32x32
        self.deconv3 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # output image
        self.out = nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=2)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 256, 8, 8)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)

        x = torch.tanh(self.out(x))
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # 32x32 → 32x32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )

        # 32x32 → 16x16
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 16x16 → 8x8
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 8x8 → 4x4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 256 * 4 * 4 = 4096 → 512
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        self.fc_out = nn.Linear(512, 1)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        features = x

        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        logits = self.fc_out(x)
        prob = torch.sigmoid(logits)

        if return_features:
            return prob, features
        return prob

class VaeGan(nn.Module):
    def __init__(self, z_dim=128, lr=3e-4, gamma=1.0, prior_weight=0.1):
        super().__init__()

        self.z_dim = z_dim
        self.gamma = gamma
        self.prior_weight = prior_weight

        # Networks
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.discriminator = Discriminator()

        # Optimizers
        self.opt_enc = torch.optim.RMSprop(self.encoder.parameters(), lr=lr)
        self.opt_dec = torch.optim.RMSprop(self.decoder.parameters(), lr=lr)
        self.opt_dis = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr)

        # Loss
        self.bce = nn.BCELoss()

    # -------------------------
    # Utils
    # -------------------------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def switch_mode(self, model, mode):
      for p in model.parameters():
        p.requires_grad_(mode)

    # -------------------------
    # Training step
    # -------------------------
    def train_step(self, x, epoch=None):
        batch_size = x.size(0)
        device = x.DEVICE

        ones = torch.ones(batch_size, 1, device=device)
        zeros = torch.zeros(batch_size, 1, device=device)

        # =======================
        # Forward
        # =======================
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)

        z_p = torch.randn_like(z)
        x_p = self.decoder(z_p)

        # =======================
        # Feature reconstruction loss (freeze D)
        # =======================
        for p in self.discriminator.parameters():
            p.requires_grad_(False)

        _, feat_real = self.discriminator(x, return_features=True)
        _, feat_fake = self.discriminator(x_hat, return_features=True)

        loss_rec = F.mse_loss(feat_fake, feat_real)

        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        # =======================
        # KL loss
        # =======================
        loss_prior = self.prior_weight * -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        # =======================
        # GAN loss
        # =======================
        prob_real = self.discriminator(x)
        prob_fake = self.discriminator(x_hat.detach())
        prob_prior = self.discriminator(x_p.detach())

        loss_dis = (
            self.bce(prob_real, ones) +
            self.bce(prob_fake, zeros) +
            self.bce(prob_prior, zeros)
        )

        # Generator GAN loss (no detach)
        prob_fake_g = self.discriminator(x_hat)
        prob_prior_g = self.discriminator(x_p)

        loss_gan = self.bce(prob_fake_g, ones) + self.bce(prob_prior_g, ones)

        # =======================
        # Total loss
        # =======================
        loss_total = (
            loss_dis +               # discriminator
            loss_prior +             # encoder
            self.gamma * loss_rec +  # encoder + decoder
            loss_gan                 # decoder
        )

        # =======================
        # Backprop (single step)
        # =======================
        self.opt_enc.zero_grad()
        self.opt_dec.zero_grad()
        self.opt_dis.zero_grad()

        loss_total.backward()

        self.opt_enc.step()
        self.opt_dec.step()
        self.opt_dis.step()

        return {
            "loss_total": loss_total.item(),
            "loss_dis": loss_dis.item(),
            "loss_rec": loss_rec.item(),
            "loss_prior": loss_prior.item(),
            "loss_gan": loss_gan.item(),
        }



    # -------------------------
    # Sampling
    # -------------------------
    @torch.no_grad()
    def sample(self, n_samples):
        device = next(self.parameters()).device
        z = torch.randn(n_samples, self.z_dim, device=device)
        x = self.decoder(z)
        return x

    # -------------------------
    # Logging helpers
    # -------------------------
    def get_init_loss_dict(self):
        return {
            "loss_total": 0.0,
            "loss_dis": 0.0,
            "loss_rec": 0.0,
            "loss_prior": 0.0,
            "loss_gan": 0.0,
        }

    def epoch_step(self):
        pass  # optional (e.g. LR schedulers)
        
    def get_model_state(self, epoch):
        return {"epoch": epoch, "weights": self.state_dict()}

    def load_state(self, checkpoint):
        self.load_state_dict(checkpoint["weights"])






