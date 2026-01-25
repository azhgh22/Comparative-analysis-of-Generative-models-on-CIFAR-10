import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Encoder
# ============================================================
class Encoder(nn.Module):
    def __init__(self, latent_dim=128, channels=3):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.conv_blocks(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


# ============================================================
# Decoder
# ============================================================
class Decoder(nn.Module):
    def __init__(self, latent_dim=128, channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 256, 4, 4)
        return self.conv_blocks(h)


# ============================================================
# Discriminator (NO sigmoid, WGAN-style)
# ============================================================
class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Linear(256 * 4 * 4, 1)

    def forward(self, x):
        h = self.blocks(x)
        features = h.view(h.size(0), -1)
        score = self.fc(features)
        return score, features


# ============================================================
# VAE-GAN (API preserved)
# ============================================================
class VAEGAN(nn.Module):
    def __init__(self, latent_dim=128, channels=3, gamma=1e-3, lr=2e-4):
        super().__init__()
        self.latent_dim = latent_dim
        self.gamma = gamma

        self.encoder = Encoder(latent_dim, channels)
        self.decoder = Decoder(latent_dim, channels)
        self.discriminator = Discriminator(channels)

        # --- Optimizers (same API as your version) ---
        self.opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(0.5, 0.9))
        self.opt_dec = torch.optim.Adam(self.decoder.parameters(), lr=lr, betas=(0.5, 0.9))
        self.opt_dis = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    # ========================================================
    # Training Step
    # ========================================================
    def train_step(self, x, epoch=None):
        batch_size = x.size(0)
        device = x.device
        
        # Labels for GAN loss (1 = Real, 0 = Fake)
        real_label = torch.ones(batch_size, 1).to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)

        # ============================================================
        # 1. Forward Pass (Shared)
        # ============================================================
        z, mu, logvar = self.encoder(x)
        x_tilde = self.decoder(z)

        # ============================================================
        # 2. Update Encoder (theta_Enc)
        # ============================================================
        self.opt_enc.zero_grad()

        # KL Divergence
        Lprior = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Feature Matching (L_llike)
        # We need features from Dis(X) and Dis(X_tilde)
        with torch.no_grad():
            _, feat_real = self.discriminator(x)
            
        _, feat_fake = self.discriminator(x_tilde)
        
        LDis_llike = F.mse_loss(feat_fake, feat_real.detach())
        
        loss_encoder = Lprior + LDis_llike
        loss_encoder.backward()
        self.opt_enc.step()

        # ============================================================
        # 3. Update Decoder (theta_Dec)
        # ============================================================
        self.opt_dec.zero_grad()

        # CRITICAL FIX 1: Detach Z to prevent "inplace operation" error
        z_detached = z.detach()
        x_tilde_dec = self.decoder(z_detached)

        # Sample from prior
        z_p = torch.randn_like(z)
        x_p = self.decoder(z_p)

        # Re-calculate L_llike for Decoder gradients
        with torch.no_grad():
            _, feat_real = self.discriminator(x)
        _, feat_fake_dec = self.discriminator(x_tilde_dec)
        LDis_llike_dec = F.mse_loss(feat_fake_dec, feat_real.detach())

        # GAN Loss (Generator view)
        pred_fake_dec, _ = self.discriminator(x_tilde_dec)
        pred_prior_dec, _ = self.discriminator(x_p)
        
        # CRITICAL FIX 2: Use BCE with LOGITS because Discriminator has no Sigmoid
        loss_gan_gen = F.binary_cross_entropy_with_logits(pred_fake_dec, real_label) + \
                       F.binary_cross_entropy_with_logits(pred_prior_dec, real_label)

        loss_decoder = (self.gamma * LDis_llike_dec) + loss_gan_gen
        
        loss_decoder.backward()
        self.opt_dec.step()

        # ============================================================
        # 4. Update Discriminator (theta_Dis)
        # ============================================================
        self.opt_dis.zero_grad()

        # Detach inputs so we don't backprop to Enc/Dec
        x_tilde_det = x_tilde_dec.detach()
        x_p_det = x_p.detach()

        pred_real, _ = self.discriminator(x)
        pred_fake, _ = self.discriminator(x_tilde_det)
        pred_prior, _ = self.discriminator(x_p_det)
        
        # CRITICAL FIX 2: Use BCE with LOGITS
        loss_gan_dis = F.binary_cross_entropy_with_logits(pred_real, real_label) + \
                       F.binary_cross_entropy_with_logits(pred_fake, fake_label) + \
                       F.binary_cross_entropy_with_logits(pred_prior, fake_label)

        loss_gan_dis.backward()
        self.opt_dis.step()

        # ============================================================
        # Return Stats
        # ============================================================
        return {
            "total_loss": loss_decoder.item() + loss_encoder.item() + loss_gan_dis.item(),
            "recon_loss": LDis_llike.item(),
            "kld_loss": Lprior.item(),
            "gan_loss": loss_gan_gen.item(),
            "disc_loss": loss_gan_dis.item(),
        }


    # -------- Utilities (unchanged API) --------
    def epoch_step(self):
        pass

    def get_init_loss_dict(self):
        return {
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "kld_loss": 0.0,
            "gan_loss": 0.0,
            "disc_loss": 0.0
        }

    def get_model_state(self, epoch):
        return {
            "epoch": epoch,
            "weights": self.state_dict(),
        }

    def load_state(self, checkpoint):
        self.load_state_dict(checkpoint["weights"])
