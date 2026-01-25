import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ----------------------------
# Residual Block
# ----------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False, upsample=False):
        super().__init__()
        self.downsample = downsample
        self.upsample = upsample

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

        self.down = nn.AvgPool2d(2) if downsample else nn.Identity()
        self.up = nn.Upsample(scale_factor=2, mode='nearest') if upsample else nn.Identity()

    def forward(self, x):
        identity = x
        if self.upsample:
            x = self.up(x)
            identity = self.up(identity)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        identity = self.skip(identity)
        out = out + identity
        if self.downsample:
            out = self.down(out)
        out = self.relu(out)
        return out

# ----------------------------
# Encoder
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.block1 = ResBlock(64, 128, downsample=True)
        self.block2 = ResBlock(128, 256, downsample=True)
        self.block3 = ResBlock(256, 512, downsample=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_logvar = nn.Linear(512, z_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# ----------------------------
# Decoder / Generator
# ----------------------------
class Decoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.fc = nn.Linear(z_dim, 512*4*4)
        self.block1 = ResBlock(512, 256, upsample=True)
        self.block2 = ResBlock(256, 128, upsample=True)
        self.block3 = ResBlock(128, 64, upsample=True)
        self.conv_out = nn.Conv2d(64, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.fc(z).view(z.size(0), 512, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.conv_out(x)
        x = self.tanh(x)
        return x

# ----------------------------
# Discriminator
# ----------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.block1 = ResBlock(64, 128, downsample=True)
        self.block2 = ResBlock(128, 256, downsample=True)
        self.block3 = ResBlock(256, 512, downsample=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, feature_layer=False):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.block1(x)
        feat = self.block2(x)  # intermediate features
        x = self.block3(feat)
        x = self.pool(x).view(x.size(0), -1)
        out = self.sigmoid(self.fc(x))
        if feature_layer:
            return feat
        return out

# ----------------------------
# VAE-GAN wrapper (3 losses)
# ----------------------------
class VAEGAN(nn.Module):
    def __init__(self, encoder, decoder, discriminator, z_dim=128, gamma=1.0, device='cuda'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.device = device
        self.z_dim = z_dim
        self.gamma = gamma

        # Optimizers
        self.optimizer_enc = optim.Adam(self.encoder.parameters(), lr=2e-4, betas=(0.5,0.999))
        self.optimizer_dec = optim.Adam(self.decoder.parameters(), lr=2e-4, betas=(0.5,0.999))
        self.optimizer_dis = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5,0.999))

    # ----------------------------
    # Reparameterization trick
    # ----------------------------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ----------------------------
    # Losses
    # ----------------------------
    def compute_Lprior(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

    def compute_LDis_llike(self, x_real, x_fake):
        with torch.no_grad():
            feat_real = self.discriminator(x_real, feature_layer=True)
        feat_fake = self.discriminator(x_fake, feature_layer=True)
        return F.mse_loss(feat_fake, feat_real)

    def compute_LGAN(self, x_real, x_fake, x_prior):
        d_real = self.discriminator(x_real)
        d_fake = self.discriminator(x_fake)
        d_prior = self.discriminator(x_prior)
        return -torch.mean(torch.log(d_real + 1e-8) +
                           torch.log(1 - d_fake + 1e-8) +
                           torch.log(1 - d_prior + 1e-8))

    # ----------------------------
    # Train step (Algorithm 1 style)
    # ----------------------------
    def train_step(self, x, epoch=None):
        x = x.to(self.device)

        # Forward pass
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_tilde = self.decoder(z)
        z_prior = torch.randn_like(z)
        x_prior = self.decoder(z_prior)

        # 3 losses exactly like paper
        Lprior = self.compute_Lprior(mu, logvar)
        LDis_llike = self.compute_LDis_llike(x, x_tilde)
        LGAN = self.compute_LGAN(x, x_tilde, x_prior)

        # --------------------
        # Update discriminator
        # --------------------
        self.optimizer_dis.zero_grad()
        LGAN.backward(retain_graph=True)  # only discriminator sees this gradient
        self.optimizer_dis.step()

        # --------------------
        # Update encoder
        # --------------------
        self.optimizer_enc.zero_grad()
        (Lprior + LDis_llike).backward(retain_graph=True)
        self.optimizer_enc.step()

        # --------------------
        # Update decoder
        # --------------------
        self.optimizer_dec.zero_grad()
        (self.gamma * LDis_llike - LGAN).backward()
        self.optimizer_dec.step()

        return {
            "total_loss": (Lprior + LDis_llike + LGAN).item(),
            "recon_loss": LDis_llike.item(),
            "kl_loss": Lprior.item(),
            "gan_loss": LGAN.item()
        }

    # -------- Utilities --------
    def epoch_step(self):
        pass

    def get_init_loss_dict(self):
        return {
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "kl_loss": 0.0,
            "gan_loss": 0.0
        }

    def get_model_state(self, epoch):
        return {
            "epoch": epoch,
            "weights": self.state_dict(),
        }

    def load_state(self, checkpoint):
        self.load_state_dict(checkpoint["weights"])
