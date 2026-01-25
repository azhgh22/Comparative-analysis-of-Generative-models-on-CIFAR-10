import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Sub-networks (Required for VAEGAN) ---
class Encoder(nn.Module):
    def __init__(self, latent_dim=128, channels=3):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        return self.reparameterize(self.fc_mu(x), self.fc_logvar(x)), self.fc_mu(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, channels=3):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1), nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z).view(z.shape[0], 256, 4, 4)
        return self.conv_blocks(out)

class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, 32, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(nn.Conv2d(32, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        ])
        self.adv_layer = nn.Sequential(nn.Linear(256 * 4 * 4, 1), nn.Sigmoid())

    def forward(self, img):
        out = img
        for block in self.blocks:
            out = block(out)
        features = out.view(out.shape[0], -1)
        validity = self.adv_layer(features)
        return validity, features

# --- Main VAEGAN Class ---
class VAEGAN(nn.Module):
    def __init__(self, latent_dim=128, channels=3, gamma=1e-6, lr=0.0003):
        super(VAEGAN, self).__init__()
        self.latent_dim = latent_dim
        self.gamma = gamma
        
        # Submodules
        self.encoder = Encoder(latent_dim, channels)
        self.decoder = Decoder(latent_dim, channels)
        self.discriminator = Discriminator(channels)
        
        # Optimizers initialized directly in constructor
        # Note: Discriminator often benefits from a lower LR in VAE-GANs
        self.opt_enc = torch.optim.RMSprop(self.encoder.parameters(), lr=lr)
        self.opt_dec = torch.optim.RMSprop(self.decoder.parameters(), lr=lr)
        self.opt_dis = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr * 0.1)

    def forward(self, x):
        """Standard VAE forward pass for inference."""
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def train_step(self, x, epoch=None):
        """
        Performs a single VAE-GAN training step with sequential updates.
        FIXED: Detaches z for Decoder update to prevent inplace operation errors.
        """
        batch_size = x.size(0)
        device = x.device
        
        # Labels
        real_label = torch.ones(batch_size, 1).to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)

        # ---------------------
        #  1. Train Discriminator
        # ---------------------
        self.opt_dis.zero_grad()
        
        # Real
        real_validity, _ = self.discriminator(x)
        d_real_loss = F.binary_cross_entropy(real_validity, real_label)
        
        # Reconstruction (Detach to stop gradients to Encoder/Decoder)
        with torch.no_grad():
            z, _, _ = self.encoder(x)
            recon_imgs = self.decoder(z)
        
        recon_validity, _ = self.discriminator(recon_imgs.detach())
        d_recon_loss = F.binary_cross_entropy(recon_validity, fake_label)
        
        # Random Sampling
        z_p = torch.randn(batch_size, self.latent_dim).to(device)
        gen_imgs = self.decoder(z_p)
        gen_validity, _ = self.discriminator(gen_imgs.detach())
        d_gen_loss = F.binary_cross_entropy(gen_validity, fake_label)
        
        d_loss = d_real_loss + d_recon_loss + d_gen_loss
        d_loss.backward()
        self.opt_dis.step()

        # ---------------------
        #  2. Train Encoder
        # ---------------------
        self.opt_enc.zero_grad()
        
        # Forward pass for Encoder
        z, mu, logvar = self.encoder(x)
        recon_imgs = self.decoder(z)
        
        # Feature Matching Loss
        # We use features from the Discriminator (frozen)
        with torch.no_grad():
            _, real_feats = self.discriminator(x)
            
        _, recon_feats = self.discriminator(recon_imgs)
        
        feature_loss = F.mse_loss(recon_feats, real_feats.detach())
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        enc_loss = kld_loss + feature_loss
        
        # We do NOT need retain_graph=True anymore because we will not reuse this graph
        enc_loss.backward()
        self.opt_enc.step()

        # ---------------------
        #  3. Train Decoder
        # ---------------------
        self.opt_dec.zero_grad()
        
        # Fresh forward pass for Decoder
        # CRITICAL FIX: We use .detach() on z. 
        # The Decoder update should NOT backprop to the Encoder.
        with torch.no_grad():
            z, _, _ = self.encoder(x)
        z = z.detach() 
        
        recon_imgs = self.decoder(z)
        
        # We need to re-calculate feature loss for the Decoder's graph
        # (It's cheap and prevents the graph error)
        with torch.no_grad():
            _, real_feats = self.discriminator(x)
        _, recon_feats = self.discriminator(recon_imgs)
        
        feature_loss_dec = F.mse_loss(recon_feats, real_feats.detach())
        
        # GAN Fooling Loss
        recon_validity, _ = self.discriminator(recon_imgs)
        g_loss = F.binary_cross_entropy(recon_validity, real_label)
        
        # Combined Decoder Loss
        dec_loss = (self.gamma * feature_loss_dec) + g_loss
        
        dec_loss.backward()
        self.opt_dec.step()

        return {
            "total_loss": dec_loss.item() + enc_loss.item() + d_loss.item(),
            "recon_loss": feature_loss.item(),
            "kld_loss": kld_loss.item(),
            "gan_loss": g_loss.item(),
            "disc_loss": d_loss.item()
        }

    # -------- Utilities --------

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