import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Encoder (CIFAR-10)
# ============================================================
class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, stride=2, padding=2)  # 32 -> 16
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)  # 16 -> 8
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 5, stride=2, padding=2)  # 8 -> 4
        self.bn3 = nn.BatchNorm2d(256)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, z_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# ============================================================
# Decoder (Generator)
# ============================================================
class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 4 * 4)
        self.bn = nn.BatchNorm1d(256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(256, 192, 5, stride=2, padding=2, output_padding=1)  # 4 -> 8
        self.bn1 = nn.BatchNorm2d(192)
        self.deconv2 = nn.ConvTranspose2d(192, 128, 5, stride=2, padding=2, output_padding=1)  # 8 -> 16
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 32, 5, stride=2, padding=2, output_padding=1)  # 16 -> 32
        self.bn3 = nn.BatchNorm2d(32)
        self.conv_out = nn.Conv2d(32, 3, 5, padding=2)

    def forward(self, z):
        x = F.relu(self.bn(self.fc(z)))
        x = x.view(-1, 256, 4, 4)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = torch.tanh(self.conv_out(x))
        return x


# ============================================================
# Discriminator
# ============================================================
class Discriminator(nn.Module):
    def __init__(self, recon_depth):
        super().__init__()
        self.recon_depth = recon_depth
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 128, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 192, 5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(192)
        self.conv4 = nn.Conv2d(192, 256, 5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = F.relu(self.bn5(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x

    def get_features(self, x, depth):
        if depth <= 0:
            return x
        x = F.relu(self.conv1(x))
        if depth == 1:
            return x
        x = F.relu(self.bn2(self.conv2(x)))
        if depth == 2:
            return x
        x = F.relu(self.bn3(self.conv3(x)))
        if depth == 3:
            return x
        x = F.relu(self.bn4(self.conv4(x)))
        if depth == 4:
            return x
        x = self.flatten(x)
        x = F.relu(self.bn5(self.fc1(x)))
        if depth == 5:
            return x
        x = torch.sigmoid(self.fc2(x))
        return x


# ============================================================
# VAE-GAN class
# ============================================================
class VAEGAN(nn.Module):
    def __init__(self, z_dim=128, recon_depth=3, gamma=1e-5, real_vs_gen_weight=0.5,
                 discriminate_ae_recon=True, discriminate_sample_z=True, device='cuda'):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.real_vs_gen_weight = real_vs_gen_weight
        self.discriminate_ae_recon = discriminate_ae_recon
        self.discriminate_sample_z = discriminate_sample_z

        self.encoder = Encoder(z_dim).to(device)
        self.decoder = Decoder(z_dim).to(device)
        self.discriminator = Discriminator(recon_depth).to(device)

        self.opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=2e-4, betas=(0.5, 0.9))
        self.opt_dec = torch.optim.Adam(self.decoder.parameters(), lr=2e-4, betas=(0.5, 0.9))
        self.opt_dis = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps
        recon = self.decoder(z)
        return recon

    def train(self):
      self.encoder.train()
      self.decoder.train()
      self.discriminator.train()
    
    def eval(self):
      self.encoder.eval()
      self.decoder.eval()
      self.discriminator.eval()

    def switch_mode(self,model, mode):
        for param in model.parameters():
            param.requires_grad = mode

    def train_step(self, x, epoch=None):
        # x = x.to(self.device)
        batch_size = x.size(0)

        # ============================================================
        # 1) ENCODER UPDATE: KL + FEATURE LOSS
        # ============================================================
        self.encoder.train()
        self.decoder.eval()
        self.discriminator.eval()  # freeze discriminator

        self.switch_mode(self.encoder,True)
        self.switch_mode(self.decoder,False)
        self.switch_mode(self.discriminator,False)

        self.opt_enc.zero_grad()

        # Encode + reparameterization
        mu, logvar = self.encoder(x)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps

        # Feature reconstruction loss (encoder sees decoder output)
        # with torch.no_grad():
        recon = self.decoder(z)
        real_feat = self.discriminator.get_features(x, self.discriminator.recon_depth)
        recon_feat = self.discriminator.get_features(recon, self.discriminator.recon_depth)

        # KL divergence
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        feat_loss = F.mse_loss(recon_feat, real_feat)

        # Encoder loss
        enc_loss = 0.1*kld + feat_loss
        enc_loss.backward()
        self.opt_enc.step()

        # ============================================================
        # 2) DISCRIMINATOR UPDATE: maximize LGAN
        # LGAN = log(D(x)) + log(1-D(Dec(z))) + log(1-D(Dec(Enc(x))))
        # ============================================================
        self.discriminator.train()
        self.decoder.eval()
        self.encoder.eval()

        self.switch_mode(self.encoder,False)
        self.switch_mode(self.decoder,False)
        self.switch_mode(self.discriminator,True)

        self.opt_dis.zero_grad()

        # Real
        d_real = self.discriminator(x)
        loss_real = F.binary_cross_entropy(d_real, torch.ones_like(d_real))

        # Fake from prior
        with torch.no_grad():
            z_prior = torch.randn(batch_size, mu.size(1), device=self.device)
            fake_z = self.decoder(z_prior).detach()
        
        d_fake_z = self.discriminator(fake_z)
        loss_fake_z = F.binary_cross_entropy(d_fake_z, torch.zeros_like(d_fake_z))

        # Fake from autoencoder
        fake_rec = recon.detach()
        d_fake_rec = self.discriminator(fake_rec)
        loss_fake_rec = F.binary_cross_entropy(d_fake_rec, torch.zeros_like(d_fake_rec))

        d_loss = loss_real + loss_fake_z + loss_fake_rec
        d_loss.backward()
        self.opt_dis.step()

        # ============================================================
        # 3) DECODER UPDATE: γ * FEATURE LOSS − LGAN
        # ============================================================
        self.encoder.eval()
        self.discriminator.eval()
        self.decoder.train()

        self.switch_mode(self.encoder,False)
        self.switch_mode(self.decoder,True)
        self.switch_mode(self.discriminator,False)

        self.opt_dec.zero_grad()

        # Recompute decoder output (clean graph)
        mu, logvar = self.encoder(x)
        # eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps
        recon = self.decoder(z.detach())

        # Feature loss (decoder wants to match real features)
        # with torch.no_grad():
        real_feat = self.discriminator.get_features(x, self.discriminator.recon_depth)
        recon_feat = self.discriminator.get_features(recon, self.discriminator.recon_depth)
        feat_loss_dec = F.mse_loss(recon_feat, real_feat)

        # GAN loss (decoder wants D(fake) ~ 1)
        # z_prior = torch.randn(batch_size, mu.size(1), device=self.device)
        fake_z = self.decoder(z_prior)
        # with torch.no_grad():
        g_fake_z = self.discriminator(fake_z)
        g_fake_rec = self.discriminator(recon)

        gan_loss = F.binary_cross_entropy(g_fake_z, torch.ones_like(g_fake_z)) + \
                  F.binary_cross_entropy(g_fake_rec, torch.ones_like(g_fake_rec))

        dec_loss = self.gamma * feat_loss_dec - gan_loss
        dec_loss.backward()
        self.opt_dec.step()

        return {
            'kld': kld.item(),
            'recon': feat_loss.item(),
            'd_loss': d_loss.item(),
            'g_loss': gan_loss.item(),
        }



    def get_init_loss_dict(self):
        return {
            'kld': 0.0,
            'recon': 0.0,
            'd_loss': 0.0,
            'g_loss': 0.0,
        }

    def get_model_state(self, epoch):
        return {
            "epoch": epoch,
            "weights": self.state_dict(),
        }

    def load_state(self, checkpoint):
        self.load_state_dict(checkpoint["weights"])

    def epoch_step(self):
        pass