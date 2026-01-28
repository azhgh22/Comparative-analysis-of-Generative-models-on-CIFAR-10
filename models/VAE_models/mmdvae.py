import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_mmd(mu, prior_mu, sigmas=[1,2,4,8,16]):
    def kernel(x, y, sigma):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return torch.exp(-((x - y) ** 2).sum(2) / (2 * sigma ** 2))

    mmd = 0
    for s in sigmas:
        Kxx = kernel(mu, mu, s)
        Kyy = kernel(prior_mu, prior_mu, s)
        Kxy = kernel(mu, prior_mu, s)

        # REMOVE diagonal (self-similarity)
        n = mu.size(0)
        Kxx = (Kxx.sum() - torch.diagonal(Kxx).sum()) / (n*(n-1))
        Kyy = (Kyy.sum() - torch.diagonal(Kyy).sum()) / (n*(n-1))
        Kxy = Kxy.mean()

        mmd += Kxx + Kyy - 2*Kxy

    return mmd

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



# ---------- MMD-VAE ----------
class MMDVAE(nn.Module):
    def __init__(self, z_dim=128, beta=10.0, lr=2e-4, mmd_type=0, device="cuda"):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

        self.z_dim = z_dim
        self.beta = beta
        self.device = device
        self.mmd_type = mmd_type

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, z, mu, logvar

    # ---------- LOSS ----------
    def loss(self, x, x_hat, mu, logvar):
        recon = F.mse_loss(x_hat, x, reduction='sum')/x.shape[0]

        prior_mu = torch.randn_like(mu)
        mmd = compute_mmd(mu, prior_mu)

        # kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total = recon + self.beta * mmd #+ 0.1 * kl
        return total, recon, mmd

    # ---------- TRAIN STEP ----------
    def train_step(self, x,epoch=None):
        self.train()
        x = x.to(self.device)

        x_hat, z, mu, logvar = self.forward(x)
        loss, recon, mmd = self.loss(x, x_hat, mu, logvar)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "recon": recon.item(),
            "mmd": mmd.item(),
            # "kl" : kl.item()
        }

    # ---------- SAMPLE ----------
    @torch.no_grad()
    def sample(self, n):
        self.eval()
        z = torch.randn(n, self.z_dim).to(self.device)
        x = self.decoder(z)
        return x


    def get_init_loss_dict(self):
        return {
            "loss": 0.0,
            "recon": 0.0,
            "mmd": 0.0,
            # "kl": 0.0
        }

    def epoch_step(self):
        pass  # optional (e.g. LR schedulers)
        
    def get_model_state(self, epoch):
        return {"epoch": epoch, "weights": self.state_dict()}

    def load_state(self, checkpoint):
        self.load_state_dict(checkpoint["weights"])
