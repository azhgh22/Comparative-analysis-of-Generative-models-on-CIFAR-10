import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- MMD Kernel ----------
def compute_mmd(z, prior_z, sigma=1.0):
    def kernel(x, y):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return torch.exp(-((x - y)**2).sum(2) / (2*sigma**2))

    Kxx = kernel(z, z).mean()
    Kyy = kernel(prior_z, prior_z).mean()
    Kxy = kernel(z, prior_z).mean()
    return Kxx + Kyy - 2*Kxy


def compute_mmd1(z, prior_z, sigmas=[1,2,4,8,16]):
    def kernel(x, y, sigma):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return torch.exp(-((x - y)**2).sum(2) / (2*sigma**2))

    mmd = 0
    for s in sigmas:
        Kxx = kernel(z,z,s).mean()
        Kyy = kernel(prior_z,prior_z,s).mean()
        Kxy = kernel(z,prior_z,s).mean()
        mmd += Kxx + Kyy - 2*Kxy
    return mmd

# ---------- Encoder ----------
class Encoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256*4*4, z_dim)
        self.fc_logvar = nn.Linear(256*4*4, z_dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


# ---------- Decoder ----------
class Decoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256*4*4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,3,4,2,1),
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0),256,4,4)
        return self.deconv(h)


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
    def loss(self, x, x_hat, z):
        recon = F.mse_loss(x_hat, x, reduction='mean')
        prior_z = torch.randn_like(z)
        if not self.mmd_type:
          mmd = compute_mmd(z, prior_z)
        else:
          mmd = compute_mmd1(z, prior_z)
        total = recon + self.beta * mmd
        return total, recon, mmd

    # ---------- TRAIN STEP ----------
    def train_step(self, x,epoch=None):
        self.train()
        x = x.to(self.device)

        x_hat, z, mu, logvar = self.forward(x)
        loss, recon, mmd = self.loss(x, x_hat, z)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "recon": recon.item(),
            "mmd": mmd.item()
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
            "mmd": 0.0
        }

    def epoch_step(self):
        pass  # optional (e.g. LR schedulers)
        
    def get_model_state(self, epoch):
        return {"epoch": epoch, "weights": self.state_dict()}

    def load_state(self, checkpoint):
        self.load_state_dict(checkpoint["weights"])
