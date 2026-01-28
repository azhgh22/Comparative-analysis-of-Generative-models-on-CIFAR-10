import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Noise schedule (matches training.noise_std)
# ============================================================

def get_sigmas(
    sigma_min=0.01,
    sigma_max=1.0,
    num_sigmas=10,
    device="cuda"
):
    return torch.exp(
        torch.linspace(
            torch.log(torch.tensor(sigma_max)),
            torch.log(torch.tensor(sigma_min)),
            num_sigmas
        )
    ).to(device)


# ============================================================
# Sigma-conditioned ResBlock (author-style)
# ============================================================

class UNetResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, down=False, up=False):
        super().__init__()
        self.down = down
        self.up = up

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.emb_proj = nn.Linear(emb_dim, out_ch)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

        if down:
            self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        if up:
            self.upsample = nn.ConvTranspose2d(
                out_ch, out_ch, 4, stride=2, padding=1
            )

    def forward(self, x, emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = F.elu(h)

        h = self.conv2(h)
        h = self.norm2(h)

        h = h + self.skip(x)
        h = F.elu(h)

        if self.down:
            h = self.downsample(h)
        if self.up:
            h = self.upsample(h)

        return h


# ============================================================
# UNet-style Score Network (CIFAR-10)
# ============================================================

class UNetScoreNet(nn.Module):
    def __init__(self, base_ch=32, emb_dim=128):
        super().__init__()

        # sigma embedding
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.ELU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # down path
        self.conv_in = nn.Conv2d(3, base_ch, 3, padding=1)

        self.down1 = UNetResBlock(base_ch, base_ch, emb_dim)
        self.down2 = UNetResBlock(base_ch, base_ch * 2, emb_dim, down=True)
        self.down3 = UNetResBlock(base_ch * 2, base_ch * 2, emb_dim, down=True)

        # bottleneck
        self.mid = UNetResBlock(base_ch * 2, base_ch * 2, emb_dim)

        # up path
        self.up3 = UNetResBlock(base_ch * 2, base_ch * 2, emb_dim, up=True)
        self.up2 = UNetResBlock(base_ch * 4, base_ch, emb_dim, up=True)
        self.up1 = UNetResBlock(base_ch * 2, base_ch, emb_dim)

        self.conv_out = nn.Conv2d(base_ch, 3, 3, padding=1)

    def forward(self, x, sigma):
        emb = self.sigma_embed(sigma[:, None])

        # config.data.logit_transform == false
        x = 2.0 * x - 1.0

        h1 = self.down1(self.conv_in(x), emb)   # 32×32
        h2 = self.down2(h1, emb)                # 16×16
        h3 = self.down3(h2, emb)                # 8×8

        h = self.mid(h3, emb)

        h = self.up3(h, emb)                    # 16×16
        h = torch.cat([h, h2], dim=1)

        h = self.up2(h, emb)                    # 32×32
        h = torch.cat([h, h1], dim=1)

        h = self.up1(h, emb)

        return self.conv_out(h)                 # score ∇x log qσ(x)


# ============================================================
# Noise Conditional Score Network (API-aligned)
# ============================================================

class NCSN(nn.Module):
    def __init__(self, lr=1e-3, device="cuda"):
        super().__init__()

        self.device = device

        # model.model.nef = 32
        self.net = UNetScoreNet(
            base_ch=32,
            emb_dim=128
        ).to(device)

        # training.noise_std = 0.01
        self.sigmas = get_sigmas(
            sigma_min=0.01,
            sigma_max=1.0,
            num_sigmas=10,
            device=device
        )

        # optim settings
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,              # optim.lr
            betas=(0.9, 0.999),
            weight_decay=0.0
        )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, x, sigma):
        return self.net(x, sigma)

    # --------------------------------------------------------
    # Weighted DSM / SSM Loss
    # --------------------------------------------------------
    def loss(self, x):
        B = x.size(0)
        device = x.device

        sigma = self.sigmas[
            torch.randint(0, len(self.sigmas), (B,), device=device)
        ]

        noise = torch.randn_like(x) * sigma[:, None, None, None]
        x_tilde = x + noise

        score = self.forward(x_tilde, sigma)
        target = -noise / (sigma[:, None, None, None] ** 2)

        per_sample = (score - target).pow(2).sum(dim=(1, 2, 3))
        loss = 0.5 * (sigma ** 2 * per_sample).mean()

        return loss

    # --------------------------------------------------------
    # Train step (API-compatible)
    # --------------------------------------------------------
    def train_step(self, x, epoch=None):
        self.train()
        x = x.to(self.device)

        loss = self.loss(x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    # --------------------------------------------------------
    # Annealed Langevin Sampling
    # --------------------------------------------------------
    @torch.no_grad()
    def sample(self, n, steps_per_sigma=100, eps=2e-5):
        self.eval()

        x = torch.rand(n, 3, 32, 32, device=self.device)
        sigma_L = self.sigmas[-1]

        for sigma in self.sigmas:
            alpha = eps * (sigma ** 2) / (sigma_L ** 2)

            for _ in range(steps_per_sigma):
                z = torch.randn_like(x)
                score = self.forward(
                    x, torch.full((n,), sigma, device=self.device)
                )
                x = x + 0.5 * alpha * score + torch.sqrt(alpha) * z

        return x.clamp(0.0, 1.0)

    # --------------------------------------------------------
    # API helpers
    # --------------------------------------------------------
    def get_init_loss_dict(self):
        return {"loss": 0.0}

    def epoch_step(self):
        pass

    def get_model_state(self, epoch):
        return {"epoch": epoch, "weights": self.state_dict()}

    def load_state(self, checkpoint):
        self.load_state_dict(checkpoint["weights"])
