import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=256, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1 / num_embeddings, 1 / num_embeddings
        )

    def forward(self, z):
        """
        z: [B, C, H, W]
        returns:
            quantized: [B, C, H, W]
            vq_loss: scalar (mean over batch)
            encoding_indices: [B, H*W]
        """
        B, C, H, W = z.shape

        # Permute for convenience
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]
        flat_z = z_perm.view(B, -1, C)               # [B, H*W, C]

        # Compute distances
        # distances shape: [B, H*W, num_embeddings]
        distances = (
            flat_z.pow(2).sum(-1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )

        # Find nearest embedding per latent
        encoding_indices = torch.argmin(distances, dim=2)           # [B, H*W]
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()  # [B,H*W,K]

        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight)  # [B,H*W,C]
        quantized = quantized.view(B, H, W, C)                       # [B,H,W,C]

        # -------- Compute VQ Losses --------
        # sum over latents per sample, then mean over batch
        e_latent_loss = ((quantized.detach() - z_perm) ** 2).sum(dim=(1,2,3))  # [B]
        q_latent_loss = ((quantized - z_perm.detach()) ** 2).sum(dim=(1,2,3))  # [B]
        vq_loss = (q_latent_loss + self.commitment_cost * e_latent_loss).mean()

        # Straight-through estimator
        quantized = z_perm + (quantized - z_perm).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # back to [B,C,H,W]

        return quantized, vq_loss, encoding_indices


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            ResidualBlock(256),
            ResidualBlock(256),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)


import torch
import torch.nn as nn
import torch.nn.functional as F

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=256, commitment_cost=0.25):
        super().__init__()

        self.encoder = Encoder()
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=256,
            commitment_cost=commitment_cost,
        )
        self.decoder = Decoder()

        # Paper optimizer: Adam, lr = 2e-4
        lr = 2e-4
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    # -------- Core API --------

    def encode(self, x):
        """
        x: [B, 3, 32, 32]
        returns z_e: [B, 256, 8, 8]
        """
        return self.encoder(x)

    def quantize(self, z_e):
        """
        z_e: [B, 256, 8, 8]
        returns:
          z_q: [B, 256, 8, 8]
          vq_loss: scalar
          indices: [B * 8 * 8]
        """
        return self.vq(z_e)

    def decode(self, z_q):
        """
        z_q: [B, 256, 8, 8]
        returns recon: [B, 3, 32, 32]
        """
        return self.decoder(z_q)

    def forward(self, x):
        z_e = self.encode(x)
        z_q, vq_loss, indices = self.quantize(z_e)
        recon = self.decode(z_q)
        return recon, vq_loss, indices

    # -------- Loss --------

    def vqvae_loss(self, recon_x, x, vq_loss):
        """
        recon_x: [B, 3, 32, 32]
        x:       [B, 3, 32, 32]
        vq_loss: scalar
        """
        # Reconstruction loss (paper uses MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)

        total_loss = recon_loss + vq_loss
        return total_loss, recon_loss

    # -------- Training step --------

    def train_step(self, x, epoch=None):
        """
        Performs a single VQ-VAE training step
        """
        optimizer = self.optimizer

        # Forward
        recon_x, vq_loss, _ = self.forward(x)

        # Loss
        total_loss, recon_loss = self.vqvae_loss(recon_x, x, vq_loss)

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "vq_loss": vq_loss.item(),
        }

    # -------- Utilities --------

    def epoch_step(self):
      pass

    def get_init_loss_dict(self):
        return {
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "vq_loss": 0.0,
        }

    def get_model_state(self, epoch):
        return {
            "epoch": epoch,
            "weights": self.state_dict(),
        }

    def load_state(self, checkpoint):
        self.load_state_dict(checkpoint["weights"])


    

