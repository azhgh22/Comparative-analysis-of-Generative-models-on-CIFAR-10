import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding)
        assert mask_type in ['A', 'B']
        self.mask_type = mask_type

        # Register mask buffer
        self.register_buffer('mask', torch.ones_like(self.weight))
        self.create_mask()

    def create_mask(self):
        kH, kW = self.kernel_size
        yc, xc = kH // 2, kW // 2
        self.mask[:, :, yc+1:, :] = 0
        self.mask[:, :, yc, xc + (self.mask_type == 'B'):] = 0

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias,
                        stride=self.stride, padding=self.padding, dilation=self.dilation)

class PixelCNN(nn.Module):
    def __init__(self, vqvae, num_embeddings ,hidden_dim=128, num_layers=7, kernel_size=3):
        super().__init__()
        self.vqvae = vqvae
        self.num_embeddings = num_embeddings
        padding = kernel_size // 2

        # First layer Mask-A
        self.input_conv = MaskedConv2d('A', self.num_embeddings, hidden_dim, kernel_size, padding)
        # Hidden layers Mask-B
        self.hidden_convs = nn.ModuleList([
            MaskedConv2d('B', hidden_dim, hidden_dim, kernel_size, padding) for _ in range(num_layers)
        ])
        # Output logits
        self.output_conv = nn.Conv2d(hidden_dim, self.num_embeddings, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)

    def forward(self, x_indices):
        # x_indices: [B,H,W]
        B, H, W = x_indices.shape
        x_onehot = F.one_hot(x_indices, self.num_embeddings).float()  # [B,H,W,K]
        x_onehot = x_onehot.permute(0, 3, 1, 2).contiguous()           # [B,K,H,W]

        h = F.relu(self.input_conv(x_onehot))
        for conv in self.hidden_convs:
            h = F.relu(conv(h))
        logits = self.output_conv(h)
        return logits

    def train_step(self, x, epoch=None):
        self.train()
        self.optimizer.zero_grad()

        # Get latent indices from VQ-VAE
        with torch.no_grad():
            _, _, indices_flat = self.vqvae(x)  # [B,H*W]
            B, HW = indices_flat.shape
            H = W = int(HW ** 0.5)
            x_indices = indices_flat.view(B, H, W)

        logits = self.forward(x_indices)        # [B,K,H,W]
        loss = F.cross_entropy(logits, x_indices)

        loss.backward()
        self.optimizer.step()
        return {"pixelcnn_loss": loss.item()}

    @torch.no_grad()
    def sample(self, batch_size=16):
        self.eval()
        vq = self.vqvae.vq
        # Use VQ-VAE latent size
        sample_input = torch.zeros(1, 3, 32, 32).cuda()
        _, _ ,_, indices_flat = self.vqvae(sample_input)
        B, HW = batch_size, indices_flat.shape[1]
        H = W = int(HW ** 0.5)

        sampled_indices = torch.zeros(B, H, W, dtype=torch.long).cuda()

        for i in range(H):
            for j in range(W):
                logits = self.forward(sampled_indices)
                probs = F.softmax(logits[:, :, i, j], dim=1)
                sampled_indices[:, i, j] = torch.multinomial(probs, 1).squeeze(-1)

        # Map indices to embeddings
        B, H, W = sampled_indices.shape
        C = vq.embedding_dim
        quantized = vq.embedding(sampled_indices.view(B, -1))      # [B,H*W,C]
        quantized = quantized.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Decode to images
        images = self.vqvae.decoder(quantized)
        return images

    def epoch_step(self):
        pass

    def get_init_loss_dict(self):
        return {"pixelcnn_loss": 0.0}

    def get_model_state(self, epoch):
        return {"epoch": epoch, "weights": self.state_dict()}

    def load_state(self, checkpoint):
        self.load_state_dict(checkpoint["weights"])
