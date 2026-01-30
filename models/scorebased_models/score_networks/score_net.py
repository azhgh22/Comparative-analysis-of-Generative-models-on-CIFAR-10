from . import *
from ..ncsn import CondRefineBlock, ConditionalInstanceNorm2d


class ScoreNet(nn.Module):
    def __init__(self, channels=128, num_scales=10, image_size=32, in_channels=3):
        super().__init__()

        # Gaussian Random Feature embedding for sigma (optional but stable)
        # Or simple One-hot/Linear embedding. Paper often used simple Dense layers.
        self.embed_dim = 256
        self.noise_level_embed = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.ELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # U-Net Encoder
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.block1 = CondRefineBlock(channels, channels, num_classes=self.embed_dim)
        self.block2 = CondRefineBlock(channels, 2 * channels, num_classes=self.embed_dim)
        self.block3 = CondRefineBlock(2 * channels, 2 * channels, dilation=2, num_classes=self.embed_dim)  # Dilated

        # Middle
        self.mid_block = CondRefineBlock(2 * channels, 2 * channels, dilation=4, num_classes=self.embed_dim)

        # U-Net Decoder (RefineNet style often combines features differently,
        # but this U-Net structure closely mimics the official NCSN code for CIFAR-10)
        self.up_block1 = CondRefineBlock(2 * channels, channels, num_classes=self.embed_dim)
        self.up_block2 = CondRefineBlock(channels, channels, num_classes=self.embed_dim)

        self.norm_out = ConditionalInstanceNorm2d(channels, self.embed_dim)
        self.conv_out = nn.Conv2d(channels, in_channels, 3, padding=1)

        # Initialize final conv layer to near zero to start as identity
        self.conv_out.weight.data.normal_(0, 1e-10)
        if self.conv_out.bias is not None:
            self.conv_out.bias.data.zero_()

    def forward(self, x, sigma, sigma_idx=None):
        """
        Args:
            x: Input image batch (B, C, H, W)
            sigma: Noise level (B,) or (B, 1)

        Returns:
            score: Predicted score (gradient of log density) (B, C, H, W)
        """
        # if sigma_idx is None:
        #     sigma_emb = self.noise_level_embed(torch.log(sigma))

        # Embed noise level
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1)

        # Log-transform sigma often helps stability
        sigma_emb = self.noise_level_embed(torch.log(sigma))

        # 2. Encoder
        h = self.conv_in(x)
        h1 = self.block1(h, sigma_emb)
        h2 = self.block2(F.avg_pool2d(h1, 2), sigma_emb)
        h3 = self.block3(F.avg_pool2d(h2, 2), sigma_emb)

        # 3. Middle
        h_mid = self.mid_block(h3, sigma_emb)

        # 4. Decoder (with upsampling)
        # Upsample h_mid
        # decoder stage 1
        h_up1 = F.interpolate(h_mid, scale_factor=2, mode='nearest')
        h_up1 = (
                h_up1
                + h2
                + F.interpolate(h3, scale_factor=2, mode="nearest")
        )
        h_up1 = self.up_block1(h_up1, sigma_emb)

        # decoder stage 2
        h_up2 = F.interpolate(h_up1, scale_factor=2, mode='nearest')
        h_up2 = h_up2 + h1
        h_up2 = self.up_block2(h_up2, sigma_emb)

        # 5. Output
        out = self.norm_out(h_up2, sigma_emb)
        out = F.elu(out)
        out = self.conv_out(out)

        return out