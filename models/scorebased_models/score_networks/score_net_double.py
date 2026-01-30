from . import *
from ..ncsn import CondRefineBlock, ConditionalInstanceNorm2d

class DoubleScoreNet(nn.Module):
    def __init__(self, channels=128, num_scales=10, image_size=32, in_channels=3):
        super().__init__()

        self.embed_dim = 256
        self.noise_level_embed = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.ELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # Input conv
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)

        # Encoder: 8 blocks (matching ncsn_erm structure)
        # res1: 2 blocks at channels
        self.block1a = CondRefineBlock(channels, channels, num_classes=self.embed_dim)
        self.block1b = CondRefineBlock(channels, channels, num_classes=self.embed_dim)

        # res2: 2 blocks, downsample to 2*channels
        self.block2a = CondRefineBlock(channels, 2 * channels, num_classes=self.embed_dim)
        self.block2b = CondRefineBlock(2 * channels, 2 * channels, num_classes=self.embed_dim)

        # res3: 2 blocks with dilation=2
        self.block3a = CondRefineBlock(2 * channels, 2 * channels, dilation=2, num_classes=self.embed_dim)
        self.block3b = CondRefineBlock(2 * channels, 2 * channels, dilation=2, num_classes=self.embed_dim)

        # res4: 2 blocks with dilation=4
        self.block4a = CondRefineBlock(2 * channels, 2 * channels, dilation=4, num_classes=self.embed_dim)
        self.block4b = CondRefineBlock(2 * channels, 2 * channels, dilation=4, num_classes=self.embed_dim)

        # Decoder: 4 blocks (matching ncsn_erm refine blocks)
        self.up_block1 = CondRefineBlock(2 * channels, 2 * channels, num_classes=self.embed_dim)
        self.up_block2 = CondRefineBlock(2 * channels, 2 * channels, num_classes=self.embed_dim)
        self.up_block3 = CondRefineBlock(2 * channels, channels, num_classes=self.embed_dim)
        self.up_block4 = CondRefineBlock(channels, channels, num_classes=self.embed_dim)

        self.norm_out = ConditionalInstanceNorm2d(channels, self.embed_dim)
        self.conv_out = nn.Conv2d(channels, in_channels, 3, padding=1)

        self.conv_out.weight.data.normal_(0, 1e-10)
        if self.conv_out.bias is not None:
            self.conv_out.bias.data.zero_()

    def forward(self, x, sigma, sigma_idx=None):
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1)
        sigma_emb = self.noise_level_embed(torch.log(sigma))

        # Encoder
        h = self.conv_in(x)

        # res1 (no downsample)
        layer1 = self.block1a(h, sigma_emb)
        layer1 = self.block1b(layer1, sigma_emb)

        # res2 (downsample)
        layer2 = self.block2a(F.avg_pool2d(layer1, 2), sigma_emb)
        layer2 = self.block2b(layer2, sigma_emb)

        # res3 (downsample + dilation=2)
        layer3 = self.block3a(F.avg_pool2d(layer2, 2), sigma_emb)
        layer3 = self.block3b(layer3, sigma_emb)

        # res4 (downsample + dilation=4)
        layer4 = self.block4a(F.avg_pool2d(layer3, 2), sigma_emb)
        layer4 = self.block4b(layer4, sigma_emb)

        # Decoder with multi-input skip connections
        ref1 = self.up_block1(layer4, sigma_emb)

        ref2 = F.interpolate(ref1, size=layer3.shape[2:], mode='nearest') + layer3
        ref2 = self.up_block2(ref2, sigma_emb)

        ref3 = F.interpolate(ref2, size=layer2.shape[2:], mode='nearest') + layer2
        ref3 = self.up_block3(ref3, sigma_emb)

        ref4 = F.interpolate(ref3, size=layer1.shape[2:], mode='nearest') + layer1
        output = self.up_block4(ref4, sigma_emb)

        # Output
        output = self.norm_out(output, sigma_emb)
        output = F.elu(output)
        output = self.conv_out(output)

        return output
