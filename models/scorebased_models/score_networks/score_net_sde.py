from models.scorebased_models.score_networks import *

# ============================================================================
# Score Network Architecture (NCSNv2 / DDPM++ style)
# ============================================================================

class ScoreNetworkSDE(nn.Module):
    """
    Score Network for SDE-based diffusion models.

    Based on the NCSN++ architecture from the score SDE paper.
    Uses time conditioning via Gaussian Fourier features or sinusoidal embeddings.
    """

    def __init__(self,
                 in_channels=3,
                 channels=128,
                 out_channels=None,
                 ch_mult=(1, 2, 2, 2),
                 num_res_blocks=2,
                 attn_resolutions=(16,),
                 dropout=0.1,
                 resamp_with_conv=True,
                 image_size=32,
                 embedding_type='fourier',
                 fourier_scale=16.):
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels
        out_channels = out_channels if out_channels is not None else in_channels
        self.out_channels = out_channels
        self.ch_mult = ch_mult
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.image_size = image_size

        # Time embedding
        self.temb_ch = channels * 4
        if embedding_type == 'fourier':
            self.time_embed = nn.Sequential(
                GaussianFourierProjection(channels, scale=fourier_scale),
                nn.Linear(channels, self.temb_ch),
                nn.SiLU(),
                nn.Linear(self.temb_ch, self.temb_ch),
            )
        else:  # 'positional'
            self.time_embed = nn.Sequential(
                nn.Linear(channels, self.temb_ch),
                nn.SiLU(),
                nn.Linear(self.temb_ch, self.temb_ch),
            )
        self.embedding_type = embedding_type

        # Downsampling
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)

        curr_res = image_size
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * in_ch_mult[i_level]
            block_out = channels * ch_mult[i_level]

            for _ in range(num_res_blocks):
                block.append(ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=dropout
                ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout
        )

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * ch_mult[i_level]
            skip_in = channels * ch_mult[i_level]

            for i_block in range(num_res_blocks + 1):
                if i_block == num_res_blocks:
                    skip_in = channels * in_ch_mult[i_level]
                block.append(ResnetBlock(
                    in_channels=block_in + skip_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=dropout
                ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # Output
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, 3, padding=1)

        # Initialize final layer to near zero
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, t):
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)
            t: Time tensor (B,) in [0, 1]

        Returns:
            Score estimate (B, C, H, W)
        """
        # Time embedding
        if self.embedding_type == 'fourier':
            temb = self.time_embed(t)
        else:
            temb = get_timestep_embedding(t * 1000, self.channels)  # Scale t for better embeddings
            temb = self.time_embed(temb)

        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Output
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h

