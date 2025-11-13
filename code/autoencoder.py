import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.GroupNorm(32, out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0),
                nn.GroupNorm(32, out_channels)
            )
        self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(residual)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.block1 = ResidualBlock(3, 64, 2)
        self.block2 = ResidualBlock(64, 128, 2)
        self.block3 = ResidualBlock(128, 256, 2)
        self.block4 = ResidualBlock(256, 512, 2)
        self.block5 = ResidualBlock(512, 1024, 2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(4096, latent_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skip1 = self.block1(x)
        skip2 = self.block2(skip1)
        skip3 = self.block3(skip2)
        skip4 = self.block4(skip3)
        skip5 = self.block5(skip4)
        z = self.linear(self.flatten(skip5))
        skips = [skip1, skip2, skip3, skip4, skip5]
        return z, skips


class UNetDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Initial projection from latent to feature map (start at 2x2 like encoder ends)
        self.linear = nn.Linear(latent_dim, 1024 * 2 * 2)

        # U-Net decoder matching encoder's spatial resolutions
        # Encoder outputs: skips[4]=2x2 (1024), [3]=4x4 (512), [2]=8x8 (256), [1]=16x16 (128), [0]=32x32 (64)

        self.up1 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)  # 2x2 -> 4x4 (1024ch)
        self.conv1 = nn.Conv2d(1536, 512, 3, 1, 1)  # After concat with interpolated skips[4] (512+1024=1536)

        self.up2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)   # 4x4 -> 8x8 (512ch)
        self.conv2 = nn.Conv2d(768, 256, 3, 1, 1)   # After concat with skips[3] (256+512=768)

        self.up3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)   # 8x8 -> 16x16 (256ch)
        self.conv3 = nn.Conv2d(384, 128, 3, 1, 1)   # After concat with skips[2] (128+256=384)

        self.up4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)    # 16x16 -> 32x32 (128ch)
        self.conv4 = nn.Conv2d(192, 64, 3, 1, 1)    # After concat with skips[1] (64+128=192)

        self.up5 = nn.ConvTranspose2d(64, 32, 4, 2, 1)     # 32x32 -> 64x64 (64ch)
        self.conv5 = nn.Conv2d(96, 32, 3, 1, 1)     # After concat with skips[0] (32+64=96)

        self.final = nn.Conv2d(32, 3, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z, skips):
        x = self.linear(z).view(-1, 1024, 2, 2)  # Start at 2x2 like encoder

        # Level 1: 2x2 -> 4x4, concat with skips[4] (2x2->4x4)
        x = F.relu(self.up1(x))  # 512x4x4
        x = torch.cat([x, F.interpolate(skips[4], size=(4, 4), mode='bilinear', align_corners=False)], dim=1)  # 1024+512=1536x4x4
        x = F.relu(self.conv1(x))  # 512x4x4

        # Level 2: 4x4 -> 8x8, concat with skips[3] (4x4->8x8)
        x = F.relu(self.up2(x))  # 256x8x8
        x = torch.cat([x, F.interpolate(skips[3], size=(8, 8), mode='bilinear', align_corners=False)], dim=1)  # 768x8x8
        x = F.relu(self.conv2(x))  # 256x8x8

        # Level 3: 8x8 -> 16x16, concat with skips[2] (8x8->16x16)
        x = F.relu(self.up3(x))  # 128x16x16
        x = torch.cat([x, F.interpolate(skips[2], size=(16, 16), mode='bilinear', align_corners=False)], dim=1)  # 384x16x16
        x = F.relu(self.conv3(x))  # 128x16x16

        # Level 4: 16x16 -> 32x32, concat with skips[1] (16x16->32x32)
        x = F.relu(self.up4(x))  # 64x32x32
        x = torch.cat([x, F.interpolate(skips[1], size=(32, 32), mode='bilinear', align_corners=False)], dim=1)  # 192x32x32
        x = F.relu(self.conv4(x))  # 64x32x32

        # Level 5: 32x32 -> 64x64, concat with skips[0] (32x32->64x64)
        x = F.relu(self.up5(x))  # 32x64x64
        x = torch.cat([x, F.interpolate(skips[0], size=(64, 64), mode='bilinear', align_corners=False)], dim=1)  # 96x64x64
        x = F.relu(self.conv5(x))  # 32x64x64

        x = self.final(x)  # 3x64x64
        return torch.tanh(x)