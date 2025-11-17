import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import l_recon, l_div, kl_divergence_loss, PerceptualLoss, Discriminator, adversarial_loss


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
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512 * 4, latent_dim)
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
        z = self.linear(self.flatten(skip4))
        skips = [skip1, skip2, skip3, skip4]
        return z, skips


class UNetDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Initial projection from latent to feature map (start at 2x2 like encoder ends)
        self.linear = nn.Linear(latent_dim, 512 * 2 * 2)

        # Simplified decoder for 32x32 output
        self.up1 = nn.ConvTranspose2d(512, 256, 2, 2, 0)   # 2x2 -> 4x4
        self.conv1 = nn.Conv2d(256, 256, 1, 1, 0)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2, 0)   # 4x4 -> 8x8
        self.conv2 = nn.Conv2d(128, 128, 1, 1, 0)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2, 0)    # 8x8 -> 16x16
        self.conv3 = nn.Conv2d(64, 64, 1, 1, 0)

        self.up4 = nn.ConvTranspose2d(64, 32, 2, 2, 0)     # 16x16 -> 32x32
        self.conv4 = nn.Conv2d(32, 32, 1, 1, 0)

        self.final = nn.Conv2d(32, 3, 1, 1, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z, skips=None):
        x = self.linear(z).view(-1, 512, 2, 2)  # Start at 2x2

        x = F.relu(self.up1(x))
        x = F.relu(self.conv1(x))

        x = F.relu(self.up2(x))
        x = F.relu(self.conv2(x))

        x = F.relu(self.up3(x))
        x = F.relu(self.conv3(x))

        x = F.relu(self.up4(x))
        x = F.relu(self.conv4(x))

        x = self.final(x)
        return torch.tanh(x)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = UNetDecoder(latent_dim)
        self.perceptual_loss = PerceptualLoss()
        self.discriminator = Discriminator()

    def encode(self, x):
        z, _ = self.encoder(x)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, skips = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def compute_losses(self, x, x_hat, z, omega_dict):
        """
        Compute the composite loss using omega_dict weights.

        Args:
            x: Original input
            x_hat: Reconstructed output
            z: Latent representation
            omega_dict: Dictionary with loss weights and gates

        Returns:
            dict: Individual losses and total loss
        """
        losses = {}

        # Reconstruction loss
        losses['L_rec'] = l_recon(x, x_hat)

        # Adversarial loss (only if gate is active)
        if 'g_adv' in omega_dict and omega_dict['g_adv'] > 0.5:
            _, g_loss = adversarial_loss(self.discriminator, x, x_hat)
            losses['L_adv'] = g_loss
        else:
            losses['L_adv'] = torch.tensor(0.0, device=x.device)

        # Diversity loss
        losses['L_div'] = l_div(z)

        # Perceptual loss
        losses['L_perc'] = self.perceptual_loss(x, x_hat)

        # KL divergence loss
        losses['L_KL'] = kl_divergence_loss(z)

        # Compute total loss
        total_loss = 0.0
        if 'alpha_rec' in omega_dict:
            total_loss += omega_dict['alpha_rec'] * losses['L_rec']
        if 'alpha_adv' in omega_dict and 'g_adv' in omega_dict:
            total_loss += omega_dict['alpha_adv'] * omega_dict['g_adv'] * losses['L_adv']
        elif 'alpha_adv' in omega_dict:
            total_loss += omega_dict['alpha_adv'] * losses['L_adv']
        if 'alpha_div' in omega_dict:
            total_loss += omega_dict['alpha_div'] * losses['L_div']
        if 'alpha_perc' in omega_dict:
            total_loss += omega_dict['alpha_perc'] * losses['L_perc']
        if 'alpha_KL' in omega_dict:
            total_loss += omega_dict['alpha_KL'] * losses['L_KL']

        losses['L_total'] = total_loss
        return losses