import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import torchvision.models as models
from torchvision.models import VGG16_Weights


def l_recon(x, x_hat):
    """
    L1 reconstruction loss between input and reconstructed output.

    Args:
        x (torch.Tensor): Original input tensor.
        x_hat (torch.Tensor): Reconstructed output tensor.

    Returns:
        torch.Tensor: L1 loss value.
    """
    return F.l1_loss(x, x_hat)


def l_div(z, delta_cov=1e-6, epsilon_num=1e-12):
    """
    Feature covariance log-det loss for diversity, with regularization and numerical stabilizers.

    Args:
        z (torch.Tensor): Latent features, shape (batch_size, latent_dim).
        delta_cov (float): Regularization added to covariance diagonal.
        epsilon_num (float): Numerical stabilizer for log det.

    Returns:
        torch.Tensor: Normalized log determinant of regularized covariance matrix.
    """
    batch, dim = z.shape
    if batch < 2:
        # Not enough samples for covariance, return stabilizer
        return torch.as_tensor(epsilon_num, device=z.device, dtype=z.dtype)

    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.t() @ z_centered) / (batch - 1)
    cov = (cov + cov.t()) / 2  # Symmetrize covariance for stability
    cov += delta_cov * torch.eye(dim, device=z.device, dtype=z.dtype)

    sign, log_det = torch.slogdet(cov)
    if sign <= 0 or torch.isnan(log_det) or torch.isinf(log_det):
        log_det = torch.as_tensor(epsilon_num, device=z.device, dtype=z.dtype)

    return log_det / dim  # Normalize by dimension


class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        # Use LPIPS for perceptual loss
        try:
            self.lpips = lpips.LPIPS(net='vgg').to(device)
            self.available = True
        except:
            self.available = False
            self.lpips = None

    def forward(self, x, y):
        if self.available:
            return self.lpips(x, y).mean()
        else:
            # Fallback to L1 if LPIPS not available
            return F.l1_loss(x, y)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def adversarial_loss(discriminator, real, fake):
    """
    Adversarial loss for GAN training.
    Returns discriminator loss and generator loss.
    """
    real_pred = discriminator(real)
    fake_pred = discriminator(fake)

    d_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred)) + \
             F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))

    g_loss = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))

    return d_loss, g_loss


def kl_divergence_loss(z, mu=None, logvar=None):
    """
    KL divergence loss for VAE-style latent regularization.
    If mu and logvar are provided, uses standard VAE KL.
    Otherwise, assumes z is already the latent and uses isotropic Gaussian prior.
    """
    if mu is not None and logvar is not None:
        # Standard VAE KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl_loss.mean()
    else:
        # Assume z is from N(0,1) and compute KL to standard normal
        return 0.5 * torch.sum(z.pow(2) - 1 + torch.log(1e-8 + z.pow(2)), dim=-1).mean()