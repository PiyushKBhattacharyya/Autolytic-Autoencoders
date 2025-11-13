import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from code.losses import l_recon, l_div, PerceptualLoss, adversarial_loss, kl_divergence_loss
from code.autoencoder import Encoder, UNetDecoder
import numpy as np
from typing import Tuple, Dict

def preprocess_omega(omega: torch.Tensor, T_inner: int, device: str):
    """
    omega: raw tensor shape (9,)
    returns: alphas (5,), g_adv (float), rho_theta (float), rho_phi (float), t_start (int)
    """
    omega = omega.to(device)
    alpha_logits = omega[:5]
    sp = F.softplus(alpha_logits)
    sp = torch.clamp(sp, min=1e-6)
    alphas = sp / (sp.sum() + 1e-8)
    g_adv = torch.sigmoid(omega[5]).item()
    rho_theta = float(F.softplus(omega[6]).clamp(min=1e-3, max=10.0).item())
    rho_phi = float(F.softplus(omega[7]).clamp(min=1e-3, max=10.0).item())
    t_start = int(torch.sigmoid(omega[8]).item() * float(max(1, T_inner)))
    return alphas, g_adv, rho_theta, rho_phi, t_start


def clone_model_to_device(model, device: str):
    m = model.__class__(**getattr(model, "_init_args", {})) if hasattr(model, "_init_args") else model.__class__()
    m.load_state_dict(model.state_dict())
    return m.to(device)


def inner_training_loop(
    encoder: Encoder,
    decoder: UNetDecoder,
    data: torch.Tensor,
    omega: torch.Tensor,
    T_inner: int = 100,
    base_lr_theta: float = 2e-4,
    base_lr_phi: float = 1e-4,
    device: str = "cpu",
    use_amp: bool = False,
) -> Tuple[Dict, Dict, Dict]:
    """
    Warm-start inner training for T_inner steps with objective controlled by omega.
    Returns encoder_state_dict (cpu), decoder_state_dict (cpu), metrics dict.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    assert data is not None
    data = data.to(device)

    # Preprocess omega
    alphas, g_adv, rho_theta, rho_phi, t_start = preprocess_omega(omega, T_inner, device)

    # Clone models and move to device
    latent_dim = encoder.linear.out_features if hasattr(encoder, "linear") else 128
    encoder_inner = Encoder(latent_dim=latent_dim).to(device)
    decoder_inner = UNetDecoder(latent_dim=latent_dim).to(device)
    encoder_inner.load_state_dict(encoder.state_dict())
    decoder_inner.load_state_dict(decoder.state_dict())

    # Optimizers
    opt_enc = torch.optim.Adam(encoder_inner.parameters(), lr=base_lr_phi * rho_phi, betas=(0.9, 0.999), eps=1e-8)
    opt_dec = torch.optim.Adam(decoder_inner.parameters(), lr=base_lr_theta * rho_theta, betas=(0.9, 0.999), eps=1e-8)

    scaler = GradScaler(enabled=use_amp and (device.startswith("cuda")))

    # Initialize optional loss components (placeholders - will be computed in loop)
    perceptual_loss_fn = PerceptualLoss(device=device)
    l_perc_loss = torch.tensor(0.0, device=device)
    l_adv_loss = torch.tensor(0.0, device=device)
    l_kl_loss = torch.tensor(0.0, device=device)

    # training loop
    for step in range(T_inner):
        encoder_inner.train()
        decoder_inner.train()

        opt_enc.zero_grad()
        opt_dec.zero_grad()

        with autocast(enabled=use_amp and (device.startswith("cuda"))):
            z, skips = encoder_inner(data)
            x_hat = decoder_inner(z, skips)

            l_recon_loss = l_recon(data, x_hat)
            l_div_loss = l_div(z)

            # Compute optional losses inside the loop
            if alphas[1] > 0.01:  # perceptual
                l_perc_loss = perceptual_loss_fn(data, x_hat)
            else:
                l_perc_loss = torch.tensor(0.0, device=device)

            # Adversarial loss (simplified - in practice would need proper GAN training per step)
            if alphas[2] > 0.01:
                from code.losses import Discriminator
                discriminator = Discriminator().to(device)
                disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
                d_loss, g_loss = adversarial_loss(discriminator, data, x_hat)
                l_adv_loss = g_loss
                # Quick discriminator update
                disc_optimizer.zero_grad()
                d_loss.backward(retain_graph=True)
                disc_optimizer.step()
            else:
                l_adv_loss = torch.tensor(0.0, device=device)

            # KL loss (assumes VAE-style latent with mu/logvar, simplified here)
            if alphas[3] > 0.01:
                l_kl_loss = kl_divergence_loss(z)
            else:
                l_kl_loss = torch.tensor(0.0, device=device)

            # primitive losses ordered as [rec, perc, adv, kl, div]
            primitive_losses = [l_recon_loss, l_perc_loss, l_adv_loss, l_kl_loss, l_div_loss]
            # composite
            L_base = sum(alpha * loss for alpha, loss in zip(alphas, primitive_losses))

        if use_amp and (device.startswith("cuda")):
            scaler.scale(L_base).backward()
            scaler.unscale_(opt_enc)
            torch.nn.utils.clip_grad_norm_(encoder_inner.parameters(), 0.5)
            scaler.unscale_(opt_dec)
            torch.nn.utils.clip_grad_norm_(decoder_inner.parameters(), 0.5)
            scaler.step(opt_enc)
            scaler.step(opt_dec)
            scaler.update()
        else:
            L_base.backward()
            torch.nn.utils.clip_grad_norm_(encoder_inner.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(decoder_inner.parameters(), 0.5)
            opt_enc.step()
            opt_dec.step()

    # compute simple metrics on this batch for meta-reward
    encoder_inner.eval()
    decoder_inner.eval()
    with torch.no_grad():
        z, _ = encoder_inner(data)
        x_hat = decoder_inner(z, skips)
        recon = float(l_recon(data, x_hat).cpu().item())
        div = float(l_div(z).cpu().item())

    # return CPU state dicts
    encoder_state = {k: v.detach().cpu().clone() for k, v in encoder_inner.state_dict().items()}
    decoder_state = {k: v.detach().cpu().clone() for k, v in decoder_inner.state_dict().items()}
    metrics = {"recon": recon, "diversity": div, "g_adv": g_adv, "t_start": t_start}
    return encoder_state, decoder_state, metrics