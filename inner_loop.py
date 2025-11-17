import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Tuple, Dict, Any, Optional
from code.losses import l_recon, l_div, adversarial_loss, kl_divergence_loss
from code.autoencoder import Encoder, UNetDecoder

def preprocess_omega(omega: torch.Tensor, T_inner: int):
    """Convert raw omega tensor to usable scalars."""
    with torch.no_grad():
        alpha_logits = omega[:5]
        sp = F.softplus(alpha_logits)
        alphas = sp / (sp.sum() + 1e-8)
        g_adv = float(torch.sigmoid(omega[5]).item())
        rho_theta = float(torch.exp(omega[6]).item())
        rho_phi = float(torch.exp(omega[7]).item())
        t_start = int(round(torch.sigmoid(omega[8]).item() * T_inner))
    return alphas, g_adv, rho_theta, rho_phi, t_start


def _make_model_copy(model: torch.nn.Module, device: torch.device):
    """Instantiate a fresh copy of the model and load weights (assumes no-arg ctor or _init_kwargs)."""
    if hasattr(model, "_init_kwargs"):
        new_model = model.__class__(**model._init_kwargs)
    else:
        new_model = model.__class__()
    new_model.load_state_dict(model.state_dict())
    return new_model.to(device)


def inner_training_loop(
    encoder: Encoder,
    decoder: UNetDecoder,
    data: torch.Tensor,
    omega: torch.Tensor,
    T_inner: int = 100,
    base_lr_theta: float = 2e-4,
    base_lr_phi: float = 1e-4,
    device: Any = "cpu",
    use_amp: bool = False,
    perceptual_fn: Optional[callable] = None,
    discriminator: Optional[torch.nn.Module] = None,
    disc_optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Inner warm-start training loop.

    Args:
        encoder, decoder: global models (will be cloned for inner-loop).
        data: input batch tensor.
        omega: raw policy vector tensor (len 9).
        T_inner: inner-loop steps.
        base_lr_theta, base_lr_phi: base LRs for decoder/encoder.
        device: 'cpu'/'cuda' or torch.device.
        use_amp: enable AMP when using CUDA.
        perceptual_fn: callable (x, x_hat) -> scalar tensor; pass eval_metrics.lpips_model or None.
        discriminator: optional discriminator module (created once outside).
        disc_optimizer: optimizer for discriminator (created outside if discriminator provided).

    Returns:
        (encoder_state_dict_cpu, decoder_state_dict_cpu, metrics)
    """
    # Normalize device
    if isinstance(device, str):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        device = torch.device("cpu")

    data = data.to(device)

    # Preprocess omega
    alphas, g_adv, rho_theta, rho_phi, t_start = preprocess_omega(omega, T_inner)

    # Clone models and move to device
    encoder_inner = _make_model_copy(encoder, device)
    decoder_inner = _make_model_copy(decoder, device)

    # Optimizers
    opt_enc = torch.optim.Adam(encoder_inner.parameters(), lr=base_lr_phi * rho_phi, betas=(0.9, 0.999), eps=1e-8)
    opt_dec = torch.optim.Adam(decoder_inner.parameters(), lr=base_lr_theta * rho_theta, betas=(0.9, 0.999), eps=1e-8)

    # If discriminator provided but optimizer not, create a default one
    if (discriminator is not None) and (disc_optimizer is None):
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    # Move discriminator to device if provided
    if discriminator is not None:
        discriminator = discriminator.to(device)

    scaler = GradScaler(enabled=use_amp and (device.type == "cuda"))

    final_skips = None

    for step in range(T_inner):
        encoder_inner.train()
        decoder_inner.train()

        opt_enc.zero_grad()
        opt_dec.zero_grad()
        if disc_optimizer is not None:
            disc_optimizer.zero_grad()

        with autocast(enabled=use_amp and (device.type == "cuda")):
            z, skips = encoder_inner(data)
            x_hat = decoder_inner(z, skips)
            final_skips = skips  # keep for final eval

            # primitive losses
            l_recon_loss = l_recon(data, x_hat)
            l_div_loss = l_div(z)

            # perceptual (use external function if provided)
            if (perceptual_fn is not None) and (float(alphas[3]) > 0.0):
                try:
                    # perceptual_fn typically expects normalized images in [-1,1] or [0,1] depending on your setup
                    l_perc_loss = perceptual_fn(data, x_hat)
                except Exception:
                    l_perc_loss = torch.tensor(0.0, device=device)
            else:
                l_perc_loss = torch.tensor(0.0, device=device)

            # adversarial (only if discriminator passed and gate & step conditions satisfied)
            if (discriminator is not None) and (float(alphas[1]) > 0.0) and (step >= t_start) and (g_adv > 0.5):
                try:
                    d_loss, g_loss = adversarial_loss(discriminator, data, x_hat)
                    l_adv_loss = g_loss
                except Exception:
                    l_adv_loss = torch.tensor(0.0, device=device)
            else:
                l_adv_loss = torch.tensor(0.0, device=device)

            # KL (if applicable)
            if float(alphas[4]) > 0.0:
                try:
                    l_kl_loss = kl_divergence_loss(z)
                except Exception:
                    l_kl_loss = torch.tensor(0.0, device=device)
            else:
                l_kl_loss = torch.tensor(0.0, device=device)

            primitive_losses = [l_recon_loss, l_adv_loss, l_div_loss, l_perc_loss, l_kl_loss]
            effective_alphas = alphas.clone()
            effective_alphas[1] *= g_adv  # apply g_adv multiplier to α_adv
            L_base = sum((effective_alpha * loss) for effective_alpha, loss in zip(effective_alphas, primitive_losses))
            L_base = L_base.mean()  # Ensure scalar output for backward()

        # Update discriminator first (if available)
        if (discriminator is not None) and ('d_loss' in locals()) and (step >= t_start):
            try:
                if use_amp and (device.type == "cuda"):
                    scaler.scale(d_loss).backward(retain_graph=True)
                    scaler.unscale_(disc_optimizer)
                    scaler.step(disc_optimizer)
                else:
                    d_loss.backward(retain_graph=True)
                    disc_optimizer.step()
            except Exception:
                # safe: don't crash training if discriminator step fails
                pass

        # Update generator (encoder+decoder)
        if use_amp and (device.type == "cuda"):
            scaler.scale(L_base).backward()
            scaler.unscale_(opt_enc)
            torch.nn.utils.clip_grad_norm_(encoder_inner.parameters(), max_norm=0.5)
            scaler.unscale_(opt_dec)
            torch.nn.utils.clip_grad_norm_(decoder_inner.parameters(), max_norm=0.5)
            scaler.step(opt_enc)
            scaler.step(opt_dec)
            scaler.update()
        else:
            L_base.backward()
            torch.nn.utils.clip_grad_norm_(encoder_inner.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(decoder_inner.parameters(), max_norm=0.5)
            opt_enc.step()
            opt_dec.step()

    # Final metrics on this batch
    encoder_inner.eval()
    decoder_inner.eval()
    with torch.no_grad():
        z, _ = encoder_inner(data)
        x_hat = decoder_inner(z, final_skips)
        recon_val = float(l_recon(data, x_hat).cpu().item())
        div_val = float(l_div(z).cpu().item())

    # Return CPU state dicts and metrics
    encoder_state = {k: v.detach().cpu().clone() for k, v in encoder_inner.state_dict().items()}
    decoder_state = {k: v.detach().cpu().clone() for k, v in decoder_inner.state_dict().items()}
    metrics = {"recon": recon_val, "diversity": div_val, "g_adv": g_adv, "t_start": t_start}
    return encoder_state, decoder_state, metrics