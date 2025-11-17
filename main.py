    #!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import random
import time
import warnings

# local imports (use code.* for project modules)
from code.autoencoder import Encoder, UNetDecoder
from code.meta_policy import MetaPolicy
from code.ppo_update import PPOBuffer, PPO
from inner_loop import inner_training_loop
from code.eval_metrics import EvalMetrics
from code.logger import Logger
from code.safety import EnsembleRewardCritic, MetaDynamicsTracker
from code.losses import Discriminator
from code.datasets import load_dataset

warnings.filterwarnings("ignore")

# Repro
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Hyperparams
N_pop = 16
K = 4
lr_psi = 3e-4
beta_ent = 0.01
eps_clip = 0.2
latent_dim = 128
T_inner = 100
# New hyperparameters for global model update
top_k = 3  # Number of top performers to average
alpha = 0.9  # EMA coefficient (0.9 means 90% old, 10% new)
# Hyperparams (keep same as you had)
N_pop = 16
K = 4
lr_psi = 3e-4
beta_ent = 0.01
eps_clip = 0.2
latent_dim = 128
T_inner = 100
def average_state_dicts(state_dicts, weights=None):
    """
    Average a list of state dictionaries.
    
    Args:
        state_dicts: List of state dictionaries to average
        weights: Optional list of weights for each state dict (must sum to 1.0)
    
    Returns:
        Averaged state dictionary
    """
    if not state_dicts:
        raise ValueError("No state dictionaries provided")
    
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    elif len(weights) != len(state_dicts):
        raise ValueError("Number of weights must match number of state dictionaries")
    
    # Ensure weights sum to 1
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Get all keys from first state dict
    keys = state_dicts[0].keys()
    
    # Check that all state dicts have the same keys
    for sd in state_dicts[1:]:
        if set(sd.keys()) != set(keys):
            raise ValueError("All state dictionaries must have the same keys")
    
    # Average parameters
    averaged_state = {}
    for key in keys:
        params = [sd[key] for sd in state_dicts]
        # Stack and average along new dimension
        averaged_param = torch.stack(params, dim=0)
        averaged_param = torch.sum(averaged_param * torch.tensor(weights, device=averaged_param.device).unsqueeze(-1).unsqueeze(-1), dim=0)
        averaged_state[key] = averaged_param
    
    return averaged_state


def ema_update(target_state, source_state, alpha):
    """
    Perform Exponential Moving Average update: target = alpha * target + (1-alpha) * source
    
    Args:
        target_state: Current EMA state dictionary
        source_state: New state dictionary to incorporate
        alpha: EMA coefficient (decay factor)
    
    Returns:
        Updated EMA state dictionary
    """
    if set(target_state.keys()) != set(source_state.keys()):
        raise ValueError("State dictionaries must have the same keys")
    
    updated_state = {}
    for key in target_state.keys():
        updated_state[key] = alpha * target_state[key] + (1 - alpha) * source_state[key]
    
    return updated_state



def main():
    # Force CPU for testing (or detect GPU)
    device = torch.device('cpu')

    # ---------- DATA (Windows safe) ----------
    # IMPORTANT: set num_workers=0 on Windows / spawn to avoid re-imports
    train_loader, test_loader = load_dataset("cifar10", batch_size=64, resolution=32)
    data_iter = iter(train_loader)
    test_iter = iter(test_loader)

    # ---------- MODELS & COMPONENTS (created once) ----------
    encoder = Encoder(latent_dim=latent_dim).to(device)
    decoder = UNetDecoder(latent_dim=latent_dim).to(device)
    meta_policy = MetaPolicy(state_dim=7, action_dim=9).to(device)

    # PPO and buffer
    ppo_buffer = PPOBuffer()
    ppo = PPO(policy=meta_policy, lr=lr_psi, epochs=K, mini_batch_size=max(2, N_pop),
              epsilon=eps_clip, c_v=1.0, c_e=beta_ent)

    # Create EvalMetrics / Logger / Safety ONCE (outside loops)
    eval_metrics = EvalMetrics(device=device)
    logger = Logger()
    ensemble_reward = EnsembleRewardCritic(eval_metrics)
    meta_dynamics = MetaDynamicsTracker()

    # Create discriminator and optimizer for adversarial training (once outside loops)
    discriminator = Discriminator(in_channels=3).to(device)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    # ---------- fetch one batch for inner loop warm-start ----------
    try:
        data, _ = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        data, _ = next(data_iter)
    data = data.to(device)

    # ---------- state ----------
    s = torch.zeros(7, dtype=torch.float32, device=device)
    last_fids = [100.0] * 5
    moving_avg_recon = 1.0
    step_indicator = 0.0

    # ---------- main meta loop ----------
    meta_iter = 0
    max_meta_iters = 10  # Increased for better convergence testing
    while meta_iter < max_meta_iters:
        start_time = time.time()
        s[0:5] = torch.tensor(last_fids[-5:], dtype=torch.float32, device=device)
        s[5] = torch.tensor(moving_avg_recon, dtype=torch.float32, device=device)
        s[6] = torch.tensor(step_indicator, dtype=torch.float32, device=device)

        population_encoder_states = []
        population_decoder_states = []
        population_omegas = []
        population_rewards = []

        for i in range(N_pop):
            omega, log_prob, value = meta_policy.get_action(s)

            enc_state, dec_state, inner_metrics = inner_training_loop(
                encoder, decoder, data, omega, T_inner, device=str(device), use_amp=False,
                perceptual_fn=eval_metrics.lpips_model, discriminator=discriminator, disc_optimizer=disc_optimizer
            )

            # evaluation samples
            try:
                test_batch, _ = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                test_batch, _ = next(test_iter)
            test_batch = test_batch.to(device)[:16]

            # eval with temporary models (no heavy global re-init)
            temp_encoder = Encoder(latent_dim=latent_dim).to(device)
            temp_decoder = UNetDecoder(latent_dim=latent_dim).to(device)
            temp_encoder.load_state_dict(enc_state)
            temp_decoder.load_state_dict(dec_state)
            temp_encoder.eval()
            temp_decoder.eval()

            with torch.no_grad():
                # Get both z and the skip connections from the encoder (one forward pass)
                z, skips = temp_encoder(test_batch)

                # Pass skips to decoder
                gen_samples = temp_decoder(z, skips)

            # compute actual metrics using EvalMetrics
            # Convert tensors to numpy arrays for evaluation (EvalMetrics handles both tensors and arrays)
            # Ensure image shape is (N, H, W, C) for numpy arrays (EvalMetrics expects this)
            real_imgs = test_batch.cpu().numpy().transpose(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            gen_imgs = gen_samples.cpu().numpy().transpose(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            final_metrics = eval_metrics.evaluate(real_imgs, gen_imgs)
            initial_metrics = {'FID': last_fids[-1], 'LPIPS': 0.5}  # Use previous FID as baseline

            validated_reward, _ = ensemble_reward.compute_ensemble_reward(
                initial_metrics, final_metrics, population_rewards[-1:] if population_rewards else []
            )
            reward = float(validated_reward)

            # store (ensure cpu tensors and consistent shapes)
            ppo_buffer.store(s.detach().cpu(), omega.detach().cpu(), log_prob.detach().cpu(), value.detach().cpu(), reward)

            population_encoder_states.append(enc_state)
            population_decoder_states.append(dec_state)
            population_omegas.append(omega.detach().cpu().numpy())
            population_rewards.append(reward)

            moving_avg_recon = 0.9 * moving_avg_recon + 0.1 * float(inner_metrics.get('recon', 0.0))
            last_fids.append(final_metrics['FID'])
            last_fids = last_fids[-5:]

        # PPO update (passes buffer object)
        diagnostics = ppo.update(ppo_buffer)

        # Choose top-K performers and average their parameters
        rewards_with_indices = list(enumerate(population_rewards))
        rewards_with_indices.sort(key=lambda x: x[1], reverse=True)  # Sort by reward descending
        top_k_indices = [idx for idx, _ in rewards_with_indices[:top_k]]

        # Average the top-K encoder and decoder states
        top_k_encoder_states = [population_encoder_states[idx] for idx in top_k_indices]
        top_k_decoder_states = [population_decoder_states[idx] for idx in top_k_indices]

        avg_encoder_state = average_state_dicts(top_k_encoder_states)
        avg_decoder_state = average_state_dicts(top_k_decoder_states)

        # Apply EMA update: θ_global = α θ_global + (1-α) θ^(averaged top-K)
        current_encoder_state = encoder.state_dict()
        current_decoder_state = decoder.state_dict()

        updated_encoder_state = ema_update(current_encoder_state, avg_encoder_state, alpha)
        updated_decoder_state = ema_update(current_decoder_state, avg_decoder_state, alpha)

        encoder.load_state_dict(updated_encoder_state)
        decoder.load_state_dict(updated_decoder_state)

        # Use the best performer from top-K for logging and meta-dynamics
        best_top_k_idx = top_k_indices[0]  # Best among the top-K
        best_omega = population_omegas[best_top_k_idx]

        meta_dynamics.update(best_omega, population_rewards[best_top_k_idx], last_fids[-1], meta_iter)
        emergent_metrics = meta_dynamics.get_emergent_metrics()

        wall_time = time.time() - start_time
        logger.log_meta_iteration(meta_iter, best_omega, population_rewards[best_top_k_idx], final_metrics,
                                  diagnostics.get('entropy_loss', 0.0), wall_time,
                                  emergent_metrics=emergent_metrics)
        logger.save_checkpoint(decoder, encoder, meta_iter)

        step_indicator = min(step_indicator + 0.01, 1.0)
        meta_iter += 1

    print("Training complete.")

if __name__ == "__main__":
    # Windows spawn-safe
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()