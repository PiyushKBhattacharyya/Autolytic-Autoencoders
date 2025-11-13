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
from code.datasets import load_dataset

warnings.filterwarnings("ignore")

# Repro
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Hyperparams (keep same as you had)
N_pop = 16
K = 4
lr_psi = 3e-4
beta_ent = 0.01
eps_clip = 0.2
latent_dim = 128
T_inner = 100

def main():
    # Force CPU for testing (or detect GPU)
    device = torch.device('cpu')

    # ---------- DATA (Windows safe) ----------
    # IMPORTANT: set num_workers=0 on Windows / spawn to avoid re-imports
    train_loader, test_loader = load_dataset("cifar10", batch_size=64, resolution=64)
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
    max_meta_iters = 2
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
                encoder, decoder, data, omega, T_inner, device=str(device), use_amp=False
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

                # Pass skips to decoder — do NOT pass None
                gen_samples = temp_decoder(z, skips) # adapt if your decoder needs skips

            # compute metrics (placeholder — replace with eval_metrics.evaluate if desired)
            initial_metrics = {'FID': last_fids[-1], 'LPIPS': 0.5}
            final_metrics = {'FID': max(0.0, last_fids[-1] - np.random.uniform(0, 10)), 'LPIPS': 0.4}

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

        # choose best performer
        best_idx = int(np.argmax(population_rewards))
        best_omega = population_omegas[best_idx]
        encoder.load_state_dict(population_encoder_states[best_idx])
        decoder.load_state_dict(population_decoder_states[best_idx])

        meta_dynamics.update(best_omega, population_rewards[best_idx], last_fids[-1], meta_iter)
        emergent_metrics = meta_dynamics.get_emergent_metrics()

        wall_time = time.time() - start_time
        logger.log_meta_iteration(meta_iter, best_omega, population_rewards[best_idx], final_metrics,
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