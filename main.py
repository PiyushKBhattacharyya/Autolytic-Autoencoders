import torch
import torch.nn as nn
import numpy as np
import random
import time
import sys
from code.autoencoder import Encoder, UNetDecoder
from code.meta_policy import MetaPolicy
from code.ppo_update import PPOBuffer, PPO
from inner_loop import inner_training_loop
from code.eval_metrics import EvalMetrics
from code.logger import Logger
from code.safety import EnsembleRewardCritic, MetaDynamicsTracker
from code.datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")
# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Hyperparameters
N_pop = 16  # Population size
K = 4  # PPO epochs
lr_psi = 3e-4  # Meta-policy learning rate
lr_value = 1e-3  # Not used separately, as value is part of meta-policy
beta_ent = 0.01  # Entropy coefficient
eps_clip = 0.2  # PPO clip epsilon
eps_entropy = 1e-3  # Entropy epsilon

# Other constants
latent_dim = 128
T_inner = 100  # Inner loop steps
device = torch.device('cpu')  # Force CPU for stability testing

# Initialize components
encoder = Encoder(latent_dim=latent_dim).to(device)
decoder = UNetDecoder(latent_dim=latent_dim).to(device)
meta_policy = MetaPolicy(state_dim=7, action_dim=9).to(device)
ppo_buffer = PPOBuffer()
eval_metrics = EvalMetrics(device=device)
logger = Logger()
ensemble_reward = EnsembleRewardCritic(eval_metrics)
meta_dynamics = MetaDynamicsTracker()

# PPO updater with stability improvements
ppo_updater = PPOBuffer()
ppo = PPO(meta_policy, lr=lr_psi, c_e=beta_ent, epsilon=eps_clip,
          entropy_schedule={'type': 'linear_decay', 'steps': 1000, 'final': 0.001},
          action_clip={'min': -2.0, 'max': 2.0})

# Load real dataset
train_loader, test_loader = load_dataset("cifar10", batch_size=64, resolution=64)
data_iter = iter(train_loader)
data, _ = next(data_iter)  # Get first batch
data = data.to(device)

# Initialize state s (7D: last 5 FID, moving avg recon loss, step indicator)
s = torch.zeros(7).to(device)
last_fids = [100.0] * 5  # Initial FID estimate
moving_avg_recon = 1.0
step_indicator = 0.0

# Initialize test iterator for evaluation
test_iter = iter(test_loader)

# Main training loop
meta_iter = 0
max_meta_iters = 2  # Very reduced for testing

while meta_iter < max_meta_iters:
    start_time = time.time()

    # Update state s
    s[0:5] = torch.tensor(last_fids[-5:]).to(device)
    s[5] = moving_avg_recon
    s[6] = step_indicator

    population_encoder_states = []
    population_decoder_states = []
    population_omegas = []
    population_rewards = []

    # Collect trajectories (population-based)
    for i in range(N_pop):
        # Sample omega from meta-policy
        omega, log_prob, value = meta_policy.get_action(s)

        # Run inner loop
        encoder_state, decoder_state, inner_metrics = inner_training_loop(encoder, decoder, data, omega, T_inner)

        # Compute metrics after inner loop using real evaluation
        try:
            test_batch, _ = next(test_iter)
        except StopIteration:
            test_iter = iter(test_loader)
            test_batch, _ = next(test_iter)
        test_batch = test_batch.to(device)

        # Get real and generated samples for evaluation
        with torch.no_grad():
            real_samples = test_batch[:16]  # Use subset for efficiency
            # Use the inner loop trained model for evaluation
            temp_encoder = Encoder(latent_dim=latent_dim).to(device)
            temp_decoder = UNetDecoder(latent_dim=latent_dim).to(device)
            temp_encoder.load_state_dict(encoder_state)
            temp_decoder.load_state_dict(decoder_state)
            temp_encoder.eval()
            temp_decoder.eval()

            z, _ = temp_encoder(data[:16])
            gen_samples = temp_decoder(z, temp_encoder(data[:16])[1])

        # Convert to numpy arrays properly for evaluation
        real_np = [img.permute(1, 2, 0).cpu().numpy() for img in real_samples]
        gen_np = [img.permute(1, 2, 0).cpu().numpy() for img in gen_samples]

        initial_metrics = eval_metrics.evaluate(real_np, real_np)  # Baseline
        final_metrics = eval_metrics.evaluate(real_np, gen_np)

        # Compute ensemble reward for stability
        validated_reward, individual_estimates = ensemble_reward.compute_ensemble_reward(
            initial_metrics, final_metrics, population_rewards[-1:] if population_rewards else []
        )
        reward = validated_reward

        # Store in PPO buffer
        ppo_updater.store(s, omega, log_prob, value, reward)

        population_encoder_states.append(encoder_state)
        population_decoder_states.append(decoder_state)
        population_omegas.append(omega)
        population_rewards.append(reward)

        # Update moving averages
        moving_avg_recon = 0.9 * moving_avg_recon + 0.1 * inner_metrics['recon']
        last_fids.append(final_metrics['FID'])
        last_fids = last_fids[-5:]

    # Perform PPO update with K epochs
    diagnostics = ppo.update(ppo_updater, step=meta_iter)

    # Update global models (simple: take best performer)
    best_idx = np.argmax(population_rewards)
    best_omega = population_omegas[best_idx]

    # Load best into global from population states
    decoder.load_state_dict(population_decoder_states[best_idx])
    encoder.load_state_dict(population_encoder_states[best_idx])

    # Update meta-dynamics tracker
    meta_dynamics.update(best_omega, population_rewards[best_idx], final_metrics['FID'], meta_iter)

    # Get emergent behavior metrics
    emergent_metrics = meta_dynamics.get_emergent_metrics()

    # Log
    wall_time = time.time() - start_time
    policy_entropy = diagnostics['entropy_loss']  # Assume from diagnostics
    logger.log_meta_iteration(meta_iter, best_omega, population_rewards[best_idx], final_metrics, policy_entropy, wall_time,
                             emergent_metrics=emergent_metrics)

    # Checkpoint
    logger.save_checkpoint(decoder, encoder, meta_iter)

    # Update step indicator
    step_indicator = min(step_indicator + 0.01, 1.0)  # Dummy update

    meta_iter += 1

if __name__ == "__main__":
    print("Training complete.")