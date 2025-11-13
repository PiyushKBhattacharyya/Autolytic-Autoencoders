import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import time
import warnings
from code.autoencoder import Encoder, Decoder
from code.meta_policy import MetaPolicy
from inner_loop import inner_training_loop
from code.ppo_update import PPO, PPOBuffer
from code.eval_metrics import EvalMetrics
from code.logger import Logger
from code.safety import EnsembleRewardCritic
warnings.filterwarnings("ignore")
# ARGUMENT PARSER (CLI FLAGS)
parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "cuda", "auto"])
parser.add_argument("--meta_iters", type=int, default=10)
parser.add_argument("--pop", type=int, default=2)
parser.add_argument("--inner_steps", type=int, default=20)
parser.add_argument("--batch", type=int, default=32)

args = parser.parse_args()

# DEVICE SELECT
if args.device == "cpu":
    device = torch.device("cpu")
elif args.device == "cuda":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_amp = (device.type == "cuda")

print(f"Using device: {device}")

# SEEDING
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# MODELS
latent_dim = 128

encoder = Encoder(latent_dim=latent_dim).to(device)
decoder = Decoder(latent_dim=latent_dim).to(device)
meta_policy = MetaPolicy(state_dim=7, action_dim=9).to(device)

ppo_buffer = PPOBuffer()
ppo = PPO(
    policy=meta_policy,
    lr=3e-4,
    epochs=4,
    mini_batch_size=max(2, args.pop),
    epsilon=0.2,
    c_v=1.0,
    c_e=0.01
)

evaler = EvalMetrics(device=str(device))
logger = Logger()
reward_ensemble = EnsembleRewardCritic(evaler)

# DUMMY TRAIN DATA (TOY TEST)
data = torch.randn(args.batch, 3, 64, 64).to(device)

# STATE INIT
last_fids = [100.0] * 5
moving_recon = 1.0
step_indicator = 0.0

# MAIN META LOOP
print("\n=== Starting Toy Experiment ===\n")

for meta_iter in range(args.meta_iters):

    t0 = time.time()

    s = torch.tensor(last_fids + [moving_recon, step_indicator],
                     dtype=torch.float32, device=device)

    pop_rewards = []
    pop_states_enc = []
    pop_states_dec = []
    pop_omegas = []

    # POPULATION
    for k in range(args.pop):

        omega, logp, value = meta_policy.get_action(s)

        enc_state, dec_state, metrics_inner = inner_training_loop(
            encoder, decoder,
            data,
            omega,
            args.inner_steps,
            device=str(device),
            use_amp=use_amp
        )

        # Fake validation FID (since it's a toy experiment)
        fid_before = last_fids[-1]
        fid_after = fid_before - np.random.uniform(0, 5)
        reward = fid_before - fid_after

        # store
        ppo_buffer.store(s.detach(), omega.detach(), logp.detach(), value.detach(), reward)
        pop_rewards.append(float(reward))
        pop_states_enc.append(enc_state)
        pop_states_dec.append(dec_state)
        pop_omegas.append(omega.cpu().numpy())

        # update running stats
        moving_recon = 0.9 * moving_recon + 0.1 * metrics_inner["recon"]
        last_fids.append(fid_after)
        last_fids = last_fids[-5:]

    # PPO update
    stats = ppo.update(ppo_buffer)

    # best model
    best_idx = int(np.argmax(pop_rewards))
    encoder.load_state_dict(pop_states_enc[best_idx])
    decoder.load_state_dict(pop_states_dec[best_idx])

    wall = time.time() - t0

    print(f"Meta iter {meta_iter}: Reward {pop_rewards[best_idx]:.4f}, FID approx {last_fids[-1]:.4f}")

print("\n=== Toy Experiment Finished ===\n")