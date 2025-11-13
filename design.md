# Autotelic Autoencoder Architecture Design

## Overview

The Autotelic Autoencoder (AAE) is a generative model in which the training objective itself is learned using a population-based reinforcement learning (RL) meta-controller. Instead of static loss terms (e.g., L1 + adversarial + perceptual), the AAE uses PPO to evolve dynamic objective functions that adjust loss weights, learning rates, schedule timing, and adversarial gating.
This creates a self-directing system that adaptively restructures its own loss landscape to maximize generative performance (e.g., FID reduction).

The architecture cleanly separates:

Inner Loop: Autoencoder training under a sampled objective

Outer Loop: RL evolution of objectives using PPO

## 1. Base Autoencoder (Inner Loop)

### Components
- **Encoder E_ϕ(x)**: CNN with residual blocks, outputs latent vector z ∈ ℝ^d
- **Decoder G_θ(z)**: Transposed CNN with skip connections, outputs reconstructed image x̂
- **Optional Critic C_ξ(x)**: PatchGAN discriminator, used only when adversarial gate g_adv is ON

### Primitive Losses

The AAE supports five primitive losses:
- L_rec: reconstruction loss
- L_adv: generator adversarial loss
- L_div: diversity / covariance log-det
- L_perc: perceptual/VGG loss
- L_KL: KL divergence (if variational)

### Dynamic Composite Loss

At each inner step t:
```
L_total = α_rec * L_rec + α_adv * g_adv * L_adv + α_div * L_div + α_perc * L_perc + α_KL * L_KL
```
All α terms and gating are generated from the meta-objective vector ω.

## 2. Meta-Objective / Omega Vector (ω)

Corrected Dimensionality: 9-dimensional

The meta-policy outputs a 9D omega vector:

| Component | Meaning | Activation |
|-----------|---------|------------|
| 1–5 | α_rec, α_adv, α_div, α_perc, α_KL | softplus → normalize |
| 6 | g_adv_raw | sigmoid → continuous gate |
| 7 | ρ_θ | exp → LR multiplier for decoder |
| 8 | ρ_ϕ | exp → LR multiplier for encoder |
| 9 | t_start_raw | sigmoid × T_inner |

Summary: ω ∈ ℝ^9, Processed values:

- **Loss weights**:
  ```
  α_i = softplus(α̃_i) / (sum_j softplus(α̃_j) + ε)
  ```

- **Adversarial gate**:
  ```
  g_adv = σ(g_adv_raw), g_adv_bool = (g_adv > 0.5)
  ```

- **Learning rate multipliers**:
  ```
  ρ_θ = e^{ω7}, ρ_ϕ = e^{ω8}
  ```

- **Start threshold**:
  ```
  t_start = round(σ(ω9) ⋅ T_inner)
  ```

## 3. Meta-Policy (Outer Loop)

### State Representation s

The state is 7-dimensional:

| Feature | Count |
|---------|-------|
| Last 5 FID scores | 5 |
| Moving average reconstruction loss | 1 |
| Inner-loop step indicator (normalized) | 1 |
| **Total** | **7D** |

s ∈ ℝ^7

### Policy Network

- **Input**: s ∈ ℝ^7
- **Architecture**: 2-layer MLP, 256 units each, ReLU
- **Outputs**:
  - Mean vector μ(s) ∈ ℝ^9
  - Log variance vector log σ(s) ∈ ℝ^9

### Action Distribution
```
ω ~ N(μ(s), diag(σ²(s)))
```
This gives high exploration and tunable objective generation.

## 4. Strategy 2: Evolutionary/RL Meta-Controller Process

### Per Meta-Iteration (m)

1. **Population Sampling**
   - Sample N_pop objective vectors: ω^(i) ~ π_ψ(ω|s)

2. **Inner-loop training**
   - For each ω^(i):
     - Copy global parameters
     - Train AE for T_inner steps
     - Use dynamic loss governed by ω^(i)
     - Apply LR multipliers
     - Activate adversarial only if g_adv^(i) > 0.5 and t ≥ t_start^(i)

3. **Metrics**
   - Compute:
     - FID
     - (optional) LPIPS
     - (optional) Diversity
     - Reconstruction loss

4. **Reward**
   - Primary reward: r^(i) = FID(θ_0) - FID(θ_T^(i))
   - Higher reward = better improvement.

5. **Store PPO transition**
   - Store in buffer: (s, ω^(i), log π_ψ(ω^(i)|s), V_w(s), r^(i))

6. **Update global model**
   - Choose: Best performer or EMA: θ_global ← α θ_global + (1-α) θ^(i*)

## 5. PPO Update Mechanism

### Buffer Stores
(s, ω, log π, V(s), r)

### PPO Losses
- **Policy loss**:
  ```
  L_policy = E[ min(r_t Â_t, clip(r_t, 1-ε, 1+ε) Â_t) ]
  ```

- **Value loss**:
  ```
  L_value = (V(s) - V_target)^2
  ```

- **Entropy bonus**:
  ```
  L_entropy = -H(π_ψ(⋅|s))
  ```

- **Total loss**:
  ```
  L_PPO = L_policy + c_v L_value + c_e L_entropy
  ```

### Advantage (Single-Step GAE Correction)
Since each population member = 1 transition:
```
Â = r - V(s)
```
(GAE general form is included for completeness but simplifies to this.)

## 6. Integration & Updates

### Inner → Outer communication:
- Metrics from inner loop
- Final AE parameters
- Reward

### Outer → Inner updates:
- Updated policy
- Updated global AE parameters

## 7. Modular Interfaces

### Autoencoder
```python
class Autoencoder(nn.Module):
    def encode(self, x) -> Tensor
    def decode(self, z) -> Tensor
    def forward(self, x) -> (x_hat, z)
    def compute_losses(self, x, x_hat, z, omega_dict) -> dict
```

### MetaPolicy
```python
class MetaPolicy(nn.Module):
    def get_action(self, state) -> (omega, log_prob, value)
    def evaluate_actions(self, states, actions) -> (log_probs, values, entropy)
```

### Reward Function
```python
class RewardFunction:
    def __call__(self, initial_metrics, final_metrics) -> float
```

## 8. Full Architecture Diagram (Corrected)

```mermaid
graph TD
    A[Meta-Policy π_ψ] --> B[Sample Population ω^(1..N)]
    B --> C[Inner-Loop Training Sessions]
    C --> D[Autoencoder θ_ϕ]
    D --> E[Dynamic Loss Computation]
    E --> F[Primitive Losses]

    C --> G[Validation Metrics: FID, LPIPS (opt), Diversity (opt)]
    G --> H[Reward r^(i)]
    H --> I[PPO Buffer]

    I --> J[PPO Update]
    J --> A

    C --> K[Parameter Checkpointing]
    K --> L[Global Model Selection]
    L --> M[EMA / Top-K Update]
    M --> N[Next Meta-Iteration]

    style A fill:#e1f5fe
    style D fill:#f3e5f5
    style I fill:#e8f5e8
    style L fill:#fff3e0
```

## 9. Implementation Summary (Corrected)

### Core Innovation
- Objective evolution: training objective becomes a learnable function
- Population-based meta-learning: multiple objectives evaluated in parallel
- Generative self-improvement through FID-optimized evolution

### Key Components
- Autoencoder (inner loop)
- 9D meta-objective vector
- PPO-based meta-policy
- Parallel inner-loop sessions
- Global model update (EMA or top-k)

### Emergent behaviors
- Dynamic loss scheduling
- Adaptive LR shaping
- Safe adversarial activation
- Multi-objective balancing
- Automatic training curriculum formation

### Advantages
- No manual tuning
- Can optimize non-differentiable metrics
- Robust exploration
- Self-organizing learning behavior