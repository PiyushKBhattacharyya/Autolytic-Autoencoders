# Autotelic Autoencoder Architecture Design

## Overview
The Autotelic Autoencoder (AAE) is a generative architecture where the training objective itself is learned through an evolutionary/RL meta-controller. It employs a population-based reinforcement learning approach using PPO to evolve the objective function, allowing the system to self-direct its learning dynamics based on generative performance metrics.

This design outlines a modular system with clear separation between the inner-loop autoencoder training and the outer-loop meta-policy evolution, enabling self-learning through adaptive loss function restructuring.

## 1. Base Autoencoder (Inner Loop)

### Components
- **Encoder (E_ϕ(x))**: Maps input data x to latent representation z
  - Architecture: Convolutional neural network with residual connections
  - Output: Latent vector z ∈ ℝ^d (dimension configurable)
  - Training: Updates via gradient descent on dynamic composite loss

- **Decoder (G_θ(z))**: Reconstructs data from latent z
  - Architecture: Transposed convolutional network with skip connections
  - Output: Reconstructed data x̂ matching input dimensions
  - Training: Joint optimization with encoder on reconstruction objectives

- **Optional Critic (C_ξ(x))**: Discriminator for adversarial training
  - Architecture: Patch-based discriminator network
  - Output: Real/fake classification scores
  - Activation: Conditional based on adversarial gate g_adv

### Primitive Losses
The system defines five primitive loss functions that can be dynamically weighted:

- **L_rec**: Reconstruction loss (e.g., L1, L2, or perceptual distance)
- **L_adv**: Adversarial loss (GAN discriminator loss)
- **L_div**: Diversity regularization (e.g., latent variance maximization)
- **L_perc**: Perceptual loss (feature-space distance using pre-trained networks)
- **L_KL**: KL divergence regularization (latent distribution matching)

### Dynamic Composite Loss
At each inner-loop step t, the total loss is:
```
L_total = α_rec * L_rec + α_adv * g_adv * L_adv + α_div * L_div + α_perc * L_perc + α_KL * L_KL
```
Where α terms are dynamically controlled by the meta-objective vector ω.

## 2. Meta-Objective / Omega Vector (ω)

7-dimensional vector defining the current training objective:

- **Loss Weights (α_rec, α_adv, α_div, α_perc, α_KL)** ∈ ℝ⁵: Multipliers for each primitive loss
  - Range: [0, ∞) with softplus activation for positivity
  - Interpretation: Relative emphasis on each loss component

- **Adversarial Gate (g_adv)** ∈ {0,1}: Binary switch for adversarial training
  - Activation: Bernoulli distribution based on logit
  - Purpose: Conditional adversarial training with start threshold t_start

- **Learning Rate Multipliers (ρ_θ, ρ_ϕ)** ∈ ℝ²: Scaling factors for decoder and encoder LRs
  - Range: [0.1, 10] with exponential activation
  - Allows dynamic adjustment of training dynamics

- **Start Threshold (t_start)** ∈ ℝ: Inner-loop step threshold for adversarial activation
  - Range: [0, T_inner] with sigmoid activation scaled to training steps
  - Enables delayed adversarial training for stability

The omega vector ω ∈ ℝ⁷ is sampled from the meta-policy and defines the complete training regime for each inner-loop session.

## 3. Meta-Policy (Outer Loop)

### State Representation (s)
The meta-policy state s aggregates information from recent training history:

- **Validation Metrics**: Last 5 FID scores (Fréchet Inception Distance)
- **Performance Indicators**: Moving average reconstruction loss
- **Training Context**: Inner-loop step indicator (normalized [0,1])
- **Dimensionality**: s ∈ ℝ^7 (5 FIDs + 1 reconstruction avg + 1 step indicator)

### Policy Network Architecture
- **Input**: State vector s
- **Hidden Layers**: 2-layer MLP with 256 units each, ReLU activation
- **Output Head**: Dual outputs for mean μ(s) and log-variance log σ(s) of Gaussian distribution
- **Action Space**: ω ∈ ℝ⁷ sampled from N(μ(s), σ²(s))
- **Parameters**: ψ (policy network weights)

### Action Distribution
Actions are sampled from a multivariate Gaussian:
```
ω ~ N(μ_ψ(s), Σ_ψ(s))
```
Where Σ is diagonal covariance derived from log σ outputs.

The policy network π_ψ(ω|s) defines a probability distribution over possible objective configurations, favoring those that have historically led to better generative performance.

## 4. Strategy 2: Evolutionary/RL Meta-Controller Process

### Meta-Iteration Structure
For each meta-iteration m:

1. **Population Sampling**
   - Sample N_pop omega vectors from current policy: ω^(i) ~ π_ψ(ω|s) for i=1 to N_pop
   - Each ω^(i) defines a unique objective configuration

2. **Inner-Loop Training Sessions**
   - For each sampled ω^(i):
     - Initialize autoencoder parameters from global model (θ_global, ϕ_global)
     - Train for T_inner steps using dynamic loss defined by ω^(i)
     - Apply learning rate multipliers ρ_θ, ρ_ϕ for parameter updates
     - Activate adversarial components based on g_adv and t_start
     - Produce final parameters (θ_T^(i), ϕ_T^(i)) and performance metrics

3. **Performance Evaluation**
   - Compute validation metrics for each trained model:
     - FID (Fréchet Inception Distance) on held-out data
     - LPIPS (Learned Perceptual Image Patch Similarity)
     - Diversity metrics (latent space coverage)
     - Reconstruction quality on validation set

4. **Meta-Reward Computation**
   - Primary reward: r^(i) = FID(θ_0) - FID(θ_T^(i))
     - Rewards improvement in generative quality
     - Higher positive values indicate better performance
   - Alternative rewards: combinations of multiple metrics (LPIPS, diversity, etc.)

5. **Transition Collection**
   - Store (s, ω^(i), log π_ψ(ω^(i)|s), V_w(s), r^(i)) in PPO replay buffer
   - Enables policy gradient updates with advantage estimation

6. **Global Model Update**
   - Select best-performing parameters or compute EMA across top-k performers
   - Update θ_global, ϕ_global for next meta-iteration
   - Maintain training stability through gradual parameter evolution

### Key Characteristics
- **Black-box RL**: No gradients flow from inner-loop to meta-policy
- **Population-based**: Multiple objectives evaluated in parallel
- **Non-differentiable metrics**: Can optimize FID and other complex metrics
- **Robust exploration**: Systematic search through objective space

## 5. PPO Update Mechanism

### Replay Buffer Structure
- **Transitions**: (state s, action ω, log-probability log π_ψ(ω|s), value V_w(s), reward r)
- **Capacity**: Rolling buffer of last K meta-iterations
- **Organization**: FIFO replacement with importance sampling weights

### PPO Loss Components

1. **Policy Loss (L_policy)**
   ```
   L_policy = E[ min(r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t) ]
   ```
   - Where r_t(θ) = π_ψ_new(ω|s) / π_ψ_old(ω|s) (probability ratio)
   - Â_t = advantage estimate using GAE (Generalized Advantage Estimation)
   - ε = clipping parameter (typically 0.2)

2. **Value Loss (L_value)**
   ```
   L_value = E[ (V_w(s) - V_target)^2 ]
   ```
   - V_target computed using TD(λ) returns with GAE
   - MSE loss with optional clipping for stability

3. **Entropy Bonus**
   ```
   L_entropy = -E[ H(π_ψ(·|s)) ]
   ```
   - Encourages exploration by penalizing deterministic policies
   - Coefficient typically 0.01-0.1

### Total PPO Loss
```
L_PPO = L_policy + c_value * L_value + c_entropy * L_entropy
```
Where c_value ≈ 0.5, c_entropy ≈ 0.01

### Update Procedure
- **Mini-batch updates**: Sample random mini-batches from buffer
- **Multiple epochs**: Update policy and value networks for E epochs per meta-iteration
- **Gradient clipping**: Clip gradients to prevent instability (max norm 0.5)
- **Target networks**: Optional for stable value estimation

### Advantage Estimation
- **GAE computation**: Â_t = Σ_{k=0}^∞ (γ λ)^k δ_{t+k}
- **TD residual**: δ_t = r_t + γ V_w(s_{t+1}) - V_w(s_t)
- **Hyperparameters**: γ = 0.99 (discount), λ = 0.95 (GAE parameter)

## 6. Integration Points and Parameter Updates

### Inner-Loop to Outer-Loop Communication
- **Metric Collection**: Validation hooks compute FID, LPIPS, diversity metrics after each inner-loop training
- **Parameter Checkpointing**: Save final (θ_T^(i), ϕ_T^(i)) for each population member
- **Reward Calculation**: Compare metrics against baseline (initial global model performance)

### Global Model Selection Strategy
- **Best-Performance Selection**: θ_global ← θ_T^(argmax_i r^(i))
  - Directly adopts the highest-reward parameters
  - Aggressive updates, potentially unstable

- **Exponential Moving Average (EMA)**: θ_global ← α * θ_global + (1-α) * θ_T^(best)
  - Gradual parameter updates with α ≈ 0.9-0.99
  - Increases training stability

- **Top-K Ensemble**: Average parameters across top-performing k members
  - Reduces variance through population averaging
  - k = 3-5 typically provides good balance

### Checkpointing and Recovery
- **Meta-Policy Checkpoints**: Save ψ parameters every N meta-iterations
- **Global Model Checkpoints**: Maintain θ_global, ϕ_global history
- **Replay Buffer Persistence**: Optional buffer saving for continued training

### Training Coordination
- **Synchronous Updates**: Meta-iteration completes before next begins
- **Parallel Inner-Loops**: Multiple ω^(i) can train simultaneously on different GPUs
- **Resource Management**: Monitor GPU memory, training time per inner-loop

## 7. Modular Design for Extensibility and Self-Learning

### Component Interfaces

#### Autoencoder Interface
```python
class Autoencoder(nn.Module):
    def encode(self, x) -> torch.Tensor:  # x -> z
    def decode(self, z) -> torch.Tensor:  # z -> x_hat
    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:  # (x_hat, z)
    def compute_losses(self, x, x_hat, z, omega: OmegaVector) -> dict[str, torch.Tensor]:
```

#### MetaPolicy Interface
```python
class MetaPolicy(nn.Module):
    def get_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # (omega, log_prob, value)
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # (log_probs, values, entropy)
```

#### Reward Function Interface
```python
class RewardFunction(Protocol):
    def __call__(self, initial_metrics: dict, final_metrics: dict) -> float:
```

### Validation Hooks
- **Metric Hooks**: Pluggable functions for computing FID, LPIPS, diversity
- **Custom Metrics**: Support for domain-specific evaluation functions
- **Early Stopping**: Optional hooks for terminating inner-loop training

### Extensibility Points
- **Loss Library**: Registry of primitive losses (L_rec, L_adv, L_div, etc.)
- **State Features**: Configurable state representation for meta-policy
- **Reward Combinations**: Weighted combinations of multiple metrics
- **Population Strategies**: Different sampling and selection strategies

### Self-Learning Capabilities
- **Adaptive Metrics**: Meta-policy can learn to weight different reward components
- **Curriculum Learning**: Automatic adjustment of inner-loop length T_inner
- **Meta-Meta Learning**: Higher-level policies for adapting meta-hyperparameters
- **Multi-Objective Optimization**: Pareto front exploration for competing metrics

## 8. Overall System Architecture

```mermaid
graph TD
    A[Meta-Policy π_ψ] --> B[Sample Population ω^(1..N)]
    B --> C[Inner-Loop Training Sessions]
    C --> D[Autoencoder θ_ϕ]
    D --> E[Dynamic Loss Computation]
    E --> F[Primitive Losses: L_rec, L_adv, L_div, L_perc, L_KL]
    C --> G[Validation Metrics: FID, LPIPS, Diversity]
    G --> H[Reward Computation r^(i)]
    H --> I[PPO Buffer]
    I --> J[PPO Update]
    J --> A
    C --> K[Parameter Checkpointing]
    K --> L[Global Model Selection]
    L --> M[EMA/Top-K Update]
    M --> N[Next Meta-Iteration]

    subgraph "Inner Loop (Per Population Member)"
        C
        D
        E
        F
    end

    subgraph "Outer Loop (RL Evolution)"
        A
        B
        I
        J
    end

    subgraph "Integration Points"
        G
        H
        K
        L
        M
    end

    style A fill:#e1f5fe
    style D fill:#f3e5f5
    style I fill:#e8f5e8
    style L fill:#fff3e0
```

### Data Flow Summary
1. **Initialization**: Start with random global autoencoder parameters and untrained meta-policy
2. **Meta-Iteration**: Sample population of objectives from current meta-policy
3. **Parallel Training**: Train separate autoencoder copies with different objectives
4. **Evaluation**: Compute generative quality metrics for each trained model
5. **Reward & Learning**: Calculate rewards, store transitions, update meta-policy via PPO
6. **Evolution**: Update global autoencoder parameters using best performers
7. **Iteration**: Repeat until convergence or computational budget exhausted

### Key Architectural Principles
- **Modularity**: Clear separation between inner-loop (autoencoder training) and outer-loop (objective evolution)
- **Scalability**: Population-based evaluation allows parallel training on multiple GPUs
- **Stability**: EMA updates and PPO constraints prevent catastrophic forgetting
- **Extensibility**: Pluggable components for losses, metrics, and reward functions
- **Self-Learning**: System adapts its own training objectives based on performance feedback

## 9. Implementation Summary

The Autotelic Autoencoder using Strategy 2 implements a sophisticated meta-learning framework where:

### Core Innovation
- **Objective Evolution**: Training objectives become learnable parameters controlled by an RL meta-policy
- **Population-Based Search**: Multiple objective configurations are evaluated simultaneously
- **Generative Self-Improvement**: The system learns to optimize its own training dynamics

### Key Components
1. **Base Autoencoder**: Standard encoder-decoder architecture with dynamic loss composition
2. **Meta-Objective Vector**: 7-dimensional control vector defining loss weights, gates, and hyperparameters
3. **Meta-Policy Network**: MLP that proposes objective configurations based on training history
4. **PPO Meta-Controller**: Stable policy optimization for objective evolution
5. **Integration Mechanisms**: Parameter updates, metric computation, and reward calculation

### Expected Emergent Behaviors
- **Dynamic Loss Scheduling**: Automatic adjustment of reconstruction vs. adversarial emphasis
- **Adaptive Learning Rates**: Self-tuning of training dynamics for different phases
- **Stability Control**: Conditional activation of potentially unstable components (adversarial training)
- **Multi-Objective Balance**: Learned trade-offs between perceptual quality, diversity, and fidelity

### Advantages Over Traditional Approaches
- **No Manual Tuning**: Eliminates human selection of loss weights and training schedules
- **Non-Differentiable Optimization**: Can optimize complex metrics like FID directly
- **Robust Exploration**: Population-based search avoids local optima in objective space
- **Self-Organizing Training**: Emergent training curricula adapted to data characteristics

This design transforms static, human-designed training procedures into an adaptive, self-evolving learning system capable of discovering novel training strategies that outperform fixed-objective approaches.