import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class MetaPolicy(nn.Module):
    """
    Numerically-stable Gaussian policy for 9D omega.
    Features:
    - softplus σ instead of exp() to avoid overflow
    - σ clamped to [1e-6, 1e1]
    - logσ initialized to -3 (σ ~ 0.05)
    - sanitize input states via torch.nan_to_num
    """

    def __init__(self, state_dim=7, action_dim=9, init_logstd=-3.0):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared trunk
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        # Policy heads
        self.mu_head = nn.Linear(256, action_dim)
        self.log_sigma_head = nn.Linear(256, action_dim)

        # Value head
        self.value_head = nn.Linear(256, 1)

        # Initialize weights
        self.apply(self._init_weights)

        # bias initialization for log_sigma
        with torch.no_grad():
            nn.init.constant_(self.log_sigma_head.bias, float(init_logstd))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    # σ computation (stable)
    def _safe_sigma(self, logit):
        sigma = F.softplus(logit) + 1e-6
        sigma = sigma.clamp(min=1e-6, max=10.0)
        return sigma

    # Forward
    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma_logits = self.log_sigma_head(x)
        value = self.value_head(x)
        return mu, log_sigma_logits, value

    # Sample action (for collection)
    def get_action(self, state):
        # sanitize state
        state = torch.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)

        self.eval()
        with torch.no_grad():
            mu, log_sigma_logits, value = self(state.unsqueeze(0))
            sigma = self._safe_sigma(log_sigma_logits)

            dist = Normal(mu, sigma)

            omega = dist.rsample()
            log_prob = dist.log_prob(omega).sum(dim=-1).squeeze(0)
            entropy = dist.entropy().sum(dim=-1).squeeze(0)

            return (
                omega.squeeze(0),
                log_prob.detach(),
                value.squeeze(0).detach(),
            )
            
    # Evaluate action (for PPO)
    def evaluate_actions(self, states, actions):
        states = torch.nan_to_num(states, nan=0.0, posinf=1e6, neginf=-1e6)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1e6, neginf=-1e6)

        mu, log_sigma_logits, values = self(states)
        sigma = self._safe_sigma(log_sigma_logits)

        dist = Normal(mu, sigma)

        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropies = dist.entropy().sum(dim=-1)
        values = values.squeeze(-1)

        # sanitize outputs
        log_probs = torch.nan_to_num(log_probs, nan=-1e8, posinf=1e8, neginf=-1e8)
        entropies = torch.nan_to_num(entropies, nan=0.0, posinf=1e8, neginf=-1e8)
        values = torch.nan_to_num(values, nan=0.0, posinf=1e8, neginf=-1e8)

        return log_probs, values, entropies