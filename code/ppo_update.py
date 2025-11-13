import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal

class PPOBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []

    def store(self, state, action, log_prob, value, reward):
        # Ensure correct shapes
        state = state.view(-1)      # [7]
        action = action.view(-1)    # [9]
        log_prob = log_prob.view(1) # [1]
        value = value.view(1)       # [1]

        self.states.append(state.detach().clone())
        self.actions.append(action.detach().clone())
        self.log_probs.append(log_prob.detach().clone())
        self.values.append(value.detach().clone())
        self.rewards.append(float(reward))

    def tensors(self, device):
        states = torch.stack(self.states).to(device)            # [B, 7]
        actions = torch.stack(self.actions).to(device)          # [B, 9]
        log_probs = torch.stack(self.log_probs).to(device).view(-1)  # [B]
        values = torch.stack(self.values).to(device).view(-1)        # [B]
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device).view(-1)  # [B]

        return states, actions, log_probs, values, rewards

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()

class PPO:
    def __init__(self, policy, lr=3e-4, epochs=4, mini_batch_size=32,
                 epsilon=0.2, c_v=1.0, c_e=0.01, entropy_schedule=None, action_clip=None):

        self.policy = policy
        self.initial_eps = epsilon
        self.eps = epsilon
        self.c_v = c_v
        self.initial_c_e = c_e
        self.c_e = c_e
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size

        # Entropy scheduling
        self.entropy_schedule = entropy_schedule or {'type': 'constant', 'final': c_e}
        self.entropy_step = 0

        # Action clipping
        self.action_clip = action_clip or {'min': -2.0, 'max': 2.0}

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    def update_entropy_schedule(self, step):
        """Update entropy coefficient based on schedule"""
        self.entropy_step = step
        if self.entropy_schedule['type'] == 'linear_decay':
            progress = min(step / self.entropy_schedule.get('steps', 1000), 1.0)
            self.c_e = self.initial_c_e + (self.entropy_schedule['final'] - self.initial_c_e) * progress
        elif self.entropy_schedule['type'] == 'exponential_decay':
            decay_rate = self.entropy_schedule.get('decay_rate', 0.999)
            self.c_e = self.initial_c_e * (decay_rate ** step)
        else:
            self.c_e = self.initial_c_e

    def apply_action_clipping(self, actions):
        """Apply action clipping to prevent extreme actions"""
        return torch.clamp(actions, self.action_clip['min'], self.action_clip['max'])

    def update(self, buffer: PPOBuffer, step=0):
        device = next(self.policy.parameters()).device

        # Update entropy schedule
        self.update_entropy_schedule(step)

        if len(buffer.states) == 0:
            return {"policy_loss": 0.0}

        states, actions, old_logp, old_values, rewards = buffer.tensors(device)

        # Apply action clipping to old actions
        actions = self.apply_action_clipping(actions)

        advantages = rewards - old_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Force all tensors to shape [batch]
        old_logp = old_logp.view(-1)
        old_values = old_values.view(-1)
        rewards = rewards.view(-1)
        advantages = advantages.view(-1)

        dataset = TensorDataset(states, actions, old_logp, old_values, rewards, advantages)
        loader = DataLoader(dataset, batch_size=min(self.mini_batch_size, len(dataset)), shuffle=True)

        diagnostics = {}

        for _ in range(self.epochs):
            for s, a, old_lp, old_v, r, adv in loader:

                lp, v, ent = self.policy.evaluate_actions(s, a)

                # Flatten safely
                lp = lp.view(-1)
                v = v.view(-1)
                ent = ent.view(-1)

                # ratios
                ratios = torch.exp(lp - old_lp)

                # Clamp ratios to prevent extreme values
                ratios = torch.clamp(ratios, 0.1, 10.0)

                # policy loss
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                v_loss = ((v - r) ** 2).mean() * self.c_v

                # entropy bonus
                entropy_loss = -ent.mean() * self.c_e

                total_loss = policy_loss + v_loss + entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                diagnostics = {
                    "policy_loss": float(policy_loss),
                    "value_loss": float(v_loss),
                    "entropy_loss": float(entropy_loss),
                    "total_loss": float(total_loss),
                    "entropy_coeff": float(self.c_e),
                    "clip_eps": float(self.eps)
                }

        buffer.clear()
        return diagnostics