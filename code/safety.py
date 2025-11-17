import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

class MetaDynamicsTracker:
    """
    Track emergent behavior patterns across meta-iterations for analysis
    """

    def __init__(self, buffer_size=100):
        self.buffer_size = buffer_size
        self.omega_history = []
        self.reward_history = []
        self.fid_history = []
        self.dynamics_patterns = {
            'loss_scheduling': [],
            'objective_discovery': [],
            'stability_metrics': []
        }

    def update(self, omega, reward, fid, step):
        """Update tracking with current meta-iteration data"""
        self.omega_history.append(omega.detach().cpu().numpy())
        self.reward_history.append(reward)
        self.fid_history.append(fid)

        if len(self.omega_history) > self.buffer_size:
            self.omega_history.pop(0)
            self.reward_history.pop(0)
            self.fid_history.pop(0)

        # Analyze emergent patterns
        self._analyze_emergent_behavior(step)

    def _analyze_emergent_behavior(self, step):
        """Analyze patterns in omega parameters for emergent behavior"""
        if len(self.omega_history) < 10:
            return

        omega_array = np.array(self.omega_history)
        alphas = omega_array[:, :5]  # Loss weights

        # Check for loss scheduling emergence
        alpha_means = alphas.mean(axis=0)
        alpha_stds = alphas.std(axis=0)
        scheduling_pattern = np.any(alpha_stds > 0.1)  # Significant variation

        # Check for objective discovery
        recent_alphas = alphas[-10:]
        alpha_trends = np.polyfit(range(len(recent_alphas)), recent_alphas.T, 1)[0]
        objective_discovery = np.any(np.abs(alpha_trends) > 0.01)  # Consistent trends

        # Stability metrics
        reward_stability = np.std(self.reward_history[-20:]) if len(self.reward_history) >= 20 else float('inf')

        self.dynamics_patterns['loss_scheduling'].append(scheduling_pattern)
        self.dynamics_patterns['objective_discovery'].append(objective_discovery)
        self.dynamics_patterns['stability_metrics'].append(reward_stability)

    def get_emergent_metrics(self):
        """Return current emergent behavior metrics"""
        return {
            'loss_scheduling_active': np.mean(self.dynamics_patterns['loss_scheduling'][-10:]) > 0.5,
            'objective_discovery_active': np.mean(self.dynamics_patterns['objective_discovery'][-10:]) > 0.5,
            'stability_score': np.mean(self.dynamics_patterns['stability_metrics'][-10:]) if self.dynamics_patterns['stability_metrics'] else float('inf'),
            'omega_diversity': np.std(np.array(self.omega_history), axis=0).mean() if self.omega_history else 0.0
        }


class EnsembleRewardCritic:
    """
    Ensemble of 3 reward estimators for stability:
    1. FID-based reward estimator
    2. LPIPS-based reward estimator
    3. Mini-batch average reward estimator
    """

    def __init__(self, eval_metrics, tau_disagree: float = 0.25):
        self.eval_metrics = eval_metrics
        self.tau_disagree = tau_disagree
        self.estimators = {
            'fid': self._fid_reward,
            'lpips': self._lpips_reward,
            'mini_batch_avg': self._mini_batch_avg_reward
        }

    def _fid_reward(self, initial_metrics: Dict, final_metrics: Dict) -> float:
        """FID-based reward: improvement in FID (lower is better)"""
        fid_init = initial_metrics.get('FID', float('inf'))
        fid_final = final_metrics.get('FID', float('inf'))
        return fid_init - fid_final  # Positive if improved

    def _lpips_reward(self, initial_metrics: Dict, final_metrics: Dict) -> float:
        """LPIPS-based reward: negative of LPIPS difference (lower LPIPS is better)"""
        lpips_init = initial_metrics.get('LPIPS', float('inf'))
        lpips_final = final_metrics.get('LPIPS', float('inf'))
        return lpips_init - lpips_final  # Positive if improved

    def _mini_batch_avg_reward(self, batch_rewards: List[float]) -> float:
        """Mini-batch average reward - uses recent population rewards as baseline"""
        if not batch_rewards:
            return 0.0
        # For autolytic systems, use the mean of recent rewards as the baseline
        return np.mean(batch_rewards)

    def compute_ensemble_reward(self, initial_metrics: Dict, final_metrics: Dict,
                                batch_rewards: Optional[List[float]] = None) -> Tuple[float, List[float]]:
        """
        Compute ensemble reward and individual estimates
        Returns: (validated_reward, individual_estimates)
        """
        estimates = []
        estimates.append(self._fid_reward(initial_metrics, final_metrics))
        estimates.append(self._lpips_reward(initial_metrics, final_metrics))
        if batch_rewards:
            estimates.append(self._mini_batch_avg_reward(batch_rewards))
        else:
            estimates.append(0.0)  # No batch rewards available

        # Validate reward using variance-based weighting (lower variance = higher weight)
        if len(estimates) > 1:
            mean_est = np.mean(estimates)
            if mean_est != 0:
                variances = [(est - mean_est)**2 for est in estimates]
                weights = [1.0 / (var + 1e-8) for var in variances]  # Inverse variance weighting
                weights = np.array(weights) / np.sum(weights)  # Normalize
                validated_reward = np.sum(weights * estimates)
            else:
                validated_reward = np.mean(estimates)
        else:
            validated_reward = np.mean(estimates)

        return validated_reward, estimates

class RollbackMechanism:
    """
    Rollback mechanism for model parameters if FID worsens by >50%
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold  # 50% worsening threshold
        self.checkpoints = {}

    def create_checkpoint(self, model, name: str):
        """Create a parameter checkpoint"""
        self.checkpoints[name] = {k: v.clone() for k, v in model.state_dict().items()}

    def should_rollback(self, fid_before: float, fid_after: float) -> bool:
        """Check if rollback is needed based on FID worsening"""
        if fid_before == 0:
            return False  # Avoid division by zero
        worsening = (fid_after - fid_before) / fid_before
        return worsening > self.threshold

    def rollback(self, model, name: str) -> bool:
        """Rollback model to checkpoint"""
        if name in self.checkpoints:
            model.load_state_dict(self.checkpoints[name])
            return True
        return False

class DiversityConstraint:
    """
    Diversity constraint: penalize if batch variance < var_min
    """

    def __init__(self, var_min: float = 1e-4, penalty_factor: float = 10.0):
        self.var_min = var_min
        self.penalty_factor = penalty_factor

    def compute_penalty(self, batch_features: torch.Tensor) -> float:
        """
        Compute diversity penalty based on batch variance
        batch_features: shape (batch_size, feature_dim)
        """
        if batch_features.shape[0] <= 1:
            return 0.0  # No variance with single sample

        # Compute variance across batch (mean over features for each sample, then variance)
        per_dim_var = batch_features.var(dim=0, unbiased=False)   # shape (feature_dim,)
        mean_var = per_dim_var.mean().item()
        if mean_var < self.var_min:
            penalty = self.penalty_factor * (self.var_min - mean_var)
        else:
            penalty = 0.0
        return penalty

def reward_validation(estimates: List[float], tau_disagree: float = 0.25) -> bool:
    """
    Validate reward estimates by checking disagreement threshold
    Returns True if estimates are consistent (disagreement < tau_disagree)
    """
    if len(estimates) < 2:
        return True

    mean_est = np.mean(estimates)
    if mean_est == 0:
        return True  # Avoid division

    # Compute coefficient of variation as disagreement measure
    std_est = np.std(estimates)
    cv = std_est / abs(mean_est)

    return cv < tau_disagree

def checkpoint_rollback(rollback_mechanism: RollbackMechanism, model, checkpoint_name: str) -> bool:
    """Convenience function for checkpoint rollback"""
    return rollback_mechanism.rollback(model, checkpoint_name)

def variance_check(diversity_constraint: DiversityConstraint, batch_features: torch.Tensor) -> float:
    """Convenience function for variance penalty computation"""
    return diversity_constraint.compute_penalty(batch_features)