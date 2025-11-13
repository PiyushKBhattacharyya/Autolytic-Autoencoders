import json
import os
import time
import threading
import torch
import torchvision
from torchvision.utils import make_grid
from pathlib import Path
import numpy as np

class Logger:
    """
    Thread-safe logging system for the Autotelic Autoencoder.

    Features:
    - JSON logging per meta-iteration with metrics like meta_iter, omega, reward, FID_val, LPIPS, diversity, policy_entropy, wall_time
    - Checkpoint saving/loading for theta/phi parameters (decoder/encoder)
    - Sample grid generation (64 images) saved every 5 meta-iterations in 'visuals' directory
    """

    def __init__(self, log_dir='logs', visuals_dir='visuals', checkpoint_dir='checkpoints'):
        self.log_dir = Path(log_dir)
        self.visuals_dir = Path(visuals_dir)
        self.checkpoint_dir = Path(checkpoint_dir)

        # Create directories if they don't exist
        self.log_dir.mkdir(exist_ok=True)
        self.visuals_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Thread lock for thread-safe operations
        self.lock = threading.Lock()

        # Log file path
        self.log_file = self.log_dir / 'training_log.jsonl'

    def log_meta_iteration(self, meta_iter, omega, reward, metrics, policy_entropy, wall_time):
        """
        Log a meta-iteration in JSON format.

        Args:
            meta_iter (int): Current meta-iteration number
            omega (list or torch.Tensor): Omega vector (9D)
            reward (float): Computed reward
            metrics (dict): Dictionary containing FID_val, LPIPS, diversity, etc.
            policy_entropy (float): Policy entropy
            wall_time (float): Wall clock time for the iteration
        """
        with self.lock:
            # Prepare omega as list if it's a tensor
            if isinstance(omega, torch.Tensor):
                omega = omega.tolist()

            log_entry = {
                'meta_iter': meta_iter,
                'omega': omega,
                'reward': reward,
                'FID_val': metrics.get('FID', None),
                'LPIPS': metrics.get('LPIPS', None),
                'diversity': metrics.get('diversity', None),
                'policy_entropy': policy_entropy,
                'wall_time': wall_time,
                'timestamp': time.time()
            }

            # Write to JSON Lines format (one JSON object per line)
            with open(self.log_file, 'a') as f:
                json.dump(log_entry, f)
                f.write('\n')

    def save_checkpoint(self, theta, phi, meta_iter, filename=None):
        """
        Save checkpoint for theta (decoder) and phi (encoder) parameters.

        Args:
            theta (nn.Module): Decoder parameters
            phi (nn.Module): Encoder parameters
            meta_iter (int): Current meta-iteration
            filename (str, optional): Custom filename, defaults to 'checkpoint_meta_{meta_iter}.pth'
        """
        with self.lock:
            if filename is None:
                filename = f'checkpoint_meta_{meta_iter}.pth'

            checkpoint_path = self.checkpoint_dir / filename

            checkpoint = {
                'meta_iter': meta_iter,
                'theta_state_dict': theta.state_dict(),
                'phi_state_dict': phi.state_dict(),
                'timestamp': time.time()
            }

            torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, theta, phi, checkpoint_path):
        """
        Load checkpoint for theta and phi parameters.

        Args:
            theta (nn.Module): Decoder to load into
            phi (nn.Module): Encoder to load into
            checkpoint_path (str or Path): Path to checkpoint file

        Returns:
            int: Meta-iteration number from checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        theta.load_state_dict(checkpoint['theta_state_dict'])
        phi.load_state_dict(checkpoint['phi_state_dict'])

        return checkpoint['meta_iter']

    def save_sample_grid(self, images, meta_iter, filename=None):
        """
        Generate and save a grid of 64 sample images.

        Args:
            images (torch.Tensor): Batch of images (shape: [batch_size, 3, H, W])
            meta_iter (int): Current meta-iteration
            filename (str, optional): Custom filename, defaults to 'samples_meta_{meta_iter}.png'
        """
        # Only save every 5 meta-iterations
        if meta_iter % 5 != 0:
            return

        with self.lock:
            if filename is None:
                filename = f'samples_meta_{meta_iter}.png'

            grid_path = self.visuals_dir / filename

            # Take up to 64 images
            num_images = min(64, images.size(0))
            grid_images = images[:num_images]

            # Take up to 64 images
            num_images = min(64, images.size(0))
            grid_images = images[:num_images]
            nrow = min(8, num_images)
            
            # auto-detect range
            minv, maxv = float(grid_images.min()), float(grid_images.max())
            value_range = (-1,1) if (minv >= -1.0 and maxv <= 1.0) else (0,1)
            grid = make_grid(grid_images, nrow=nrow, normalize=True, value_range=value_range)
            torchvision.utils.save_image(grid, grid_path)
