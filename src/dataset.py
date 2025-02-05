"""
dataset.py

This module provides a simple dataset for flow matching.
For each sample, it returns a pair:
    - x0: A sample from the base distribution (2D standard Gaussian)
    - x1: A sample from the target distribution (a bimodal Gaussian mixture)
"""

import torch
from torch.utils.data import Dataset


def sample_base(batch_size, device='cpu'):
    """
    Sample from a 2D standard Gaussian.
    """
    return torch.randn(batch_size, 2, device=device)


def sample_target(batch_size, device='cpu'):
    """
    Sample from a simple bimodal 2D distribution.
    We define the target distribution as an equally weighted mixture of two Gaussians.
    One centered at (-2, 0) and one at (2, 0) with a small covariance.
    """
    # Randomly choose one of the two modes
    modes = torch.randint(0, 2, (batch_size,), device=device)
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0]], device=device)
    chosen_centers = centers[modes]
    # Add small Gaussian noise around the centers
    noise = 0.3 * torch.randn(batch_size, 2, device=device)
    return chosen_centers + noise


class FlowMatchingDataset(Dataset):
    def __init__(self, num_samples=10000, device='cpu'):
        """
        Args:
            num_samples (int): Number of sample pairs.
            device (str): Device to use ('cpu' or 'cuda').
        """
        self.num_samples = num_samples
        self.device = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # For each sample, generate a base and a target sample.
        # (They are generated on the fly.)
        x0 = sample_base(1, device=self.device).squeeze(0)   # shape: (2,)
        x1 = sample_target(1, device=self.device).squeeze(0)   # shape: (2,)
        return x0, x1
