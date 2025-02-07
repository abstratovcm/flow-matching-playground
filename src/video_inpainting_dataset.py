"""
video_inpainting_dataset.py

This module defines the VideoInpaintingDataset class, a PyTorch Dataset that generates synthetic
video sequences for inpainting tasks. Each sample consists of three segments:
    - Observed frames before the missing segment.
    - Missing frames that need to be predicted.
    - Observed frames after the missing segment.

The dataset normalizes each sample using statistics computed solely from the observed segments,
ensuring that the missing segment is not used for normalization.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class VideoInpaintingDataset(Dataset):
    def __init__(self, num_samples=1000, obs_before_len=50, missing_len=25, obs_after_len=50, fps=10):
        """
        Initializes the dataset.

        Args:
            num_samples (int): Number of video sequences to generate.
            obs_before_len (int): Number of frames in the observed (before) segment.
            missing_len (int): Number of frames in the missing segment.
            obs_after_len (int): Number of frames in the observed (after) segment.
            fps (int): Frame rate of the video sequences.
        """
        self.num_samples = num_samples
        self.obs_before_len = obs_before_len
        self.missing_len = missing_len
        self.obs_after_len = obs_after_len
        self.total_len = obs_before_len + missing_len + obs_after_len
        self.fps = fps

    def __len__(self):
        """
        Returns:
            int: Total number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generates a synthetic video sequence sample.

        Steps:
            1. Create a time vector spanning 0 to 2π.
            2. Randomly sample base offsets, amplitudes, frequencies, and phases for the x and y coordinates.
            3. Generate the video sequence as a 2D sinusoidal trajectory.
            4. Split the sequence into observed (before and after) and missing segments.
            5. Compute normalization statistics from the observed segments.
            6. Normalize the entire sequence and re-split into segments.
            7. Convert the segments and normalization parameters to PyTorch tensors.

        Args:
            idx (int): Index of the sample (not used since samples are generated on the fly).

        Returns:
            tuple: (obs_before, missing, obs_after, mean, std) where:
                - obs_before (Tensor): Observed frames before the missing segment.
                - missing (Tensor): Missing frames (ground truth).
                - obs_after (Tensor): Observed frames after the missing segment.
                - mean (Tensor): Mean value used for normalization.
                - std (Tensor): Standard deviation used for normalization.
        """
        # Create a time vector.
        t = np.linspace(0, 2 * np.pi, self.total_len)

        # Sample random base offsets (centered around 100).
        x0 = np.random.uniform(90.0, 110.0)
        y0 = np.random.uniform(90.0, 110.0)

        # Sample random amplitudes.
        Ax = np.random.uniform(3.0, 6.0)
        Ay = np.random.uniform(3.0, 6.0)

        # Sample random frequencies.
        omega_x = np.random.uniform(0.5, 2.0)
        omega_y = np.random.uniform(0.5, 2.0)

        # Sample random phases.
        phi_x = np.random.uniform(0, 2 * np.pi)
        phi_y = np.random.uniform(0, 2 * np.pi)

        # Generate x and y coordinates for the video sequence.
        x = x0 + Ax * np.sin(omega_x * t + phi_x)
        y = y0 + Ay * np.sin(omega_y * t + phi_y)

        # Combine x and y into a (total_len, 2) array.
        video = np.stack([x, y], axis=1).astype(np.float32)

        # Split the video into segments (without normalization yet).
        obs_before = video[:self.obs_before_len]
        missing = video[self.obs_before_len:self.obs_before_len + self.missing_len]
        obs_after = video[self.obs_before_len + self.missing_len:]

        # Compute normalization parameters using only the observed segments.
        observed = np.concatenate([obs_before, obs_after], axis=0)
        mean = observed.mean()
        std = observed.std() if observed.std() > 0 else 1.0

        # Normalize the entire video sequence.
        video_norm = (video - mean) / std

        # Re-split the normalized video into segments.
        obs_before = video_norm[:self.obs_before_len]
        missing = video_norm[self.obs_before_len:self.obs_before_len + self.missing_len]
        obs_after = video_norm[self.obs_before_len + self.missing_len:]

        # Convert arrays to torch tensors.
        obs_before = torch.tensor(obs_before)
        missing = torch.tensor(missing)
        obs_after = torch.tensor(obs_after)
        mean = torch.tensor(mean)
        std = torch.tensor(std)

        return obs_before, missing, obs_after, mean, std
