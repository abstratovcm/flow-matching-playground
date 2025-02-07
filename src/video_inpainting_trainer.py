"""
video_inpainting_trainer.py

This module defines the VideoInpaintingTrainer class, which is responsible for training a video
inpainting model. It manages the training loop, loss computation, and teacher forcing scheduling.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn


class VideoInpaintingTrainer:
    def __init__(self, model, dataset, device='cpu', batch_size=64, lr=1e-3):
        """
        Initializes the trainer.

        Args:
            model (nn.Module): The inpainting model to be trained.
            dataset (Dataset): An instance of VideoInpaintingDataset.
            device (str): Device to run training on ('cpu' or 'cuda').
            batch_size (int): Training batch size.
            lr (float): Learning rate for the optimizer.
        """
        self.model = model.to(device)
        self.device = device
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train(self, num_epochs=20, initial_teacher_forcing_ratio=0.5):
        """
        Trains the model for a specified number of epochs using a linear decay in teacher forcing ratio.

        The teacher forcing ratio decays linearly from initial_teacher_forcing_ratio to 0 across epochs.
        When teacher forcing is applied, the model is forced to use the ground truth tokens for the decoder.
        Otherwise, the model uses its own predictions (autoregressive decoding).

        Args:
            num_epochs (int): Total number of training epochs.
            initial_teacher_forcing_ratio (float): Starting teacher forcing ratio.
        """
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0

            # Compute the teacher forcing ratio for this epoch.
            if num_epochs > 1:
                current_tf_ratio = initial_teacher_forcing_ratio * (1 - epoch / (num_epochs - 1))
            else:
                current_tf_ratio = initial_teacher_forcing_ratio

            # Iterate over batches.
            for obs_before, missing, obs_after, _, _ in tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                obs_before = obs_before.to(self.device)
                missing = missing.to(self.device)
                obs_after = obs_after.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass with scheduled teacher forcing.
                output = self.model(obs_before, obs_after, tgt=missing, teacher_forcing_ratio=current_tf_ratio)
                loss = self.criterion(output, missing)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * obs_before.size(0)

            avg_loss = epoch_loss / len(self.dataset)
            print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.6f}, Teacher Forcing Ratio = {current_tf_ratio:.4f}")
