"""
trainer.py

This module implements a Trainer class that handles the training loop for the flow matching model.
The training objective is to match the model’s predicted velocity to the target velocity defined by the
linear interpolation between base and target samples.

For each training sample:
    - Sample a time t ~ Uniform(0,1)
    - Compute the interpolated point: x_t = (1-t)*x0 + t*x1
    - The target velocity is simply: v_target = x1 - x0  (since the interpolation is linear)
    - The loss is the mean squared error between f(x_t, t) and v_target.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(self, model, dataset, device='cpu', batch_size=128, lr=1e-3):
        """
        Args:
            model (nn.Module): The flow matching model.
            dataset (Dataset): A dataset yielding (x0, x1) pairs.
            device (str): Device to use ('cpu' or 'cuda').
            batch_size (int): Training batch size.
            lr (float): Learning rate.
        """
        self.model = model.to(device)
        self.device = device
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

    def train(self, num_epochs=10):
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for x0, x1 in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # Move data to device
                x0 = x0.to(self.device)
                x1 = x1.to(self.device)
                batch_size = x0.size(0)
                # Sample a random time for each sample in the batch
                t = torch.rand(batch_size, device=self.device)
                # Compute the interpolated position: x_t = (1-t)*x0 + t*x1
                # t is of shape (batch_size,) so we unsqueeze it for broadcasting.
                t_unsqueezed = t.unsqueeze(1)
                x_t = (1 - t_unsqueezed) * x0 + t_unsqueezed * x1

                # The target velocity is (x1 - x0), independent of t.
                v_target = x1 - x0

                # Predict the velocity using the model
                v_pred = self.model(x_t, t)

                # Compute the loss (MSE)
                loss = self.criterion(v_pred, v_target)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_size

            epoch_loss /= len(self.dataset)
            print(f"Epoch {epoch+1}: Average Loss = {epoch_loss:.6f}")

    def eval_batch(self, x0, x1, t):
        """
        Utility function: Given a batch of (x0, x1) and times t,
        compute the model prediction at the interpolated points.
        """
        t_unsqueezed = t.unsqueeze(1)
        x_t = (1 - t_unsqueezed) * x0 + t_unsqueezed * x1
        v_pred = self.model(x_t, t)
        return v_pred
