"""
model.py

This module defines the FlowMatchingModel: a neural network that takes a 2D point and a time t,
and outputs a 2D velocity vector. In this example, we assume that the input to the model is
the concatenation of the 2D position and a scalar time (thus a 3-dimensional input).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatchingModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=3, output_dim=2):
        """
        Args:
            input_dim (int): Dimension of the input (2 for x plus 1 for t).
            hidden_dim (int): Number of hidden units in each layer.
            num_layers (int): Number of hidden layers.
            output_dim (int): Dimension of the output (2 for 2D velocity).
        """
        super(FlowMatchingModel, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x, t):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input positions of shape (batch_size, 2).
            t (Tensor): Time values of shape (batch_size, 1) or (batch_size,).

        Returns:
            Tensor: Velocity vectors of shape (batch_size, 2).
        """
        # Ensure t has shape (batch_size, 1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        # Concatenate x and t along the feature dimension
        xt = torch.cat([x, t], dim=1)  # shape: (batch_size, 3)
        return self.network(xt)
