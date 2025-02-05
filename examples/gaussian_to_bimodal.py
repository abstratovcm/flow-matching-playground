"""
gaussian_to_bimodal.py

This script sets up flow matching training and provides improved visualizations:
1. A scatter plot showing sample pairs from the base and target distributions.
2. A vector field plot with overlaid sample trajectories, illustrating how points move.
"""

import torch
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

# Add the project root to sys.path so `src` can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import FlowMatchingModel
from src.dataset import FlowMatchingDataset
from src.trainer import Trainer
from src.utils import set_random_seed


def visualize_samples_and_trajectories(dataset, device='cpu', num_samples=50):
    """
    Visualize sample pairs from the dataset:
      - Base samples (x0) from the simple 2D Gaussian.
      - Target samples (x1) from the bimodal distribution.
    Draw lines between each pair to show the desired transformation.
    """
    x0_list, x1_list = [], []
    for i in range(num_samples):
        x0, x1 = dataset[i]
        x0_list.append(x0.cpu().numpy())
        x1_list.append(x1.cpu().numpy())
    x0_array = np.array(x0_list)
    x1_array = np.array(x1_list)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x0_array[:, 0], x0_array[:, 1], color='green', alpha=0.6, label='Base (x0)')
    plt.scatter(x1_array[:, 0], x1_array[:, 1], color='red', alpha=0.6, label='Target (x1)')
    for i in range(num_samples):
        plt.plot([x0_array[i, 0], x1_array[i, 0]], [x0_array[i, 1], x1_array[i, 1]],
                 color='gray', alpha=0.3)
    plt.title("Sample Pairs from Base and Target Distributions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_flow_field_with_trajectories(model, device='cpu'):
    """
    Visualize the learned flow field at a fixed time (t=0.5) and overlay sample trajectories.
    The trajectories are computed using a simple Euler integration.
    """
    model.eval()
    
    # Create a grid of points for the vector field
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    grid_size = 20
    x_vals = np.linspace(x_min, x_max, grid_size)
    y_vals = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    positions = np.stack([X.ravel(), Y.ravel()], axis=1)
    positions_tensor = torch.tensor(positions, dtype=torch.float32, device=device)
    
    # Use a fixed time t = 0.5 for the vector field
    t = torch.full((positions_tensor.size(0),), 0.5, dtype=torch.float32, device=device)
    with torch.no_grad():
        v_pred = model(positions_tensor, t).cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    plt.quiver(positions[:, 0], positions[:, 1], v_pred[:, 0], v_pred[:, 1],
               color='blue', angles='xy', scale_units='xy', scale=1.5, width=0.003)
    plt.title("Learned Flow Field with Sample Trajectories (t=0.5)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    
    # Overlay trajectories: choose a few starting points and simulate their evolution.
    num_trajectories = 10
    times = np.linspace(0, 1, 6)  # For example: t = 0, 0.2, 0.4, 0.6, 0.8, 1.0
    indices = np.random.choice(len(positions), num_trajectories, replace=False)
    start_points = positions[indices]
    
    for point in start_points:
        trajectory = []
        x = torch.tensor(point, dtype=torch.float32, device=device).unsqueeze(0)  # (1,2)
        for t_val in times:
            t_tensor = torch.tensor([t_val], dtype=torch.float32, device=device)
            with torch.no_grad():
                v = model(x, t_tensor)
            # Use a simple Euler step; here dt is chosen arbitrarily for visualization
            dt = 0.2
            x = x + v * dt
            trajectory.append(x.cpu().numpy().squeeze())
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', linestyle='-', alpha=0.8)
    
    plt.show()


def main():
    # Set random seed for reproducibility.
    set_random_seed(42)

    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_samples = 10000
    batch_size = 128
    num_epochs = 10
    learning_rate = 1e-3

    # Initialize dataset, model, and trainer.
    dataset = FlowMatchingDataset(num_samples=num_samples, device=device)
    model = FlowMatchingModel(input_dim=3, hidden_dim=128, num_layers=4, output_dim=2)
    trainer = Trainer(model, dataset, device=device, batch_size=batch_size, lr=learning_rate)

    # Train the model
    trainer.train(num_epochs=num_epochs)

    # Visualize sample pairs and the desired transformation.
    visualize_samples_and_trajectories(dataset, device=device)
    
    # Visualize the learned flow field and overlay sample trajectories.
    visualize_flow_field_with_trajectories(model, device=device)


if __name__ == '__main__':
    main()
