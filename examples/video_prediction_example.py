"""
video_prediction_example.py

This script demonstrates a complete pipeline for video inpainting using a synthetic dataset.
It generates synthetic video sequences split into three segments (observed before, missing, observed after),
trains a Transformer model to inpaint the missing segment, and then visualizes the results with static plots
and an animation.

Modules imported from the src directory:
    - VideoInpaintingDataset: Generates synthetic video data.
    - VideoInpaintingTransformer: Defines the Transformer model for inpainting.
    - VideoInpaintingTrainer: Handles model training.
    - set_random_seed: Utility function to set reproducibility seeds.
"""

import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Add project root to sys.path so that modules in src/ can be imported.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.video_inpainting_dataset import VideoInpaintingDataset
from src.video_inpainting_transformer import VideoInpaintingTransformer
from src.video_inpainting_trainer import VideoInpaintingTrainer
from src.utils import set_random_seed  # Assumes you have this function in utils.py

# Define output directory for saving images and animations.
OUTPUT_DIR = os.path.join(os.getcwd(), 'images', 'video_prediction')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def denormalize(tensor, mean, std):
    """
    Denormalizes a tensor using provided mean and standard deviation.

    Args:
        tensor (Tensor): Normalized tensor.
        mean (float): Mean value used for normalization.
        std (float): Standard deviation used for normalization.

    Returns:
        Tensor: Denormalized tensor.
    """
    return tensor * std + mean


def visualize_inpainting(obs_before, missing_gt, obs_after, pred_missing, joint_idx=0, save_path=None):
    """
    Creates a static plot comparing the observed and predicted joint trajectories.

    Args:
        obs_before (Tensor): Observed frames before the missing segment (shape: [num_frames, num_joints*2]).
        missing_gt (Tensor): Ground truth for the missing segment (normalized; shape: [num_frames, num_joints*2]).
        obs_after (Tensor): Observed frames after the missing segment (shape: [num_frames, num_joints*2]).
        pred_missing (Tensor): Predicted missing frames (normalized; shape: [num_frames, num_joints*2]).
        joint_idx (int): Index of the joint to visualize.
        save_path (str, optional): If provided, the plot is saved to this path; otherwise, it is displayed.
    """
    def extract_joint(seq):
        # Reshape sequence and extract the x and y coordinates for the given joint index.
        seq = seq.view(seq.size(0), -1)
        x = seq[:, joint_idx * 2].cpu().numpy()
        y = seq[:, joint_idx * 2 + 1].cpu().numpy()
        return x, y

    before_x, before_y = extract_joint(obs_before)
    missing_gt_x, missing_gt_y = extract_joint(missing_gt)
    after_x, after_y = extract_joint(obs_after)
    pred_x, pred_y = extract_joint(pred_missing)

    plt.figure(figsize=(8, 6))
    plt.plot(before_x, before_y, 'b--', label='Observed (Before)')
    plt.plot(after_x, after_y, 'b--', label='Observed (After)')
    plt.plot(missing_gt_x, missing_gt_y, 'g-', label='True Missing')
    plt.plot(pred_x, pred_y, 'r--', label='Predicted Missing')
    plt.legend()
    plt.title(f'Joint {joint_idx} Inpainting Prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Static inpainting plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def animate_inpainting_predictions(obs_before, missing_gt, obs_after, pred_missing, fps=10, max_obs_secs=2, save_path='inpainting_prediction.gif'):
    """
    Creates an animation showing the progression of inpainting predictions over time.

    Args:
        obs_before (ndarray): Observed frames before the missing segment (shape: [num_frames, num_joints, 2]).
        missing_gt (ndarray): Ground truth for the missing segment (shape: [num_frames, num_joints, 2]).
        obs_after (ndarray): Observed frames after the missing segment (shape: [num_frames, num_joints, 2]).
        pred_missing (ndarray): Predicted missing frames (shape: [num_frames, num_joints, 2]).
        fps (int): Frames per second for the animation.
        max_obs_secs (int): Maximum seconds to display for observed segments.
        save_path (str): Filename for saving the animation (GIF).
    """
    obs_before_len = obs_before.shape[0]
    obs_after_len = obs_after.shape[0]
    missing_len = missing_gt.shape[0]
    max_obs_frames = max_obs_secs * fps

    # Limit the number of observed frames to display.
    obs_before_disp = obs_before[-max_obs_frames:] if obs_before_len > max_obs_frames else obs_before
    obs_after_disp = obs_after[:max_obs_frames] if obs_after_len > max_obs_frames else obs_after
    disp_obs_before_len = obs_before_disp.shape[0]
    disp_obs_after_len = obs_after_disp.shape[0]
    total_frames = disp_obs_before_len + missing_len + disp_obs_after_len

    fig, ax = plt.subplots(figsize=(8, 6))

    # Set plot limits based on all data points.
    all_data = np.concatenate([obs_before_disp, missing_gt, obs_after_disp, pred_missing], axis=0)
    x_min, x_max = np.min(all_data[:, :, 0]), np.max(all_data[:, :, 0])
    y_min, y_max = np.min(all_data[:, :, 1]), np.max(all_data[:, :, 1])
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)

    # Plot background trajectories for observed segments.
    for j in range(obs_before_disp.shape[1]):
        ax.plot(obs_before_disp[:, j, 0], obs_before_disp[:, j, 1], 'b--', alpha=0.5)
    for j in range(obs_after_disp.shape[1]):
        ax.plot(obs_after_disp[:, j, 0], obs_after_disp[:, j, 1], 'b--', alpha=0.5)

    scatter_true = ax.scatter([], [], c='green', s=50, label='True Missing')
    scatter_pred = ax.scatter([], [], c='red', s=50, label='Predicted Missing')
    pred_line, = ax.plot([], [], 'r--', lw=2, label='Predicted Missing (dashed)')
    ax.legend()

    def update(frame):
        """
        Update function for each frame of the animation.

        Args:
            frame (int): Current frame index.
        """
        if frame < disp_obs_before_len:
            current_true = obs_before_disp[frame]
            current_pred = obs_before_disp[frame]
            pred_line.set_data([], [])
        elif frame < disp_obs_before_len + missing_len:
            idx = frame - disp_obs_before_len
            current_true = missing_gt[idx]
            current_pred = pred_missing[idx]
            pred_points = pred_missing[:idx + 1]
            pred_line.set_data(pred_points[:, 0, 0], pred_points[:, 0, 1])
        else:
            idx = frame - disp_obs_before_len - missing_len
            current_true = obs_after_disp[idx]
            current_pred = obs_after_disp[idx]
        scatter_true.set_offsets(current_true)
        scatter_pred.set_offsets(current_pred)
        ax.set_title(f'Frame {frame + 1}/{total_frames}')
        return scatter_true, scatter_pred, pred_line

    anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=200, blit=True)
    gif_full_path = os.path.join(OUTPUT_DIR, save_path)
    anim.save(gif_full_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Inpainting animation saved as {gif_full_path}")


def main():
    """
    Main function to run the video inpainting pipeline:
        - Sets the random seed.
        - Initializes the dataset, model, and trainer.
        - Trains the model.
        - Generates predictions for one sample.
        - Denormalizes and visualizes the results.
    """
    set_random_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_samples = 1024
    obs_before_len = 30    # Number of frames observed before the missing segment.
    missing_len = 2       # Number of missing frames to predict.
    obs_after_len = 30     # Number of frames observed after the missing segment.
    fps = 10

    # Create synthetic dataset.
    dataset = VideoInpaintingDataset(num_samples=num_samples, obs_before_len=obs_before_len,
                                     missing_len=missing_len, obs_after_len=obs_after_len, fps=fps)

    input_size = 2  # (x, y) coordinates.
    # Initialize the Transformer model for video inpainting.
    model = VideoInpaintingTransformer(input_size=input_size, d_model=128, nhead=8,
                                       num_encoder_layers=4, num_decoder_layers=4,
                                       dim_feedforward=512, missing_len=missing_len, dropout=0.1)

    # Create a trainer instance with the model and dataset.
    trainer = VideoInpaintingTrainer(model, dataset, device=device, batch_size=32, lr=1e-3)

    # Train the model for 20 epochs with a linearly decaying teacher forcing ratio.
    trainer.train(num_epochs=2, initial_teacher_forcing_ratio=0.6)

    model.eval()
    with torch.no_grad():
        # Retrieve the first sample for evaluation.
        obs_before, missing_gt, obs_after, mean, std = dataset[0]
        obs_before_in = obs_before.unsqueeze(0).to(device)  # Shape: (1, obs_before_len, 2)
        obs_after_in = obs_after.unsqueeze(0).to(device)    # Shape: (1, obs_after_len, 2)
        # Generate missing frames using autoregressive decoding.
        pred_missing = model(obs_before_in, obs_after_in, tgt=None, teacher_forcing_ratio=0.0)
        pred_missing = pred_missing.squeeze(0)  # Shape: (missing_len, 2)
        print("Mean:", mean.item(), "Std:", std.item())
        print("Ground Truth (normalized):", missing_gt)
        print("Prediction (normalized):", pred_missing)

    # Denormalize the data for visualization.
    obs_before_denorm = denormalize(obs_before, mean, std)
    missing_gt_denorm = denormalize(missing_gt, mean, std)
    obs_after_denorm = denormalize(obs_after, mean, std)
    pred_missing_denorm = denormalize(pred_missing, mean, std)

    # Reshape tensors into numpy arrays for animation (adding a joint dimension).
    obs_before_np = obs_before_denorm.cpu().numpy().reshape(obs_before_len, 1, 2)
    missing_gt_np = missing_gt_denorm.cpu().numpy().reshape(missing_len, 1, 2)
    obs_after_np = obs_after_denorm.cpu().numpy().reshape(obs_after_len, 1, 2)
    pred_missing_np = pred_missing_denorm.cpu().numpy().reshape(missing_len, 1, 2)

    # Generate and save an animation of the inpainting prediction.
    animate_inpainting_predictions(obs_before_np, missing_gt_np, obs_after_np, pred_missing_np,
                                   fps=fps, max_obs_secs=2, save_path='inpainting_prediction.gif')

    # Generate and save a static plot for joint index 0.
    static_plot_path = os.path.join(OUTPUT_DIR, 'joint0_inpainting.png')
    visualize_inpainting(obs_before_denorm.view(obs_before_len, -1),
                         missing_gt_denorm.view(missing_len, -1),
                         obs_after_denorm.view(obs_after_len, -1),
                         pred_missing_denorm.view(missing_len, -1),
                         joint_idx=0,
                         save_path=static_plot_path)


if __name__ == '__main__':
    main()
