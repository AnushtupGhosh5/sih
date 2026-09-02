import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Relative imports
from .model import create_model
from .dataset import create_dataloaders


def evaluate_model(model_path: str, data_dir: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Load a trained model and evaluate it on the validation set.

    Args:
        model_path (str): Path to the trained model checkpoint (.pth).
        data_dir (str): Path to the directory containing dataset.
        device (str): Device to run evaluation on ('cuda' or 'cpu').

    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics including:
            - Position RMSE (total and per-axis)
            - Velocity RMSE (total and per-axis)
            - Max position error
            - Max velocity error
            - 95th percentile errors
            - Raw error distributions for plotting
    """
    device = torch.device(device)
    print(f"Evaluating on device: {device}")

    # Load model
    model = create_model()
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Load data (only taking validation loader for evaluation)
    _, val_loader = create_dataloaders(data_dir=data_dir, batch_size=128, seq_len=100)

    all_pos_errors = []
    all_vel_errors = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)

            # Assuming output format (B, 6) where [..., :3] is pos and [..., 3:] is vel
            pos_err = preds[..., :3] - targets[..., :3]
            vel_err = preds[..., 3:] - targets[..., 3:]

            all_pos_errors.append(pos_err.cpu().numpy())
            all_vel_errors.append(vel_err.cpu().numpy())

    # Concatenate all batches
    pos_errors_np = np.concatenate(all_pos_errors, axis=0)
    vel_errors_np = np.concatenate(all_vel_errors, axis=0)

    # Flatten across time and batch if sequence output, else just batch
    if pos_errors_np.ndim > 2:
        pos_errors_np = pos_errors_np.reshape(-1, 3)
        vel_errors_np = vel_errors_np.reshape(-1, 3)

    # Calculate metrics
    # RMSE per axis
    pos_rmse_axis = np.sqrt(np.mean(pos_errors_np ** 2, axis=0))
    vel_rmse_axis = np.sqrt(np.mean(vel_errors_np ** 2, axis=0))

    # Total RMSE
    pos_rmse_total = np.sqrt(np.mean(np.sum(pos_errors_np ** 2, axis=1)))
    vel_rmse_total = np.sqrt(np.mean(np.sum(vel_errors_np ** 2, axis=1)))

    # Max errors (Euclidean distance)
    pos_err_norm = np.linalg.norm(pos_errors_np, axis=1)
    vel_err_norm = np.linalg.norm(vel_errors_np, axis=1)

    max_pos_err = np.max(pos_err_norm)
    max_vel_err = np.max(vel_err_norm)

    # 95th Percentile errors
    p95_pos_err = np.percentile(pos_err_norm, 95)
    p95_vel_err = np.percentile(vel_err_norm, 95)

    metrics = {
        'pos_rmse_total': float(pos_rmse_total),
        'vel_rmse_total': float(vel_rmse_total),
        'pos_rmse_axis': pos_rmse_axis.tolist(),
        'vel_rmse_axis': vel_rmse_axis.tolist(),
        'max_pos_err': float(max_pos_err),
        'max_vel_err': float(max_vel_err),
        'p95_pos_err': float(p95_pos_err),
        'p95_vel_err': float(p95_vel_err),
        'raw_pos_errors': pos_errors_np,  # Included for plotting
        'raw_vel_errors': vel_errors_np
    }

    print("=== Evaluation Results ===")
    print(f"Position RMSE Total: {metrics['pos_rmse_total']:.4f}")
    print(f"Velocity RMSE Total: {metrics['vel_rmse_total']:.4f}")
    print(f"Position RMSE (N, E, D): {metrics['pos_rmse_axis']}")
    print(f"Velocity RMSE (N, E, D): {metrics['vel_rmse_axis']}")
    print(f"Max Position Error: {metrics['max_pos_err']:.4f}")
    print(f"Max Velocity Error: {metrics['max_vel_err']:.4f}")
    print(f"95th Percentile Pos Error: {metrics['p95_pos_err']:.4f}")
    print(f"95th Percentile Vel Error: {metrics['p95_vel_err']:.4f}")

    return metrics


def plot_error_distribution(errors: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Plot error histograms for position and velocity if matplotlib is available.

    Args:
        errors (Dict[str, Any]): Dictionary containing raw errors from evaluate_model.
        save_path (Optional[str]): If provided, save the plot to this path instead of showing.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib is not installed. Skipping error distribution plot.")
        return

    pos_err_norm = np.linalg.norm(errors['raw_pos_errors'], axis=1)
    vel_err_norm = np.linalg.norm(errors['raw_vel_errors'], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Position Error Histogram
    axes[0].hist(pos_err_norm, bins=50, color='blue', alpha=0.7)
    axes[0].set_title('Position Error Distribution')
    axes[0].set_xlabel('Error (m)')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)

    # Velocity Error Histogram
    axes[1].hist(vel_err_norm, bins=50, color='green', alpha=0.7)
    axes[1].set_title('Velocity Error Distribution')
    axes[1].set_xlabel('Error (m/s)')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate IMU Drift Predictor")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to data directory")
    parser.add_argument('--device', type=str, default='cuda', help="Device (cuda/cpu)")
    parser.add_argument('--save_plot', type=str, default=None, help="Path to save the error distribution plot")

    args = parser.parse_args()

    metrics = evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        device=args.device if torch.cuda.is_available() or args.device != 'cuda' else 'cpu'
    )
    
    plot_error_distribution(metrics, save_path=args.save_plot)
