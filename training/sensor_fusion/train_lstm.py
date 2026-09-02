import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Relative imports
from .model import IMUDriftPredictor, DriftPredictionLoss, create_model, count_parameters
from .dataset import create_dataloaders


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The IMU drift predictor model.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run training on.

    Returns:
        Dict[str, float]: Dictionary containing average loss, pos_rmse, and vel_rmse.
    """
    model.train()
    total_loss = 0.0
    total_pos_mse = 0.0
    total_vel_mse = 0.0
    num_samples = 0

    for batch in loader:
        # Assume batch contains inputs and targets
        # Unpack according to your dataset structure, here we assume (inputs, targets)
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(inputs)
        
        loss = criterion(preds, targets)
        loss.backward()
        
        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

        # Assuming preds and targets are of shape (B, 6) where [..., :3] is pos and [..., 3:] is vel
        with torch.no_grad():
            pos_mse = torch.nn.functional.mse_loss(preds[..., :3], targets[..., :3], reduction='sum').item()
            vel_mse = torch.nn.functional.mse_loss(preds[..., 3:], targets[..., 3:], reduction='sum').item()
            
            # Since mse_loss with sum reduction sums over all elements, we divide by (batch_size * 3) later
            total_pos_mse += pos_mse / 3.0
            total_vel_mse += vel_mse / 3.0

    avg_loss = total_loss / num_samples
    pos_rmse = (total_pos_mse / num_samples) ** 0.5
    vel_rmse = (total_vel_mse / num_samples) ** 0.5

    return {
        'loss': avg_loss,
        'pos_rmse': pos_rmse,
        'vel_rmse': vel_rmse
    }


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate the model.

    Args:
        model (nn.Module): The IMU drift predictor model.
        loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run validation on.

    Returns:
        Dict[str, float]: Dictionary containing average loss, pos_rmse, and vel_rmse.
    """
    model.eval()
    total_loss = 0.0
    total_pos_mse = 0.0
    total_vel_mse = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)
            loss = criterion(preds, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size

            pos_mse = torch.nn.functional.mse_loss(preds[..., :3], targets[..., :3], reduction='sum').item()
            vel_mse = torch.nn.functional.mse_loss(preds[..., 3:], targets[..., 3:], reduction='sum').item()
            
            total_pos_mse += pos_mse / 3.0
            total_vel_mse += vel_mse / 3.0

    avg_loss = total_loss / num_samples
    pos_rmse = (total_pos_mse / num_samples) ** 0.5
    vel_rmse = (total_vel_mse / num_samples) ** 0.5

    return {
        'loss': avg_loss,
        'pos_rmse': pos_rmse,
        'vel_rmse': vel_rmse
    }


def main(args: argparse.Namespace) -> None:
    """
    Full training loop for the IMU drift predictor.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    device = torch.device(args.device if torch.cuda.is_available() or args.device != 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # 1. Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len
    )

    # 2. Create model, loss, optimizer
    model = create_model().to(device)
    print(f"Model created with {count_parameters(model):,} parameters.")
    
    criterion = DriftPredictionLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 3. CosineAnnealingLR scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pth"
    history_path = output_dir / "training_history.json"

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    history = {'train': [], 'val': []}

    # 4. Training loop
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch:03d}/{args.epochs} [{epoch_time:.1f}s] - "
              f"Train Loss: {train_metrics['loss']:.4f}, Pos RMSE: {train_metrics['pos_rmse']:.4f}, Vel RMSE: {train_metrics['vel_rmse']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}, Pos RMSE: {val_metrics['pos_rmse']:.4f}, Vel RMSE: {val_metrics['vel_rmse']:.4f}")

        # 5 & 6. Early stopping and save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, best_model_path)
            print(f"  [*] Best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time / 60:.2f} minutes.")

    # 7. Save training history
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")

    # 8. Print summary
    print("=== Training Summary ===")
    print(f"Total Epochs: {epoch}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model Checkpoint: {best_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train IMU Drift Predictor LSTM")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to training data directory")
    parser.add_argument('--output_dir', type=str, default="checkpoints", help="Directory to save model and logs")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--seq_len', type=int, default=100, help="Sequence length for LSTM")
    parser.add_argument('--device', type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    main(args)
