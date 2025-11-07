"""
Improved ConvCNP training script with:
- Scientific notation for losses
- Early stopping
- Learning rate scheduling
- Regularization improvements
"""

import argparse
import json
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""

    def __init__(self, patience=10, min_delta=0, mode='min', verbose=True):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss, 'max' for metrics like R²
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        """Check if we should stop training."""
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        # Check if score improved
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"✓ Validation improved (best epoch: {epoch})")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠ No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"🛑 Early stopping triggered! Best epoch was {self.best_epoch}")

        return self.early_stop


def train_epoch(model, dataloader, optimizer, device, gradient_clip=1.0):
    """Train for one epoch with gradient clipping."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc='Training'):
        optimizer.zero_grad()
        batch_loss = 0

        # Process each tile in the batch
        for i in range(len(batch['context_coords'])):
            context_coords = batch['context_coords'][i].to(device)
            context_embeddings = batch['context_embeddings'][i].to(device)
            context_agbd = batch['context_agbd'][i].to(device)
            target_coords = batch['target_coords'][i].to(device)
            target_embeddings = batch['target_embeddings'][i].to(device)
            target_agbd = batch['target_agbd'][i].to(device)

            if len(target_coords) == 0:
                continue

            # Forward pass
            pred_mean, pred_log_var = model(
                context_coords,
                context_embeddings,
                context_agbd,
                target_coords,
                target_embeddings
            )

            # Compute loss (with small epsilon for numerical stability)
            loss = neural_process_loss(pred_mean, pred_log_var, target_agbd)
            batch_loss += loss

        if batch_loss > 0:
            batch_loss = batch_loss / len(batch['context_coords'])
            batch_loss.backward()

            # Gradient clipping to prevent exploding gradients
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            total_loss += batch_loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_metrics = []
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            batch_loss = 0

            for i in range(len(batch['context_coords'])):
                context_coords = batch['context_coords'][i].to(device)
                context_embeddings = batch['context_embeddings'][i].to(device)
                context_agbd = batch['context_agbd'][i].to(device)
                target_coords = batch['target_coords'][i].to(device)
                target_embeddings = batch['target_embeddings'][i].to(device)
                target_agbd = batch['target_agbd'][i].to(device)

                if len(target_coords) == 0:
                    continue

                # Forward pass
                pred_mean, pred_log_var = model(
                    context_coords,
                    context_embeddings,
                    context_agbd,
                    target_coords,
                    target_embeddings
                )

                # Compute loss
                loss = neural_process_loss(pred_mean, pred_log_var, target_agbd)
                batch_loss += loss

                # Compute metrics
                pred_std = torch.exp(0.5 * pred_log_var) if pred_log_var is not None else None
                metrics = compute_metrics(pred_mean, pred_std, target_agbd)
                all_metrics.append(metrics)

            if batch_loss > 0:
                batch_loss = batch_loss / len(batch['context_coords'])
                total_loss += batch_loss.item()
                n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)

    # Aggregate metrics
    avg_metrics = {}
    if len(all_metrics) > 0:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_loss, avg_metrics


def neural_process_loss(pred_mean, pred_log_var, target, eps=1e-8):
    """
    Compute negative log-likelihood loss with numerical stability.

    Args:
        pred_mean: Predicted means (N,)
        pred_log_var: Predicted log variances (N,) or None
        target: Target values (N,)
        eps: Small constant for numerical stability
    """
    if pred_log_var is None:
        # MSE loss if no uncertainty
        return torch.mean((pred_mean - target) ** 2)
    else:
        # Negative log-likelihood with numerical stability
        # Clamp log_var to prevent extreme values
        pred_log_var = torch.clamp(pred_log_var, min=-10, max=10)

        # NLL = 0.5 * (log(2π) + log_var + (y - μ)²/var)
        # We drop the constant log(2π) as it doesn't affect optimization
        nll = 0.5 * (pred_log_var + ((pred_mean - target) ** 2) / (torch.exp(pred_log_var) + eps))
        return torch.mean(nll)


def compute_metrics(pred_mean, pred_std, target):
    """Compute evaluation metrics."""
    pred_mean = pred_mean.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    # RMSE
    rmse = np.sqrt(np.mean((pred_mean - target) ** 2))

    # MAE
    mae = np.mean(np.abs(pred_mean - target))

    # R²
    ss_res = np.sum((target - pred_mean) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

    # Uncertainty metrics if available
    if pred_std is not None:
        pred_std = pred_std.detach().cpu().numpy()
        metrics['mean_std'] = np.mean(pred_std)

        # Calibration: how often does truth fall within predicted uncertainty
        z_scores = np.abs(pred_mean - target) / (pred_std + 1e-8)
        metrics['calib_1std'] = np.mean(z_scores < 1.0)  # Should be ~68%
        metrics['calib_2std'] = np.mean(z_scores < 2.0)  # Should be ~95%

    return metrics


def train_with_improvements(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs=100,
    early_stopping_patience=15,
    output_dir='./outputs'
):
    """
    Training loop with improvements:
    - Scientific notation for losses
    - Early stopping
    - LR scheduling
    - Better logging
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Early stopping based on validation R² (maximize)
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=0.01,  # At least 1% improvement
        mode='max',  # Maximize R²
        verbose=True
    )

    best_val_r2 = -float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'val_r2': [],
        'lr': []
    }

    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 80)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss, val_metrics = validate(model, val_loader, device)

        # Update scheduler
        if scheduler is not None:
            scheduler.step(val_metrics.get('r2', -float('inf')))

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_metrics.get('rmse', 0))
        history['val_mae'].append(val_metrics.get('mae', 0))
        history['val_r2'].append(val_metrics.get('r2', 0))
        history['lr'].append(current_lr)

        # Print metrics with scientific notation
        print(f"Train Loss: {train_loss:.6e}")
        print(f"Val Loss:   {val_loss:.6e}")
        if val_metrics:
            print(f"Val RMSE:   {val_metrics.get('rmse', 0):.4f}")
            print(f"Val MAE:    {val_metrics.get('mae', 0):.4f}")
            print(f"Val R²:     {val_metrics.get('r2', 0):.4f}")
            if 'mean_std' in val_metrics:
                print(f"Val σ:      {val_metrics.get('mean_std', 0):.4f}")
            if 'calib_1std' in val_metrics:
                print(f"Calib 1σ:   {val_metrics.get('calib_1std', 0):.1%}")
        print(f"LR:         {current_lr:.6e}")

        # Save best model (based on R²)
        val_r2 = val_metrics.get('r2', -float('inf'))
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'train_loss': train_loss
            }, output_dir / 'best_model.pt')
            print(f"✓ Saved best model (R²={val_r2:.4f})")

        # Check early stopping
        if early_stopping(val_r2, epoch):
            print(f"\n🛑 Early stopping at epoch {epoch}")
            print(f"   Best epoch: {early_stopping.best_epoch} (R²={early_stopping.best_score:.4f})")
            break

    # Save final model and history
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, output_dir / 'final_model.pt')

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation R²: {best_val_r2:.4f}")
    print(f"Models saved to: {output_dir}")
    print("=" * 80)

    return history


def create_optimizer_and_scheduler(model, lr=1e-4, weight_decay=1e-5):
    """
    Create optimizer with weight decay and learning rate scheduler.

    Args:
        model: The model
        lr: Initial learning rate
        weight_decay: L2 regularization strength
    """
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )

    # ReduceLROnPlateau: reduce LR when validation plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize R²
        factor=0.5,  # Reduce by half
        patience=5,  # Wait 5 epochs
        verbose=True,
        min_lr=1e-7
    )

    return optimizer, scheduler


# Example usage in main training script:
"""
# In your main() function, replace the training loop with:

optimizer, scheduler = create_optimizer_and_scheduler(
    model,
    lr=args.lr,
    weight_decay=1e-5  # Add L2 regularization
)

history = train_with_improvements(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device=args.device,
    epochs=args.epochs,
    early_stopping_patience=15,  # Stop after 15 epochs without improvement
    output_dir=output_dir
)
"""
