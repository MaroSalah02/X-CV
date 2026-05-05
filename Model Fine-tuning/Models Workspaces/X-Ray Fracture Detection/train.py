"""
Training pipeline for DenseNet121 fracture classification model.
Includes two-phase fine-tuning, mixed precision, and early stopping.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple, Optional

from model import build_densenet121, unfreeze_layers


def get_optimizer(model: nn.Module, lr: float = 1e-4,
                  weight_decay: float = 1e-4) -> optim.AdamW:
    """
    Create AdamW optimizer with different learning rates for different layers.

    Args:
        model (nn.Module): DenseNet121 model
        lr (float): Learning rate for classifier head
        weight_decay (float): Weight decay for regularization

    Returns:
        optim.AdamW: Configured optimizer with param groups
    """
    # Identify which parameters are in the classifier head
    classifier_params = set(model.classifier.parameters())

    # Create param groups with different learning rates
    param_groups = [
        {'params': [p for p in model.parameters() if p not in classifier_params and p.requires_grad],
         'lr': lr / 10.0,  # Lower LR for backbone
         'weight_decay': weight_decay},
        {'params': [p for p in model.classifier.parameters() if p.requires_grad],
         'lr': lr,  # Higher LR for classifier
         'weight_decay': weight_decay}
    ]

    optimizer = optim.AdamW(param_groups)
    return optimizer


def get_scheduler(optimizer: optim.Optimizer,
                  num_epochs: int) -> CosineAnnealingLR:
    """
    Create CosineAnnealingLR scheduler.

    Args:
        optimizer (optim.Optimizer): Optimizer to schedule
        num_epochs (int): Total number of epochs

    Returns:
        CosineAnnealingLR: Configured scheduler
    """
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    return scheduler


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module,
                    device: str,
                    scaler: GradScaler) -> Tuple[float, float]:
    """
    Train for one epoch with mixed precision.

    Args:
        model (nn.Module): Model to train
        loader (DataLoader): Training dataloader
        optimizer (optim.Optimizer): Optimizer
        criterion (nn.Module): Loss function
        device (str): Device to train on ('cuda' or 'cpu')
        scaler (GradScaler): Mixed precision gradient scaler

    Returns:
        Tuple[float, float]: (average_loss, average_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{running_loss / (total / images.size(0)):.4f}'})

    avg_loss = running_loss / len(loader)
    avg_acc = 100 * correct / total

    return avg_loss, avg_acc


def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: str) -> Tuple[float, float, str]:
    """
    Evaluate model on validation/test set.

    Args:
        model (nn.Module): Model to evaluate
        loader (DataLoader): Evaluation dataloader
        criterion (nn.Module): Loss function
        device (str): Device to evaluate on

    Returns:
        Tuple[float, float, str]: (average_loss, average_accuracy, classification_report)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc='Evaluating', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    avg_acc = 100 * sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=['Not Fractured', 'Fractured'],
        digits=4
    )

    return avg_loss, avg_acc, report


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int = 30,
                device: str = 'cuda',
                save_dir: str = './weights',
                patience: int = 5,
                weight_decay: float = 1e-3,
                class_weights: Optional[torch.Tensor] = None) -> Dict[str, list]:
    """
    Main training loop with two-phase fine-tuning and early stopping.

    Args:
        model (nn.Module): DenseNet121 model
        train_loader (DataLoader): Training dataloader
        val_loader (DataLoader): Validation dataloader
        num_epochs (int): Total number of epochs
        device (str): Device to train on
        save_dir (str): Directory to save checkpoints
        patience (int): Early stopping patience
        weight_decay (float): L2 regularization
        class_weights (torch.Tensor, optional): Optional weights for CrossEntropyLoss

    Returns:
        Dict[str, list]: Training history with keys: train_loss, val_loss, train_acc, val_acc
    """
    os.makedirs(save_dir, exist_ok=True)

    # Setup training
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, num_epochs)
    scaler = GradScaler()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0

    print(f"\nTraining on {device}")
    print(f"Total epochs: {num_epochs}")
    print("=" * 70)

    for epoch in range(1, num_epochs + 1):
        # Phase 1: Freeze backbone, train only classifier (epochs 1-5)
        if epoch <= 5 and epoch == 1:
            print("\n[Phase 1] Training classifier head only (frozen backbone)")
            for param in model.features.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True

        # Phase 2: Unfreeze last 2 DenseBlocks (epochs 6+)
        if epoch == 6:
            print("\n[Phase 2] Fine-tuning with unfrozen last 2 DenseBlocks")
            model = unfreeze_layers(model, num_layers_to_unfreeze=2)
            model = model.to(device)
            # Recreate optimizer with new parameter groups
            optimizer = get_optimizer(model, weight_decay=weight_decay)

        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # Evaluate on validation set
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress
        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Early stopping and checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), os.path.join(save_dir, 'densenet121_best.pth'))

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'history': history
            }
            torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_best.pt'))
            print(f"  ✓ Best model saved (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement in validation loss")
                print(f"Best validation loss: {best_val_loss:.4f} (Acc: {best_val_acc:.2f}%) at epoch {best_epoch}")
                break

    # Save last model
    torch.save(model.state_dict(), os.path.join(save_dir, 'densenet121_last.pth'))

    # Save final checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'history': history
    }
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_last.pt'))

    print("=" * 70)
    print(f"Training completed!")
    print(f"Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Models saved to: {save_dir}")

    return history


def plot_training_history(history: Dict[str, list]) -> None:
    """
    Plot training and validation loss and accuracy curves.

    Args:
        history (Dict[str, list]): Training history dictionary
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax1.plot(history['val_loss'], label='Val Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy', marker='o', markersize=3)
    ax2.plot(history['val_acc'], label='Val Accuracy', marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def load_model(checkpoint_path: str,
               device: str = 'cuda',
               num_classes: int = 2,
               pretrained: bool = True,
               dropout_rate: float = 0.5) -> nn.Module:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path (str): Path to model checkpoint (.pth file)
        device (str): Device to load model on
        num_classes (int): Number of classes
        pretrained (bool): Whether original model used pretrained weights
        dropout_rate (float): Dropout rate for classifier head

    Returns:
        nn.Module: Loaded model in eval mode
    """
    model = build_densenet121(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )

    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    print(f"Model loaded from: {checkpoint_path}")
    return model
