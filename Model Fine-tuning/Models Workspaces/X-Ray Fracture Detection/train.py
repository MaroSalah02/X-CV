"""
Training pipeline for DenseNet121 fracture classification model.

Two-phase fine-tuning with a single optimizer and a single per-step schedule:
  Phase 1 (first `freeze_epochs` epochs): backbone frozen, only the head trains.
  Phase 2: the whole backbone is unfrozen and trains at `backbone_lr_mult` x the head LR.

Model selection uses validation ROC-AUC (threshold independent). The decision
threshold is then tuned on the validation set and stored in the checkpoint together
with the preprocessing settings, so inference uses exactly what training used.
"""

import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import build_densenet121, set_backbone_trainable


def get_optimizer(model: nn.Module, lr: float = 1e-3, backbone_lr_mult: float = 0.1,
                  weight_decay: float = 1e-4) -> optim.AdamW:
    """
    AdamW with two param groups: backbone (lr * backbone_lr_mult) and classifier head (lr).

    All parameters are registered up front. Frozen parameters never get a gradient,
    and AdamW skips parameters whose .grad is None, so unfreezing the backbone later
    needs no new optimizer (and therefore no new scheduler).
    """
    head_ids = {id(p) for p in model.classifier.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    head = list(model.classifier.parameters())
    return optim.AdamW([
        {'params': backbone, 'lr': lr * backbone_lr_mult, 'weight_decay': weight_decay},
        {'params': head, 'lr': lr, 'weight_decay': weight_decay},
    ])


def get_scheduler(optimizer: optim.Optimizer, total_steps: int, warmup_steps: int,
                  min_lr_ratio: float = 0.01) -> LambdaLR:
    """Per-step linear warmup followed by cosine decay to min_lr_ratio * base LR."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def amp_supported(device: str) -> bool:
    """fp16 autocast only pays off on tensor-core GPUs (compute capability >= 7.0, e.g. T4)."""
    return device.startswith('cuda') and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: optim.Optimizer,
                    scheduler: LambdaLR,
                    criterion: nn.Module,
                    device: str,
                    scaler: torch.amp.GradScaler,
                    use_amp: bool) -> Tuple[float, float]:
    """
    Train for one epoch (optionally with mixed precision). The scheduler steps per batch.

    Returns:
        Tuple[float, float]: (average_loss, average_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    device_type = torch.device(device).type

    pbar = tqdm(loader, desc='Training', leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({'loss': f'{running_loss / total:.4f}'})

    return running_loss / total, 100 * correct / total


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: str,
                  tta: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run the model over a loader and return (P(fractured), labels, mean CE loss).
    With tta=True the softmax of the image and its horizontal mirror are averaged.
    """
    model.eval()
    use_amp = amp_supported(device)
    device_type = torch.device(device).type
    probs, labels_all = [], []
    loss_sum, n = 0.0, 0
    for images, labels in tqdm(loader, desc='Evaluating', leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            logits = model(images).float()
            flipped = model(torch.flip(images, dims=[3])).float() if tta else None
        loss_sum += nn.functional.cross_entropy(logits, labels, reduction='sum').item()
        p = torch.softmax(logits, 1)
        if tta:
            p = 0.5 * (p + torch.softmax(flipped, 1))
        probs.append(p[:, 1].float().cpu().numpy())
        labels_all.append(labels.cpu().numpy())
        n += labels.size(0)
    return np.concatenate(probs), np.concatenate(labels_all), loss_sum / n


def tune_threshold(labels: np.ndarray, probs: np.ndarray, mode: str = 'youden',
                   target_sensitivity: float = 0.90) -> float:
    """
    Pick a decision threshold on validation predictions.

    mode='youden'      -> maximise sensitivity + specificity - 1 (balanced operating point)
    mode='sensitivity' -> highest threshold that still reaches `target_sensitivity`
                          (screening setting: missed fractures are the costly error)
    """
    fpr, tpr, thr = roc_curve(labels, probs)
    thr = np.clip(thr, 0.0, 1.0)  # roc_curve's first threshold is +inf
    if mode == 'youden':
        i = int(np.argmax(tpr - fpr))
    elif mode == 'sensitivity':
        i = int(np.nonzero(tpr >= target_sensitivity)[0][0])
    else:
        raise ValueError(f"mode must be 'youden' or 'sensitivity', got {mode}")
    return float(thr[i])


def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: Optional[nn.Module],
             device: str,
             tta: bool = False) -> Tuple[float, float, str]:
    """
    Evaluate model on validation/test set at the default 0.5 threshold.

    Returns:
        Tuple[float, float, str]: (average_loss, average_accuracy, classification_report)
    """
    probs, labels, loss = predict_probs(model, loader, device, tta=tta)
    preds = (probs > 0.5).astype(int)
    report = classification_report(labels, preds, target_names=['Not Fractured', 'Fractured'], digits=4)
    return loss, 100 * float((preds == labels).mean()), report


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int = 25,
                device: str = 'cuda',
                save_dir: str = './weights',
                patience: int = 6,
                lr: float = 1e-3,
                backbone_lr_mult: float = 0.1,
                weight_decay: float = 1e-4,
                freeze_epochs: int = 2,
                warmup_epochs: float = 1.0,
                label_smoothing: float = 0.05,
                class_weights: Optional[torch.Tensor] = None,
                threshold_mode: str = 'youden',
                checkpoint_meta: Optional[Dict] = None) -> Dict[str, list]:
    """
    Main training loop with two-phase fine-tuning and early stopping on validation AUC.

    Args:
        model: DenseNet121 from build_densenet121 (backbone frozen)
        train_loader / val_loader: DataLoaders yielding (image, label)
        num_epochs: maximum number of epochs
        device: 'cuda' or 'cpu'
        save_dir: where checkpoints are written
        patience: epochs without val-AUC improvement before stopping
        lr: head learning rate (backbone uses lr * backbone_lr_mult)
        weight_decay: AdamW weight decay
        freeze_epochs: number of initial head-only epochs
        warmup_epochs: linear LR warmup length (in epochs)
        label_smoothing: CrossEntropy label smoothing
        class_weights: optional per-class loss weights (tensor of shape [2])
        threshold_mode: how the decision threshold is tuned on val ('youden' | 'sensitivity')
        checkpoint_meta: extra keys stored in the checkpoint (img_size and preprocessing
            are recorded automatically)

    Returns:
        Dict[str, list]: history with train_loss, train_acc, val_loss, val_acc, val_auc, lr
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    use_amp = amp_supported(device)

    weight = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    optimizer = get_optimizer(model, lr=lr, backbone_lr_mult=backbone_lr_mult, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = get_scheduler(optimizer, total_steps=num_epochs * steps_per_epoch,
                              warmup_steps=max(1, int(warmup_epochs * steps_per_epoch)))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': [], 'lr': []}
    best_auc, best_epoch, patience_counter = -1.0, 0, 0
    # Preprocessing that inference must reproduce. The image size is read off a real
    # validation sample so it can never disagree with what the model was trained on.
    meta = {'img_size': int(val_loader.dataset[0][0].shape[-1]),
            'preprocessing': 'pad_to_square',
            **(checkpoint_meta or {})}

    print(f"\nTraining on {device} | mixed precision: {use_amp}")
    print(f"Max epochs: {num_epochs} | head-only epochs: {freeze_epochs} | lr head {lr:g}, backbone {lr * backbone_lr_mult:g}")
    print("=" * 70)
    set_backbone_trainable(model, False)
    print("\n[Phase 1] Training classifier head only (frozen backbone)")

    for epoch in range(1, num_epochs + 1):
        if epoch == freeze_epochs + 1:
            print("\n[Phase 2] Fine-tuning the full network")
            set_backbone_trainable(model, True)

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler,
                                                criterion, device, scaler, use_amp)
        probs, labels, val_loss = predict_probs(model, val_loader, device, tta=False)
        val_auc = roc_auc_score(labels, probs)
        val_acc = 100 * float(((probs > 0.5).astype(int) == labels).mean())

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['lr'].append(optimizer.param_groups[1]['lr'])

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc, best_epoch, patience_counter = val_auc, epoch, 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'densenet121_best.pth'))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'val_auc': val_auc,
                'history': history,
                **meta,
            }, os.path.join(save_dir, 'checkpoint_best.pt'))
            print(f"  ✓ Best model saved (Val AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > freeze_epochs:
                print(f"\nEarly stopping: no val-AUC improvement for {patience} epochs")
                break

    torch.save(model.state_dict(), os.path.join(save_dir, 'densenet121_last.pth'))

    # Tune the decision threshold for the best model on the validation set (with the
    # same flip-TTA used at test time) and store it in the best checkpoint.
    best_path = os.path.join(save_dir, 'checkpoint_best.pt')
    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    probs, labels, _ = predict_probs(model, val_loader, device, tta=True)
    ckpt['threshold'] = tune_threshold(labels, probs, mode=threshold_mode)
    ckpt['threshold_mode'] = threshold_mode
    ckpt['tta'] = True
    ckpt['history'] = history
    torch.save(ckpt, best_path)

    print("=" * 70)
    print(f"Training completed! Best Val AUC: {best_auc:.4f} (epoch {best_epoch})")
    print(f"Decision threshold tuned on val ({threshold_mode}): {ckpt['threshold']:.3f}")
    print(f"Models saved to: {save_dir}")
    return history


def plot_training_history(history: Dict[str, list]) -> None:
    """Plot loss, accuracy and validation AUC curves."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o', markersize=3)
    ax2.plot(epochs, history['val_acc'], label='Val Accuracy @0.5', marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(epochs, history['val_auc'], label='Val ROC-AUC', marker='s', markersize=3, color='tab:green')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUC')
    ax3.set_title('Validation ROC-AUC', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def load_model(checkpoint_path: str,
               device: str = 'cuda',
               num_classes: int = 2,
               dropout_rate: float = 0.5) -> nn.Module:
    """
    Load a trained model from either a bare state_dict (.pth) or a checkpoint (.pt).

    Checkpoint metadata (threshold, img_size, ...) is attached as `model.fracture_meta`.
    """
    model = build_densenet121(num_classes=num_classes, pretrained=False, dropout_rate=dropout_rate)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        model.fracture_meta = {k: v for k, v in ckpt.items()
                               if k not in ('model_state_dict', 'optimizer_state_dict', 'history')}
    else:
        model.load_state_dict(ckpt)
        model.fracture_meta = {}

    model = model.to(device)
    model.eval()
    print(f"Model loaded from: {checkpoint_path} | meta: {model.fracture_meta}")
    return model
