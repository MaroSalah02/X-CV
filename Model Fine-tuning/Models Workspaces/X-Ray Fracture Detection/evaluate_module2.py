"""
Evaluation and inference utilities for fracture detection model.
"""

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from data_preprocessing import get_transforms, load_xray
from train import predict_probs


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    """AUC, accuracy, sensitivity, specificity, precision and F1 at a threshold."""
    preds = (probs > threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {
        'n': len(labels),
        'pos': int(labels.sum()),
        'auc': roc_auc_score(labels, probs) if 0 < labels.sum() < len(labels) else float('nan'),
        'accuracy': (tp + tn) / len(labels),
        'balanced_acc': (sens + spec) / 2,
        'sensitivity': sens,
        'specificity': spec,
        'precision': prec,
        'f1': 2 * prec * sens / max(prec + sens, 1e-12),
    }


def evaluate_on_test(model: nn.Module,
                     test_loader: DataLoader,
                     device: str,
                     threshold: Optional[float] = None,
                     sources: Optional[Sequence[str]] = None,
                     tta: Optional[bool] = None,
                     plot: bool = True) -> Dict:
    """
    Comprehensive evaluation on the test set: classification report, confusion matrix,
    ROC curve and a per-source breakdown (each dataset can behave very differently).

    Args:
        model: trained model (if loaded with train.load_model, its stored threshold is used)
        test_loader: DataLoader over the test split (shuffle=False)
        device: device to evaluate on
        threshold: decision threshold on P(fractured); defaults to the checkpoint's tuned value
        sources: per-sample source names aligned with test_loader.dataset (e.g. 'A', 'C')
        tta: horizontal-flip test-time augmentation; defaults to the checkpoint setting
        plot: draw the confusion matrix / ROC figures

    Returns:
        Dict with overall metrics, per-source metrics, probabilities and labels
    """
    meta = getattr(model, 'fracture_meta', {}) or {}
    if threshold is None:
        threshold = meta.get('threshold', 0.5)
    if tta is None:
        tta = meta.get('tta', True)

    print(f"\nEvaluating on test set (threshold={threshold:.3f}, flip-TTA={tta})...")
    probs, labels, _ = predict_probs(model, test_loader, device, tta=tta)
    preds = (probs > threshold).astype(int)

    print("\n" + "=" * 70)
    print("Classification Report")
    print("=" * 70)
    print(classification_report(labels, preds, target_names=['Not Fractured', 'Fractured'], digits=4))

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix")
    print(cm)

    overall = binary_metrics(labels, probs, threshold)
    per_source = None
    if sources is not None:
        sources = np.asarray(sources)
        rows = {s: binary_metrics(labels[sources == s], probs[sources == s], threshold)
                for s in sorted(set(sources))}
        rows['ALL'] = overall
        per_source = pd.DataFrame(rows).T
        print("\n" + "=" * 70)
        print("Per-source metrics")
        print("=" * 70)
        print(per_source.round(4).to_string())

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['Not Fractured', 'Fractured'],
                    yticklabels=['Not Fractured', 'Fractured'],
                    cbar_kws={'label': 'Count'})
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_title('Confusion Matrix - Test Set', fontweight='bold')

        axes[1].plot(fpr, tpr, lw=2, label=f'All (AUC = {roc_auc:.4f})')
        if sources is not None:
            for s in sorted(set(sources)):
                m = sources == s
                if 0 < labels[m].sum() < m.sum():
                    f, t, _ = roc_curve(labels[m], probs[m])
                    axes[1].plot(f, t, lw=1.2, label=f'{s} (AUC = {auc(f, t):.4f})')
        axes[1].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve - Test Set', fontweight='bold')
        axes[1].legend(loc='lower right')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    print("=" * 70)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("=" * 70)

    return {'overall': overall, 'per_source': per_source, 'probs': probs, 'labels': labels,
            'threshold': threshold}


def predict_single_image(model: nn.Module,
                         image_path: str,
                         device: str = 'cuda',
                         threshold: Optional[float] = None,
                         img_size: Optional[int] = None) -> Dict:
    """
    Predict fracture classification for a single image using the checkpoint's
    preprocessing, threshold and TTA settings.

    Returns:
        Dict: prediction, confidence, probabilities, threshold
    """
    meta = getattr(model, 'fracture_meta', {}) or {}
    threshold = meta.get('threshold', 0.5) if threshold is None else threshold
    img_size = meta.get('img_size', 224) if img_size is None else img_size

    img = load_xray(Image.open(image_path))
    img_tensor = get_transforms('test', img_size=img_size)(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(img_tensor), dim=1)
        if meta.get('tta', True):
            probs = 0.5 * (probs + torch.softmax(model(torch.flip(img_tensor, dims=[3])), dim=1))
    p_fx = probs[0, 1].item()
    pred_class = 1 if p_fx > threshold else 0

    class_labels = ['Not Fractured', 'Fractured']
    return {
        'prediction': class_labels[pred_class],
        'confidence': p_fx if pred_class == 1 else 1 - p_fx,
        'probabilities': {'Fractured': p_fx, 'Not Fractured': 1 - p_fx},
        'threshold': threshold,
    }
