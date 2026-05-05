"""
Evaluation and inference utilities for fracture detection model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from typing import Dict
from tqdm import tqdm


def evaluate_on_test(model: nn.Module,
                     test_loader: DataLoader,
                     device: str,
                     threshold: float = 0.5) -> None:
    """
    Comprehensive evaluation on test set with metrics, confusion matrix, and ROC curve.

    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test dataloader
        device (str): Device to evaluate on
        threshold (float): Decision threshold for Fracture class (default=0.4 to minimize False Negatives)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print("\nEvaluating on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Test evaluation'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            # Use parametrized threshold for Fracture inference
            predicted = (probs[:, 1] >= threshold).long()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fracture

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Classification Report
    print("\n" + "=" * 70)
    print("Classification Report")
    print("=" * 70)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=['Not Fractured', 'Fractured'],
        digits=4
    )
    print(report)

    # Confusion Matrix
    print("=" * 70)
    print("Confusion Matrix")
    print("=" * 70)
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Fractured', 'Fractured'],
                yticklabels=['Not Fractured', 'Fractured'],
                cbar_kws={'label': 'Count'})
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("=" * 70)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("=" * 70)


def predict_single_image(model: nn.Module,
                         image_path: str,
                         device: str = 'cuda',
                         threshold: float = 0.5) -> Dict:
    """
    Predict fracture classification for a single image.

    Args:
        model (nn.Module): Trained model in eval mode
        image_path (str): Path to image file
        device (str): Device to run inference on
        threshold (float): Decision threshold for Fracture class (default=0.5 to balance precision and recall)

    Returns:
        Dict: Dictionary with keys: prediction, confidence, probabilities
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')

    # Apply test transforms (same as validation)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        # Use param threshold
        pred_class = 1 if probs[0, 1].item() >= threshold else 0
        confidence = probs[0, pred_class].item()

    # Map predictions to labels
    class_labels = ['Not Fractured', 'Fractured']
    prediction = class_labels[pred_class]

    probabilities = {
        'Fractured': probs[0, 1].item(),
        'Not Fractured': probs[0, 0].item()
    }

    result = {
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': probabilities
    }

    return result
