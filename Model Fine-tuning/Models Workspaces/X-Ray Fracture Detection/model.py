"""
DenseNet121 model architecture for binary fracture classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


def build_densenet121(num_classes: int = 2,
                      pretrained: bool = True,
                      dropout_rate: float = 0.5) -> nn.Module:
    """
    Build a DenseNet121 model with replaced classifier head for binary classification.

    Args:
        num_classes (int): Number of output classes (default: 2 for binary)
        pretrained (bool): Whether to load ImageNet pretrained weights
        dropout_rate (float): Dropout rate for regularization

    Returns:
        nn.Module: DenseNet121 model with custom classifier head
    """
    # Load pretrained DenseNet121
    model = models.densenet121(pretrained=pretrained, progress=True)

    # Freeze all backbone layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Get input features for classifier
    num_features = model.classifier.in_features  # 1024

    # Replace classifier with custom head
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(512, num_classes)
    )

    # Unfreeze classifier for training
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def unfreeze_layers(model: nn.Module,
                    num_layers_to_unfreeze: int = 2) -> nn.Module:
    """
    Unfreeze the last N DenseBlocks and the classifier head for fine-tuning.

    Args:
        model (nn.Module): DenseNet121 model
        num_layers_to_unfreeze (int): Number of DenseBlocks to unfreeze from the end

    Returns:
        nn.Module: Model with unfrozen layers
    """
    # Get the DenseNet features
    features = model.features

    # DenseBlocks are: denseblock1, denseblock2, denseblock3, denseblock4
    dense_blocks = []
    for name, module in features.named_children():
        if 'denseblock' in name:
            dense_blocks.append((name, module))

    # Unfreeze last num_layers_to_unfreeze blocks
    num_to_unfreeze = min(num_layers_to_unfreeze, len(dense_blocks))

    unfrozen_names = []
    for i in range(num_to_unfreeze):
        block_idx = len(dense_blocks) - 1 - i
        name, module = dense_blocks[block_idx]
        for param in module.parameters():
            param.requires_grad = True
        unfrozen_names.append(name)

    # Ensure classifier is unfrozen
    for param in model.classifier.parameters():
        param.requires_grad = True
    unfrozen_names.append('classifier')

    print(f"Unfrozen layers: {', '.join(reversed(unfrozen_names))}")

    return model


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> None:
    """
    Print model summary including total and trainable parameters.

    Args:
        model (nn.Module): Model to summarize
        input_size (Tuple): Expected input size (batch, channels, height, width)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print("=" * 70)
    print("Model Summary")
    print("=" * 70)
    print(f"Input size:              {input_size}")
    print(f"Total parameters:        {total_params:,}")
    print(f"Trainable parameters:    {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Trainable ratio:         {100 * trainable_params / total_params:.2f}%")
    print("=" * 70)
