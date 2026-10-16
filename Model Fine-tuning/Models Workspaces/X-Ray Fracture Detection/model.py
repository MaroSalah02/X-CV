"""
DenseNet121 model architecture for binary fracture classification.

The classifier head layout (Linear-BN-ReLU-Dropout-Linear) is kept identical to the
one rebuilt in Pipeline/Inference_Modules/fracture_detection.py so checkpoints stay
loadable by the inference pipeline.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


def build_densenet121(num_classes: int = 2,
                      pretrained: bool = True,
                      dropout_rate: float = 0.5,
                      memory_efficient: bool = False) -> nn.Module:
    """
    Build a DenseNet121 model with replaced classifier head for binary classification.

    The backbone starts frozen; training code decides when to unfreeze it
    (see set_backbone_trainable / unfreeze_layers).

    Args:
        num_classes (int): Number of output classes (default: 2 for binary)
        pretrained (bool): Whether to load ImageNet pretrained weights
        dropout_rate (float): Dropout rate for regularization
        memory_efficient (bool): Use gradient checkpointing inside dense blocks
            (much lower memory at higher resolutions, ~20% slower)

    Returns:
        nn.Module: DenseNet121 model with custom classifier head
    """
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights, memory_efficient=memory_efficient)

    # Freeze all backbone layers initially
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier.in_features  # 1024

    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(512, num_classes)
    )

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> nn.Module:
    """
    Freeze or unfreeze the whole convolutional backbone (all dense blocks,
    transition layers and the final norm5). The classifier always stays trainable.
    """
    for param in model.features.parameters():
        param.requires_grad = trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


def unfreeze_layers(model: nn.Module,
                    num_layers_to_unfreeze: int = 2) -> nn.Module:
    """
    Unfreeze the last N DenseBlocks and the classifier head for fine-tuning.

    Note: this leaves transition layers and norm5 frozen. Prefer
    set_backbone_trainable(model, True) for full fine-tuning, which works better
    for X-rays because the domain is far from ImageNet.

    Args:
        model (nn.Module): DenseNet121 model
        num_layers_to_unfreeze (int): Number of DenseBlocks to unfreeze from the end

    Returns:
        nn.Module: Model with unfrozen layers
    """
    features = model.features

    dense_blocks = []
    for name, module in features.named_children():
        if 'denseblock' in name:
            dense_blocks.append((name, module))

    num_to_unfreeze = min(num_layers_to_unfreeze, len(dense_blocks))

    unfrozen_names = []
    for i in range(num_to_unfreeze):
        block_idx = len(dense_blocks) - 1 - i
        name, module = dense_blocks[block_idx]
        for param in module.parameters():
            param.requires_grad = True
        unfrozen_names.append(name)

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
