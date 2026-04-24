"""
Data preprocessing module for X-ray fracture detection.
Handles loading, augmentation, and batch preparation for both datasets.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_transforms(phase: str) -> transforms.Compose:
    """
    Get image transforms for the specified phase.

    Args:
        phase (str): One of 'train', 'val', or 'test'

    Returns:
        transforms.Compose: Composition of transforms for the phase

    Raises:
        ValueError: If phase is not 'train', 'val', or 'test'
    """
    if phase not in ['train', 'val', 'test']:
        raise ValueError(f"phase must be 'train', 'val', or 'test', got {phase}")

    if phase == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_dataset_A(root_dir: str, split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                  phase: str = 'train', seed: int = 42) -> Subset:
    """
    Load Dataset A (Bone_Fracture_Binary_Classification) with manual train/val/test split.

    This dataset has no predefined splits, only fracture/ and normal/ folders.

    Args:
        root_dir (str): Path to Bone_Fracture_Binary_Classification directory (containing fracture/ and normal/)
        split_ratios (Tuple[float, float, float]): (train_ratio, val_ratio, test_ratio)
        phase (str): One of 'train', 'val', or 'test'
        seed (int): Random seed for reproducibility

    Returns:
        Subset: Subset of dataset for the specified phase

    Raises:
        ValueError: If phase is invalid or split_ratios don't sum to 1.0
        FileNotFoundError: If directory structure is incorrect
    """
    if phase not in ['train', 'val', 'test']:
        raise ValueError(f"phase must be 'train', 'val', or 'test', got {phase}")

    if not (0.99 <= sum(split_ratios) <= 1.01):
        raise ValueError(f"split_ratios must sum to 1.0, got {sum(split_ratios)}")

    fracture_dir = os.path.join(root_dir, 'fracture')
    normal_dir = os.path.join(root_dir, 'normal')

    if not os.path.exists(fracture_dir) or not os.path.exists(normal_dir):
        raise FileNotFoundError(f"Dataset structure incorrect. Expected fracture/ and normal/ directories in {root_dir}")

    # Load all image paths and labels
    images = []
    labels = []

    # Load fractured images (label 1)
    for img_name in os.listdir(fracture_dir):
        img_path = os.path.join(fracture_dir, img_name)
        if os.path.isfile(img_path) and img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            images.append(img_path)
            labels.append(1)

    # Load normal images (label 0)
    for img_name in os.listdir(normal_dir):
        img_path = os.path.join(normal_dir, img_name)
        if os.path.isfile(img_path) and img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            images.append(img_path)
            labels.append(0)

    if len(images) == 0:
        raise FileNotFoundError(f"No images found in {fracture_dir} or {normal_dir}")

    # Split into train, val, test
    train_ratio, val_ratio, test_ratio = split_ratios

    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        np.arange(len(images)),
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=np.array(labels)
    )

    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_size),
        random_state=seed,
        stratify=np.array(labels)[temp_indices]
    )

    # Create a simple dataset class for Dataset A
    class SimpleImageDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label

    # Create full dataset with all images
    full_dataset = SimpleImageDataset(
        images,
        labels,
        transform=get_transforms(phase)
    )

    # Return subset for the requested phase
    if phase == 'train':
        indices = train_indices
    elif phase == 'val':
        indices = val_indices
    else:  # test
        indices = test_indices

    return Subset(full_dataset, indices)


def get_dataset_B(root_dir: str, phase: str) -> ImageFolder:
    """
    Load Dataset B (Bone fracture dataset) with predefined train/val/test splits.

    This dataset has predefined splits: train/, val/, test/ folders,
    each containing fractured/ and not fractured/ subfolders.

    Args:
        root_dir (str): Path to Bone fracture dataset directory (containing train/, val/, test/)
        phase (str): One of 'train', 'val', or 'test'

    Returns:
        ImageFolder: Dataset with correct class mapping

    Raises:
        ValueError: If phase is invalid
        FileNotFoundError: If directory structure is incorrect
    """
    if phase not in ['train', 'val', 'test']:
        raise ValueError(f"phase must be 'train', 'val', or 'test', got {phase}")

    phase_dir = os.path.join(root_dir, phase)
    if not os.path.exists(phase_dir):
        raise FileNotFoundError(f"Phase directory not found: {phase_dir}")

    # Create dataset with ImageFolder
    dataset = ImageFolder(
        root=phase_dir,
        transform=get_transforms(phase)
    )

    # Ensure correct class mapping: 'not fractured' -> 0, 'fractured' -> 1
    original_classes = dataset.classes
    if len(original_classes) != 2:
        raise ValueError(f"Expected 2 classes, found {len(original_classes)}: {original_classes}")

    # Remap to ensure correct labels regardless of alphabetical order
    dataset.class_to_idx = {'fractured': 1, 'not fractured': 0}
    dataset.classes = ['not fractured', 'fractured']

    # Remap existing indices
    for idx, (path, old_class) in enumerate(dataset.imgs):
        folder_name = os.path.basename(os.path.dirname(path))
        if folder_name == 'fractured':
            dataset.imgs[idx] = (path, 1)
            dataset.samples[idx] = (path, 1)
        elif folder_name == 'not fractured':
            dataset.imgs[idx] = (path, 0)
            dataset.samples[idx] = (path, 0)

    return dataset


def get_class_weights(dataset: torch.utils.data.Dataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for handling class imbalance.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to compute weights for

    Returns:
        torch.Tensor: Class weights as float tensor of shape (num_classes,)
    """
    # Count samples per class
    class_counts = {}

    for _, label in tqdm(dataset, desc="Computing class weights"):
        label = int(label)
        class_counts[label] = class_counts.get(label, 0) + 1

    # Compute inverse-frequency weights
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    weights = torch.zeros(num_classes)

    for class_idx, count in class_counts.items():
        weights[class_idx] = total_samples / (num_classes * count)

    return weights


def get_combined_dataloader(root_A: str, root_B: str,
                           phase: str,
                           batch_size: int = 32,
                           num_workers: int = 4) -> DataLoader:
    """
    Create combined dataloader from Dataset A and Dataset B with proper sampling.

    Args:
        root_A (str): Path to Bone_Fracture_Binary_Classification directory (contains fracture/ and normal/)
        root_B (str): Path to Bone fracture dataset directory (contains train/, val/, test/)
        phase (str): One of 'train', 'val', or 'test'
        batch_size (int): Batch size for dataloader
        num_workers (int): Number of worker processes

    Returns:
        DataLoader: Combined dataloader with appropriate sampling strategy
    """
    # Load both datasets
    dataset_a = get_dataset_A(root_A, phase=phase)
    dataset_b = get_dataset_B(root_B, phase=phase)

    # Combine datasets
    combined_dataset = ConcatDataset([dataset_a, dataset_b])

    # For training, use WeightedRandomSampler to handle class imbalance
    if phase == 'train':
        class_weights = get_class_weights(combined_dataset)

        # Get labels for all samples
        sample_weights = []
        for idx in range(len(combined_dataset)):
            _, label = combined_dataset[idx]
            sample_weights.append(class_weights[int(label)])

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        dataloader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        dataloader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return dataloader


def show_batch(dataloader: DataLoader, n: int = 8) -> None:
    """
    Visualize a batch of images with class labels.

    Args:
        dataloader (DataLoader): DataLoader to visualize from
        n (int): Number of images to display
    """
    # Get one batch
    images, labels = next(iter(dataloader))

    # Denormalize images (reverse normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    # Create grid
    n = min(n, len(images))
    rows = int(np.sqrt(n))
    cols = (n + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1 or cols == 1:
        axes = [[ax] for ax in axes] if rows == 1 else [axes]

    axes_flat = [ax for row in axes for ax in row]

    class_labels = ['Not Fractured', 'Fractured']

    for idx in range(n):
        ax = axes_flat[idx]

        # Get image and convert to numpy
        img = images[idx].permute(1, 2, 0).numpy()
        label = class_labels[int(labels[idx])]

        ax.imshow(img)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.axis('off')

    # Hide empty subplots
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()
    plt.show()
