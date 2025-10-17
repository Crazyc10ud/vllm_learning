from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets


def build_cifar100_dataloaders(
    image_processor: Callable,
    batch_size: int,
    num_workers: int = 4,
    val_split: float = 0.1,
    root: str = "data",
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    transform = image_processor
    full_train = datasets.CIFAR100(root=root, train=True, transform=transform, download=True)
    test_set = datasets.CIFAR100(root=root, train=False, transform=transform, download=True)

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    num_classes = 100
    return train_loader, val_loader, test_loader, num_classes
