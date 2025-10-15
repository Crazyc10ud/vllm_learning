from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float


@dataclass
class TrainingHistory:
    records: List[EpochMetrics] = field(default_factory=list)

    def append(self, metrics: EpochMetrics) -> None:
        self.records.append(metrics)

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "epoch": [m.epoch for m in self.records],
            "train_loss": [m.train_loss for m in self.records],
            "val_loss": [m.val_loss for m in self.records],
            "val_accuracy": [m.val_accuracy for m in self.records],
        }


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool = True,
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    for batch in tqdm(dataloader, leave=False):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)

        if train:
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    return total_loss / total_samples


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs).logits
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> TrainingHistory:
    history = TrainingHistory()
    for epoch in range(1, num_epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer=None, device=device, train=False)
        val_acc = evaluate(model, val_loader, device)
        if scheduler is not None:
            scheduler.step()
        history.append(EpochMetrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_accuracy=val_acc))
    return history
