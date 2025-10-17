from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def plot_training_history(history: Dict[str, list], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Training and Validation Loss")

    axes[1].plot(epochs, history["val_accuracy"], label="Val Accuracy", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(output_dir / "training_metrics.png")
    plt.close(fig)
