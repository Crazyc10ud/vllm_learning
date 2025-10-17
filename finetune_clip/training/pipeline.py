from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from ..datasets.cifar import build_cifar100_dataloaders
from ..datasets.isic import build_isic_dataloaders
from ..datasets.transforms import ProcessorTransform
from ..models.backbones import create_backbone
from ..models.classifier import VisionClassifier
from ..visualization.cam import compute_vit_gradcam, save_cam_grid
from ..visualization.plots import plot_training_history
from .engine import evaluate, train_model
from .lora import LoRAConfig, inject_lora_layers, lora_parameters

DATASET_BUILDERS = {
    "cifar100": build_cifar100_dataloaders,
    "isic2018": build_isic_dataloaders,
}


@dataclass
class ExperimentConfig:
    model_family: str
    pretrained_name: str
    dataset: str
    finetune_mode: str
    output_dir: Path
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lora_rank: int = 8
    lora_alpha: float = 16.0


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_dataloaders(self, image_processor: ProcessorTransform) -> tuple:
        builder = DATASET_BUILDERS.get(self.config.dataset)
        if builder is None:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")
        return builder(
            image_processor=image_processor,
            batch_size=self.config.batch_size,
        )

    def run(self) -> Dict[str, float]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        backbone_bundle = create_backbone(
            self.config.model_family, self.config.pretrained_name, self.device
        )
        processor_transform = ProcessorTransform(backbone_bundle.image_processor)
        train_loader, val_loader, test_loader, num_classes = self._prepare_dataloaders(processor_transform)

        model = VisionClassifier(backbone_bundle.model, backbone_bundle.hidden_size, num_classes).to(self.device)

        if self.config.finetune_mode == "linear-probing":
            model.freeze_backbone()
        elif self.config.finetune_mode == "lora":
            model.freeze_backbone()
            lora_cfg = LoRAConfig(rank=self.config.lora_rank, alpha=self.config.lora_alpha)
            inject_lora_layers(model.backbone, lora_cfg)
            for param in lora_parameters(model.backbone):
                param.requires_grad = True
        else:
            model.unfreeze_backbone()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        sample_batch = next(iter(train_loader))
        sample_images, _ = sample_batch
        cam_dir = self.config.output_dir / "cam"
        cam_dir.mkdir(parents=True, exist_ok=True)

        cams_before, images_before = compute_vit_gradcam(model, sample_images[:8])
        save_cam_grid(cam_dir, cams_before, images_before, prefix="before")

        history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            self.device,
            self.config.epochs,
        )

        history_dict = history.to_dict()
        plot_training_history(history_dict, self.config.output_dir)

        torch.save(model.state_dict(), self.config.output_dir / "model.pt")

        test_accuracy = evaluate(model, test_loader, self.device)

        cams_after, images_after = compute_vit_gradcam(model, sample_images[:8])
        save_cam_grid(cam_dir, cams_after, images_after, prefix="after")

        metrics = {"test_accuracy": test_accuracy}
        with open(self.config.output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({**history_dict, **metrics}, f, indent=2)
        return metrics
