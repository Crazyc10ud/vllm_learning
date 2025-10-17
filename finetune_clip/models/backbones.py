from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModel,
    SiglipImageProcessor,
    SiglipVisionModel,
)


@dataclass
class BackboneBundle:
    """Container with everything required to use a pretrained backbone."""

    model: nn.Module
    image_processor: Callable
    hidden_size: int
    patch_size: int
    num_patches: int


def _inspect_vit_structure(model: nn.Module) -> Tuple[int, int]:
    """Infer the patch size and number of image tokens for ViT style models."""
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "image_size") and hasattr(config, "patch_size"):
        image_size = config.image_size
        patch_size = config.patch_size
    else:
        raise ValueError("Unable to infer image/patch size from model configuration.")

    grid = image_size // patch_size
    num_patches = grid * grid
    return patch_size, num_patches


def load_clip_backbone(pretrained_model_name: str, device: torch.device) -> BackboneBundle:
    vision_model = CLIPVisionModel.from_pretrained(pretrained_model_name)
    image_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name)

    patch_size, num_patches = _inspect_vit_structure(vision_model)
    hidden_size = vision_model.config.hidden_size

    return BackboneBundle(
        model=vision_model.to(device),
        image_processor=image_processor,
        hidden_size=hidden_size,
        patch_size=patch_size,
        num_patches=num_patches,
    )


def load_siglip_backbone(pretrained_model_name: str, device: torch.device) -> BackboneBundle:
    vision_model = SiglipVisionModel.from_pretrained(pretrained_model_name)
    image_processor = SiglipImageProcessor.from_pretrained(pretrained_model_name)

    patch_size, num_patches = _inspect_vit_structure(vision_model)
    hidden_size = vision_model.config.hidden_size

    return BackboneBundle(
        model=vision_model.to(device),
        image_processor=image_processor,
        hidden_size=hidden_size,
        patch_size=patch_size,
        num_patches=num_patches,
    )


BACKBONE_LOADERS: Dict[str, Callable[[str, torch.device], BackboneBundle]] = {
    "clip": load_clip_backbone,
    "siglip": load_siglip_backbone,
}


def create_backbone(
    family: str,
    pretrained_name: str,
    device: torch.device,
) -> BackboneBundle:
    try:
        loader = BACKBONE_LOADERS[family]
    except KeyError as exc:
        raise ValueError(f"Unsupported backbone family: {family}") from exc

    return loader(pretrained_name, device)
