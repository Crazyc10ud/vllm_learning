from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def _get_last_encoder_layer(backbone: nn.Module) -> nn.Module:
    encoder = getattr(backbone, "encoder", None)
    if encoder is None:
        raise ValueError("Backbone does not expose an encoder attribute for CAM computation.")
    return encoder.layers[-1]


def compute_vit_gradcam(
    model: nn.Module,
    images: torch.Tensor,
    target_classes: torch.Tensor | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)

    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise ValueError("Model must expose a backbone attribute.")

    target_module = _get_last_encoder_layer(backbone)

    activations: torch.Tensor | None = None
    gradients: torch.Tensor | None = None

    def forward_hook(_module, _input, output):
        nonlocal activations
        activations = output[0] if isinstance(output, tuple) else output

    def backward_hook(_module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    handle_fwd = target_module.register_forward_hook(forward_hook)
    handle_bwd = target_module.register_full_backward_hook(backward_hook)

    outputs = model(images)
    logits = outputs.logits
    if target_classes is None:
        target_classes = torch.argmax(logits, dim=1)
    loss = logits.gather(1, target_classes.unsqueeze(1)).sum()

    model.zero_grad()
    loss.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    assert activations is not None and gradients is not None

    activations = activations[:, 1:, :]
    gradients = gradients[:, 1:, :]

    weights = gradients.mean(dim=2, keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=2))

    batch_size, num_tokens = cam.shape
    grid_size = int(num_tokens ** 0.5)
    cam = cam.reshape(batch_size, grid_size, grid_size)
    cam = cam.detach().cpu().numpy()

    images_np = images.detach().cpu().numpy()
    return cam, images_np


def overlay_cam_on_image(image: np.ndarray, cam: np.ndarray) -> np.ndarray:
    cam_resized = np.clip(cam, 0, None)
    cam_resized -= cam_resized.min()
    if cam_resized.max() > 0:
        cam_resized /= cam_resized.max()
    cam_resized = np.uint8(255 * cam_resized)
    cam_resized = np.stack([cam_resized] * 3, axis=-1)
    cam_resized = np.array(Image.fromarray(cam_resized).resize((image.shape[-1], image.shape[-2])))
    image = image.transpose(1, 2, 0)
    image = (image - image.min()) / (image.max() - image.min() + 1e-6)
    heatmap = plt.cm.jet(cam_resized[..., 0] / 255.0)[..., :3]
    overlay = 0.5 * image + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay


def save_cam_grid(
    output_dir: Path,
    cams: np.ndarray,
    images: np.ndarray,
    prefix: str,
    max_items: int = 8,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    num_items = min(max_items, cams.shape[0])
    fig, axes = plt.subplots(num_items, 2, figsize=(6, 3 * num_items))
    if num_items == 1:
        axes = np.expand_dims(axes, axis=0)
    for idx in range(num_items):
        overlay = overlay_cam_on_image(images[idx], cams[idx])
        axes[idx, 0].imshow(images[idx].transpose(1, 2, 0))
        axes[idx, 0].axis("off")
        axes[idx, 0].set_title("Input")
        axes[idx, 1].imshow(overlay)
        axes[idx, 1].axis("off")
        axes[idx, 1].set_title("CAM")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_cam.png")
    plt.close(fig)
