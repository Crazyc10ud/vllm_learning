from __future__ import annotations

from typing import Any, Callable

import torch
from PIL import Image


class ProcessorTransform:
    """Wrap a huggingface image processor so that it behaves like a torchvision transform."""

    def __init__(self, processor: Callable, normalize: bool = True) -> None:
        self.processor = processor
        self.normalize = normalize

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"][0]
        if not self.normalize:
            # Some processors (e.g. SigLIP) already normalize. Option is for completeness.
            pixel_values = pixel_values.clamp(0.0, 1.0)
        return pixel_values
