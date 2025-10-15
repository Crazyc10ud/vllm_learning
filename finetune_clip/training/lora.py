from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj")


class LoRALinear(nn.Module):
    """A hand-written LoRA layer that wraps a frozen linear transformation."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, rank: int, alpha: float) -> None:
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.weight = nn.Parameter(weight.clone().detach(), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.clone().detach(), requires_grad=False)
        else:
            self.bias = None

        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora_weight = torch.matmul(self.lora_B, self.lora_A)
        lora_output = F.linear(x, lora_weight, None) * self.scaling
        return base + lora_output

    def merge_weights(self) -> torch.Tensor:
        lora_weight = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        return self.weight + lora_weight


def _should_replace(name: str, target_modules: Tuple[str, ...]) -> bool:
    return any(name.endswith(target) for target in target_modules)


def inject_lora_layers(module: nn.Module, config: LoRAConfig) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and _should_replace(name, config.target_modules):
            new_layer = LoRALinear(child.weight.data, child.bias.data if child.bias is not None else None, config.rank, config.alpha)
            setattr(module, name, new_layer)
        else:
            inject_lora_layers(child, config)


def lora_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield from (child.lora_A, child.lora_B)
