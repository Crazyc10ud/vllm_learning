from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn


@dataclass
class ForwardOutputs:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    attentions: Optional[List[torch.Tensor]]


class VisionClassifier(nn.Module):
    """A light-weight classification head on top of a frozen or trainable backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> ForwardOutputs:
        outputs = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=True,
            output_attentions=True,
        )
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        hidden_states = outputs.hidden_states[-1]
        attentions = list(outputs.attentions) if outputs.attentions is not None else None
        return ForwardOutputs(logits=logits, hidden_states=hidden_states, attentions=attentions)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield param

    def forward_with_features(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.forward(pixel_values)
        return {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
