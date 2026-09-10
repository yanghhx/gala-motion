from __future__ import annotations

import torch
from torch import nn


class ModelEMA:
    """Exponential moving average of floating-point parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self._backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        if not self.shadow:
            return
        for name, parameter in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def apply(self, model: nn.Module):
        self._backup = {}
        for name, parameter in model.named_parameters():
            if name not in self.shadow:
                continue
            self._backup[name] = parameter.detach().clone()
            parameter.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, parameter in model.named_parameters():
            if name in self._backup:
                parameter.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = float(state.get("decay", self.decay))
        shadow = state.get("shadow") or {}
        for name, tensor in shadow.items():
            if name in self.shadow:
                self.shadow[name] = tensor.detach().clone()
