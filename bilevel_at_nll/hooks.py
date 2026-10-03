# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

_LAYER_PATH_CANDIDATES = (
    "model.layers",
    "model.model.layers",
    "gpt_neox.layers",
    "transformer.h",
    "base_model.model.layers",
    "base_model.model.model.layers",
    "base_model.gpt_neox.layers",
    "base_model.transformer.h",
)


def rgetattr(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def rhasattr(obj: Any, path: str) -> bool:
    try:
        rgetattr(obj, path)
        return True
    except Exception:
        return False


def resolve_layers_path_safe(model: nn.Module, preferred: Optional[str] = None) -> str:
    if preferred is not None and rhasattr(model, preferred):
        return preferred
    for p in _LAYER_PATH_CANDIDATES:
        if rhasattr(model, p):
            layers = rgetattr(model, p)
            if hasattr(layers, "__len__") and hasattr(layers, "__getitem__"):
                return p
    raise ValueError(
        f"Could not resolve transformer layers path for model={type(model)}. "
        f"Tried: {', '.join(_LAYER_PATH_CANDIDATES)}"
    )


def get_layers_safe(model: nn.Module, preferred: Optional[str] = None):
    return rgetattr(model, resolve_layers_path_safe(model, preferred))


def infer_input_device(model: nn.Module) -> torch.device:
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HiddenCaptureHook:
    def __init__(self, model: nn.Module, layer_idx: int, *, layers_path: Optional[str] = None):
        self.hidden: Optional[torch.Tensor] = None
        layers = get_layers_safe(model, layers_path)
        self._handle = layers[int(layer_idx)].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        if torch.is_tensor(h) and h.dim() == 3 and h.shape[1] > 1:
            self.hidden = h

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()
        return False


class AddVectorMaskHook:
    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        delta: torch.Tensor,
        *,
        position_mask: torch.Tensor,
        layers_path: Optional[str] = None,
    ):
        self.delta = delta
        self.position_mask = position_mask
        layers = get_layers_safe(model, layers_path)
        self._handle = layers[int(layer_idx)].register_forward_hook(self._hook)

    def _apply(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() != 3 or hidden.shape[1] == 1:
            return hidden
        mask = self.position_mask.to(hidden.device)
        delta = self.delta.to(hidden.device, dtype=hidden.dtype).view(1, 1, -1)
        return hidden + mask.unsqueeze(-1).to(hidden.dtype) * delta

    def _hook(self, module, inputs, output):
        if isinstance(output, (tuple, list)):
            h = output[0]
            h2 = self._apply(h)
            if isinstance(output, tuple):
                return (h2,) + tuple(output[1:])
            out = list(output)
            out[0] = h2
            return out
        return self._apply(output)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()
        return False


class PrefillOnlyAddVectorHook:
    def __init__(self, model: nn.Module, layer_idx: int, delta: torch.Tensor, *, positions=None, layers_path: Optional[str] = None):
        self.delta = delta
        self.positions = positions
        layers = get_layers_safe(model, layers_path)
        self._handle = layers[int(layer_idx)].register_forward_hook(self._hook)

    def _apply(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() != 3 or hidden.shape[1] == 1:
            return hidden
        delta = self.delta.to(hidden.device, dtype=hidden.dtype)
        if self.positions is None:
            return hidden + delta.view(1, 1, -1)
        out = hidden.clone()
        if isinstance(self.positions, slice):
            out[:, self.positions, :] = out[:, self.positions, :] + delta.view(1, 1, -1)
            return out
        idx = torch.tensor(list(self.positions), device=hidden.device, dtype=torch.long)
        out.index_add_(1, idx, delta.view(1, 1, -1).expand(hidden.size(0), idx.numel(), -1))
        return out

    def _hook(self, module, inputs, output):
        if isinstance(output, (tuple, list)):
            h = output[0]
            h2 = self._apply(h)
            if isinstance(output, tuple):
                return (h2,) + tuple(output[1:])
            out = list(output)
            out[0] = h2
            return out
        return self._apply(output)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()
        return False


class BatchPrefillOnlyAddVectorHook:
    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        deltas: torch.Tensor,
        *,
        positions=None,
        position_mask: Optional[torch.Tensor] = None,
        layers_path: Optional[str] = None,
    ):
        self.deltas = deltas
        self.positions = positions
        self.position_mask = position_mask
        layers = get_layers_safe(model, layers_path)
        self._handle = layers[int(layer_idx)].register_forward_hook(self._hook)

    def _apply(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() != 3 or hidden.shape[1] == 1:
            return hidden
        bsz, _, d = hidden.shape
        deltas = self.deltas.to(hidden.device, dtype=hidden.dtype)
        if deltas.shape != (bsz, d):
            raise RuntimeError(f"Batch steering mismatch: hidden={tuple(hidden.shape)} deltas={tuple(deltas.shape)}")
        if self.position_mask is not None:
            mask = self.position_mask.to(hidden.device, dtype=hidden.dtype)
            return hidden + mask.unsqueeze(-1) * deltas.unsqueeze(1)
        if self.positions is None:
            return hidden + deltas.unsqueeze(1)
        out = hidden.clone()
        if isinstance(self.positions, slice):
            out[:, self.positions, :] = out[:, self.positions, :] + deltas.unsqueeze(1)
            return out
        idx = torch.tensor(list(self.positions), device=hidden.device, dtype=torch.long)
        out[:, idx, :] = out[:, idx, :] + deltas.unsqueeze(1)
        return out

    def _hook(self, module, inputs, output):
        if isinstance(output, (tuple, list)):
            h = output[0]
            h2 = self._apply(h)
            if isinstance(output, tuple):
                return (h2,) + tuple(output[1:])
            out = list(output)
            out[0] = h2
            return out
        return self._apply(output)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()
        return False
