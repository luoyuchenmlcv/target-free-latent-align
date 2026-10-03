# -*- coding: utf-8 -*-
from __future__ import annotations

import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model


_SUFFIXES = [
    'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
    'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
    'attention.query_key_value', 'attention.dense',
    'mlp.dense_h_to_4h', 'mlp.dense_4h_to_h',
    'attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj',
]


def _candidate_proj_names_for_layer(layer_idx: int) -> list[str]:
    prefixes = [
        f'model.layers.{layer_idx}',
        f'model.model.layers.{layer_idx}',
        f'transformer.h.{layer_idx}',
        f'gpt_neox.layers.{layer_idx}',
    ]
    return [f'{p}.{leaf}' for p in prefixes for leaf in _SUFFIXES]


def _parse_layer_spec(model: nn.Module, peft_layers: str) -> list[int]:
    if str(peft_layers).strip().lower() == 'all':
        layer_ids = set()
        for name, _ in model.named_modules():
            for anchor in ('.layers.', '.h.'):
                if anchor in name:
                    rest = name.split(anchor, 1)[1]
                    idx = rest.split('.', 1)[0]
                    if idx.isdigit():
                        layer_ids.add(int(idx))
        if not layer_ids:
            raise ValueError('Could not infer model layers for peft_layers=all')
        return sorted(layer_ids)

    out: list[int] = []
    for part in str(peft_layers).split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            a, b = part.split(':', 1)
            start = int(a) if a != '' else 0
            stop = int(b)
            out.extend(list(range(start, stop)))
        else:
            out.append(int(part))
    out = sorted(set(out))
    if not out:
        raise ValueError(f'Invalid peft_layers={peft_layers!r}')
    return out


def find_existing_target_modules_for_layers(model: nn.Module, layer_indices: list[int]) -> list[str]:
    name_to_module = dict(model.named_modules())
    out: list[str] = []
    for layer_idx in layer_indices:
        existing = [name for name in _candidate_proj_names_for_layer(layer_idx) if name in name_to_module]
        if existing:
            out.extend(existing)
            continue
        tags = [f'.layers.{layer_idx}.', f'.h.{layer_idx}.']
        for name, _ in model.named_modules():
            if any(tag in name for tag in tags) and any(name.endswith(s) for s in [
                'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
                'query_key_value', 'dense_h_to_4h', 'dense_4h_to_h', 'dense', 'c_attn', 'c_proj', 'c_fc',
            ]):
                out.append(name)
    out = sorted(set(out))
    if not out:
        raise ValueError(f'Could not find PEFT target modules for layers {layer_indices}')
    return out


def attach_lora(model: nn.Module, peft_layers: str, r: int, alpha: int, dropout: float):
    layer_indices = _parse_layer_spec(model, peft_layers)
    target_modules = find_existing_target_modules_for_layers(model, layer_indices)
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=target_modules,
        bias='none',
    )
    model = get_peft_model(model, cfg)
    return model, target_modules
