# model_activations.py
# -*- coding: utf-8 -*-
"""
A thin, reusable "model + activations" layer for:
  1) Loading HF causal LMs + tokenizers (chat-template friendly)
  2) Extracting hidden states at specific layers/positions
  3) Running a sliced forward between layers (used by delta-acts)
  4) Injecting additive steering vectors via forward hooks (test-time steering)

Designed to be architecture-agnostic across common HF decoder-only models:
  - LLaMA/Vicuna style: model.model.layers
  - Qwen2.*: model.layers or model.model.layers (varies by repo)
  - GPT-NeoX style: gpt_neox.layers
  - GPT2 style: transformer.h

This file is intentionally dependency-light: torch + transformers only.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------
# Small reflection helpers
# ----------------------------

def rgetattr(obj: Any, path: str) -> Any:
    """Recursive getattr: rgetattr(x, 'a.b.c') == x.a.b.c"""
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur

def rhasattr(obj: Any, path: str) -> bool:
    """Recursive hasattr for dotted paths."""
    try:
        rgetattr(obj, path)
        return True
    except Exception:
        return False

def _as_dtype(dtype: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
    if dtype is None or dtype == "auto":
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    d = str(dtype).lower()
    if d in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if d in {"fp16", "float16", "half"}:
        return torch.float16
    if d in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")

# ----------------------------
# Layer path resolution
# ----------------------------

_LAYER_PATH_CANDIDATES: Tuple[str, ...] = (
    # LLaMA/Vicuna/Mistral/Qwen (often)
    "model.layers",
    "model.model.layers",
    # GPT-NeoX
    "gpt_neox.layers",
    # GPT2 / OPT-ish
    "transformer.h",
)

def resolve_layers_path(model: nn.Module, preferred: Optional[str] = None) -> str:
    """
    Return the dotted path to the list/ModuleList of transformer blocks.
    """
    if preferred is not None:
        if not rhasattr(model, preferred):
            raise ValueError(f"preferred layers_path='{preferred}' not found on model={type(model)}")
        return preferred

    for p in _LAYER_PATH_CANDIDATES:
        if rhasattr(model, p):
            layers = rgetattr(model, p)
            # Heuristic: must be indexable and have len
            if hasattr(layers, "__len__") and hasattr(layers, "__getitem__"):
                return p

    raise ValueError(
        "Could not resolve transformer layers path. "
        "Tried: " + ", ".join(_LAYER_PATH_CANDIDATES) +
        ". Pass layers_path=... explicitly."
    )

def get_layers(model: nn.Module, layers_path: Optional[str] = None) -> Sequence[nn.Module]:
    path = resolve_layers_path(model, layers_path)
    layers = rgetattr(model, path)
    return layers

# ----------------------------
# Chat templating
# ----------------------------

def format_chat_prompt(
    tokenizer,
    user_text: str,
    system_prompt: Optional[str] = None,
    add_generation_prompt: bool = True,
) -> str:
    """
    Best-effort chat formatting:
      - If tokenizer has apply_chat_template, use it.
      - Otherwise, fall back to a simple instruction format.
    """
    messages = []
    # if system_prompt:
    #     messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})

    if hasattr(tokenizer, "apply_chat_template"):
        try: 
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
                add_special_tokens=False,
            )
        except TypeError:
            # older signature variants
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
        except Exception:
            pass

    # Fallback (Vicuna-ish but generic)
    return f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: {user_text} ASSISTANT:"

# ----------------------------
# Model bundle
# ----------------------------

@dataclass
class ModelBundle:
    model: nn.Module
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype
    layers_path: str
    system_prompt: Optional[str] = None

    def chat(self, text: str, add_generation_prompt: bool = True) -> str:
        return format_chat_prompt(self.tokenizer, text, self.system_prompt, add_generation_prompt)

def load_causal_lm(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    *,
    device_map: Union[str, Dict[str, Any], None] = "auto",
    dtype: Union[str, torch.dtype, None] = "auto",
    trust_remote_code: bool = True,
    padding_side: str = "left",
    truncation_side: str = "left",
    attn_implementation: Optional[str] = "eager",
    low_cpu_mem_usage: bool = True,
    system_prompt: Optional[str] = None,
    layers_path: Optional[str] = None,
    **model_kwargs,
) -> ModelBundle:
    """
    Load an AutoModelForCausalLM + tokenizer with sane defaults for:
      - chat templating
      - left padding (safe for generation with batch)
      - eager attention (torch.func friendliness for vmap/jvp/vjp use cases)

    Note: `attn_implementation` is passed as `_attn_implementation` if supported.
    """
    tok_name = tokenizer_name_or_path or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = truncation_side

    if tokenizer.pad_token is None:
        # For decoder-only LMs, pad_token often unset; use eos_token
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = _as_dtype(dtype)

    # Some transformers versions accept `_attn_implementation`, others accept `attn_implementation`.
    # We try `_attn_implementation` first, but keep it optional.
    if attn_implementation is not None and "_attn_implementation" not in model_kwargs:
        model_kwargs["_attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage,
        **model_kwargs,
    )

    # Model device: for device_map="auto" it's distributed; model.device usually exists but may be meta.
    # We'll pick the first parameter's device as "default".
    try:
        param = next(model.parameters())
        device = param.device
        dtype = param.dtype
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved_layers_path = resolve_layers_path(model, layers_path)

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype = dtype,
        layers_path=resolved_layers_path,
        system_prompt=system_prompt,
    )

# ----------------------------
# Activation capture utilities
# ----------------------------

class LayerOutputCollector:
    """
    Collect the output hidden states of a specific transformer block.
    Works for modules where forward output is either:
      - Tensor [B,S,D], or
      - tuple/list whose first element is Tensor [B,S,D]
    """
    def __init__(self, model: nn.Module, layer_idx: int, layers_path: Optional[str] = None):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.layers_path = resolve_layers_path(model, layers_path)
        layers = rgetattr(model, self.layers_path)
        if not (0 <= self.layer_idx < len(layers)):
            raise ValueError(f"layer_idx out of range: {self.layer_idx} not in [0, {len(layers)-1}]")

        self.hidden: Optional[torch.Tensor] = None
        self._handle = layers[self.layer_idx].register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any):
        if isinstance(output, (tuple, list)):
            self.hidden = output[0]
        else:
            self.hidden = output

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

@torch.no_grad()
def get_activation_for_text(
    bundle: ModelBundle,
    text: str,
    *,
    layer_idx: int,
    pos: int = -1,
    max_len: int = 2048,
    add_generation_prompt: bool = True,
    return_cpu_float: bool = True,
) -> torch.Tensor:
    """
    Single text -> single forward pass (batch_size=1) -> capture layer activation at position `pos`.

    Returns:
      Tensor [D] (float32 by default if return_cpu_float=True)
    """
    prompt = bundle.chat(text, add_generation_prompt=add_generation_prompt)
    # print("prompt", prompt)
    enc = bundle.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(bundle.model.device)
    # print("enc", enc.input_ids.shape)
    # quit()
    collector = LayerOutputCollector(bundle.model, layer_idx, layers_path=bundle.layers_path)
    _ = bundle.model(**enc)
    H = collector.hidden
    collector.remove()
    if H is None:
        raise RuntimeError("Failed to capture hidden state (hook returned None).")

    # H: [1, S, D]
    S = H.shape[1]
    p = pos if pos >= 0 else S + pos
    if not (0 <= p < S):
        raise ValueError(f"pos out of range: seq_len={S}, pos={pos}")
    h = H[0, p, :]
    if return_cpu_float:
        return h.detach().float().cpu()
    return h.detach()

@torch.no_grad()
def forward_hidden_states(
    bundle: ModelBundle,
    prompts: Sequence[str],
    *,
    max_length: int,
    padding: Union[bool, str] = "max_length",
    truncation: bool = True,
    add_generation_prompt: bool = True,
    output_hidden_states: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
    """
    Batched forward for a list of prompts. Returns (encodings, hidden_states tuple).
    hidden_states is the standard HF tuple: (embeddings, layer1, layer2, ..., final)
    """
    texts = [bundle.chat(t, add_generation_prompt=add_generation_prompt) for t in prompts]
    # print("texts", texts)
    enc = bundle.tokenizer(
        texts,
        return_tensors="pt",
        truncation=truncation,
        padding=padding,
        max_length=max_length,
    ).to(bundle.model.device)

    # print("enc", enc.input_ids.shape)
    # quit()
    
    out = bundle.model(**enc, output_hidden_states=output_hidden_states)
    if not output_hidden_states:
        raise ValueError("output_hidden_states must be True to return hidden states.")
    return enc, out.hidden_states  # type: ignore[attr-defined]

# ----------------------------
# Sliced model forward (for delta-acts)
# ----------------------------

class SlicedModel(nn.Module):
    """
    Run only a contiguous block of transformer layers [start_layer, end_layer] on an input hidden state tensor.

    This is adapted from your implementation and relies on temporarily swapping model's layers list.
    It assumes the model supports calling forward with `inputs_embeds=...` and `output_hidden_states=True`.

    IMPORTANT:
      - Some models rely on internal layer_idx values for attention (e.g., flash-attn caching). We best-effort reset.
      - For torch.func vmap/jvp/vjp, using eager attention is strongly recommended.
    """
    def __init__(
        self,
        model: nn.Module,
        start_layer: int,
        end_layer: int,
        *,
        layers_path: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer)

        self.layers_path = resolve_layers_path(model, layers_path)
        self._layers_owner_path, self._layers_attr = self.layers_path.rsplit(".", 1)
        self._layers_owner = rgetattr(model, self._layers_owner_path)
        self._full_layers = rgetattr(model, self.layers_path)

        if not (0 <= self.start_layer <= self.end_layer < len(self._full_layers)):
            raise ValueError(
                f"Invalid slice [{self.start_layer}, {self.end_layer}] for num_layers={len(self._full_layers)}"
            )

        # Some models use config.num_hidden_layers; keep original if present.
        self._has_num_hidden_layers = hasattr(getattr(model, "config", object()), "num_hidden_layers")
        self._orig_num_hidden_layers = getattr(model.config, "num_hidden_layers", None) if self._has_num_hidden_layers else None

    def _reset(self):
        # restore full layer list
        setattr(self._layers_owner, self._layers_attr, self._full_layers)
        # restore config.num_hidden_layers if present
        if self._has_num_hidden_layers and self._orig_num_hidden_layers is not None:
            setattr(self.model.config, "num_hidden_layers", self._orig_num_hidden_layers)
        # best-effort: restore attention layer indices
        layers = rgetattr(self.model, self.layers_path)
        for i, layer in enumerate(layers):
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = i

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # swap in sliced layers
        layers = self._full_layers
        sliced = layers[self.start_layer : self.end_layer + 1]
        setattr(self._layers_owner, self._layers_attr, sliced)

        if self._has_num_hidden_layers:
            # the number used internally varies by model
            setattr(self.model.config, "num_hidden_layers", self.end_layer - self.start_layer)

        # best-effort reindex
        for i, layer in enumerate(rgetattr(self.model, self.layers_path)):
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = i

        # run: inputs_embeds expects [B,S,D]
        out = self.model(inputs_embeds=h, output_hidden_states=True)
        # hidden_states indexing: 0 is embedding output, then each layer
        # We want the output after the last sliced layer.
        hs = out.hidden_states  # type: ignore[attr-defined]
        result = hs[self.end_layer - self.start_layer]  

        # restore model
        self._reset()
        return result

# ----------------------------
# Steering injection via hooks
# ----------------------------

def _apply_addition(
    hidden: torch.Tensor,
    delta: torch.Tensor,
    positions: Union[slice, Sequence[int], torch.Tensor, None],
) -> torch.Tensor:
    """
    hidden: [B,S,D]
    delta:  [D] or [1,1,D] or [S,D] or [1,S,D] or [B,S,D]
    positions: which token positions to add on (None => all)
    """
    if delta.dim() == 1:
        delta = delta.view(1, 1, -1)
    if delta.dim() == 2:
        # [S,D]
        delta = delta.unsqueeze(0)
    # now delta is [*,*,D] with dim==3
    if positions is None:
        return hidden + delta

    if isinstance(positions, slice):
        hidden = hidden.clone()
        hidden[:, positions, :] = hidden[:, positions, :] + delta
        return hidden

    if isinstance(positions, torch.Tensor):
        idx = positions
        if idx.dtype != torch.long:
            idx = idx.long()
        hidden = hidden.clone()
        hidden.index_add_(1, idx.to(hidden.device), delta.expand(hidden.size(0), idx.numel(), -1))
        return hidden

    # sequence of ints
    idx = torch.tensor(list(positions), device=hidden.device, dtype=torch.long)
    hidden = hidden.clone()
    hidden.index_add_(1, idx, delta.expand(hidden.size(0), idx.numel(), -1))
    return hidden

class AddVectorHook:
    """
    Add a vector (or matrix) to the output hidden states of a specific transformer layer.

    Typical use:
        with AddVectorHook(model, layer_idx, v, positions=slice(-3,None)):
            out = model(**enc)
    """
    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        delta: torch.Tensor,
        *,
        positions: Union[slice, Sequence[int], torch.Tensor, None] = None,
        layers_path: Optional[str] = None,
    ):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.delta = delta
        self.positions = positions
        self.layers_path = resolve_layers_path(model, layers_path)
        layers = rgetattr(model, self.layers_path)
        if not (0 <= self.layer_idx < len(layers)):
            raise ValueError(f"layer_idx out of range: {self.layer_idx} not in [0, {len(layers)-1}]")
        self._handle = layers[self.layer_idx].register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any):
        if isinstance(output, (tuple, list)):
            h = output[0]
            h2 = _apply_addition(h, self.delta.to(h.device), self.positions)
            # preserve tuple structure
            if isinstance(output, tuple):
                return (h2,) + tuple(output[1:])
            else:
                out_list = list(output)
                out_list[0] = h2
                return out_list
        else:
            h = output
            return _apply_addition(h, self.delta.to(h.device), self.positions)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()
        return False

# ----------------------------
# Convenience: build X/Y tensors 
# ----------------------------

@torch.no_grad()
def build_xy_from_prompts(
    bundle: ModelBundle,
    prompts: Sequence[str],
    *,
    source_layer_idx: int,
    target_layer_idx: int,
    max_seq_len: int,
    forward_batch_size: int = 1,
    token_positions: Union[slice, Sequence[int], None] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build X (source hidden states) and Y (unsteered target hidden states) 

    Returns:
      X: [N, T, D]  on CPU (float32)
      Y: [N, T, D]  on CPU (float32)
    """
    model = bundle.model
    d_model = getattr(model.config, "hidden_size", None) or getattr(model.config, "hidden_dim", None)
    if d_model is None:
        raise ValueError("Could not infer d_model from model.config.hidden_size/hidden_dim")

    sliced = SlicedModel(model, start_layer=source_layer_idx, end_layer=target_layer_idx, layers_path=bundle.layers_path)

    N = len(prompts)
    X = torch.zeros((N, max_seq_len, d_model), device="cuda")
    Y = torch.zeros((N, max_seq_len, d_model), device="cuda")

    for i in range(0, N, forward_batch_size):
        batch = prompts[i : i + forward_batch_size]
        # _, hs = forward_hidden_states(
        #     bundle,
        #     batch,
        #     max_length=max_seq_len,
        #     padding="max_length",
        #     truncation=True,
        #     add_generation_prompt=True,
        #     output_hidden_states=True,
        # )
        # print("batch", batch)
        model_inputs = bundle.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length", max_length=27).to(bundle.model.device)
        # print("model_inputs", model_inputs["input_ids"])
        hs = bundle.model(model_inputs["input_ids"], output_hidden_states=True).hidden_states
        h_source = hs[source_layer_idx]  # [B,T,D]
        y_target = sliced(h_source)      # [B,T,D]
        X[i : i + len(batch)] = h_source.detach()
        Y[i : i + len(batch)] = y_target.detach()
    return X, Y
