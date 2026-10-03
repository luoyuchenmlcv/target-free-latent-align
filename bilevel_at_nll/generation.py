# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence

import torch

from .hooks import AddVectorMaskHook, BatchPrefillOnlyAddVectorHook, PrefillOnlyAddVectorHook, infer_input_device
from .prompting import build_prompt_content_mask_batch


def make_source_hook(bundle, queries: Sequence[str], full_lens: Sequence[int], seq_len: int, layer_idx: int, v: torch.Tensor, steer_positions: str, steer_prompt_content_only: bool, max_prompt_tokens: int):
    mode = str(steer_positions).lower()
    if steer_prompt_content_only:
        mode = "content"
    if mode == "content":
        mask = build_prompt_content_mask_batch(bundle, queries, full_lens, seq_len, max_prompt_tokens)
        return AddVectorMaskHook(bundle.model, layer_idx, v, position_mask=mask, layers_path=None)
    if mode == "all":
        return PrefillOnlyAddVectorHook(bundle.model, layer_idx, v, positions=None, layers_path=None)
    if mode == "last":
        return PrefillOnlyAddVectorHook(bundle.model, layer_idx, v, positions=slice(-1, None), layers_path=None)
    raise ValueError(f"Unsupported steer_positions={steer_positions!r}; expected one of: last, all, content")


@torch.no_grad()
def generate_clean(bundle, query: str, max_prompt_tokens: int, max_new_tokens: int, do_sample: bool, temperature: float) -> str:
    tok = bundle.tokenizer
    dev = infer_input_device(bundle.model)
    prompt = bundle.chat(query, add_generation_prompt=True)
    enc = tok(
        prompt,
        return_tensors="pt",
        truncation=(max_prompt_tokens > 0),
        max_length=(max_prompt_tokens if max_prompt_tokens > 0 else None),
        add_special_tokens=False,
    ).to(dev)
    prefix_len = int(enc["input_ids"].shape[1])
    kwargs = dict(**enc, max_new_tokens=max_new_tokens, pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id), do_sample=do_sample)
    if do_sample:
        kwargs["temperature"] = temperature
    out_ids = bundle.model.generate(**kwargs)
    return tok.decode(out_ids[0, prefix_len:], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_with_prefill_steer(bundle, query: str, v: torch.Tensor, source_layer_idx: int, steer_positions: str, steer_prompt_content_only: bool, max_prompt_tokens: int, max_new_tokens: int, do_sample: bool, temperature: float) -> str:
    tok = bundle.tokenizer
    dev = infer_input_device(bundle.model)
    prompt = bundle.chat(query, add_generation_prompt=True)
    enc = tok(
        prompt,
        return_tensors="pt",
        truncation=(max_prompt_tokens > 0),
        max_length=(max_prompt_tokens if max_prompt_tokens > 0 else None),
        add_special_tokens=False,
    ).to(dev)
    prefix_len = int(enc["input_ids"].shape[1])
    kwargs = dict(**enc, max_new_tokens=max_new_tokens, pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id), do_sample=do_sample)
    if do_sample:
        kwargs["temperature"] = temperature
    full_lens = [prefix_len]
    seq_len = prefix_len
    source_hook = make_source_hook(bundle, [query], full_lens, seq_len, source_layer_idx, v, steer_positions, steer_prompt_content_only, max_prompt_tokens)
    with source_hook:
        out_ids = bundle.model.generate(**kwargs)
    return tok.decode(out_ids[0, prefix_len:], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_with_batch_prefill_steer(bundle, query: str, vs: torch.Tensor, source_layer_idx: int, steer_positions: str, steer_prompt_content_only: bool, max_prompt_tokens: int, max_new_tokens: int, do_sample: bool, temperature: float) -> list[str]:
    tok = bundle.tokenizer
    dev = infer_input_device(bundle.model)
    k = int(vs.shape[1])
    prompts = [query] * k
    prompt_texts = [bundle.chat(query, add_generation_prompt=True) for _ in range(k)]
    enc = tok(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=(max_prompt_tokens > 0),
        max_length=(max_prompt_tokens if max_prompt_tokens > 0 else None),
        add_special_tokens=False,
    )
    prefix_len = int(enc["input_ids"].shape[1])
    full_lens = enc["attention_mask"].sum(dim=1).tolist()
    enc = {kk: vv.to(dev) for kk, vv in enc.items()}
    mode = str(steer_positions).lower()
    if steer_prompt_content_only:
        mode = "content"
    deltas = vs.T.to(dev)
    if mode == "content":
        mask = build_prompt_content_mask_batch(bundle, prompts, full_lens, prefix_len, max_prompt_tokens)
        source_hook = BatchPrefillOnlyAddVectorHook(bundle.model, source_layer_idx, deltas, position_mask=mask, layers_path=None)
    elif mode == "all":
        source_hook = BatchPrefillOnlyAddVectorHook(bundle.model, source_layer_idx, deltas, positions=None, layers_path=None)
    elif mode == "last":
        source_hook = BatchPrefillOnlyAddVectorHook(bundle.model, source_layer_idx, deltas, positions=slice(-1, None), layers_path=None)
    else:
        raise ValueError(f"Unsupported steer_positions={steer_positions}")

    kwargs = dict(**enc, max_new_tokens=max_new_tokens, pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id), do_sample=do_sample)
    if do_sample:
        kwargs["temperature"] = temperature
    with source_hook:
        out_ids = bundle.model.generate(**kwargs)
    texts = tok.batch_decode(out_ids[:, prefix_len:], skip_special_tokens=True)
    return [t.strip() for t in texts]
