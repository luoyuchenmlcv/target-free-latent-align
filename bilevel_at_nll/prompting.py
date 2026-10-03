# -*- coding: utf-8 -*-
from __future__ import annotations

import torch

from .hooks import infer_input_device


def _find_user_text_span_tokens(tokenizer, prompt_text: str, user_text: str) -> tuple[int, int]:
    idx = prompt_text.find(user_text)
    if idx < 0:
        raise ValueError("Could not find raw user text inside chat-formatted prompt")
    prefix = prompt_text[:idx]
    upto = prompt_text[: idx + len(user_text)]
    prefix_ids = tokenizer(prefix, add_special_tokens=False, truncation=False)["input_ids"]
    upto_ids = tokenizer(upto, add_special_tokens=False, truncation=False)["input_ids"]
    return len(prefix_ids), len(upto_ids)


def build_prompt_batch(bundle, queries, max_prompt_tokens: int):
    tok = bundle.tokenizer
    texts = [bundle.chat(q, add_generation_prompt=True) for q in queries]
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=(max_prompt_tokens > 0),
        max_length=(max_prompt_tokens if max_prompt_tokens > 0 else None),
        add_special_tokens=False,
    )
    full_lens = enc["attention_mask"].sum(dim=1).tolist()
    seq_len = int(enc["input_ids"].shape[1])
    dev = infer_input_device(bundle.model)
    enc = {k: v.to(dev) for k, v in enc.items()}
    return enc, [int(x) for x in full_lens], seq_len


def build_fixed_prompt_batch(bundle, queries, seq_len: int):
    tok = bundle.tokenizer
    texts = [bundle.chat(q, add_generation_prompt=True) for q in queries]
    enc = tok(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        add_special_tokens=False,
    )
    full_lens = enc["attention_mask"].sum(dim=1).tolist()
    dev = infer_input_device(bundle.model)
    enc = {k: v.to(dev) for k, v in enc.items()}
    return enc, [int(x) for x in full_lens], int(enc["input_ids"].shape[1])


def build_prompt_completion_batch(bundle, prompt_generation_rows, max_prompt_tokens: int):
    tok = bundle.tokenizer
    prompt_texts = [bundle.chat(row["prompt"], add_generation_prompt=True) for row in prompt_generation_rows]
    completion_texts = [str(row["generation"]) for row in prompt_generation_rows]
    prompt_ids = [tok(text, add_special_tokens=False, truncation=False)["input_ids"] for text in prompt_texts]
    completion_ids = [tok(text, add_special_tokens=False, truncation=False)["input_ids"] for text in completion_texts]

    merged_ids = [p + c for p, c in zip(prompt_ids, completion_ids)]
    if max_prompt_tokens > 0:
        merged_ids = [ids[-max_prompt_tokens:] for ids in merged_ids]

    prompt_token_counts: list[int] = []
    for p_ids, merged in zip(prompt_ids, merged_ids):
        prompt_token_counts.append(min(len(p_ids), len(merged)))

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id")

    seq_len = max(len(ids) for ids in merged_ids) if merged_ids else 0
    bsz = len(merged_ids)
    input_ids = torch.full((bsz, seq_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, seq_len), dtype=torch.long)
    completion_mask = torch.zeros((bsz, seq_len), dtype=torch.bool)

    for i, (ids, prompt_len) in enumerate(zip(merged_ids, prompt_token_counts)):
        ids_t = torch.tensor(ids, dtype=torch.long)
        left_pad = seq_len - len(ids)
        input_ids[i, left_pad:] = ids_t
        attention_mask[i, left_pad:] = 1
        comp_start = left_pad + prompt_len
        if comp_start < seq_len:
            completion_mask[i, comp_start:] = True

    dev = infer_input_device(bundle.model)
    enc = {
        "input_ids": input_ids.to(dev),
        "attention_mask": attention_mask.to(dev),
    }
    return enc, completion_mask.to(dev), [int(x) for x in prompt_token_counts], seq_len


def build_prompt_content_mask_batch(bundle, queries, full_lens, seq_len: int, max_prompt_tokens: int):
    tok = bundle.tokenizer
    mask = torch.zeros((len(queries), seq_len), dtype=torch.bool)
    for i, (q, fl) in enumerate(zip(queries, full_lens)):
        prompt_text = bundle.chat(q, add_generation_prompt=True)
        start_raw, end_raw = _find_user_text_span_tokens(tok, prompt_text, q)
        if max_prompt_tokens > 0:
            prompt_ids_no_trunc = tok(
                prompt_text,
                return_tensors=None,
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
            trim_left = max(0, len(prompt_ids_no_trunc) - max_prompt_tokens)
            start = max(0, start_raw - trim_left)
            end = max(0, end_raw - trim_left)
        else:
            start, end = start_raw, end_raw
        left_pad = seq_len - fl
        start = min(start, fl)
        end = min(end, fl)
        if end > start:
            mask[i, left_pad + start : left_pad + end] = True
    return mask
