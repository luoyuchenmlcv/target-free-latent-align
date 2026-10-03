# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence

import torch

from nuclear import SlicedModel, DeltaActivations, Nuclear

from .config import parse_positions
from .hooks import infer_input_device, resolve_layers_path_safe


@torch.no_grad()
def collect_xy(bundle, examples: Sequence[str], source_layer_idx: int, target_layer_idx: int, max_seq_len: int, forward_batch_size: int):
    model = bundle.model
    tok = bundle.tokenizer
    sliced_model = SlicedModel(model, start_layer=source_layer_idx, end_layer=target_layer_idx, layers_name=resolve_layers_path_safe(model))
    d_model = model.config.hidden_size
    X = torch.zeros(len(examples), max_seq_len, d_model, dtype=getattr(bundle, "dtype", torch.float32), device=bundle.device)
    Y = torch.zeros(len(examples), max_seq_len, d_model, dtype=getattr(bundle, "dtype", torch.float32), device=bundle.device)

    for t in range(0, len(examples)):
        batch = examples[t : t + forward_batch_size]
        model_inputs = tok(batch, return_tensors="pt", truncation=True, padding="max_length", max_length=max_seq_len).to(infer_input_device(model))
        hidden_states = model(
            model_inputs["input_ids"],
            attention_mask=model_inputs.get("attention_mask"),
            output_hidden_states=True,
        ).hidden_states
        h_source = hidden_states[source_layer_idx]
        unsteered_target = sliced_model(h_source)
        X[t : t + len(batch)] = h_source
        Y[t : t + len(batch)] = unsteered_target
    return X, Y


def _last_or_default(xs, default=0.0):
    if xs is None:
        return float(default)
    if len(xs) == 0:
        return float(default)
    return float(xs[-1])


def _run_inner_impl(bundle, harmful_queries: Sequence[str], args):
    examples = [bundle.chat(q, add_generation_prompt=True) for q in harmful_queries[: args.inner_num_samples]]
    X, Y = collect_xy(
        bundle,
        examples,
        source_layer_idx=args.source_layer_idx,
        target_layer_idx=args.target_layer_idx,
        max_seq_len=args.inner_max_seq_len,
        forward_batch_size=args.inner_forward_batch_size,
    )

    sliced_model = SlicedModel(
        bundle.model,
        start_layer=args.source_layer_idx,
        end_layer=args.target_layer_idx,
        layers_name=resolve_layers_path_safe(bundle.model),
    )
    delta_acts_single = DeltaActivations(sliced_model, target_position_indices=parse_positions(args.inner_token_idxs))

    input_scale = float(args.inner_input_scale)

    nuclear = Nuclear(num_factors=args.inner_num_factors)
    inner_method = str(getattr(args, "inner_method", "nuclear")).lower()

 
    U, V = nuclear.fit_nuclear(
        delta_acts_single,
        X,
        Y,
        batch_size=args.inner_backward_batch_size,
        factor_batch_size=args.inner_factor_batch_size,
        init=args.inner_init,
        d_proj=args.inner_dim_output_projection,
        input_scale=input_scale,
        max_iters=args.inner_num_iters,
        lr=float(getattr(args, "inner_lr", 1e-2)),
    )

    # The returned indices are sorted in descending score order.
    rank_scores, rank_indices = nuclear.rank(
        delta_acts_single,
        X,
        Y,
        batch_size=args.inner_backward_batch_size,
        factor_batch_size=args.inner_factor_batch_size,
        target_vec=None,
    )

    return {
        "U": U.detach().cpu(),
        "V": V.detach().cpu(),
        "V_scaled": V.detach().cpu() * input_scale,
        "rank_scores": rank_scores.detach().cpu(),
        "rank_indices": rank_indices.detach().cpu(),
        "mean_shift_avg_norm": _last_or_default(getattr(nuclear, 'mean_shift_avg_norm_values', [])),
        "mean_shift_dir_sim": _last_or_default(getattr(nuclear, 'mean_shift_dir_sim_values', [])),
        "v_self_sim": _last_or_default(getattr(nuclear, 'v_self_sim_values', [])),
        "intra_prompt_consistency": _last_or_default(getattr(nuclear, 'intra_prompt_consistency_values', [])),
        "step_stats": list(getattr(nuclear, 'step_stats', [])),
        "V_history": [v.clone() for v in getattr(nuclear, 'v_history', [])],
        "V_scaled_history": [v.clone() for v in getattr(nuclear, 'v_scaled_history', [])],
        "C_history": [c.clone() for c in getattr(nuclear, 'c_history', [])],
        "input_scale": float(input_scale),
        "inner_method": inner_method,
    }


def run_inner(bundle, harmful_queries: Sequence[str], args):
    """Run the inner adversary while treating the language model as fixed.

    The inner objective only optimizes the steering directions.  Leaving LoRA
    parameters trainable here makes autograd retain parameter-gradient state
    for every factor, even though those gradients are discarded before the
    outer update, and can exhaust GPU memory on 7B models.
    """
    grad_states = [(param, param.requires_grad) for param in bundle.model.parameters()]
    for param, requires_grad in grad_states:
        if requires_grad:
            param.requires_grad_(False)
    try:
        return _run_inner_impl(bundle, harmful_queries, args)
    finally:
        for param, requires_grad in grad_states:
            if param.requires_grad != requires_grad:
                param.requires_grad_(requires_grad)
