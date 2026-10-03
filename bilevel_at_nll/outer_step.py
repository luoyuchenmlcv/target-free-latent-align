from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence

import torch
from tqdm import tqdm

from .hooks import BatchPrefillOnlyAddVectorHook, infer_input_device
from .prompting import build_prompt_content_mask_batch


def _adapter_disabled_ctx(model):
    disable_adapter = getattr(model, "disable_adapter", None)
    if callable(disable_adapter):
        return disable_adapter()
    return nullcontext()


def _build_steering_matrix_with_zero(vs_scaled: torch.Tensor) -> torch.Tensor:
    """
    vs_scaled: [d_model, K]
    return   : [d_model, K+1], first column is zero-vector
    """
    if vs_scaled.dim() != 2:
        raise ValueError(f"Expected vs_scaled to have shape [d, K], got {tuple(vs_scaled.shape)}")
    zero = torch.zeros(
        vs_scaled.shape[0],
        1,
        device=vs_scaled.device,
        dtype=vs_scaled.dtype,
    )
    return torch.cat([zero, vs_scaled], dim=1)


def _round_up_to_bucket(x: int, bucket_size: int, max_seq_len: int) -> int:
    if bucket_size <= 0:
        raise ValueError(f"bucket_size must be positive, got {bucket_size}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    rounded = ((int(x) + bucket_size - 1) // bucket_size) * bucket_size
    return min(rounded, int(max_seq_len))


def _resolve_seq_bucket_len(
    args,
    *,
    prompt_len: int,
    generation_len_full: int,
) -> int:
    """
    Resolve per-example padded length with fixed-size bucketing.

    Defaults
    --------
    - seq_bucket_size = 32
    - num_seq_buckets = 16
    - max_seq_len = 512

    Final effective maximum:
        min(args.max_seq_len, args.seq_bucket_size * args.num_seq_buckets)

    Behavior
    --------
    - If prompt itself exceeds the effective maximum, raise immediately.
    - Otherwise, choose the smallest bucket that can hold:
          prompt_len + truncated_generation_len
    - Generation will later be truncated to fit the chosen bucket.
    """
    bucket_size = int(getattr(args, "seq_bucket_size", 32))
    num_buckets = int(getattr(args, "num_seq_buckets", 16))
    user_max_seq_len = int(getattr(args, "max_seq_len", 512))

    max_bucket_len = int(bucket_size * num_buckets)
    max_seq_len = int(min(user_max_seq_len, max_bucket_len))

    if bucket_size <= 0:
        raise ValueError(f"seq_bucket_size must be positive, got {bucket_size}")
    if num_buckets <= 0:
        raise ValueError(f"num_seq_buckets must be positive, got {num_buckets}")
    if max_seq_len <= 0:
        raise ValueError(
            f"Resolved non-positive max_seq_len={max_seq_len} "
            f"(user_max_seq_len={user_max_seq_len}, bucket_size={bucket_size}, num_buckets={num_buckets})"
        )

    if prompt_len > max_seq_len:
        raise ValueError(
            "Teacher-forced batch prompt exceeds maximum bucketed length. "
            f"prompt_tokens={prompt_len} max_seq_len={max_seq_len} "
            f"(bucket_size={bucket_size}, num_buckets={num_buckets})."
        )

    needed = prompt_len + generation_len_full
    needed = min(needed, max_seq_len)

    bucket_len = _round_up_to_bucket(
        needed,
        bucket_size=bucket_size,
        max_seq_len=max_seq_len,
    )

    if bucket_len < prompt_len:
        bucket_len = _round_up_to_bucket(
            prompt_len,
            bucket_size=bucket_size,
            max_seq_len=max_seq_len,
        )

    return int(bucket_len)


def _single_example_to_teacher_forced_batch(
    bundle,
    example: dict[str, str],
    num_steers: int,
    args,
):
    """
    Build a bucketed teacher-forced batch for ONE example, repeated num_steers times.

    Semantics
    ---------
    - prompt is always fully preserved, never truncated
    - generation is truncated to fit the chosen bucket
    - completion_mask is False on prompt tokens, True on generation tokens
    - output tensors have shape [num_steers, seq_len_bucket]
    """
    tok = bundle.tokenizer

    query = example["prompt"]
    generation = str(example.get("generation", ""))

    prompt = bundle.chat(query, add_generation_prompt=True)

    prompt_ids = tok(
        prompt,
        return_tensors=None,
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )["input_ids"]

    generation_ids_full = tok(
        generation,
        return_tensors=None,
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )["input_ids"]

    seq_len = _resolve_seq_bucket_len(
        args,
        prompt_len=len(prompt_ids),
        generation_len_full=len(generation_ids_full),
    )

    max_ctx = getattr(bundle.model.config, "max_position_embeddings", None)
    if max_ctx is not None and seq_len > int(max_ctx):
        raise ValueError(
            "Resolved bucket length exceeds model context window. "
            f"bucket_seq_len={seq_len} max_position_embeddings={int(max_ctx)}. "
            "Reduce args.max_seq_len / num_seq_buckets / seq_bucket_size, or shorten the prompt."
        )

    max_generation_tokens = int(seq_len - len(prompt_ids))
    generation_ids = generation_ids_full[:max_generation_tokens]
    merged_ids = prompt_ids + generation_ids

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id")

    input_ids = torch.full((num_steers, seq_len), fill_value=pad_id, dtype=torch.long)
    attention_mask = torch.zeros((num_steers, seq_len), dtype=torch.long)
    completion_mask = torch.zeros((num_steers, seq_len), dtype=torch.bool)

    merged_t = torch.tensor(merged_ids, dtype=torch.long)
    true_len = len(merged_ids)

    input_ids[:, :true_len] = merged_t.unsqueeze(0)
    attention_mask[:, :true_len] = 1
    if len(generation_ids) > 0:
        completion_mask[:, len(prompt_ids):true_len] = True

    batch_cpu = {
        "input_ids": input_ids.cpu(),
        "attention_mask": attention_mask.cpu(),
    }
    full_lens = [true_len] * num_steers
    queries = [query] * num_steers
    return batch_cpu, completion_mask.cpu(), full_lens, seq_len, queries


def _single_example_to_teacher_forced_batch_one_row(
    bundle,
    example: dict[str, str],
    args,
):
    return _single_example_to_teacher_forced_batch(
        bundle=bundle,
        example=example,
        num_steers=1,
        args=args,
    )


def _make_batch_steer_hook(
    bundle,
    queries: Sequence[str],
    full_lens: Sequence[int],
    seq_len: int,
    vs: torch.Tensor,  # [d_model, B]
    args,
):
    """
    Mirror generation.generate_with_batch_prefill_steer:
      - same prompt repeated B times
      - each row gets its own steering vector
    """
    mode = str(args.steer_positions).lower()
    if args.steer_prompt_content_only:
        mode = "content"

    deltas = vs.T  # [B, d_model]

    if mode == "content":
        mask = build_prompt_content_mask_batch(
            bundle,
            queries,
            full_lens,
            seq_len,
            args.max_prompt_tokens,
        )
        return BatchPrefillOnlyAddVectorHook(
            bundle.model,
            args.source_layer_idx,
            deltas,
            position_mask=mask,
            layers_path=None,
        )
    if mode == "all":
        return BatchPrefillOnlyAddVectorHook(
            bundle.model,
            args.source_layer_idx,
            deltas,
            positions=None,
            layers_path=None,
        )
    if mode == "last":
        return BatchPrefillOnlyAddVectorHook(
            bundle.model,
            args.source_layer_idx,
            deltas,
            positions=slice(-1, None),
            layers_path=None,
        )

    raise ValueError(f"Unsupported steer_positions={args.steer_positions!r}")


def _get_lm_head(model):
    emb = None
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        emb = getter()
    if emb is not None:
        return emb
    for name in ("lm_head", "embed_out"):
        if hasattr(model, name):
            return getattr(model, name)
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        getter = getattr(base_model, "get_output_embeddings", None)
        if callable(getter):
            emb = getter()
        if emb is not None:
            return emb
        for name in ("lm_head", "embed_out"):
            if hasattr(base_model, name):
                return getattr(base_model, name)
    raise ValueError(f"Could not resolve lm_head for model type {type(model)}")


def _resolve_backbone_model(model):
    """
    Return the transformer backbone that produces final hidden states without
    running the lm_head.

    Robust to PEFT / LoRA wrappers:
    - Prefer actual backbone modules over CausalLM shells
    - Recurse through common wrapper attributes
    """
    seen = set()

    def _visit(obj):
        if obj is None:
            return None

        oid = id(obj)
        if oid in seen:
            return None
        seen.add(oid)

        if callable(getattr(obj, "forward", None)) and not hasattr(obj, "lm_head"):
            if any(hasattr(obj, attr) for attr in ("layers", "embed_tokens", "norm", "rotary_emb")):
                return obj

        for attr in ("model", "transformer", "gpt_neox", "base_model", "backbone"):
            sub = getattr(obj, attr, None)
            found = _visit(sub)
            if found is not None:
                return found

        return None

    candidates = [model]

    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        try:
            candidates.append(getter())
        except Exception:
            pass

    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        candidates.append(base_model)

    for cand in candidates:
        found = _visit(cand)
        if found is not None:
            return found

    raise ValueError(f"Could not resolve backbone model for type {type(model)}")


def _forward_hidden_batch_steer(
    bundle,
    batch_cpu,
    *,
    queries: Sequence[str],
    full_lens: Sequence[int],
    seq_len: int,
    vs: torch.Tensor | None,   # [d_model, B] or None
    args,
) -> torch.Tensor:
    """
    Forward teacher-forced final hidden states for ONE chunk only.
    Returns [B, T, d_model].

    Robust behavior:
    1) Prefer pure backbone forward and read last_hidden_state
    2) Fall back to full model with output_hidden_states=True if needed
    """
    dev = infer_input_device(bundle.model)
    batch = {k: v.to(dev) for k, v in batch_cpu.items()}
    backbone = _resolve_backbone_model(bundle.model)

    def _extract_hidden_from_outputs(outputs):
        hidden = getattr(outputs, "last_hidden_state", None)
        if torch.is_tensor(hidden):
            return hidden

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None and len(hidden_states) > 0 and torch.is_tensor(hidden_states[-1]):
            return hidden_states[-1]

        if isinstance(outputs, (tuple, list)):
            for x in outputs:
                if torch.is_tensor(x) and x.dim() == 3:
                    return x

        if hasattr(outputs, "keys"):
            for k in ("last_hidden_state", "hidden_states"):
                if k in outputs:
                    v = outputs[k]
                    if torch.is_tensor(v):
                        return v
                    if isinstance(v, (tuple, list)) and len(v) > 0 and torch.is_tensor(v[-1]):
                        return v[-1]
            for _, v in outputs.items():
                if torch.is_tensor(v) and v.dim() == 3:
                    return v
                if isinstance(v, (tuple, list)):
                    for vv in v:
                        if torch.is_tensor(vv) and vv.dim() == 3:
                            return vv

        return None

    def _run_forward():
        outputs = backbone(
            **batch,
            use_cache=False,
            return_dict=True,
        )
        hidden = _extract_hidden_from_outputs(outputs)
        if torch.is_tensor(hidden):
            return hidden

        outputs = bundle.model(
            **batch,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden = _extract_hidden_from_outputs(outputs)
        if torch.is_tensor(hidden):
            return hidden

        raise RuntimeError(
            f"Expected final hidden state tensor, got {type(hidden)} "
            f"(backbone_type={type(backbone)}, model_type={type(bundle.model)})"
        )

    if vs is None:
        return _run_forward()

    vs = vs.to(device=dev)
    hook = _make_batch_steer_hook(
        bundle=bundle,
        queries=queries,
        full_lens=full_lens,
        seq_len=seq_len,
        vs=vs,
        args=args,
    )
    with hook:
        return _run_forward()


def _gather_valid_hidden_and_labels_per_row(
    hidden: torch.Tensor,             # [B, T, d]
    input_ids_cpu: torch.Tensor,      # [B, T]
    completion_mask_cpu: torch.Tensor,# [B, T]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Per row, apply the standard causal-LM one-token shift and keep only the
    completion-region tokens.
    """
    device = hidden.device
    B = int(hidden.shape[0])
    hidden_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []

    for b in range(B):
        hidden_b = hidden[b, :-1, :]
        labels_b = input_ids_cpu[b, 1:].to(device=device, dtype=torch.long)
        mask_b = completion_mask_cpu[b, 1:].to(device=device, dtype=torch.bool)
        if int(mask_b.sum().item()) <= 0:
            hidden_rows.append(hidden_b.new_zeros((0, hidden_b.shape[-1])))
            label_rows.append(labels_b.new_zeros((0,), dtype=torch.long))
            continue
        hidden_rows.append(hidden_b[mask_b])
        label_rows.append(labels_b[mask_b])

    return hidden_rows, label_rows


def _get_vocab_chunk_size(args, lm_head) -> int:
    return max(
        1,
        int(
            getattr(
                args,
                "outer_vocab_chunk_size",
                lm_head.weight.shape[0],
            )
        ),
    )


def _get_steer_chunk_size(args, num_steers: int) -> int:
    if hasattr(args, "steer_chunk_size"):
        return max(1, int(getattr(args, "steer_chunk_size", 64)))
    return num_steers


def _get_token_chunk_size(args) -> int:
    return max(1, int(getattr(args, "outer_token_chunk_size", 64)))


def _iter_vocab_chunks(vocab_size: int, chunk_size: int, *, desc: str | None = None, enable_tqdm: bool = False):
    it = range(0, vocab_size, chunk_size)
    if enable_tqdm:
        it = tqdm(
            it,
            desc=desc or "vocab_chunks",
            dynamic_ncols=True,
            leave=False,
        )
    for s in it:
        e = min(vocab_size, s + chunk_size)
        yield s, e


def _iter_token_chunks(num_tokens: int, chunk_size: int):
    for s in range(0, num_tokens, chunk_size):
        e = min(num_tokens, s + chunk_size)
        yield s, e


def _project_vocab_chunk(hidden_tokens: torch.Tensor, lm_head, start: int, end: int) -> torch.Tensor:
    """
    hidden_tokens: [N, d]
    returns: [N, V_chunk]

    IMPORTANT:
    - Do projection on lm_head.device, not on hidden_tokens.device
    """
    if hidden_tokens.numel() == 0:
        return hidden_tokens.new_zeros((0, end - start))

    weight = lm_head.weight[start:end]
    bias = None if getattr(lm_head, "bias", None) is None else lm_head.bias[start:end]

    proj_device = weight.device
    proj_dtype = weight.dtype

    if bias is not None:
        bias = bias.to(device=proj_device, dtype=proj_dtype)

    h = hidden_tokens.to(device=proj_device, dtype=proj_dtype, non_blocking=False)
    logits = h @ weight.transpose(0, 1)
    if bias is not None:
        logits = logits + bias
    return logits


def _exact_chunked_ce_sum_count_per_row(
    hidden_rows: list[torch.Tensor],
    label_rows: list[torch.Tensor],
    lm_head,
    vocab_chunk_size: int,
    token_chunk_size: int,
    *,
    out_dtype: torch.dtype,
    out_device: torch.device,
    enable_vocab_tqdm: bool = False,
    vocab_tqdm_desc: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact token-level CE computed by streaming over vocab chunks.
    Returns per-row token-summed NLL and valid-token counts.
    """
    vocab_size = int(lm_head.weight.shape[0])
    proj_device = lm_head.weight.device

    row_nll_sums = []
    row_token_counts = []

    for row_idx, (H, y) in enumerate(zip(hidden_rows, label_rows)):
        n_valid = int(y.shape[0])
        if n_valid <= 0:
            row_nll_sums.append(torch.tensor(0.0, device=out_device, dtype=out_dtype))
            row_token_counts.append(torch.tensor(0.0, device=out_device, dtype=out_dtype))
            continue

        y = y.to(device=proj_device, dtype=torch.long, non_blocking=False)
        token_nll_parts = []

        for token_chunk_idx, (ts, te) in enumerate(_iter_token_chunks(n_valid, token_chunk_size)):
            H_tok = H[ts:te]
            y_tok = y[ts:te]
            n_tok = int(y_tok.shape[0])

            lse_total = None
            target_logits = torch.full((n_tok,), float("-inf"), device=proj_device, dtype=torch.float32)

            vocab_iter = _iter_vocab_chunks(
                vocab_size,
                vocab_chunk_size,
                desc=(
                    vocab_tqdm_desc
                    if vocab_tqdm_desc is not None
                    else f"vocab_chunks[row={row_idx},tok={token_chunk_idx}]"
                ),
                enable_tqdm=enable_vocab_tqdm,
            )
            for s, e in vocab_iter:
                logits_chunk = _project_vocab_chunk(H_tok, lm_head, s, e)
                logits_chunk_f32 = logits_chunk.float()
                chunk_lse = torch.logsumexp(logits_chunk_f32, dim=-1)
                lse_total = chunk_lse if lse_total is None else torch.logaddexp(lse_total, chunk_lse)

                in_chunk = (y_tok >= s) & (y_tok < e)
                if in_chunk.any():
                    row_idx_local = torch.nonzero(in_chunk, as_tuple=False).squeeze(-1)
                    col_idx = (y_tok[row_idx_local] - s).to(dtype=torch.long)
                    target_logits[row_idx_local] = logits_chunk_f32[row_idx_local, col_idx]

                del logits_chunk
                del logits_chunk_f32

            if torch.isinf(target_logits).any():
                raise RuntimeError("Failed to recover some target logits while streaming CE over vocab chunks.")

            token_nll_parts.append(lse_total - target_logits)

            del H_tok, y_tok, lse_total, target_logits

        token_nll = torch.cat(token_nll_parts, dim=0) if len(token_nll_parts) > 1 else token_nll_parts[0]
        row_nll_sums.append(token_nll.sum().to(device=out_device, dtype=out_dtype))
        row_token_counts.append(torch.tensor(float(n_valid), device=out_device, dtype=out_dtype))

        del token_nll
        del token_nll_parts

    return torch.stack(row_nll_sums, dim=0), torch.stack(row_token_counts, dim=0)


def _exact_chunked_kl_sum_count_per_row(
    train_hidden_rows: list[torch.Tensor],
    ref_hidden_rows: list[torch.Tensor],
    lm_head,
    vocab_chunk_size: int,
    token_chunk_size: int,
    *,
    out_dtype: torch.dtype,
    out_device: torch.device,
    enable_vocab_tqdm: bool = False,
    vocab_tqdm_desc: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact token-level KL(ref || train) computed by streaming over vocab chunks.
    Returns per-row token-summed KL and valid-token counts.
    """
    vocab_size = int(lm_head.weight.shape[0])
    proj_device = lm_head.weight.device

    row_kl_sums = []
    row_token_counts = []

    for row_idx, (H_train, H_ref) in enumerate(zip(train_hidden_rows, ref_hidden_rows)):
        n_valid = int(H_train.shape[0])
        if n_valid <= 0:
            row_kl_sums.append(torch.tensor(0.0, device=out_device, dtype=out_dtype))
            row_token_counts.append(torch.tensor(0.0, device=out_device, dtype=out_dtype))
            continue

        token_kl_parts = []

        for token_chunk_idx, (ts, te) in enumerate(_iter_token_chunks(n_valid, token_chunk_size)):
            H_train_tok = H_train[ts:te]
            H_ref_tok = H_ref[ts:te]
            n_tok = int(H_train_tok.shape[0])

            lse_train = None
            lse_ref = None

            vocab_iter_first = _iter_vocab_chunks(
                vocab_size,
                vocab_chunk_size,
                desc=(
                    vocab_tqdm_desc
                    if vocab_tqdm_desc is not None
                    else f"vocab_chunks[row={row_idx},tok={token_chunk_idx}]"
                ),
                enable_tqdm=enable_vocab_tqdm,
            )
            for s, e in vocab_iter_first:
                z_train_chunk = _project_vocab_chunk(H_train_tok, lm_head, s, e)
                z_ref_chunk = _project_vocab_chunk(H_ref_tok, lm_head, s, e)

                z_train_chunk_f32 = z_train_chunk.float()
                z_ref_chunk_f32 = z_ref_chunk.float()

                chunk_lse_train = torch.logsumexp(z_train_chunk_f32, dim=-1)
                chunk_lse_ref = torch.logsumexp(z_ref_chunk_f32, dim=-1)

                lse_train = chunk_lse_train if lse_train is None else torch.logaddexp(lse_train, chunk_lse_train)
                lse_ref = chunk_lse_ref if lse_ref is None else torch.logaddexp(lse_ref, chunk_lse_ref)

                del z_train_chunk, z_ref_chunk, z_train_chunk_f32, z_ref_chunk_f32

            token_kl = torch.zeros((n_tok,), device=proj_device, dtype=torch.float32)

            vocab_iter_second = _iter_vocab_chunks(
                vocab_size,
                vocab_chunk_size,
                desc=(
                    vocab_tqdm_desc
                    if vocab_tqdm_desc is not None
                    else f"vocab_chunks[row={row_idx},tok={token_chunk_idx}]"
                ),
                enable_tqdm=enable_vocab_tqdm,
            )
            for s, e in vocab_iter_second:
                z_train_chunk = _project_vocab_chunk(H_train_tok, lm_head, s, e)
                z_ref_chunk = _project_vocab_chunk(H_ref_tok, lm_head, s, e)

                z_train_chunk_f32 = z_train_chunk.float()
                z_ref_chunk_f32 = z_ref_chunk.float()

                log_p_train_chunk = z_train_chunk_f32 - lse_train.unsqueeze(-1)
                log_p_ref_chunk = z_ref_chunk_f32 - lse_ref.unsqueeze(-1)
                p_ref_chunk = torch.exp(log_p_ref_chunk)

                token_kl = token_kl + (p_ref_chunk * (log_p_ref_chunk - log_p_train_chunk)).sum(dim=-1)

                del z_train_chunk, z_ref_chunk, z_train_chunk_f32, z_ref_chunk_f32
                del log_p_train_chunk, log_p_ref_chunk, p_ref_chunk

            token_kl_parts.append(token_kl)

            del H_train_tok, H_ref_tok, lse_train, lse_ref

        token_kl = torch.cat(token_kl_parts, dim=0) if len(token_kl_parts) > 1 else token_kl_parts[0]
        row_kl_sums.append(token_kl.sum().to(device=out_device, dtype=out_dtype))
        row_token_counts.append(torch.tensor(float(n_valid), device=out_device, dtype=out_dtype))

        del token_kl
        del token_kl_parts

    return torch.stack(row_kl_sums, dim=0), torch.stack(row_token_counts, dim=0)


def _build_harmful_refusal_example(example: dict[str, str]) -> dict[str, str]:
    return {
        "prompt": str(example["prompt"]),
        "generation": str(example["generation"]),
    }


def _slice_batch_cpu(batch_cpu, start: int, end: int):
    return {
        "input_ids": batch_cpu["input_ids"][start:end],
        "attention_mask": batch_cpu["attention_mask"][start:end],
    }


def _slice_completion_mask_cpu(completion_mask_cpu: torch.Tensor, start: int, end: int):
    return completion_mask_cpu[start:end]


def _count_valid_rows(completion_mask_cpu: torch.Tensor) -> int:
    shifted = completion_mask_cpu[:, 1:]
    if shifted.numel() == 0:
        return 0
    return int((shifted.sum(dim=1) > 0).sum().item())


def _single_example_harmful_nll(
    bundle,
    example_idx: int,
    example: dict[str, str],
    vs_with_zero: torch.Tensor,
    args,
) -> torch.Tensor:
    """
    No-grad / evaluation path:
        mean_v NLL(y | x, v)
    """
    refusal_example = _build_harmful_refusal_example(example)

    num_steers = int(vs_with_zero.shape[1])
    dev = infer_input_device(bundle.model)
    steer_chunk_size = _get_steer_chunk_size(args, num_steers)
    token_chunk_size = _get_token_chunk_size(args)
    lm_head = _get_lm_head(bundle.model)
    vocab_chunk_size = _get_vocab_chunk_size(args, lm_head)

    batch_cpu_full, completion_mask_cpu_full, full_lens_full, seq_len_full, queries_full = (
        _single_example_to_teacher_forced_batch(
            bundle,
            refusal_example,
            num_steers=num_steers,
            args=args,
        )
    )

    if completion_mask_cpu_full[:, 1:].sum().item() <= 0:
        return torch.tensor(0.0, device=dev)

    total_nll_sum = None
    total_row_count = 0.0

    steer_iter = tqdm(
        range(0, num_steers, steer_chunk_size),
        desc=f"harmful_steer_chunks[ex={example_idx}]",
        dynamic_ncols=True,
        leave=False,
    )
    for s in steer_iter:
        e = min(num_steers, s + steer_chunk_size)
        vs_chunk = vs_with_zero[:, s:e]

        batch_cpu_chunk = _slice_batch_cpu(batch_cpu_full, s, e)
        completion_mask_cpu_chunk = _slice_completion_mask_cpu(completion_mask_cpu_full, s, e)
        full_lens_chunk = full_lens_full[s:e]
        queries_chunk = queries_full[s:e]

        if completion_mask_cpu_chunk[:, 1:].sum().item() <= 0:
            continue

        train_hidden_chunk = _forward_hidden_batch_steer(
            bundle,
            batch_cpu_chunk,
            queries=queries_chunk,
            full_lens=full_lens_chunk,
            seq_len=seq_len_full,
            vs=vs_chunk,
            args=args,
        )

        hidden_rows_chunk, label_rows_chunk = _gather_valid_hidden_and_labels_per_row(
            hidden=train_hidden_chunk,
            input_ids_cpu=batch_cpu_chunk["input_ids"],
            completion_mask_cpu=completion_mask_cpu_chunk,
        )

        row_nll_sums_chunk, row_counts_chunk = _exact_chunked_ce_sum_count_per_row(
            hidden_rows=hidden_rows_chunk,
            label_rows=label_rows_chunk,
            lm_head=lm_head,
            vocab_chunk_size=vocab_chunk_size,
            token_chunk_size=token_chunk_size,
            out_dtype=train_hidden_chunk.dtype,
            out_device=train_hidden_chunk.device,
            enable_vocab_tqdm=(s == 0),
            vocab_tqdm_desc=f"harmful_vocab[ex={example_idx},chunk={s // steer_chunk_size}]",
        )

        valid_rows = row_counts_chunk > 0
        if valid_rows.any():
            row_means_chunk = row_nll_sums_chunk[valid_rows] / row_counts_chunk[valid_rows]
            total_row_count += float(valid_rows.sum().item())
            total_nll_sum = row_means_chunk.sum() if total_nll_sum is None else (total_nll_sum + row_means_chunk.sum())
            steer_iter.set_postfix(valid_rows=int(total_row_count))
            del row_means_chunk

        del batch_cpu_chunk, completion_mask_cpu_chunk, full_lens_chunk, queries_chunk
        del train_hidden_chunk, hidden_rows_chunk, label_rows_chunk, row_nll_sums_chunk, row_counts_chunk

    if total_nll_sum is None or total_row_count <= 0:
        return torch.tensor(0.0, device=dev, dtype=vs_with_zero.dtype)

    return total_nll_sum / total_row_count


def _single_example_benign_kl_simple(
    bundle,
    example_idx: int,
    example: dict[str, str],
    args,
) -> torch.Tensor:
    """
    No-grad / evaluation path:
        KL(p_ref || p_train) on benign example
    where:
        - ref = LoRA disabled
        - train = LoRA enabled
        - no steer
        - single row only
    """
    del example_idx

    dev = infer_input_device(bundle.model)
    lm_head = _get_lm_head(bundle.model)
    vocab_chunk_size = _get_vocab_chunk_size(args, lm_head)
    token_chunk_size = _get_token_chunk_size(args)

    batch_cpu, completion_mask_cpu, full_lens, seq_len, queries = (
        _single_example_to_teacher_forced_batch_one_row(
            bundle=bundle,
            example=example,
            args=args,
        )
    )

    if completion_mask_cpu[:, 1:].sum().item() <= 0:
        return torch.tensor(0.0, device=dev)

    train_hidden = _forward_hidden_batch_steer(
        bundle,
        batch_cpu,
        queries=queries,
        full_lens=full_lens,
        seq_len=seq_len,
        vs=None,
        args=args,
    )

    with torch.no_grad():
        with _adapter_disabled_ctx(bundle.model):
            ref_hidden = _forward_hidden_batch_steer(
                bundle,
                batch_cpu,
                queries=queries,
                full_lens=full_lens,
                seq_len=seq_len,
                vs=None,
                args=args,
            )

    train_hidden_rows, _ = _gather_valid_hidden_and_labels_per_row(
        hidden=train_hidden,
        input_ids_cpu=batch_cpu["input_ids"],
        completion_mask_cpu=completion_mask_cpu,
    )
    ref_hidden_rows, _ = _gather_valid_hidden_and_labels_per_row(
        hidden=ref_hidden,
        input_ids_cpu=batch_cpu["input_ids"],
        completion_mask_cpu=completion_mask_cpu,
    )

    row_kl_sums, row_counts = _exact_chunked_kl_sum_count_per_row(
        train_hidden_rows=train_hidden_rows,
        ref_hidden_rows=ref_hidden_rows,
        lm_head=lm_head,
        vocab_chunk_size=vocab_chunk_size,
        token_chunk_size=token_chunk_size,
        out_dtype=train_hidden.dtype,
        out_device=train_hidden.device,
        enable_vocab_tqdm=False,
        vocab_tqdm_desc="benign_vocab_simple",
    )

    valid_rows = row_counts > 0
    if not valid_rows.any():
        return torch.tensor(0.0, device=dev, dtype=train_hidden.dtype)

    row_means = row_kl_sums[valid_rows] / row_counts[valid_rows]
    return row_means.mean()


def _single_example_harmful_nll_backward(
    bundle,
    example_idx: int,
    example: dict[str, str],
    vs_with_zero: torch.Tensor,
    args,
    *,
    backward_scale: float,
) -> float:
    """
    Training path:
        do per-steer-chunk backward immediately
        return detached unweighted mean_v NLL as a Python float
    """
    refusal_example = _build_harmful_refusal_example(example)

    num_steers = int(vs_with_zero.shape[1])
    steer_chunk_size = _get_steer_chunk_size(args, num_steers)
    token_chunk_size = _get_token_chunk_size(args)
    lm_head = _get_lm_head(bundle.model)
    vocab_chunk_size = _get_vocab_chunk_size(args, lm_head)

    batch_cpu_full, completion_mask_cpu_full, full_lens_full, seq_len_full, queries_full = (
        _single_example_to_teacher_forced_batch(
            bundle,
            refusal_example,
            num_steers=num_steers,
            args=args,
        )
    )

    total_valid_rows = _count_valid_rows(completion_mask_cpu_full)
    if total_valid_rows <= 0:
        return 0.0

    running_mean_sum = 0.0
    running_row_count = 0.0

    steer_iter = tqdm(
        range(0, num_steers, steer_chunk_size),
        desc=f"harmful_steer_chunks[ex={example_idx}]",
        dynamic_ncols=True,
        leave=False,
    )
    for s in steer_iter:
        e = min(num_steers, s + steer_chunk_size)
        vs_chunk = vs_with_zero[:, s:e]

        batch_cpu_chunk = _slice_batch_cpu(batch_cpu_full, s, e)
        completion_mask_cpu_chunk = _slice_completion_mask_cpu(completion_mask_cpu_full, s, e)
        full_lens_chunk = full_lens_full[s:e]
        queries_chunk = queries_full[s:e]

        if completion_mask_cpu_chunk[:, 1:].sum().item() <= 0:
            continue

        train_hidden_chunk = _forward_hidden_batch_steer(
            bundle,
            batch_cpu_chunk,
            queries=queries_chunk,
            full_lens=full_lens_chunk,
            seq_len=seq_len_full,
            vs=vs_chunk,
            args=args,
        )

        hidden_rows_chunk, label_rows_chunk = _gather_valid_hidden_and_labels_per_row(
            hidden=train_hidden_chunk,
            input_ids_cpu=batch_cpu_chunk["input_ids"],
            completion_mask_cpu=completion_mask_cpu_chunk,
        )

        row_nll_sums_chunk, row_counts_chunk = _exact_chunked_ce_sum_count_per_row(
            hidden_rows=hidden_rows_chunk,
            label_rows=label_rows_chunk,
            lm_head=lm_head,
            vocab_chunk_size=vocab_chunk_size,
            token_chunk_size=token_chunk_size,
            out_dtype=train_hidden_chunk.dtype,
            out_device=train_hidden_chunk.device,
            enable_vocab_tqdm=(s == 0),
            vocab_tqdm_desc=f"harmful_vocab[ex={example_idx},chunk={s // steer_chunk_size}]",
        )

        valid_rows = row_counts_chunk > 0
        if valid_rows.any():
            row_means_chunk = row_nll_sums_chunk[valid_rows] / row_counts_chunk[valid_rows]
            n_valid_chunk = int(valid_rows.sum().item())

            chunk_weight = float(n_valid_chunk) / float(total_valid_rows)
            chunk_loss = row_means_chunk.mean()
            (backward_scale * chunk_weight * chunk_loss).backward()

            running_mean_sum += float(row_means_chunk.detach().sum().item())
            running_row_count += float(n_valid_chunk)
            steer_iter.set_postfix(valid_rows=int(running_row_count))

            del chunk_loss
            del row_means_chunk

        del batch_cpu_chunk, completion_mask_cpu_chunk, full_lens_chunk, queries_chunk
        del train_hidden_chunk, hidden_rows_chunk, label_rows_chunk, row_nll_sums_chunk, row_counts_chunk

    return running_mean_sum / max(running_row_count, 1.0)


def _single_example_benign_kl_simple_backward(
    bundle,
    example_idx: int,
    example: dict[str, str],
    args,
    *,
    backward_scale: float,
) -> float:
    """
    Training path:
        KL(p_ref || p_train) on benign example
    where:
        - ref = LoRA disabled
        - train = LoRA enabled
        - no steer
        - single row only
    """
    del example_idx

    lm_head = _get_lm_head(bundle.model)
    vocab_chunk_size = _get_vocab_chunk_size(args, lm_head)
    token_chunk_size = _get_token_chunk_size(args)

    batch_cpu, completion_mask_cpu, full_lens, seq_len, queries = (
        _single_example_to_teacher_forced_batch_one_row(
            bundle=bundle,
            example=example,
            args=args,
        )
    )

    total_valid_rows = _count_valid_rows(completion_mask_cpu)
    if total_valid_rows <= 0:
        return 0.0

    train_hidden = _forward_hidden_batch_steer(
        bundle,
        batch_cpu,
        queries=queries,
        full_lens=full_lens,
        seq_len=seq_len,
        vs=None,
        args=args,
    )

    with torch.no_grad():
        with _adapter_disabled_ctx(bundle.model):
            ref_hidden = _forward_hidden_batch_steer(
                bundle,
                batch_cpu,
                queries=queries,
                full_lens=full_lens,
                seq_len=seq_len,
                vs=None,
                args=args,
            )

    train_hidden_rows, _ = _gather_valid_hidden_and_labels_per_row(
        hidden=train_hidden,
        input_ids_cpu=batch_cpu["input_ids"],
        completion_mask_cpu=completion_mask_cpu,
    )
    ref_hidden_rows, _ = _gather_valid_hidden_and_labels_per_row(
        hidden=ref_hidden,
        input_ids_cpu=batch_cpu["input_ids"],
        completion_mask_cpu=completion_mask_cpu,
    )

    row_kl_sums, row_counts = _exact_chunked_kl_sum_count_per_row(
        train_hidden_rows=train_hidden_rows,
        ref_hidden_rows=ref_hidden_rows,
        lm_head=lm_head,
        vocab_chunk_size=vocab_chunk_size,
        token_chunk_size=token_chunk_size,
        out_dtype=train_hidden.dtype,
        out_device=train_hidden.device,
        enable_vocab_tqdm=False,
        vocab_tqdm_desc="benign_vocab_simple",
    )

    valid_rows = row_counts > 0
    if not valid_rows.any():
        return 0.0

    row_means = row_kl_sums[valid_rows] / row_counts[valid_rows]
    loss = row_means.mean()
    (backward_scale * loss).backward()

    return float(loss.detach().item())


def _compute_harmful_nll_benign_kl(
    *,
    train_bundle,
    harmful_examples: Sequence[dict[str, str]],
    benign_examples: Sequence[dict[str, str]],
    harmful_vs_with_zero: torch.Tensor,
    args,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute unweighted harmful refusal NLL and simple benign KL.
    """
    dev = infer_input_device(train_bundle.model)

    harmful_vals: list[torch.Tensor] = []
    benign_vals: list[torch.Tensor] = []

    for i, ex in enumerate(harmful_examples):
        harmful_vals.append(
            _single_example_harmful_nll(
                bundle=train_bundle,
                example_idx=i,
                example=ex,
                vs_with_zero=harmful_vs_with_zero,
                args=args,
            )
        )

    for i, ex in enumerate(benign_examples):
        benign_vals.append(
            _single_example_benign_kl_simple(
                bundle=train_bundle,
                example_idx=i,
                example=ex,
                args=args,
            )
        )

    harmful_nll = (
        torch.stack(harmful_vals).mean()
        if harmful_vals
        else torch.tensor(0.0, device=dev)
    )
    benign_kl = (
        torch.stack(benign_vals).mean()
        if benign_vals
        else torch.tensor(0.0, device=dev)
    )

    return harmful_nll, benign_kl


@torch.no_grad()
def evaluate_outer_loss(
    train_bundle,
    harmful_examples: Sequence[dict[str, str]],
    benign_examples: Sequence[dict[str, str]],
    harmful_grouped_vs_scaled: Sequence[torch.Tensor],
    harmful_group_sizes: Sequence[int],
    benign_vs_scaled: torch.Tensor | None,
    args,
):
    """
    Evaluate the current grouped outer loss objective without taking optimizer steps.

    - harmful examples are chunked according to harmful_group_sizes
    - harmful group g uses only harmful_grouped_vs_scaled[g]
    - benign KL is no-steer and does not depend on grouping
    """
    del benign_vs_scaled

    dev = infer_input_device(train_bundle.model)
    model_dtype = next(train_bundle.model.parameters()).dtype

    harmful_groups, benign_list = _build_grouped_example_batches(
        harmful_examples=harmful_examples,
        benign_examples=benign_examples,
        harmful_group_sizes=harmful_group_sizes,
    )
    grouped_vs = _normalize_grouped_vs_list(
        harmful_grouped_vs_scaled,
        device=dev,
        dtype=model_dtype,
    )

    harmful_value, benign_value = _compute_grouped_harmful_nll_benign_kl(
        train_bundle=train_bundle,
        harmful_grouped_examples=harmful_groups,
        benign_examples=benign_list,
        harmful_grouped_vs_scaled=grouped_vs,
        args=args,
    )

    lambda_h = float(getattr(args, "lambda_harmful", 0.0))
    lambda_b = float(getattr(args, "lambda_benign", 0.0))

    harmful_value_f = float(harmful_value.detach().item())
    benign_value_f = float(benign_value.detach().item())

    num_groups = int(len(grouped_vs))
    dirs_per_group = [int(v.shape[1]) for v in grouped_vs]
    total_dirs = int(sum(dirs_per_group))
    total_steers = int(sum(k + 1 for k in dirs_per_group))

    return {
        "harmful_unweighted": harmful_value_f,
        "benign_unweighted": benign_value_f,
        "loss_harmful": float(lambda_h * harmful_value_f),
        "loss_benign": float(lambda_b * benign_value_f),
        "loss": float(lambda_h * harmful_value_f + lambda_b * benign_value_f),
        "num_harmful": int(len(harmful_examples)),
        "num_benign": int(len(benign_examples)),
        "num_groups": num_groups,
        "group_sizes": [int(x) for x in harmful_group_sizes],
        "num_dirs_total": total_dirs,
        "num_dirs_per_group": dirs_per_group,
        "num_steers_per_group_including_zero": [int(k + 1) for k in dirs_per_group],
        "num_steers_total_including_group_zeros": total_steers,
    }


def _clone_trainable_state(model):
    return {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def _restore_trainable_state(model, state_dict):
    name_to_param = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state_dict.items():
            p = name_to_param[name]
            p.copy_(value.to(device=p.device, dtype=p.dtype))


def _chunk_examples_by_sizes(
    examples: Sequence[dict[str, str]],
    group_sizes: Sequence[int],
    *,
    name: str,
) -> list[list[dict[str, str]]]:
    group_sizes = [int(x) for x in group_sizes]
    if any(x <= 0 for x in group_sizes):
        raise ValueError(f"{name}: all group sizes must be positive, got {group_sizes}")

    total_needed = int(sum(group_sizes))
    if len(examples) != total_needed:
        raise ValueError(
            f"{name}: expected {total_needed} examples from harmful_group_sizes={group_sizes}, "
            f"but got {len(examples)}"
        )

    out: list[list[dict[str, str]]] = []
    ptr = 0
    for size in group_sizes:
        out.append(list(examples[ptr : ptr + size]))
        ptr += size
    return out


def _build_grouped_example_batches(
    *,
    harmful_examples: Sequence[dict[str, str]],
    benign_examples: Sequence[dict[str, str]],
    harmful_group_sizes: Sequence[int],
) -> tuple[list[list[dict[str, str]]], list[dict[str, str]]]:
    harmful_groups = _chunk_examples_by_sizes(
        harmful_examples,
        harmful_group_sizes,
        name="harmful_examples",
    )
    benign_list = list(benign_examples)
    return harmful_groups, benign_list


def _normalize_grouped_vs_list(
    grouped_vs: Sequence[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for g, vs in enumerate(grouped_vs):
        if not torch.is_tensor(vs):
            raise TypeError(f"group {g}: expected a torch.Tensor, got {type(vs)}")
        if vs.dim() != 2:
            raise ValueError(f"group {g}: expected [d_model, K], got {tuple(vs.shape)}")
        out.append(vs.to(device=device, dtype=dtype))
    if not out:
        raise ValueError("harmful_grouped_vs_scaled is empty")
    return out


@torch.no_grad()
def _compute_grouped_harmful_nll_benign_kl(
    train_bundle,
    harmful_grouped_examples: Sequence[Sequence[dict[str, str]]],
    benign_examples: Sequence[dict[str, str]],
    harmful_grouped_vs_scaled: Sequence[torch.Tensor],
    args,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact grouped objective evaluation.

    - harmful group g uses only V_g for its harmful subgroup
    - benign KL is computed over the full benign set with NO steer
    """
    dev = infer_input_device(train_bundle.model)
    ref_dtype = next(train_bundle.model.parameters()).dtype

    num_groups = int(len(harmful_grouped_vs_scaled))
    if len(harmful_grouped_examples) != num_groups:
        raise ValueError(
            "Grouped length mismatch: "
            f"num_vs={num_groups} "
            f"num_harmful_groups={len(harmful_grouped_examples)}"
        )

    total_harmful = int(sum(len(g) for g in harmful_grouped_examples))
    total_benign = int(len(benign_examples))

    harmful_sum = torch.tensor(0.0, device=dev, dtype=ref_dtype)
    benign_sum = torch.tensor(0.0, device=dev, dtype=ref_dtype)

    harmful_global_idx = 0
    benign_global_idx = 0

    for harmful_group, vs_group in zip(harmful_grouped_examples, harmful_grouped_vs_scaled):
        vs_with_zero = _build_steering_matrix_with_zero(vs_group)

        for ex in harmful_group:
            harmful_sum = harmful_sum + _single_example_harmful_nll(
                bundle=train_bundle,
                example_idx=harmful_global_idx,
                example=ex,
                vs_with_zero=vs_with_zero,
                args=args,
            )
            harmful_global_idx += 1

    for ex in benign_examples:
        benign_sum = benign_sum + _single_example_benign_kl_simple(
            bundle=train_bundle,
            example_idx=benign_global_idx,
            example=ex,
            args=args,
        )
        benign_global_idx += 1

    harmful_mean = harmful_sum / float(max(1, total_harmful))
    benign_mean = benign_sum / float(max(1, total_benign))
    return harmful_mean, benign_mean


def outer_step(
    train_bundle,
    harmful_examples: Sequence[dict[str, str]],
    benign_examples: Sequence[dict[str, str]],
    harmful_grouped_vs_scaled: Sequence[torch.Tensor],
    harmful_group_sizes: Sequence[int],
    benign_vs_scaled: torch.Tensor | None,
    optimizer,
    args,
    eval_harmful_grouped_vs_scaled: Sequence[torch.Tensor] | None = None,
    eval_benign_vs_scaled: torch.Tensor | None = None,
):
    """
    Grouped outer optimization.

    - harmful examples are grouped and use their corresponding V_g
    - benign examples are NOT grouped and do NOT use steer
    - benign_vs_scaled is ignored, kept only for train.py compatibility

    Exact objective:
        lambda_h * mean_i mean_{v in V_group(i)} loss_harmful(i, v)
      + lambda_b * mean_j KL( p_ref(j) || p_train(j) )
    """
    del benign_vs_scaled
    del eval_benign_vs_scaled  # benign eval remains no-steer in the current objective

    num_steps = max(1, int(args.outer_steps_per_round))
    dev = infer_input_device(train_bundle.model)

    model_dtype = next(train_bundle.model.parameters()).dtype
    grouped_vs = _normalize_grouped_vs_list(
        harmful_grouped_vs_scaled,
        device=dev,
        dtype=model_dtype,
    )
    if eval_harmful_grouped_vs_scaled is None:
        eval_harmful_grouped_vs_scaled = harmful_grouped_vs_scaled
    eval_grouped_vs = _normalize_grouped_vs_list(
        eval_harmful_grouped_vs_scaled,
        device=dev,
        dtype=model_dtype,
    )
    harmful_groups, benign_list = _build_grouped_example_batches(
        harmful_examples=harmful_examples,
        benign_examples=benign_examples,
        harmful_group_sizes=harmful_group_sizes,
    )

    if len(grouped_vs) != len(harmful_groups):
        raise ValueError(
            "Mismatch between grouped Vs and grouped examples: "
            f"num_vs={len(grouped_vs)} num_groups={len(harmful_groups)}"
        )
    if len(eval_grouped_vs) != len(harmful_groups):
        raise ValueError(
            "Mismatch between eval grouped Vs and grouped examples: "
            f"num_eval_vs={len(eval_grouped_vs)} num_groups={len(harmful_groups)}"
        )

    lambda_h = float(getattr(args, "lambda_harmful", 0.0))
    lambda_b = float(getattr(args, "lambda_benign", 0.0))

    num_harmful = int(len(harmful_examples))
    num_benign = int(len(benign_examples))
    num_groups = int(len(grouped_vs))
    dirs_per_group = [int(v.shape[1]) for v in grouped_vs]
    total_dirs = int(sum(dirs_per_group))
    total_steers = int(sum(k + 1 for k in dirs_per_group))

    start_eval = evaluate_outer_loss(
        train_bundle=train_bundle,
        harmful_examples=harmful_examples,
        benign_examples=benign_examples,
        harmful_grouped_vs_scaled=eval_grouped_vs,
        harmful_group_sizes=harmful_group_sizes,
        benign_vs_scaled=None,
        args=args,
    )

    last_loss = None
    last_grad_norm = None
    last_stats = None
    history: list[dict[str, float]] = []

    best_loss_value = None
    best_stats = None
    best_grad_norm = None
    best_state = None
    best_outer_local = None

    outer_iter = tqdm(
        range(num_steps),
        desc="outer_step",
        dynamic_ncols=True,
    )
    for outer_local in outer_iter:
        optimizer.zero_grad(set_to_none=True)

        harmful_sum_scalar = 0.0
        benign_sum_scalar = 0.0

        harmful_global_idx = 0
        benign_global_idx = 0

        for harmful_group, vs_group in zip(harmful_groups, grouped_vs):
            vs_with_zero = _build_steering_matrix_with_zero(vs_group)

            if len(harmful_group) > 0 and lambda_h > 0:
                per_example_scale_h = lambda_h / float(max(1, num_harmful))
                for ex in harmful_group:
                    ex_value = _single_example_harmful_nll_backward(
                        bundle=train_bundle,
                        example_idx=harmful_global_idx,
                        example=ex,
                        vs_with_zero=vs_with_zero,
                        args=args,
                        backward_scale=per_example_scale_h,
                    )
                    harmful_sum_scalar += float(ex_value)
                    harmful_global_idx += 1
            else:
                harmful_global_idx += len(harmful_group)

        if len(benign_list) > 0 and lambda_b > 0:
            per_example_scale_b = lambda_b / float(max(1, num_benign))
            for ex in benign_list:
                ex_value = _single_example_benign_kl_simple_backward(
                    bundle=train_bundle,
                    example_idx=benign_global_idx,
                    example=ex,
                    args=args,
                    backward_scale=per_example_scale_b,
                )
                benign_sum_scalar += float(ex_value)
                benign_global_idx += 1
        else:
            benign_global_idx += len(benign_list)

        params = [p for p in train_bundle.model.parameters() if p.requires_grad]
        if params:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.outer_grad_clip)
            optimizer.step()
        else:
            grad_norm = torch.tensor(0.0, device=dev)

        harmful_value = harmful_sum_scalar / max(1, num_harmful) if lambda_h > 0 else 0.0
        benign_value = benign_sum_scalar / max(1, num_benign) if lambda_b > 0 else 0.0
        loss_harmful_value = lambda_h * harmful_value
        loss_benign_value = lambda_b * benign_value
        loss_value = loss_harmful_value + loss_benign_value

        grad_norm_value = float(
            grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm
        )

        stats = {
            "loss": float(loss_value),
            "loss_harmful": float(loss_harmful_value),
            "loss_benign": float(loss_benign_value),
            "harmful_unweighted": float(harmful_value),
            "benign_unweighted": float(benign_value),
            "grad_norm": grad_norm_value,
            "num_harmful": num_harmful,
            "num_benign": num_benign,
            "num_groups": num_groups,
            "group_sizes": [int(x) for x in harmful_group_sizes],
            "num_dirs_total": total_dirs,
            "num_dirs_per_group": list(dirs_per_group),
            "num_steers_per_group_including_zero": [int(k + 1) for k in dirs_per_group],
            "num_steers_total_including_group_zeros": total_steers,
            "outer_local": int(outer_local),
        }

        history.append(dict(stats))
        last_loss = torch.tensor(loss_value, device=dev, dtype=model_dtype)
        last_grad_norm = grad_norm
        last_stats = stats

        if best_loss_value is None or loss_value < best_loss_value:
            best_loss_value = float(loss_value)
            best_stats = dict(stats)
            best_grad_norm = grad_norm.detach().clone() if torch.is_tensor(grad_norm) else grad_norm
            best_state = _clone_trainable_state(train_bundle.model)
            best_outer_local = outer_local

        outer_iter.set_postfix(
            loss=f"{stats['loss']:.6f}",
            harmful=f"{stats['loss_harmful']:.6f}",
            benign=f"{stats['loss_benign']:.6f}",
            grad=f"{stats['grad_norm']:.6f}",
            steers=int(stats["num_steers_total_including_group_zeros"]),
        )

        print(
            f"[outer_step] step={outer_local:03d} "
            f"loss={stats['loss']:.6f} "
            f"loss_harmful={stats['loss_harmful']:.6f} "
            f"loss_benign={stats['loss_benign']:.6f} "
            f"harmful_unweighted={stats['harmful_unweighted']:.6f} "
            f"benign_unweighted={stats['benign_unweighted']:.6f} "
            f"grad_norm={stats['grad_norm']:.6f} "
            f"num_steers={stats['num_steers_total_including_group_zeros']}"
        )

    if best_stats is None:
        best_stats = {
            "loss": 0.0,
            "loss_harmful": 0.0,
            "loss_benign": 0.0,
            "harmful_unweighted": 0.0,
            "benign_unweighted": 0.0,
            "grad_norm": 0.0,
            "num_harmful": num_harmful,
            "num_benign": num_benign,
            "num_groups": num_groups,
            "group_sizes": [int(x) for x in harmful_group_sizes],
            "num_dirs_total": total_dirs,
            "num_dirs_per_group": list(dirs_per_group),
            "num_steers_per_group_including_zero": [int(k + 1) for k in dirs_per_group],
            "num_steers_total_including_group_zeros": total_steers,
            "outer_local": -1,
        }

    best_stats["loss_start"] = float(start_eval.get("loss", 0.0))
    best_stats["loss_harmful_start"] = float(start_eval.get("loss_harmful", 0.0))
    best_stats["loss_benign_start"] = float(start_eval.get("loss_benign", 0.0))
    best_stats["harmful_unweighted_start"] = float(start_eval.get("harmful_unweighted", 0.0))
    best_stats["benign_unweighted_start"] = float(start_eval.get("benign_unweighted", 0.0))

    if best_state is not None:
        _restore_trainable_state(train_bundle.model, best_state)
        print(
            f"[outer_step] restore_best "
            f"best_outer_local={best_outer_local:03d} "
            f"best_loss={best_loss_value:.6f}"
        )

    end_eval = evaluate_outer_loss(
        train_bundle=train_bundle,
        harmful_examples=harmful_examples,
        benign_examples=benign_examples,
        harmful_grouped_vs_scaled=eval_grouped_vs,
        harmful_group_sizes=harmful_group_sizes,
        benign_vs_scaled=None,
        args=args,
    )

    best_stats["loss_end"] = float(end_eval.get("loss", 0.0))
    best_stats["loss_harmful_end"] = float(end_eval.get("loss_harmful", 0.0))
    best_stats["loss_benign_end"] = float(end_eval.get("loss_benign", 0.0))
    best_stats["harmful_unweighted_end"] = float(end_eval.get("harmful_unweighted", 0.0))
    best_stats["benign_unweighted_end"] = float(end_eval.get("benign_unweighted", 0.0))
    best_stats["eval_num_dirs_total"] = int(end_eval.get("num_dirs_total", 0))
    best_stats["eval_num_dirs_per_group"] = list(end_eval.get("num_dirs_per_group", []))
    best_stats["train_num_dirs_total"] = int(total_dirs)
    best_stats["train_num_dirs_per_group"] = list(dirs_per_group)

    best_loss = torch.tensor(
        best_loss_value if best_loss_value is not None else 0.0,
        device=dev,
        dtype=model_dtype,
    )

    return best_loss, best_stats, best_grad_norm, history