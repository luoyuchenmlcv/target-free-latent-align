from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .generation import generate_clean, generate_with_batch_prefill_steer

# external file in project root
from strong_reject import StrongREJECTEvaluator


def load_strongreject_evaluator(model_name: str, device_map: str = "auto") -> StrongREJECTEvaluator:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", truncation_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return StrongREJECTEvaluator(model, tokenizer)


@torch.no_grad()
def validate_styled(
    bundle,
    styled_queries: Sequence[str],
    strongreject: StrongREJECTEvaluator,
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    scores: List[float] = []

    pbar = tqdm(styled_queries, desc="validation", dynamic_ncols=True)
    for q in pbar:
        out = generate_clean(bundle, q, max_prompt_tokens, max_new_tokens, do_sample, temperature)
        sr = strongreject.calculate_score(q, out)
        rows.append({"query": q, "output": out, "strongreject": float(sr)})
        scores.append(float(sr))
        running_mean = float(sum(scores) / len(scores))
        pbar.set_postfix(running_mean_sr=f"{running_mean:.4f}")

    return {
        "mean_strongreject": float(sum(scores) / max(1, len(scores))),
        "rows": rows,
    }


def _short_query(text: str, max_len: int = 72) -> str:
    s = " ".join(str(text).split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _score_outputs_for_query(
    strongreject: StrongREJECTEvaluator,
    query: str,
    outputs: Sequence[str],
) -> List[float]:
    pairs = [(query, out) for out in outputs]
    if hasattr(strongreject, "calculate_score_batch"):
        try:
            scores = strongreject.calculate_score_batch(
                pairs,
                mini_batch_size=8,
                return_single=True,
            )
            # StrongREJECT returns a NumPy array when return_single=True.
            # Accept any sized iterable instead of silently falling back to
            # the single-example path.
            if hasattr(scores, "__len__") and len(scores) == len(outputs):
                return [float(x) for x in scores]
        except Exception:
            pass
    return [float(strongreject.calculate_score(query, out)) for out in outputs]


def _tensor_to_order(x, num_dirs: int) -> list[int]:
    if x is None:
        return list(range(num_dirs))
    if torch.is_tensor(x):
        x = x.detach().cpu().long().tolist()
    order = [int(i) for i in x]
    order = [i for i in order if 0 <= i < num_dirs]
    seen = set(order)
    order.extend([i for i in range(num_dirs) if i not in seen])
    return order[:num_dirs]


def _tensor_to_float_list(x, expected_len: int) -> list[float | None]:
    if x is None:
        return [None] * expected_len
    if torch.is_tensor(x):
        vals = x.detach().cpu().float().tolist()
    else:
        vals = list(x)
    vals = [float(v) for v in vals[:expected_len]]
    vals.extend([None] * max(0, expected_len - len(vals)))
    return vals


@torch.no_grad()
def evaluate_inner_step_batch_steer(
    bundle,
    query_groups,              # list[list[str]] or list[str]
    vs_groups,                 # list[Tensor] each [D,K], or Tensor [D,K]
    strongreject: StrongREJECTEvaluator,
    args,
    rank_indices_groups=None,  # list[Tensor] each [K], sorted high -> low
    rank_scores_groups=None,   # list[Tensor] each [K], aligned with rank_indices
) -> Dict[str, Any]:
    """
    Vector-major global inner batch-steer eval.

    This function guarantees the file-level structure the experiment needs:
        K source directions + N prompts  ->  K records, not K*N records.

    Important semantics
    -------------------
    1. We still generate efficiently in prompt-major order:
          for each prompt: batch-generate K steered outputs.
    2. After generation, we TRANSPOSE the results into vector-major records:
          record[rank_pos] contains outputs for all prompts under that ranked v slot.
    3. If query_groups/vs_groups contains multiple groups, we aggregate by `rank_pos` globally:
          record rank_pos=r collects prompt outputs from every group using that group's r-th ranked vector.
       This avoids writing len(groups) * K records. For 128 vectors and 12 prompts, the saved JSON/JSONL
       will contain exactly 128 records.

    Returned keys
    -------------
    - records: nested vector-major records, one record per rank_pos.
    - rows: flat compatibility rows, one row per (rank_pos, prompt) pair. These are NOT used by
      save_inner_eval_results for the main JSON/JSONL.
    """
    # -----------------------------
    # Normalize inputs
    # -----------------------------
    if torch.is_tensor(vs_groups):
        vs_groups = [vs_groups]
    else:
        vs_groups = list(vs_groups)

    # query_groups may be list[str] or list[list[str]]
    if len(query_groups) > 0 and isinstance(query_groups[0], str):
        query_groups = [list(query_groups)]
    else:
        query_groups = [list(g) for g in query_groups]

    if len(query_groups) != len(vs_groups):
        raise ValueError(
            f"query_groups and vs_groups must have the same number of groups, "
            f"got {len(query_groups)} vs {len(vs_groups)}. "
            "If you want one global V evaluated on all prompts, pass query_groups=[all_queries] and vs_groups=[V]."
        )

    num_groups = len(vs_groups)
    if num_groups == 0:
        return {"mean_strongreject": 0.0, "records": [], "rows": []}

    num_dirs_per_group = [int(v.shape[1]) for v in vs_groups]
    if len(set(num_dirs_per_group)) != 1:
        raise ValueError(
            f"All V groups must have the same number of directions for global vector-major saving, "
            f"got {num_dirs_per_group}."
        )
    num_dirs = num_dirs_per_group[0]

    # Resolve per-group rank order and scores.
    orders: list[list[int]] = []
    rank_scores_by_group: list[list[float | None]] = []
    for group_idx in range(num_groups):
        rank_indices = None
        rank_scores = None
        if rank_indices_groups is not None and group_idx < len(rank_indices_groups):
            rank_indices = rank_indices_groups[group_idx]
        if rank_scores_groups is not None and group_idx < len(rank_scores_groups):
            rank_scores = rank_scores_groups[group_idx]
        orders.append(_tensor_to_order(rank_indices, num_dirs))
        rank_scores_by_group.append(_tensor_to_float_list(rank_scores, num_dirs))

    # -----------------------------
    # Initialize exactly K records.
    # -----------------------------
    records: list[dict[str, Any]] = []
    for rank_pos in range(num_dirs):
        original_indices_by_group = {
            str(group_idx): int(orders[group_idx][rank_pos])
            for group_idx in range(num_groups)
        }
        scores_this_rank = [rank_scores_by_group[g][rank_pos] for g in range(num_groups)]
        valid_scores = [float(x) for x in scores_this_rank if x is not None]
        mean_rank_score = float(sum(valid_scores) / len(valid_scores)) if valid_scores else None

        records.append({
            # For single-group eval this is the real group id. For multi-group eval,
            # this record aggregates all groups by the same rank_pos.
            "group_idx": 0 if num_groups == 1 else "all",
            "group_indices": [int(g) for g in range(num_groups)],
            "rank_pos": int(rank_pos),
            # Compatibility field: valid when num_groups == 1; for multi-group it is the first group's index.
            "source_vec_original_idx": int(original_indices_by_group["0"]),
            # Explicit multi-group mapping. Use this when num_groups > 1.
            "source_vec_original_indices_by_group": original_indices_by_group,
            "rank_score": mean_rank_score,
            "rank_scores_by_group": {
                str(g): (None if rank_scores_by_group[g][rank_pos] is None else float(rank_scores_by_group[g][rank_pos]))
                for g in range(num_groups)
            },
            "mean_strongreject": 0.0,
            "num_prompts": 0,
            "prompts": [],
        })

    rows: list[dict[str, Any]] = []
    all_scores: list[float] = []
    global_prompt_idx = 0

    outer_pbar = tqdm(
        range(num_groups),
        desc="inner_steered_eval_groups",
        dynamic_ncols=True,
        position=0,
    )

    # -----------------------------
    # Generate prompt-major, append vector-major.
    # -----------------------------
    for group_idx in outer_pbar:
        group_queries = list(query_groups[group_idx])
        vs = vs_groups[group_idx]  # [D, K]
        order = orders[group_idx]
        order_t = torch.tensor(order, dtype=torch.long, device=vs.device)
        vs_ranked = vs[:, order_t]  # [D, K] sorted high -> low for this group

        inner_pbar = tqdm(
            enumerate(group_queries),
            total=len(group_queries),
            desc=f"group_{group_idx:02d}_queries",
            dynamic_ncols=True,
            position=1,
            leave=False,
        )

        for local_prompt_idx, q in inner_pbar:
            outputs = generate_with_batch_prefill_steer(
                bundle=bundle,
                query=q,
                vs=vs_ranked,
                source_layer_idx=args.source_layer_idx,
                steer_positions=args.steer_positions,
                steer_prompt_content_only=args.steer_prompt_content_only,
                max_prompt_tokens=args.max_prompt_tokens,
                max_new_tokens=args.inner_eval_max_new_tokens,
                do_sample=args.validation_do_sample,
                temperature=args.validation_temperature,
            )
            scores = _score_outputs_for_query(strongreject, q, outputs)
            if len(outputs) != num_dirs or len(scores) != num_dirs:
                raise RuntimeError(
                    f"Expected {num_dirs} outputs/scores, got outputs={len(outputs)}, scores={len(scores)}"
                )

            all_scores.extend(float(x) for x in scores)
            mean_sr_prompt = float(sum(scores) / max(1, len(scores)))
            inner_pbar.set_postfix(
                query=_short_query(q),
                mean_sr=f"{mean_sr_prompt:.4f}",
                num_dirs=num_dirs,
            )

            for rank_pos in range(num_dirs):
                original_idx = int(order[rank_pos])
                rank_score = rank_scores_by_group[group_idx][rank_pos]
                sr = float(scores[rank_pos])
                output = outputs[rank_pos]

                prompt_record = {
                    "global_prompt_idx": int(global_prompt_idx),
                    "group_idx": int(group_idx),
                    "prompt_idx": int(local_prompt_idx),
                    "query": q,
                    "output": output,
                    "strongreject": sr,
                    # These two fields disambiguate multi-group eval.
                    "source_vec_original_idx": original_idx,
                    "rank_score": rank_score,
                }
                records[rank_pos]["prompts"].append(prompt_record)

                rows.append({
                    "rank_pos": int(rank_pos),
                    "group_idx": int(group_idx),
                    "prompt_idx": int(local_prompt_idx),
                    "global_prompt_idx": int(global_prompt_idx),
                    "source_vec_original_idx": original_idx,
                    "rank_score": rank_score,
                    "query": q,
                    "output": output,
                    "strongreject": sr,
                })

            global_prompt_idx += 1

        running_mean_sr = float(sum(all_scores) / max(1, len(all_scores)))
        outer_pbar.set_postfix(
            group=group_idx,
            group_size=len(group_queries),
            num_dirs=num_dirs,
            running_mean_sr=f"{running_mean_sr:.4f}",
        )

    # -----------------------------
    # Finalize each v record.
    # -----------------------------
    for record in records:
        v_scores = [float(p["strongreject"]) for p in record["prompts"]]
        record["num_prompts"] = int(len(record["prompts"]))
        record["mean_strongreject"] = float(sum(v_scores) / max(1, len(v_scores)))

    overall = float(sum(all_scores) / max(1, len(all_scores)))
    return {
        "mean_strongreject": overall,
        "records": records,
        "rows": rows,
    }


def save_inner_eval_results(
    out_dir: Path,
    inner_round: int,
    inner_eval: Dict[str, Any],
    *,
    save_json: bool,
    save_jsonl: bool,
) -> None:
    """
    Save ONLY the nested vector-major records as the main artifacts.

    Guarantees:
    - inner_eval_XXX.json has exactly K records under payload["records"].
    - inner_eval_XXX.jsonl has exactly K lines, one nested record per ranked v.

    The flat compatibility rows are returned by evaluate_inner_step_batch_steer for logging,
    but are intentionally NOT saved here, because they caused confusion with K*N records.
    """
    target_dir = Path(out_dir) / "inner_eval"
    target_dir.mkdir(parents=True, exist_ok=True)

    records = inner_eval.get("records", [])

    if save_json:
        payload = {
            "inner_round": int(inner_round),
            "mean_strongreject": float(inner_eval["mean_strongreject"]),
            "num_records": int(len(records)),
            "records": records,
        }
        (target_dir / f"inner_eval_{inner_round:03d}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if save_jsonl:
        path = target_dir / f"inner_eval_{inner_round:03d}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                payload = {"inner_round": int(inner_round), **record}
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
