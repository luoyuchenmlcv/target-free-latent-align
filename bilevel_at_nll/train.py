from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model_activations import load_causal_lm

from .config import TrainSummary, build_parser
from .inner_step import run_inner
from .io_utils import MixedCsvRoundDataLoader, read_examples
from .logging_utils import (
    finalize_wandb,
    init_wandb,
    log_eval_batch_steer,
    log_inner_step,
    log_outer_step,
    log_inner_step_history,
)
from .outer_step import outer_step
from .peft_utils import attach_lora
from .test_utils import maybe_save_lora_checkpoint
from .validation import (
    evaluate_inner_step_batch_steer,
    load_strongreject_evaluator,
    save_inner_eval_results,
)

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import logging
logging.set_verbosity_error()


def _chunk_list(xs, chunk_size: int):
    chunk_size = max(1, int(chunk_size))
    return [list(xs[i : i + chunk_size]) for i in range(0, len(xs), chunk_size)]



def _safe_float_mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return float(np.mean(xs)) if xs else 0.0


def _save_inner_artifacts(out_dir: Path, round_idx: int, grouped_inner_results):
    root = out_dir / "inner_artifacts" / f"round_{int(round_idx):04d}"
    root.mkdir(parents=True, exist_ok=True)
    for group_idx, result in enumerate(grouped_inner_results):
        gdir = root / f"group_{int(group_idx):02d}"
        gdir.mkdir(parents=True, exist_ok=True)

        meta = {
            "group_idx": int(group_idx),
            "inner_method": str(result.get("inner_method", "nuclear")),
            "input_scale": float(result.get("input_scale", 0.0)),
            "num_steps": int(len(result.get("step_stats", []))),
            "mean_shift_avg_norm": float(result.get("mean_shift_avg_norm", 0.0)),
            "mean_shift_dir_sim": float(result.get("mean_shift_dir_sim", 0.0)),
            "v_self_sim": float(result.get("v_self_sim", 0.0)),
            "intra_prompt_consistency": float(result.get("intra_prompt_consistency", 0.0)),
        }
        (gdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        (gdir / "step_stats.json").write_text(json.dumps(result.get("step_stats", []), ensure_ascii=False, indent=2), encoding="utf-8")

        for name in ("U", "V", "V_scaled", "rank_scores", "rank_indices"):
            tensor = result.get(name)
            if torch.is_tensor(tensor):
                torch.save(tensor, gdir / f"{name}_final.pt")

        for i, tensor in enumerate(result.get("V_history", [])):
            torch.save(tensor, gdir / f"V_step_{i:04d}.pt")
        for i, tensor in enumerate(result.get("V_scaled_history", [])):
            torch.save(tensor, gdir / f"V_scaled_step_{i:04d}.pt")
        for i, tensor in enumerate(result.get("C_history", [])):
            torch.save(tensor, gdir / f"C_step_{i:04d}.pt")

def set_seed(seed: int = 325):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def train(args):
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("WANDB INIT")
    init_wandb(args)

    round_loader = MixedCsvRoundDataLoader(
        harmful_file=args.harmful_file,
        harmful_col=args.harmful_col,
        harmful_generation_col=args.harmful_generation_col,
        benign_file=args.benign_file,
        benign_col=args.benign_col,
        benign_generation_col=args.benign_generation_col,
        borderline_file=args.borderline_file,
        borderline_col=args.borderline_col,
        borderline_generation_col=args.borderline_generation_col,
        harmful_max_samples=args.harmful_max_samples,
        benign_max_samples=args.benign_max_samples,
        borderline_max_samples=args.borderline_max_samples,
        ratio_harmful=args.round_ratio_harmful,
        ratio_benign=args.round_ratio_benign,
        ratio_borderline=args.round_ratio_borderline,
        base_unit=args.round_base_unit,
        shuffle=False,
    )



    # Single-model version: only load one base model, then attach LoRA on it.
    train_bundle = load_causal_lm(
        args.model_name,
        tokenizer_name_or_path=args.tokenizer_name,
        device_map=args.device_map,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        system_prompt=args.system_prompt,
        layers_path=None,
    )

    train_bundle.model, peft_target_modules = attach_lora(
        train_bundle.model,
        peft_layers=args.peft_layers,
        r=args.peft_r,
        alpha=args.peft_alpha,
        dropout=args.peft_dropout,
    )
    train_bundle.model.train()

    trainable_params = [p for p in train_bundle.model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found after attaching LoRA.")

    validation_every = int(args.validation_every_round)
    strongreject = None
    if validation_every > 0:
        strongreject = load_strongreject_evaluator(
            args.strongreject_model_name,
            device_map=args.strongreject_device_map,
        )

    logs = []
    final_stats = None
    final_grad_norm = None
    final_val = None

    for round_idx in range(args.rounds):

        saved_ckpt = maybe_save_lora_checkpoint(
            train_bundle,
            out_dir,
            round_idx,
            getattr(args, "save_checkpoint_every_rounds", 1),
        )
        
        round_batch = round_loader.sample_round()

        harmful_batch_examples = round_batch.harmful_examples
        benign_batch_examples = round_batch.benign_examples
        harmful_batch_queries = [x["prompt"] for x in harmful_batch_examples]

        print(
            f"\n[round {round_idx}] "
            f"harmful={len(round_batch.harmful_examples)} "
            f"benign={len(round_batch.benign_pure_examples)} "
            f"borderline={len(round_batch.borderline_examples)} "
            f"merged_benign={len(round_batch.benign_examples)}"
        )

        # Inner step runs on grouped harmful prompts only.
        harmful_grouped_examples = _chunk_list(
            harmful_batch_examples,
            max(1, int(getattr(args, "harmful_inner_minibatch_size", len(harmful_batch_examples) or 1))),
        )
        harmful_grouped_queries = [
            [x["prompt"] for x in group]
            for group in harmful_grouped_examples
        ]
        harmful_group_sizes = [len(group) for group in harmful_grouped_examples]
        train_bundle.model.eval()

        grouped_inner_results = [
            run_inner(train_bundle, group_queries, args)
            for group_queries in harmful_grouped_queries
            if len(group_queries) > 0
        ]
        if not grouped_inner_results:
            raise ValueError("Grouped inner step produced no results; harmful batch is empty.")

        harmful_grouped_vs_scaled = [r["V_scaled"] for r in grouped_inner_results]
        benign_global_vs_scaled = torch.cat(harmful_grouped_vs_scaled, dim=1)

        inner_result = {
            "V_scaled": benign_global_vs_scaled,
            "num_groups": int(len(grouped_inner_results)),
            "group_sizes": list(harmful_group_sizes),
            "mean_shift_avg_norm": _safe_float_mean([r.get("mean_shift_avg_norm", 0.0) for r in grouped_inner_results]),
            "mean_shift_dir_sim": _safe_float_mean([r.get("mean_shift_dir_sim", 0.0) for r in grouped_inner_results]),
            "v_self_sim": _safe_float_mean([r.get("v_self_sim", 0.0) for r in grouped_inner_results]),
            "intra_prompt_consistency": _safe_float_mean([r.get("intra_prompt_consistency", 0.0) for r in grouped_inner_results]),
            "inner_method": str(getattr(args, "inner_method", "nuclear")),
        }
        log_inner_step(round_idx, inner_result, step=round_idx)
        flat_step_history = []
        for group_idx, r in enumerate(grouped_inner_results):
            for item in r.get("step_stats", []):
                row = dict(item)
                row["group_idx"] = int(group_idx)
                flat_step_history.append(row)
        log_inner_step_history(round_idx, flat_step_history, step=round_idx)
        _save_inner_artifacts(out_dir, round_idx, grouped_inner_results)

        inner_eval = None
        if strongreject is not None and round_idx % validation_every == 0:
            all_query_groups = _chunk_list(harmful_batch_queries, args.harmful_inner_minibatch_size)
            eval_limit = max(0, int(args.inner_eval_num_prompts))
            remaining = eval_limit
            selected_group_indices = []
            harmful_query_groups = []
            for group_idx, query_group in enumerate(all_query_groups):
                if remaining <= 0:
                    break
                selected_queries = list(query_group[:remaining])
                if selected_queries:
                    selected_group_indices.append(group_idx)
                    harmful_query_groups.append(selected_queries)
                    remaining -= len(selected_queries)

            vs_groups = [grouped_inner_results[i]["V_scaled"] for i in selected_group_indices]
            rank_indices_groups = [grouped_inner_results[i].get("rank_indices", None) for i in selected_group_indices]
            rank_scores_groups = [grouped_inner_results[i].get("rank_scores", None) for i in selected_group_indices]

            inner_eval = evaluate_inner_step_batch_steer(
                train_bundle,
                harmful_query_groups,
                vs_groups,
                strongreject,
                args,
                rank_indices_groups=rank_indices_groups,
                rank_scores_groups=rank_scores_groups,
            )
            final_val = inner_eval

            save_inner_eval_results(
                out_dir,
                round_idx,
                inner_eval,
                save_json=args.inner_eval_save_json,
                save_jsonl=args.inner_eval_save_jsonl,
            )
            log_eval_batch_steer(
                round_idx,
                inner_eval,
                prefix="inner_nuclear_eval",
                step=round_idx,
            )

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.outer_lr,
            weight_decay=args.outer_weight_decay,
        )
        train_bundle.model.train()

        _, final_stats, final_grad_norm, outer_history = outer_step(
            train_bundle=train_bundle,
            harmful_examples=harmful_batch_examples,
            benign_examples=benign_batch_examples,  # benign + borderline merged
            harmful_grouped_vs_scaled=harmful_grouped_vs_scaled,
            harmful_group_sizes=harmful_group_sizes,
            benign_vs_scaled=benign_global_vs_scaled,
            optimizer=optimizer,
            args=args,
        )
        train_bundle.model.eval()

        for hist in outer_history:
            row = {
                "type": "outer",
                "round": round_idx,
                "num_harmful_round": len(round_batch.harmful_examples),
                "num_benign_round": len(round_batch.benign_pure_examples),
                "num_borderline_round": len(round_batch.borderline_examples),
                "num_merged_benign_round": len(round_batch.benign_examples),
                **hist,
            }
            logs.append(row)

        if final_stats is not None:
            log_outer_step(round_idx, final_stats, step=round_idx)
            print(
                f"[round {round_idx}] "
                f"loss={final_stats['loss']:.6f} "
                f"loss_harmful={final_stats.get('loss_harmful', 0.0):.6f} "
                f"loss_benign={final_stats.get('loss_benign', 0.0):.6f}"
            )


        if saved_ckpt is not None:
            print(f"[round {round_idx}] saved LoRA checkpoint to: {saved_ckpt}")

    train_bundle.model.eval()

    adapter_dir = out_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    train_bundle.model.save_pretrained(adapter_dir)
    train_bundle.tokenizer.save_pretrained(adapter_dir)

    (out_dir / "logs.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    final_validation_mean_sr = (
        float(final_val["mean_strongreject"])
        if final_val is not None
        else None
    )

    summary = TrainSummary(
        model_name=args.model_name,
        source_layer_idx=args.source_layer_idx,
        target_layer_idx=args.target_layer_idx,
        rounds=args.rounds,
        outer_steps_per_round=args.outer_steps_per_round,
        peft_target_modules=peft_target_modules,
        final_validation_mean_sr=final_validation_mean_sr,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    wandb_final = {"num_logs": len(logs)}
    if final_validation_mean_sr is not None:
        wandb_final["final_validation_mean_sr"] = final_validation_mean_sr

    finalize_wandb(
        final_val=wandb_final,
        bank_size=int(getattr(args, "inner_num_factors", 0)),
        peft_target_modules=peft_target_modules,
    )



def main():
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
