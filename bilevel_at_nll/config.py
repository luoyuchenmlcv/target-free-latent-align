# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class TrainSummary:
    model_name: str
    source_layer_idx: int
    target_layer_idx: int
    rounds: int
    outer_steps_per_round: int
    peft_target_modules: list[str]
    final_validation_mean_sr: float | None


def parse_positions(s: str | None):
    if s is None:
        return None
    ss = str(s).strip().lower()
    if ss in {"", "all", "none"}:
        return None
    if ss == "last":
        return slice(-1, None)
    if ":" in ss:
        a, b = ss.split(":", 1)
        start = int(a) if a != "" else None
        stop = int(b) if b != "" else None
        return slice(start, stop)
    if "," in ss:
        return [int(x.strip()) for x in ss.split(",") if x.strip()]
    return [int(ss)]



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Target-free Latent Safety Alignment")

    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--tokenizer_name", type=str, default=None)
    p.add_argument("--system_prompt", type=str, default=None)

    # harmful csv
    p.add_argument("--harmful_file", type=str, required=True)
    p.add_argument("--harmful_col", type=str, default="prompt")
    p.add_argument("--harmful_generation_col", type=str, default="generation")

    # benign csv
    p.add_argument("--benign_file", type=str, required=True)
    p.add_argument("--benign_col", type=str, default="prompt")
    p.add_argument("--benign_generation_col", type=str, default="generation")

    # borderline csv
    p.add_argument("--borderline_file", type=str, required=True)
    p.add_argument("--borderline_col", type=str, default="prompt")
    p.add_argument("--borderline_generation_col", type=str, default="generation")

    # validation
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--source_layer_idx", type=int, required=True)
    p.add_argument("--target_layer_idx", type=int, required=True)
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="bf16")
    p.add_argument("--attn_implementation", type=str, default="eager")

    # main rounds
    p.add_argument("--rounds", type=int, default=5)

    # per-round sampler
    p.add_argument("--round_ratio_harmful", type=int, default=5)
    p.add_argument("--round_ratio_benign", type=int, default=5)
    p.add_argument("--round_ratio_borderline", type=int, default=2)
    p.add_argument("--round_base_unit", type=int, default=1)

    # inner step
    p.add_argument("--inner_num_samples", type=int, default=32)
    p.add_argument("--inner_max_seq_len", type=int, default=256)
    p.add_argument("--inner_forward_batch_size", type=int, default=1)
    p.add_argument("--inner_backward_batch_size", type=int, default=1)
    p.add_argument("--inner_num_factors", type=int, default=128)
    p.add_argument("--inner_factor_batch_size", type=int, default=64)
    p.add_argument("--inner_dim_output_projection", type=int, default=32)
    p.add_argument("--inner_num_iters", type=int, default=10)
    p.add_argument("--inner_init", type=str, default="jacobian", choices=["random", "similarity"])
    p.add_argument("--inner_beta", type=float, default=1.0)
    p.add_argument("--inner_input_scale", type=float, default=-1.0)
    p.add_argument("--inner_calibration_target_ratio", type=float, default=0.5)
    p.add_argument("--inner_token_idxs", type=str, default="-3:")
    p.add_argument("--inner_method", type=str, default="nuclear", choices=["nuclear"])
    p.add_argument("--inner_lr", type=float, default=1e-2)

    # grouped harmful inner step
    p.add_argument("--harmful_inner_minibatch_size", type=int, default=12)

    # outer step
    p.add_argument("--outer_steps_per_round", type=int, default=20)
    p.add_argument("--outer_lr", type=float, default=5e-4)
    p.add_argument("--outer_weight_decay", type=float, default=0.0)
    p.add_argument("--outer_grad_clip", type=float, default=1.0)
    p.add_argument("--steer_chunk_size", type=int, default=32)

    # peft
    p.add_argument("--peft_layers", type=str, default="all")
    p.add_argument("--peft_r", type=int, default=16)
    p.add_argument("--peft_alpha", type=int, default=32)
    p.add_argument("--peft_dropout", type=float, default=0.05)

    # losses
    p.add_argument("--lambda_nuclear", type=float, default=1.0)
    p.add_argument("--lambda_harmful", type=float, default=1.0)
    p.add_argument("--lambda_benign", type=float, default=1.0)

    # causal ablation: train with V subset, evaluate NLL proxy with full V
    p.add_argument("--v_subset_mode", type=str, default="all", choices=["no_steer", "all", "top_similar", "top_dissimilar"])
    p.add_argument(
        "--v_subset_k",
        type=int,
        default=0,
        help="0 means use all V. Otherwise select K directions per harmful group for outer-step training.",
    )
    p.add_argument(
        "--save_nll_curve",
        action="store_true",
        help="Save full-V NLL proxy start/end stats for causal ablation.",
    )

    # steering / prompt
    p.add_argument("--steer_positions", type=str, default="all", choices=["last", "all", "content"])
    p.add_argument("--steer_prompt_content_only", action="store_true")
    p.add_argument("--max_prompt_tokens", type=int, default=64)

    # inner eval
    p.add_argument("--inner_eval_num_prompts", type=int, default=8)
    p.add_argument("--inner_eval_max_new_tokens", type=int, default=128)
    p.add_argument("--inner_eval_save_json", action="store_true")
    p.add_argument("--inner_eval_save_jsonl", action="store_true")
    p.add_argument("--inner_eval_vector_batch_size", type=int, default=64)

    # validation
    p.add_argument("--validation_every_round", type=int, default=1)
    p.add_argument("--validation_max_new_tokens", type=int, default=128)
    p.add_argument("--validation_do_sample", action="store_true")
    p.add_argument("--validation_temperature", type=float, default=1.0)
    p.add_argument("--strongreject_model_name", type=str, default="qylu4156/strongreject-15k-v1")
    p.add_argument("--strongreject_device_map", type=str, default="auto")

    # checkpointing for later vLLM+LoRA tests
    p.add_argument(
        "--save_checkpoint_every_rounds",
        type=int,
        default=1,
        help=(
            "If > 0, save a LoRA checkpoint to "
            "{out_dir}/lora_checkpoints/round_XXXX every K rounds. "
            "Round index in the folder name matches train.py's 0-based round_idx."
        ),
    )

    # dataset trimming
    p.add_argument("--harmful_max_samples", type=int, default=0)
    p.add_argument("--benign_max_samples", type=int, default=0)
    p.add_argument("--borderline_max_samples", type=int, default=0)
    p.add_argument("--validation_max_samples", type=int, default=0)

    # misc
    p.add_argument("--seed", type=int, default=42)

    # wandb
    p.add_argument("--wandb_project", type=str, default="bilevel-at")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="disabled")
    p.add_argument("--wandb_tags", type=str, default="")

    return p
