from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

try:
    from .test_utils import (
        VLLMGenerationConfig,
        build_prompt_texts,
        ensure_dir,
        generate_with_vllm_lora,
        round_dir,
        save_csv,
        save_json,
        save_jsonl,
    )
except ImportError:
    from test_utils import (
        VLLMGenerationConfig,
        build_prompt_texts,
        ensure_dir,
        generate_with_vllm_lora,
        round_dir,
        save_csv,
        save_json,
        save_jsonl,
    )


PAIRWISE_JUDGE_PROMPT = """You are an expert evaluator for instruction-following assistants.

Given one user instruction and two candidate responses, decide which response is better overall for the user.
Judge based on helpfulness, correctness, relevance, completeness, and clarity.
Do not favor verbosity by itself. A longer answer is only better if it is actually more helpful.
Reply with exactly one token: "1" if Response 1 is better, or "2" if Response 2 is better.

Instruction:
{instruction}

Response 1:
{output_1}

Response 2:
{output_2}

Winner:"""


def infer_lora_adapter_path(out_dir: str | Path, round_idx: int) -> Path:
    path = Path(out_dir) / "lora_checkpoints" / f"round_{int(round_idx):04d}"
    if not path.exists():
        raise FileNotFoundError(
            f"Inferred LoRA adapter path does not exist: {path}. "
            "Make sure train.py has already saved the periodic checkpoint for this round."
        )
    return path


def _cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _safe_mean(xs: list[float]) -> float:
    return float(mean(xs)) if xs else float("nan")


def _safe_stderr(xs: list[float]) -> float:
    if len(xs) <= 1:
        return float("nan")
    arr = np.asarray(xs, dtype=np.float64)
    return float(arr.std(ddof=1) / math.sqrt(len(arr)))


def _sanitize_name(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    return out.strip("._") or "model"


def _generate_base_vllm(
    *,
    model_name_or_path: str,
    prompts: list[str],
    sampling: VLLMGenerationConfig,
) -> list[str]:
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=model_name_or_path,
        tokenizer=model_name_or_path,
        trust_remote_code=True,
        disable_log_stats=True,
        tensor_parallel_size=int(sampling.tensor_parallel_size),
        gpu_memory_utilization=float(sampling.gpu_memory_utilization),
        dtype=sampling.dtype,
    )
    if sampling.max_model_len is not None:
        llm_kwargs["max_model_len"] = int(sampling.max_model_len)

    llm = LLM(**llm_kwargs)
    params = SamplingParams(
        temperature=float(sampling.temperature),
        top_p=float(sampling.top_p),
        max_tokens=int(sampling.max_new_tokens),
    )
    outputs = llm.generate(prompts, params)
    texts = [out.outputs[0].text.strip() if out.outputs else "" for out in outputs]

    del llm
    _cleanup_memory()
    return texts


# def _load_alpacaeval_instructions(max_samples: int = 0) -> list[dict[str, Any]]:
#     ds = load_dataset(
#         "tatsu-lab/alpaca_eval",
#         "alpaca_eval_gpt4_baseline",
#         split="eval",
#         trust_remote_code=True,
#     )
#     rows = []
#     for i, ex in enumerate(ds):
#         rows.append(
#             {
#                 "instruction": str(ex["instruction"]),
#                 "dataset": ex.get("dataset", ""),
#                 "reference_output": ex.get("output", ""),
#                 "reference_generator": ex.get("generator", ""),
#                 "example_id": i,
#             }
#         )
#         if max_samples > 0 and len(rows) >= int(max_samples):
#             break
#     return rows


def _load_alpacaeval_instructions(max_samples: int = 0) -> list[dict[str, Any]]:
    """
    Load AlpacaEval prompts directly from the dataset repo files,
    without relying on `datasets.load_dataset`, because the repo
    still contains a legacy loading script (`alpaca_eval.py`) and
    newer `datasets` versions reject dataset scripts.
    """
    local_path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        repo_type="dataset",
        filename="alpaca_eval.json",
    )

    data = json.loads(Path(local_path).read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for i, ex in enumerate(data):
        rows.append(
            {
                "instruction": str(ex["instruction"]),
                "dataset": ex.get("dataset", ""),
                "reference_output": ex.get("output", ""),
                "reference_generator": ex.get("generator", ""),
                "example_id": i,
            }
        )
        if max_samples > 0 and len(rows) >= int(max_samples):
            break

    return rows


def _base_cache_paths(out_dir: str | Path, base_model_name_or_path: str) -> tuple[Path, Path]:
    cache_dir = ensure_dir(Path(out_dir) / "tests" / "alpaca_eval_cache")
    stem = _sanitize_name(base_model_name_or_path)
    return (
        cache_dir / f"base_outputs__{stem}.jsonl",
        cache_dir / f"base_outputs__{stem}.csv",
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))
    return rows


def _get_or_create_base_outputs(
    *,
    out_dir: str | Path,
    base_model_name_or_path: str,
    system_prompt: str | None,
    rows: list[dict[str, Any]],
    sampling: VLLMGenerationConfig,
) -> list[dict[str, Any]]:
    jsonl_path, csv_path = _base_cache_paths(out_dir, base_model_name_or_path)
    cached = _load_jsonl(jsonl_path)
    if cached and len(cached) >= len(rows):
        return cached[: len(rows)]

    raw_prompts = [r["instruction"] for r in rows]
    prompt_texts = build_prompt_texts(
        raw_prompts,
        base_model_name_or_path,
        system_prompt=system_prompt,
    )
    outputs = _generate_base_vllm(
        model_name_or_path=base_model_name_or_path,
        prompts=prompt_texts,
        sampling=sampling,
    )

    base_rows: list[dict[str, Any]] = []
    for src, output in zip(rows, outputs):
        base_rows.append(
            {
                "example_id": int(src["example_id"]),
                "instruction": src["instruction"],
                "dataset": src.get("dataset", ""),
                "output": output,
                "generator": _sanitize_name(base_model_name_or_path),
            }
        )

    save_jsonl(jsonl_path, base_rows)
    save_csv(csv_path, base_rows)
    return base_rows


def _seeded_swap_decision(instruction: str, idx: int) -> bool:
    h = hashlib.md5(f"{idx}||{instruction}".encode("utf-8")).hexdigest()
    return int(h, 16) % 2 == 1


def _build_pairwise_prompt(instruction: str, output_1: str, output_2: str) -> str:
    return PAIRWISE_JUDGE_PROMPT.format(
        instruction=instruction,
        output_1=output_1,
        output_2=output_2,
    )


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    name = str(dtype_name).strip().lower()
    if name in {"bf16", "bfloat16", "auto"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected single-token label for {text!r}, got ids={ids}")
    return int(ids[0])


def _judge_pairwise_weighted_preferences(
    *,
    judge_model_name_or_path: str,
    pairs: list[dict[str, Any]],
    batch_size: int = 8,
    judge_dtype: str = "bf16",
    judge_device_map: str = "auto",
) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(
        judge_model_name_or_path,
        trust_remote_code=True,
        padding_side="left",
        truncation_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        judge_model_name_or_path,
        trust_remote_code=True,
        device_map=judge_device_map,
        torch_dtype=_resolve_torch_dtype(judge_dtype),
    )
    model.eval()

    token_1 = _single_token_id(tokenizer, "1")
    token_2 = _single_token_id(tokenizer, "2")

    judged: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(pairs), batch_size), desc="alpacaeval_judge", dynamic_ncols=True):
        chunk = pairs[start : start + batch_size]
        prompts = []

        for row in chunk:
            prompts.append(
                _build_pairwise_prompt(
                    instruction=row["instruction"],
                    output_1=row["presented_output_1"],
                    output_2=row["presented_output_2"],
                )
            )

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits[:, -1, :]
            choice_logits = logits[:, [token_1, token_2]]
            probs = torch.softmax(choice_logits.float(), dim=-1)

        probs_np = probs.detach().cpu().numpy()
        for row, p in zip(chunk, probs_np):
            p1 = float(p[0])
            p2 = float(p[1])

            # preference convention follows AlpacaEval head2head:
            # preference in [1,2], and win_rate = mean(preference - 1)
            # where output_2 is the evaluated model output after de-randomization.
            if row["is_swapped"]:
                # presented output_1 is candidate, presented output_2 is base
                # p2 = P(base wins in presented order) => candidate wins prob = p1
                candidate_win_prob = p1
            else:
                # presented output_1 is base, presented output_2 is candidate
                candidate_win_prob = p2

            judged_row = dict(row)
            judged_row["judge_prob_presented_1"] = p1
            judged_row["judge_prob_presented_2"] = p2
            judged_row["candidate_win_prob"] = float(candidate_win_prob)
            judged_row["preference"] = 1.0 + float(candidate_win_prob)
            judged.append(judged_row)

    del model
    _cleanup_memory()
    return judged


def _char_len(text: str) -> int:
    return len(str(text).strip())


def _fit_length_controlled_preferences(
    judged_rows: list[dict[str, Any]],
    *,
    num_steps: int = 2000,
    lr: float = 0.1,
    reg_instr: float = 1e-2,
    reg_beta: float = 1e-2,
    reg_alpha: float = 1e-4,
    device: str | None = None,
) -> dict[str, Any]:
    n = len(judged_rows)
    if n == 0:
        return {
            "lc_preferences": [],
            "raw_preferences": [],
            "length_deltas": [],
            "raw_win_rate": float("nan"),
            "length_controlled_win_rate": float("nan"),
            "standard_error": float("nan"),
            "length_controlled_standard_error": float("nan"),
        }

    raw_pref = np.asarray([float(r["candidate_win_prob"]) for r in judged_rows], dtype=np.float32)
    base_len = np.asarray([_char_len(r["base_output"]) for r in judged_rows], dtype=np.float32)
    cand_len = np.asarray([_char_len(r["candidate_output"]) for r in judged_rows], dtype=np.float32)
    delta_len = cand_len - base_len

    std = float(delta_len.std()) if float(delta_len.std()) > 1e-8 else 1.0
    length_feat = np.tanh(delta_len / std).astype(np.float32)

    # one instruction id per row; with regularization this still gives a stable
    # fixed-baseline approximation to the official LC fit.
    instr_ids = np.arange(n, dtype=np.int64)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    y = torch.tensor(raw_pref, dtype=torch.float32, device=device)
    x = torch.tensor(length_feat, dtype=torch.float32, device=device)
    idx = torch.tensor(instr_ids, dtype=torch.long, device=device)

    alpha = torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device=device))
    beta = torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device=device))
    gamma = torch.nn.Parameter(torch.zeros((n,), dtype=torch.float32, device=device))

    opt = torch.optim.Adam([alpha, beta, gamma], lr=lr)

    for _ in range(int(num_steps)):
        opt.zero_grad()
        logits = alpha + beta * x + gamma[idx]
        pred = torch.sigmoid(logits)

        bce = F.binary_cross_entropy(pred, y)
        reg = reg_alpha * alpha.pow(2) + reg_beta * beta.pow(2) + reg_instr * gamma.pow(2).mean()
        loss = bce + reg
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = alpha + beta * x + gamma[idx]
        pred = torch.sigmoid(logits)
        lc_pred = torch.sigmoid(alpha + gamma[idx])

    raw_preferences = pred.detach().cpu().numpy().astype(float).tolist()
    lc_preferences = lc_pred.detach().cpu().numpy().astype(float).tolist()

    return {
        "lc_preferences": lc_preferences,
        "raw_preferences": raw_preferences,
        "length_deltas": delta_len.astype(float).tolist(),
        "raw_win_rate": float(np.mean(raw_pref) * 100.0),
        "length_controlled_win_rate": float(np.mean(lc_pred.detach().cpu().numpy()) * 100.0),
        "standard_error": float(np.std(raw_pref, ddof=1) / math.sqrt(n) * 100.0) if n > 1 else float("nan"),
        "length_controlled_standard_error": float(np.std(lc_pred.detach().cpu().numpy(), ddof=1) / math.sqrt(n) * 100.0) if n > 1 else float("nan"),
        "length_feature_std": float(std),
    }


def run_general_utility_test(
    *,
    round_idx: int,
    out_dir: str | Path,
    base_model_name_or_path: str,
    system_prompt: str | None = None,
    max_samples: int = 0,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    max_model_len: int | None = None,
    vllm_dtype: str = "auto",
    judge_model_name_or_path: str = "/workspace/models/Qwen2.5-7B-Instruct",
    judge_batch_size: int = 8,
    judge_dtype: str = "bf16",
    judge_device_map: str = "auto",
) -> dict[str, Any]:
    test_dir = ensure_dir(round_dir(out_dir, round_idx) / "alpaca_eval")
    lora_adapter_path = infer_lora_adapter_path(out_dir, round_idx)

    eval_rows = _load_alpacaeval_instructions(max_samples=max_samples)

    gen_sampling = VLLMGenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=vllm_dtype,
    )

    base_rows = _get_or_create_base_outputs(
        out_dir=out_dir,
        base_model_name_or_path=base_model_name_or_path,
        system_prompt=system_prompt,
        rows=eval_rows,
        sampling=gen_sampling,
    )

    raw_prompts = [r["instruction"] for r in eval_rows]
    prompt_texts = build_prompt_texts(
        raw_prompts,
        base_model_name_or_path,
        system_prompt=system_prompt,
    )

    lora_outputs = generate_with_vllm_lora(
        base_model_name_or_path=base_model_name_or_path,
        tokenizer_name_or_path=base_model_name_or_path,
        prompts=prompt_texts,
        lora_adapter_path=lora_adapter_path,
        sampling=gen_sampling,
        lora_name=f"alpaca_eval_round_{int(round_idx):04d}",
        lora_int_id=int(round_idx) + 1,
    )

    lora_rows: list[dict[str, Any]] = []
    for src, out in zip(eval_rows, lora_outputs):
        lora_rows.append(
            {
                "example_id": int(src["example_id"]),
                "instruction": src["instruction"],
                "dataset": src.get("dataset", ""),
                "output": out,
                "generator": f"lora_round_{int(round_idx):04d}",
            }
        )

    save_jsonl(test_dir / "lora_outputs.jsonl", lora_rows)
    save_csv(test_dir / "lora_outputs.csv", lora_rows)
    _cleanup_memory()

    pairs: list[dict[str, Any]] = []
    for base_row, cand_row in zip(base_rows, lora_rows):
        instruction = str(base_row["instruction"])
        base_output = str(base_row["output"])
        candidate_output = str(cand_row["output"])

        is_swapped = _seeded_swap_decision(instruction, int(base_row["example_id"]))
        if is_swapped:
            presented_output_1 = candidate_output
            presented_output_2 = base_output
        else:
            presented_output_1 = base_output
            presented_output_2 = candidate_output

        pairs.append(
            {
                "example_id": int(base_row["example_id"]),
                "instruction": instruction,
                "dataset": base_row.get("dataset", ""),
                "base_output": base_output,
                "candidate_output": candidate_output,
                "presented_output_1": presented_output_1,
                "presented_output_2": presented_output_2,
                "is_swapped": bool(is_swapped),
            }
        )

    judged_rows = _judge_pairwise_weighted_preferences(
        judge_model_name_or_path=judge_model_name_or_path,
        pairs=pairs,
        batch_size=judge_batch_size,
        judge_dtype=judge_dtype,
        judge_device_map=judge_device_map,
    )

    lc = _fit_length_controlled_preferences(judged_rows)

    final_rows: list[dict[str, Any]] = []
    for row, lc_pref, fitted_raw in zip(
        judged_rows,
        lc["lc_preferences"],
        lc["raw_preferences"],
    ):
        out = dict(row)
        out["fitted_raw_candidate_win_prob"] = float(fitted_raw)
        out["length_controlled_candidate_win_prob"] = float(lc_pref)
        out["base_length"] = int(_char_len(row["base_output"]))
        out["candidate_length"] = int(_char_len(row["candidate_output"]))
        out["length_delta"] = int(out["candidate_length"] - out["base_length"])
        final_rows.append(out)

    save_jsonl(test_dir / "pairwise_annotations.jsonl", final_rows)
    save_csv(test_dir / "pairwise_annotations.csv", final_rows)

    summary = {
        "round_idx": int(round_idx),
        "num_samples": int(len(final_rows)),
        "base_model_name_or_path": str(base_model_name_or_path),
        "judge_model_name_or_path": str(judge_model_name_or_path),
        "lora_adapter_path": str(lora_adapter_path),
        "win_rate": float(_safe_mean([r["candidate_win_prob"] for r in final_rows]) * 100.0),
        "length_controlled_win_rate": float(lc["length_controlled_win_rate"]),
        "standard_error": float(_safe_stderr([r["candidate_win_prob"] for r in final_rows]) * 100.0) if final_rows else float("nan"),
        "length_controlled_standard_error": float(lc["length_controlled_standard_error"]),
        "avg_base_length": float(_safe_mean([float(_char_len(r["base_output"])) for r in final_rows])),
        "avg_candidate_length": float(_safe_mean([float(_char_len(r["candidate_output"])) for r in final_rows])),
        "avg_length_delta": float(_safe_mean([float(_char_len(r["candidate_output"]) - _char_len(r["base_output"])) for r in final_rows])),
        "base_outputs_cache": str(_base_cache_paths(out_dir, base_model_name_or_path)[0]),
        "notes": {
            "raw_win_rate_formula": "mean(preference - 1) * 100, matching AlpacaEval's head2head convention.",
            "length_controlled_method": "Fixed-baseline logistic correction using model intercept + tanh-normalized length term + regularized instruction effects, then zeroing the length term.",
        },
    }

    save_json(test_dir / "summary.json", summary)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run AlpacaEval-style general utility test with cached base generations, LoRA generations, local judge, and length-controlled win rate.")
    p.add_argument("--round_idx", type=int, default=0)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--base_model_name_or_path", type=str, required=True)
    p.add_argument("--system_prompt", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=0)

    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    p.add_argument("--max_model_len", type=int, default=0)
    p.add_argument("--vllm_dtype", type=str, default="auto")

    p.add_argument("--judge_model_name_or_path", type=str, required=True)
    p.add_argument("--judge_batch_size", type=int, default=8)
    p.add_argument("--judge_dtype", type=str, default="bf16")
    p.add_argument("--judge_device_map", type=str, default="auto")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    max_model_len = None if int(args.max_model_len) <= 0 else int(args.max_model_len)

    summary = run_general_utility_test(
        round_idx=args.round_idx,
        out_dir=args.out_dir,
        base_model_name_or_path=args.base_model_name_or_path,
        system_prompt=args.system_prompt,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        vllm_dtype=args.vllm_dtype,
        judge_model_name_or_path=args.judge_model_name_or_path,
        judge_batch_size=args.judge_batch_size,
        judge_dtype=args.judge_dtype,
        judge_device_map=args.judge_device_map,
    )

    summary_path = Path(args.out_dir) / "tests" / f"round_{int(args.round_idx):04d}" / "alpaca_eval" / "summary.json"
    print("[test_general_utility] done")
    print(f"[test_general_utility] summary_path={summary_path}")
    print(
        "[test_general_utility] "
        f"win_rate={summary['win_rate']:.4f}, "
        f"length_controlled_win_rate={summary['length_controlled_win_rate']:.4f}, "
        f"num_samples={summary['num_samples']}"
    )


if __name__ == "__main__":
    main()
