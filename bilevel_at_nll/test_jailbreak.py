from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
from statistics import mean
from typing import Any

from tqdm import tqdm

try:
    from .test_utils import (
        VLLMGenerationConfig,
        attack_name_from_path,
        build_prompt_texts,
        ensure_dir,
        parse_csv_path_list,
        read_prompt_rows,
        round_dir,
        save_csv,
        save_json,
        save_jsonl,
    )
    from .validation import load_strongreject_evaluator
except ImportError:
    from test_utils import (
        VLLMGenerationConfig,
        attack_name_from_path,
        build_prompt_texts,
        ensure_dir,
        parse_csv_path_list,
        read_prompt_rows,
        round_dir,
        save_csv,
        save_json,
        save_jsonl,
    )
    from validation import load_strongreject_evaluator


def infer_lora_adapter_path(out_dir: str | Path, round_idx: int) -> Path:
    """
    Infer the periodic LoRA checkpoint path from:
        {out_dir}/lora_checkpoints/round_{round_idx:04d}
    """
    path = Path(out_dir) / "lora_checkpoints" / f"round_{int(round_idx):04d}"
    if not path.exists():
        raise FileNotFoundError(
            f"Inferred LoRA adapter path does not exist: {path}. "
            "Make sure train.py has already saved the periodic checkpoint for this round."
        )
    return path


def _score_pairs_one_by_one(strongreject, pairs: list[tuple[str, str]]) -> list[float]:
    scores: list[float] = []
    for prompt, response in tqdm(pairs, desc="strongreject", dynamic_ncols=True, leave=False):
        scores.append(float(strongreject.calculate_score(prompt, response)))
    return scores


def _generate_with_vllm_lora_once(
    *,
    base_model_name_or_path: str,
    tokenizer_name_or_path: str | None,
    prompts: list[str],
    lora_adapter_path: str | Path | None,
    sampling: VLLMGenerationConfig,
    lora_name: str = "round_adapter",
    lora_int_id: int = 1,
) -> list[str]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if not prompts:
        return []

    llm_kwargs = {
        "model": base_model_name_or_path,
        "enable_lora": lora_adapter_path is not None,
        "tensor_parallel_size": int(sampling.tensor_parallel_size),
        "gpu_memory_utilization": float(sampling.gpu_memory_utilization),
        "dtype": sampling.dtype,
        "trust_remote_code": bool(sampling.trust_remote_code),
        "enforce_eager": bool(sampling.enforce_eager),
        "swap_space": float(sampling.swap_space),
        "max_lora_rank": 128,

    }
    if tokenizer_name_or_path:
        llm_kwargs["tokenizer"] = tokenizer_name_or_path
    if sampling.max_model_len is not None:
        llm_kwargs["max_model_len"] = int(sampling.max_model_len)
    if sampling.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = int(sampling.max_num_seqs)

    llm = LLM(**llm_kwargs)
    sp = SamplingParams(
        max_tokens=int(sampling.max_new_tokens),
        temperature=float(sampling.temperature),
        top_p=float(sampling.top_p),
    )

    lora_request = None
    if lora_adapter_path is not None:
        lora_request = LoRARequest(str(lora_name), int(lora_int_id), str(lora_adapter_path))

    try:
        outputs = llm.generate(list(prompts), sp, lora_request=lora_request)
        texts: list[str] = []
        for out in outputs:
            if not out.outputs:
                texts.append("")
                continue
            texts.append(out.outputs[0].text.strip())
        return texts
    finally:
        try:
            del llm
            gc.collect()
        except Exception:
            pass


def run_jailbreak_test(
    *,
    round_idx: int,
    out_dir: str | Path,
    base_model_name_or_path: str,
    tokenizer_name_or_path: str,
    attack_files: list[str],
    prompt_col: str = "prompt",
    system_prompt: str | None = None,
    max_samples_per_attack: int = 0,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    vllm_dtype: str = "auto",
    strongreject_model_name: str = "qylu4156/strongreject-15k-v1",
    strongreject_device_map: str = "auto",
) -> dict[str, Any]:
    test_dir = ensure_dir(round_dir(out_dir, round_idx) / "jailbreak")
    lora_adapter_path = infer_lora_adapter_path(out_dir, round_idx)

    strongreject = load_strongreject_evaluator(
        strongreject_model_name,
        device_map=strongreject_device_map,
    )

    sampling = VLLMGenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=vllm_dtype,
    )

    # 1) 先把所有 attack 文件读到一个大列表里
    all_source_rows: list[dict[str, Any]] = []
    attack_meta: list[dict[str, Any]] = []

    attack_pbar = tqdm(attack_files, desc="load_attacks", dynamic_ncols=True)
    for attack_file in attack_pbar:
        attack_name = attack_name_from_path(attack_file)
        attack_rows = read_prompt_rows(
            attack_file,
            prompt_col=prompt_col,
            max_samples=max_samples_per_attack,
        )

        attack_meta.append(
            {
                "attack_name": attack_name,
                "source_file": str(attack_file),
                "num_samples": len(attack_rows),
            }
        )

        for row_idx, src_row in enumerate(attack_rows):
            merged_row = {
                "round_idx": int(round_idx),
                "attack_name": attack_name,
                "source_file": str(attack_file),
                "source_row_idx": int(row_idx),
                "prompt": src_row[prompt_col],
            }
            for k, v in src_row.items():
                if k not in merged_row:
                    merged_row[k] = v
            all_source_rows.append(merged_row)

        attack_pbar.set_postfix(attack=attack_name, n=len(attack_rows), total=len(all_source_rows))

    # 2) 对大列表一次性构造 prompts，并只调用一次 vLLM.generate
    prompts = [row[prompt_col] for row in all_source_rows]
    prompt_texts = build_prompt_texts(
        prompts,
        tokenizer_name_or_path,
        system_prompt=system_prompt,
    )

    outputs = _generate_with_vllm_lora_once(
        base_model_name_or_path=base_model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        prompts=prompt_texts,
        lora_adapter_path=lora_adapter_path,
        sampling=sampling,
        lora_name=f"jailbreak_round_{int(round_idx):04d}",
        lora_int_id=int(round_idx) + 1,
    )

    if len(outputs) != len(all_source_rows):
        raise RuntimeError(
            f"Generation count mismatch: got {len(outputs)} outputs for {len(all_source_rows)} prompts."
        )

    # 3) StrongREJECT 仍按逐条打分，避免 batch 对齐问题
    pairs = list(zip(prompts, outputs))
    scores = _score_pairs_one_by_one(strongreject, pairs)

    all_rows: list[dict[str, Any]] = []
    per_attack_buckets: dict[str, list[dict[str, Any]]] = {}

    for src_row, output, score in zip(all_source_rows, outputs, scores):
        row = dict(src_row)
        row["output"] = output
        row["strongreject"] = float(score)
        all_rows.append(row)
        per_attack_buckets.setdefault(str(row["attack_name"]), []).append(row)

    # 4) 再按 attack_name 分桶保存与统计
    per_attack: list[dict[str, Any]] = []
    summary_pbar = tqdm(attack_meta, desc="save_attack_results", dynamic_ncols=True)
    for meta in summary_pbar:
        attack_name = str(meta["attack_name"])
        scored_rows = per_attack_buckets.get(attack_name, [])
        attack_scores = [float(x["strongreject"]) for x in scored_rows]

        attack_summary = {
            "round_idx": int(round_idx),
            "attack_name": attack_name,
            "num_samples": len(scored_rows),
            "mean_strongreject": float(mean(attack_scores)) if attack_scores else float("nan"),
            "max_strongreject": float(max(attack_scores)) if attack_scores else float("nan"),
            "min_strongreject": float(min(attack_scores)) if attack_scores else float("nan"),
            "source_file": str(meta["source_file"]),
        }
        per_attack.append(attack_summary)

        summary_pbar.set_postfix(
            attack=attack_name,
            mean_sr=(f"{attack_summary['mean_strongreject']:.4f}" if attack_scores else "nan"),
            n=len(scored_rows),
        )

        save_csv(test_dir / f"{attack_name}_scored.csv", scored_rows)
        save_jsonl(test_dir / f"{attack_name}_scored.jsonl", scored_rows)
        save_json(test_dir / f"{attack_name}_summary.json", attack_summary)

    overall_scores = [float(row["strongreject"]) for row in all_rows]
    macro_scores = [
        float(x["mean_strongreject"])
        for x in per_attack
        if x["num_samples"] > 0 and not math.isnan(float(x["mean_strongreject"]))
    ]
    overall = {
        "round_idx": int(round_idx),
        "lora_adapter_path": str(lora_adapter_path),
        "base_model_name_or_path": str(base_model_name_or_path),
        "tokenizer_name_or_path": str(tokenizer_name_or_path),
        "num_attacks": len(per_attack),
        "num_total_rows": len(all_rows),
        "micro_mean_strongreject": float(mean(overall_scores)) if overall_scores else float("nan"),
        "macro_mean_strongreject": float(mean(macro_scores)) if macro_scores else float("nan"),
        "attacks": per_attack,
    }

    save_csv(test_dir / "all_scored.csv", all_rows)
    save_jsonl(test_dir / "all_scored.jsonl", all_rows)
    save_json(test_dir / "summary.json", overall)
    return overall


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run jailbreak robustness test with vLLM + LoRA + StrongREJECT.")
    p.add_argument("--round_idx", type=int, default=0)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--base_model_name_or_path", type=str, required=True)
    p.add_argument("--attack_files", type=str, required=True, help="Comma-separated CSV file list.")
    p.add_argument("--prompt_col", type=str, default="prompt")
    p.add_argument("--system_prompt", type=str, default=None)
    p.add_argument("--max_samples_per_attack", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=0)
    p.add_argument("--vllm_dtype", type=str, default="auto")
    p.add_argument("--strongreject_model_name", type=str, default="qylu4156/strongreject-15k-v1")
    p.add_argument("--strongreject_device_map", type=str, default="auto")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    attack_files = parse_csv_path_list(args.attack_files)
    if not attack_files:
        raise ValueError("No attack files provided. Use --attack_files with a comma-separated csv list.")

    tokenizer_name_or_path = args.base_model_name_or_path
    max_model_len = None if int(args.max_model_len) <= 0 else int(args.max_model_len)

    summary = run_jailbreak_test(
        round_idx=args.round_idx,
        out_dir=args.out_dir,
        base_model_name_or_path=args.base_model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        attack_files=attack_files,
        prompt_col=args.prompt_col,
        system_prompt=args.system_prompt,
        max_samples_per_attack=args.max_samples_per_attack,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        vllm_dtype=args.vllm_dtype,
        strongreject_model_name=args.strongreject_model_name,
        strongreject_device_map=args.strongreject_device_map,
    )

    summary_path = Path(args.out_dir) / "tests" / f"round_{int(args.round_idx):04d}" / "jailbreak" / "summary.json"
    print("[test_jailbreak] done")
    print(f"[test_jailbreak] summary_path={summary_path}")
    print(
        "[test_jailbreak] "
        f"micro_mean_strongreject={summary['micro_mean_strongreject']}, "
        f"macro_mean_strongreject={summary['macro_mean_strongreject']}, "
        f"num_attacks={summary['num_attacks']}, "
        f"num_total_rows={summary['num_total_rows']}"
    )


if __name__ == "__main__":
    main()
