from __future__ import annotations

import csv
import json
import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from transformers import AutoTokenizer


@dataclass
class VLLMGenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    dtype: str = "auto"
    trust_remote_code: bool = True

    # 改成默认 True，避免忘记传时又回到 compile
    enforce_eager: bool = True

    swap_space: float = 4.0
    max_num_seqs: int | None = None


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_csv_path_list(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def round_dir(out_dir: str | Path, round_idx: int) -> Path:
    return ensure_dir(Path(out_dir) / "tests" / f"round_{int(round_idx):04d}")


def checkpoint_dir(out_dir: str | Path, round_idx: int) -> Path:
    return ensure_dir(Path(out_dir) / "lora_checkpoints" / f"round_{int(round_idx):04d}")


def save_lora_checkpoint(bundle, out_dir: str | Path, round_idx: int) -> Path:
    target = checkpoint_dir(out_dir, round_idx)
    bundle.model.save_pretrained(target)
    bundle.tokenizer.save_pretrained(target)
    return target


def maybe_save_lora_checkpoint(bundle, out_dir: str | Path, round_idx: int, every_k_rounds: int) -> Path | None:
    k = int(every_k_rounds)
    if k <= 0:
        return None
    if (int(round_idx) + 1) % k != 0:
        return None
    return save_lora_checkpoint(bundle, out_dir, round_idx)


def read_prompt_rows(csv_path: str | Path, prompt_col: str = "prompt", max_samples: int = 0) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    if prompt_col not in df.columns:
        raise ValueError(f"CSV {csv_path} missing prompt column '{prompt_col}'. Available columns: {list(df.columns)}")
    if max_samples and int(max_samples) > 0:
        df = df.head(int(max_samples))
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        prompt = str(row[prompt_col]).strip()
        if not prompt or prompt.lower() == "nan":
            continue
        obj = {k: row[k] for k in df.columns}
        obj[prompt_col] = prompt
        rows.append(obj)
    return rows


def attack_name_from_path(path: str | Path) -> str:
    return Path(path).stem


def save_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def save_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    path = Path(path)
    rows = list(rows)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            pass
        return path
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _build_prompt_text(tokenizer, prompt: str, system_prompt: str | None = None) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    if system_prompt:
        return f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
    return prompt


def build_prompt_texts(
    prompts: Sequence[str],
    tokenizer_name_or_path: str,
    *,
    system_prompt: str | None = None,
    trust_remote_code: bool = True,
) -> list[str]:
    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=trust_remote_code)
    return [_build_prompt_text(tok, p, system_prompt=system_prompt) for p in prompts]


def batched(xs: Sequence[Any], batch_size: int) -> list[list[Any]]:
    bsz = max(1, int(batch_size))
    return [list(xs[i : i + bsz]) for i in range(0, len(xs), bsz)]


def generate_with_vllm_lora(
    *,
    base_model_name_or_path: str,
    tokenizer_name_or_path: str | None,
    prompts: Sequence[str],
    lora_adapter_path: str | Path | None,
    sampling: VLLMGenerationConfig,
    lora_name: str = "round_adapter",
    lora_int_id: int = 1,
) -> list[str]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if not prompts:
        return []

    # 这些环境变量不是必须，但能减少误入 compile 路径的概率
    if bool(sampling.enforce_eager):
        os.environ["TORCH_COMPILE_DISABLE"] = "1"

    llm_kwargs = {
        "model": base_model_name_or_path,
        "enable_lora": lora_adapter_path is not None,
        "tensor_parallel_size": int(sampling.tensor_parallel_size),
        "gpu_memory_utilization": float(sampling.gpu_memory_utilization),
        "dtype": sampling.dtype,
        "trust_remote_code": bool(sampling.trust_remote_code),
        "enforce_eager": bool(sampling.enforce_eager),
        "swap_space": float(sampling.swap_space),
        "disable_log_stats": True,
    }
    if tokenizer_name_or_path:
        llm_kwargs["tokenizer"] = tokenizer_name_or_path
    if sampling.max_model_len is not None:
        llm_kwargs["max_model_len"] = int(sampling.max_model_len)
    if sampling.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = int(sampling.max_num_seqs)

    print(f"[generate_with_vllm_lora] enforce_eager={llm_kwargs['enforce_eager']}")
    print(f"[generate_with_vllm_lora] enable_lora={llm_kwargs['enable_lora']}")
    print(f"[generate_with_vllm_lora] tensor_parallel_size={llm_kwargs['tensor_parallel_size']}")
    print(f"[generate_with_vllm_lora] gpu_memory_utilization={llm_kwargs['gpu_memory_utilization']}")

    llm = LLM(**llm_kwargs)

    sp = SamplingParams(
        max_tokens=int(sampling.max_new_tokens),
        temperature=float(sampling.temperature),
        top_p=float(sampling.top_p),
    )

    lora_request = None
    if lora_adapter_path is not None:
        lora_request = LoRARequest(str(lora_name), int(lora_int_id), str(lora_adapter_path))

    outputs = llm.generate(list(prompts), sp, lora_request=lora_request)

    texts: list[str] = []
    for out in outputs:
        if not out.outputs:
            texts.append("")
            continue
        texts.append(out.outputs[0].text.strip())

    try:
        del llm
        gc.collect()
    except Exception:
        pass

    return texts