# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence


def _read_lines(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _read_jsonl_prompts(path: Path) -> list[str]:
    out: list[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        obj = json.loads(ln)
        for k in ("prompt", "text", "goal", "query"):
            if k in obj and isinstance(obj[k], str) and obj[k].strip():
                out.append(obj[k].strip())
                break
    return out


def _read_csv_prompts(path: Path, col: Optional[str] = None) -> list[str]:
    import pandas as pd

    df = pd.read_csv(path)
    use_col = col if col is not None else "prompt"
    if use_col not in df.columns:
        raise ValueError(f"CSV {path} missing column '{use_col}', got {list(df.columns)}")

    vals: list[str] = []
    for x in df[use_col].tolist():
        sx = str(x).strip()
        if sx and sx.lower() != "nan":
            vals.append(sx)
    return vals


def read_prompts(path: str | Path, col: Optional[str] = None) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    if suf in {".txt", ".md"}:
        return _read_lines(path)
    if suf == ".jsonl":
        return _read_jsonl_prompts(path)
    if suf == ".csv":
        return _read_csv_prompts(path, col=col)
    raise ValueError(f"Unsupported prompt file: {path}")


def _normalize_cell(x) -> str:
    sx = str(x).strip()
    if not sx or sx.lower() == "nan":
        return ""
    return sx


def read_examples(
    path: str | Path,
    prompt_col: Optional[str] = "prompt",
    generation_col: Optional[str] = "generation",
) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()

    if suf in {".txt", ".md"}:
        return [{"prompt": x, "generation": ""} for x in _read_lines(path)]

    if suf == ".jsonl":
        out: list[dict[str, str]] = []
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)

            use_prompt = prompt_col if prompt_col is not None else "prompt"
            p = ""
            if use_prompt in obj and isinstance(obj[use_prompt], str):
                p = obj[use_prompt].strip()

            if not p:
                continue

            g = ""
            if generation_col and generation_col in obj and isinstance(obj[generation_col], str):
                g = obj[generation_col].strip()

            out.append({"prompt": p, "generation": g})
        return out

    if suf == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        use_prompt = prompt_col if prompt_col is not None else "prompt"

        if use_prompt not in df.columns:
            raise ValueError(f"CSV {path} missing prompt column '{use_prompt}', got {list(df.columns)}")

        use_generation = generation_col if (generation_col is not None and generation_col in df.columns) else None

        out: list[dict[str, str]] = []
        for _, row in df.iterrows():
            p = _normalize_cell(row[use_prompt])
            if not p:
                continue
            g = _normalize_cell(row[use_generation]) if use_generation is not None else ""
            out.append({"prompt": p, "generation": g})
        return out

    raise ValueError(f"Unsupported example file: {path}")


def chunked(xs: Sequence, batch_size: int) -> Iterator[list]:
    for i in range(0, len(xs), batch_size):
        yield list(xs[i : i + batch_size])


@dataclass
class MixedRoundBatch:
    harmful_examples: list[dict[str, str]]
    benign_examples: list[dict[str, str]]         # benign + borderline merged
    benign_pure_examples: list[dict[str, str]]
    borderline_examples: list[dict[str, str]]


class RoundRobinSampler:
    """
    Sample sequentially from a shuffled order.
    If one round needs more samples than remain, continue into the next reshuffled pass.
    """

    def __init__(self, data: list[dict[str, str]], *, shuffle: bool = True):
        self.data = list(data)
        self.shuffle = bool(shuffle)

        if len(self.data) == 0:
            raise ValueError("Sampler got empty dataset")

        self._order = list(range(len(self.data)))
        self._ptr = 0
        if self.shuffle:
            random.shuffle(self._order)

    def _refresh(self):
        self._order = list(range(len(self.data)))
        self._ptr = 0
        if self.shuffle:
            random.shuffle(self._order)

    def sample(self, n: int) -> list[dict[str, str]]:
        n = int(n)
        if n <= 0:
            return []

        out: list[dict[str, str]] = []
        while len(out) < n:
            if self._ptr >= len(self._order):
                self._refresh()

            remain = len(self._order) - self._ptr
            need = n - len(out)
            take = min(remain, need)

            idxs = self._order[self._ptr : self._ptr + take]
            out.extend([self.data[i] for i in idxs])
            self._ptr += take

        return out


class MixedCsvRoundDataLoader:
    """
    Every round samples:
        harmful   : benign   : borderline = 5 : 5 : 2  (scaled by base_unit)

    Then benign and borderline are merged into one benign_examples list.
    """

    def __init__(
        self,
        *,
        harmful_file: str,
        harmful_col: Optional[str] = "prompt",
        harmful_generation_col: Optional[str] = "generation",
        benign_file: str,
        benign_col: Optional[str] = "prompt",
        benign_generation_col: Optional[str] = "generation",
        borderline_file: str,
        borderline_col: Optional[str] = "prompt",
        borderline_generation_col: Optional[str] = "generation",
        harmful_max_samples: int = 0,
        benign_max_samples: int = 0,
        borderline_max_samples: int = 0,
        ratio_harmful: int = 5,
        ratio_benign: int = 5,
        ratio_borderline: int = 2,
        base_unit: int = 1,
        shuffle: bool = True,
    ):
        self.ratio_harmful = int(ratio_harmful)
        self.ratio_benign = int(ratio_benign)
        self.ratio_borderline = int(ratio_borderline)
        self.base_unit = int(base_unit)

        if self.ratio_harmful < 0 or self.ratio_benign < 0 or self.ratio_borderline < 0:
            raise ValueError("Ratios must be non-negative")
        if self.ratio_harmful + self.ratio_benign + self.ratio_borderline <= 0:
            raise ValueError("At least one ratio must be positive")
        if self.base_unit <= 0:
            raise ValueError("base_unit must be positive")

        harmful_examples = read_examples(
            harmful_file,
            harmful_col,
            harmful_generation_col,
        )
        benign_examples = read_examples(
            benign_file,
            benign_col,
            benign_generation_col,
        )
        borderline_examples = read_examples(
            borderline_file,
            borderline_col,
            borderline_generation_col,
        )

        if harmful_max_samples > 0:
            harmful_examples = harmful_examples[: int(harmful_max_samples)]
        if benign_max_samples > 0:
            benign_examples = benign_examples[: int(benign_max_samples)]
        if borderline_max_samples > 0:
            borderline_examples = borderline_examples[: int(borderline_max_samples)]

        if len(harmful_examples) == 0:
            raise ValueError("No harmful training examples found")
        if len(benign_examples) == 0:
            raise ValueError("No benign training examples found")
        if len(borderline_examples) == 0:
            raise ValueError("No borderline training examples found")

        self.harmful_examples = harmful_examples
        self.benign_examples = benign_examples
        self.borderline_examples = borderline_examples

        self.harmful_sampler = RoundRobinSampler(harmful_examples, shuffle=shuffle)
        self.benign_sampler = RoundRobinSampler(benign_examples, shuffle=shuffle)
        self.borderline_sampler = RoundRobinSampler(borderline_examples, shuffle=shuffle)

    @property
    def harmful_per_round(self) -> int:
        return self.ratio_harmful * self.base_unit

    @property
    def benign_per_round(self) -> int:
        return self.ratio_benign * self.base_unit

    @property
    def borderline_per_round(self) -> int:
        return self.ratio_borderline * self.base_unit

    def sample_round(self) -> MixedRoundBatch:
        harmful_batch = self.harmful_sampler.sample(self.harmful_per_round)
        benign_batch = self.benign_sampler.sample(self.benign_per_round)
        borderline_batch = self.borderline_sampler.sample(self.borderline_per_round)

        merged_benign = list(benign_batch) + list(borderline_batch)
        random.shuffle(merged_benign)

        return MixedRoundBatch(
            harmful_examples=harmful_batch,
            benign_examples=merged_benign,
            benign_pure_examples=benign_batch,
            borderline_examples=borderline_batch,
        )