#!/usr/bin/env bash
set -euo pipefail

#qwen
# ROUND_IDX="${ROUND_IDX:-10}"
# OUT_DIR="${OUT_DIR:-./runs/bilevel_at_qwen7b_nll}"
# BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:-/workspace/models/Qwen2.5-7B-Instruct}"

#llama8b
# ROUND_IDX="${ROUND_IDX:-10}"
# OUT_DIR="${OUT_DIR:-./runs/bilevel_at_llama8b_nll}"
# BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:-/workspace/models/Llama-3-8B-Instruct}"

#mistral7b
ROUND_IDX="${ROUND_IDX:-10}"
OUT_DIR="${OUT_DIR:-./runs/bilevel_at_mistral7b_nll}"
BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:-/workspace/models/Mistral-7B-Instruct-v0.2}"


SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
JUDGE_MODEL_NAME_OR_PATH="${JUDGE_MODEL_NAME_OR_PATH:-/workspace/models/Qwen2.5-32B-Instruct}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-8}"
JUDGE_DTYPE="${JUDGE_DTYPE:-bf16}"
JUDGE_DEVICE_MAP="${JUDGE_DEVICE_MAP:-auto}"

export PYTHONPATH="$(pwd)"

python -m bilevel_at_nll.test_general_utility \
  --round_idx "$ROUND_IDX" \
  --out_dir "$OUT_DIR" \
  --base_model_name_or_path "$BASE_MODEL_NAME_OR_PATH" \
  ${SYSTEM_PROMPT:+--system_prompt "$SYSTEM_PROMPT"} \
  --max_samples "$MAX_SAMPLES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --max_model_len "$MAX_MODEL_LEN" \
  --vllm_dtype "$VLLM_DTYPE" \
  --judge_model_name_or_path "$JUDGE_MODEL_NAME_OR_PATH" \
  --judge_batch_size "$JUDGE_BATCH_SIZE" \
  --judge_dtype "$JUDGE_DTYPE" \
  --judge_device_map "$JUDGE_DEVICE_MAP"
