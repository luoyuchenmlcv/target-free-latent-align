#!/usr/bin/env bash
set -euo pipefail

ROUND_IDX="${ROUND_IDX:-10}"
#qwen
# ROUND_IDX="${ROUND_IDX:-7}"
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


ORBENCH_TEST_FILE="${ORBENCH_TEST_FILE:-attacks/orbench_test_csv.csv}"
PROMPT_COL="${PROMPT_COL:-prompt}"
CATEGORY_COL="${CATEGORY_COL:-category}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
JUDGE_MODEL_NAME_OR_PATH="${JUDGE_MODEL_NAME_OR_PATH:-/workspace/models/Qwen2.5-7B-Instruct}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-128}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
JUDGE_TOP_P="${JUDGE_TOP_P:-1.0}"
JUDGE_TENSOR_PARALLEL_SIZE="${JUDGE_TENSOR_PARALLEL_SIZE:-1}"
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.50}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-0}"
JUDGE_VLLM_DTYPE="${JUDGE_VLLM_DTYPE:-auto}"

export PYTHONPATH="$(pwd)"

python -m bilevel_at_nll.test_over_refusal \
  --round_idx "$ROUND_IDX" \
  --out_dir "$OUT_DIR" \
  --base_model_name_or_path "$BASE_MODEL_NAME_OR_PATH" \
  --orbench_test_file "$ORBENCH_TEST_FILE" \
  --prompt_col "$PROMPT_COL" \
  --category_col "$CATEGORY_COL" \
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
  --judge_max_new_tokens "$JUDGE_MAX_NEW_TOKENS" \
  --judge_temperature "$JUDGE_TEMPERATURE" \
  --judge_top_p "$JUDGE_TOP_P" \
  --judge_tensor_parallel_size "$JUDGE_TENSOR_PARALLEL_SIZE" \
  --judge_gpu_memory_utilization "$JUDGE_GPU_MEMORY_UTILIZATION" \
  --judge_max_model_len "$JUDGE_MAX_MODEL_LEN" \
  --judge_vllm_dtype "$JUDGE_VLLM_DTYPE"
