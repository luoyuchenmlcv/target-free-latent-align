 
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_TRITON=0
export VLLM_DISABLE_CUSTOM_KERNELS=1

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


TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-$BASE_MODEL_NAME_OR_PATH}"
ATTACK_FILES="${ATTACK_FILES:-attacks/DirectRequest.csv,attacks/GCG.csv,attacks/Codechameleon.csv,attacks/Multilingual.csv,attacks/AutoDAN.csv,attacks/GPTFuzz.csv,attacks/TAP.csv,attacks/FewShot.csv}"
PROMPT_COL="${PROMPT_COL:-prompt}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
MAX_SAMPLES_PER_ATTACK="${MAX_SAMPLES_PER_ATTACK:-300}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
STRONGREJECT_MODEL_NAME="${STRONGREJECT_MODEL_NAME:-qylu4156/strongreject-15k-v1}"
STRONGREJECT_DEVICE_MAP="${STRONGREJECT_DEVICE_MAP:-auto}"

export PYTHONPATH="$(pwd)"

python -m bilevel_at_nll.test_jailbreak \
  --round_idx "$ROUND_IDX" \
  --out_dir "$OUT_DIR" \
  --base_model_name_or_path "$BASE_MODEL_NAME_OR_PATH" \
  --attack_files "$ATTACK_FILES" \
  --prompt_col "$PROMPT_COL" \
  ${SYSTEM_PROMPT:+--system_prompt "$SYSTEM_PROMPT"} \
  --max_samples_per_attack "$MAX_SAMPLES_PER_ATTACK" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --max_model_len "$MAX_MODEL_LEN" \
  --vllm_dtype "$VLLM_DTYPE" \
  --strongreject_model_name "$STRONGREJECT_MODEL_NAME" \
  --strongreject_device_map "$STRONGREJECT_DEVICE_MAP"
