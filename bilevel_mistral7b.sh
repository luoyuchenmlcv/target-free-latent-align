#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="/workspace/models/Mistral-7B-Instruct-v0.2"
TOKENIZER_NAME="$MODEL_NAME"

HARMFUL_FILE="prompt_generation_csvs/harmful.csv"
HARMFUL_COL="prompt"

BENIGN_FILE="prompt_generation_csvs/Benign.csv"
BENIGN_COL="prompt"

BORDERLINE_FILE="prompt_generation_csvs/Boundary.csv"
BORDERLINE_COL="prompt"

OUT_DIR="./runs/bilevel_at_mistral7b_nll"

SOURCE_LAYER_IDX=10
TARGET_LAYER_IDX=20
DEVICE_MAP="auto"
DTYPE="bf16"
ATTN_IMPLEMENTATION="eager"
MAX_PROMPT_TOKENS=32
SEED=42

# =============================
# Main adversarial training rounds
# =============================
ROUNDS=20

# harmful : benign : borderline = 5 : 5 : 2
ROUND_RATIO_HARMFUL=12
ROUND_RATIO_BENIGN=6
ROUND_RATIO_BORDERLINE=6
ROUND_BASE_UNIT=1

# =============================
# Inner step
# =============================

#nuclear
INNER_NUM_SAMPLES=12
INNER_MAX_SEQ_LEN=27
INNER_FORWARD_BATCH_SIZE=1
INNER_BACKWARD_BATCH_SIZE=1
INNER_NUM_FACTORS=64
INNER_FACTOR_BATCH_SIZE="${INNER_FACTOR_BATCH_SIZE:-32}"
INNER_DIM_OUTPUT_PROJECTION=32
INNER_NUM_ITERS=3
INNER_INIT="random"
INNER_BETA=1.0
INNER_INPUT_SCALE=2.5
INNER_CALIBRATION_TARGET_RATIO=0.5
INNER_TOKEN_IDXS="-3:"
INNER_EVAL_NUM_PROMPTS=1
INNER_EVAL_MAX_NEW_TOKENS=256


# =============================
# Outer step
# =============================
OUTER_STEPS_PER_ROUND=20
OUTER_LR=2e-5
OUTER_WEIGHT_DECAY=0.0
OUTER_GRAD_CLIP=1.0
STEER_CHUNK_SIZE=16
PEFT_R=16
PEFT_ALPHA=32
PEFT_DROPOUT=0.0
PEFT_LAYERS="all"

LAMBDA_NUCLEAR=1.0
LAMBDA_HARMFUL=1.0
LAMBDA_BENIGN=1.0

# =============================
# Validation
# =============================
VALIDATION_EVERY_ROUND="${VALIDATION_EVERY_ROUND:-1}"
VALIDATION_MAX_NEW_TOKENS=128
VALIDATION_TEMPERATURE=1.0

STRONGREJECT_MODEL_NAME="qylu4156/strongreject-15k-v1"
STRONGREJECT_DEVICE_MAP="auto"

# =============================
# Optional dataset trimming
# =============================
HARMFUL_MAX_SAMPLES=0
BENIGN_MAX_SAMPLES=0
BORDERLINE_MAX_SAMPLES=0
VALIDATION_MAX_SAMPLES=0

# =============================
# WandB
# =============================
WANDB_PROJECT="bilevel_at"
WANDB_NAME="mistral7b_nll"
WANDB_MODE="${WANDB_MODE:-disabled}"

export PYTHONPATH="$(pwd)"

python -m bilevel_at_nll.train \
  --model_name="$MODEL_NAME" \
  --tokenizer_name="$TOKENIZER_NAME" \
  --harmful_file="$HARMFUL_FILE" \
  --harmful_col="$HARMFUL_COL" \
  --harmful_generation_col="generation" \
  --benign_file="$BENIGN_FILE" \
  --benign_col="$BENIGN_COL" \
  --benign_generation_col="generation" \
  --borderline_file="$BORDERLINE_FILE" \
  --borderline_col="$BORDERLINE_COL" \
  --borderline_generation_col="generation" \
  --out_dir="$OUT_DIR" \
  --source_layer_idx="$SOURCE_LAYER_IDX" \
  --target_layer_idx="$TARGET_LAYER_IDX" \
  --device_map="$DEVICE_MAP" \
  --dtype="$DTYPE" \
  --attn_implementation="$ATTN_IMPLEMENTATION" \
  --rounds="$ROUNDS" \
  --round_ratio_harmful="$ROUND_RATIO_HARMFUL" \
  --round_ratio_benign="$ROUND_RATIO_BENIGN" \
  --round_ratio_borderline="$ROUND_RATIO_BORDERLINE" \
  --round_base_unit="$ROUND_BASE_UNIT" \
  --inner_method="nuclear" \
  --v_subset_mode all \
  --v_subset_k 0 \
  --inner_num_samples="$INNER_NUM_SAMPLES" \
  --inner_max_seq_len="$INNER_MAX_SEQ_LEN" \
  --inner_forward_batch_size="$INNER_FORWARD_BATCH_SIZE" \
  --inner_backward_batch_size="$INNER_BACKWARD_BATCH_SIZE" \
  --inner_num_factors="$INNER_NUM_FACTORS" \
  --inner_factor_batch_size="$INNER_FACTOR_BATCH_SIZE" \
  --inner_dim_output_projection="$INNER_DIM_OUTPUT_PROJECTION" \
  --inner_num_iters="$INNER_NUM_ITERS" \
  --inner_init="$INNER_INIT" \
  --inner_beta="$INNER_BETA" \
  --inner_input_scale="$INNER_INPUT_SCALE" \
  --inner_calibration_target_ratio="$INNER_CALIBRATION_TARGET_RATIO" \
  --inner_token_idxs="$INNER_TOKEN_IDXS" \
  --inner_eval_num_prompts="$INNER_EVAL_NUM_PROMPTS" \
  --inner_eval_max_new_tokens="$INNER_EVAL_MAX_NEW_TOKENS" \
  --outer_steps_per_round="$OUTER_STEPS_PER_ROUND" \
  --outer_lr="$OUTER_LR" \
  --outer_weight_decay="$OUTER_WEIGHT_DECAY" \
  --outer_grad_clip="$OUTER_GRAD_CLIP" \
  --steer_chunk_size="$STEER_CHUNK_SIZE" \
  --peft_r="$PEFT_R" \
  --peft_alpha="$PEFT_ALPHA" \
  --peft_dropout="$PEFT_DROPOUT" \
  --peft_layers="$PEFT_LAYERS" \
  --lambda_nuclear="$LAMBDA_NUCLEAR" \
  --lambda_harmful="$LAMBDA_HARMFUL" \
  --lambda_benign="$LAMBDA_BENIGN" \
  --max_prompt_tokens="$MAX_PROMPT_TOKENS" \
  --validation_every_round="$VALIDATION_EVERY_ROUND" \
  --validation_max_new_tokens="$VALIDATION_MAX_NEW_TOKENS" \
  --validation_temperature="$VALIDATION_TEMPERATURE" \
  --inner_eval_save_jsonl \
  --strongreject_model_name="$STRONGREJECT_MODEL_NAME" \
  --strongreject_device_map="$STRONGREJECT_DEVICE_MAP" \
  --harmful_max_samples="$HARMFUL_MAX_SAMPLES" \
  --benign_max_samples="$BENIGN_MAX_SAMPLES" \
  --borderline_max_samples="$BORDERLINE_MAX_SAMPLES" \
  --validation_max_samples="$VALIDATION_MAX_SAMPLES" \
  --wandb_project="$WANDB_PROJECT" \
  --wandb_name="$WANDB_NAME" \
  --wandb_mode="$WANDB_MODE" \
  --seed="$SEED"
