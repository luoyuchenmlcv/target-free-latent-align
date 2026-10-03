# Target-Free Latent Safety Alignment

This repository implements a Target-Free adversarial training framework for LLM safety alignment.

In each training round, the inner loop learns a set of latent steering directions by maximizing a nuclear-norm objective over hidden-state shifts. The outer loop then trains a LoRA adapter to preserve safe behavior under these latent perturbations while maintaining benign utility.

## Project Structure

```text
.
├── prompt_generation_csvs/     # Harmful, benign, and boundary training data
├── attacks/                    # Datasets used by evaluation scripts
├── bilevel_at_nll/
│   ├── train.py                # Main training entry point
│   ├── inner_step.py           # Inner latent-direction optimization
│   ├── outer_step.py           # Outer LoRA optimization
│   ├── generation.py           # Steered generation
│   ├── validation.py           # Inner evaluation with StrongREJECT
│   └── config.py               # Command-line arguments
├── nuclear.py                  # Nuclear-norm objective
├── strong_reject.py            # StrongREJECT evaluator
├── bilevel_qwen7b.sh           # Qwen2.5-7B training
├── bilevel_llama8b.sh          # Llama-3-8B training
├── bilevel_mistral7b.sh        # Mistral-7B training
├── test_jailbreak.sh
├── test_over_refusal.sh
├── test_general_utility.sh
└── test_knowlege.sh
```

## Installation

Create a Python environment and install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install a PyTorch build compatible with your CUDA version first.
pip install transformers==4.46.3 peft==0.13.2 accelerate==1.1.1 \
  datasets==3.1.0 wandb==0.18.7 pandas numpy scipy tqdm \
  sentencepiece safetensors huggingface-hub
```

The evaluation scripts additionally require `vllm`; `test_knowlege.sh` requires `lm-eval`.

## Data

The training scripts use the following files:

```text
prompt_generation_csvs/harmful.csv
prompt_generation_csvs/Benign.csv
prompt_generation_csvs/Boundary.csv
```

Each CSV must contain `prompt` and `generation` columns:

```csv
prompt,generation
"Example instruction","Reference response"
```

## Models

Set the local model path near the top of the corresponding training script:

```bash
# bilevel_qwen7b.sh
MODEL_NAME="/workspace/models/Qwen2.5-7B-Instruct"

# bilevel_llama8b.sh
MODEL_NAME="/workspace/models/Meta-Llama-3-8B-Instruct"

# bilevel_mistral7b.sh
MODEL_NAME="/workspace/models/Mistral-7B-Instruct-v0.2"
```

Inner-generation evaluation uses `qylu4156/strongreject-15k-v1`, which is based on `google/gemma-2b`. Download both models after obtaining Hugging Face access:

```bash
hf auth login
hf download qylu4156/strongreject-15k-v1
hf download google/gemma-2b \
  config.json generation_config.json model.safetensors.index.json \
  model-00001-of-00002.safetensors model-00002-of-00002.safetensors
```

## Training

Run a training script from the repository root:

```bash
bash bilevel_qwen7b.sh
bash bilevel_llama8b.sh
bash bilevel_mistral7b.sh
```

For Qwen2.5-7B, the following command enables inner generation and StrongREJECT evaluation after every round:

```bash
VALIDATION_EVERY_ROUND=1 \
STRONGREJECT_DEVICE_MAP=cpu \
WANDB_MODE=disabled \
bash bilevel_qwen7b.sh
```

To train without inner-generation evaluation:

```bash
VALIDATION_EVERY_ROUND=0 bash bilevel_qwen7b.sh
```

The main settings, including layer indices, number of directions, optimization steps, LoRA configuration, and output directory, are defined at the top of each training script. Complete command-line options are available in `bilevel_at_nll/config.py`.

## Inner Generation

When validation is enabled, the learned steering directions are used to generate responses for harmful prompts. StrongREJECT scores each response, and the results are saved as JSONL files:

```text
runs/bilevel_at_qwen7b_nll/inner_eval/inner_eval_000.jsonl
runs/bilevel_at_qwen7b_nll/inner_eval/inner_eval_001.jsonl
...
```

The relevant settings are:

```bash
INNER_NUM_FACTORS=64
INNER_EVAL_NUM_PROMPTS=1
INNER_EVAL_MAX_NEW_TOKENS=256
VALIDATION_EVERY_ROUND=1
```

## Outputs

A Qwen training run writes to `runs/bilevel_at_qwen7b_nll/` by default:

```text
runs/bilevel_at_qwen7b_nll/
├── inner_artifacts/        # Learned directions, rankings, and per-step statistics
├── inner_eval/             # Steered generations and StrongREJECT scores
├── lora_checkpoints/       # Per-round LoRA checkpoints
├── lora_adapter/           # Final LoRA adapter
├── logs.json               # Training history
└── summary.json            # Final run summary
```

## Evaluation

The repository includes scripts for jailbreak robustness, over-refusal, general utility, and general knowledge evaluation.

Example for a Qwen checkpoint:

```bash
ROUND_IDX=0 \
OUT_DIR=./runs/bilevel_at_qwen7b_nll \
BASE_MODEL_NAME_OR_PATH=/workspace/models/Qwen2.5-7B-Instruct \
bash test_jailbreak.sh
```

The other evaluations use the same environment-variable pattern:

```bash
bash test_over_refusal.sh
bash test_general_utility.sh
bash test_knowlege.sh
```
