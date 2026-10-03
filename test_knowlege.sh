lm-eval \
  --model vllm \
  --model_args pretrained=/workspace/models/your_peft_finetuned_model,dtype=bfloat16,tensor_parallel_size=1,max_model_len=2048,gpu_memory_utilization=0.9 \
  --tasks mmlu hellaswag,arc_challenge,truthfulqa_mc2,gsm8k  \
  --num_fewshot 5 \
  --output_path ./runs/utility_suite \
  --log_samples \
  --apply_chat_template 