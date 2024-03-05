export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
python generate.py --model_type tune_prune_LLM --ckpt prune_log/llama_prune/pytorch_model.bin --lora_ckpt tune_log/llama_prune_tuned