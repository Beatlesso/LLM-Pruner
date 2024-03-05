export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
CUDA_VISIBLE_DEVICES=0 python post_training.py --prune_model prune_log/llama_prune/pytorch_model.bin \
      --data_path yahma/alpaca-cleaned \
      --lora_r 8 \
      --num_epochs 2 \
      --learning_rate 1e-4 \
      --batch_size 64 \
      --output_dir tune_log/llama_prune_tuned \
      --wandb_project llama_tune

echo "[INFO] - The pruned model is at {prune_log/llama_prune/pytorch_model.bin}, and the recovery weight is at {tune_log/llama_prune_tuned}/"