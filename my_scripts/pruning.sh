export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
CUDA_VISIBLE_DEVICES=0
python hf_prune.py --pruning_ratio 0.25 \
      --block_wise \
      --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
      --block_attention_layer_start 4 --block_attention_layer_end 30 \
      --pruner_type taylor \
      --test_after_train \
      --device cpu  --eval_device cuda \
      --save_ckpt_log_name llama_prune \
      --save_model

echo "[INFO] - The pruned model is at {prune_log/llama_prune/pytorch_model.bin}"
