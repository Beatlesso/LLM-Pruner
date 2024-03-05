cd /home/luoyicong/LLM-Pruner/tune_log/llama_prune_tuned
export epoch=200
cp adapter_config.json checkpoint-$epoch/
mv checkpoint-$epoch/pytorch_model.bin checkpoint-$epoch/adapter_model.bin