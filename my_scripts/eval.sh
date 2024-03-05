export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export PYTHONPATH='.'
PATH_TO_PRUNE_MODEL=/home/luoyicong/LLM-Pruner/prune_log/llama_prune/pytorch_model.bin
PATH_TO_SAVE_TUNE_MODEL=/home/luoyicong/LLM-Pruner/tune_log/llama_prune_tuned
PATH_OR_NAME_TO_BASE_MODEL=/mnt/llms/model/meta-llama/Llama-2-7b-hf
PATH_TO_SAVE_EVALUATION_LOG=/home/luoyicong/LLM-Pruner/eval_log
python lm-evaluation-harness/main.py --model hf-causal-experimental \
       --model_args checkpoint=$PATH_TO_PRUNE_MODEL,peft=$PATH_TO_SAVE_TUNE_MODEL,config_pretrained=$PATH_OR_NAME_TO_BASE_MODEL \
       --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
       --device cuda:0 --no_cache \
       --output_path $PATH_TO_SAVE_EVALUATION_LOG 