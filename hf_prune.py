import os
import gc
import sys
import time
import json
import copy
import random
import argparse
from typing import Tuple

import torch
import numpy as np
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP

import LLMPruner.torch_pruning as tp 
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts

# 为所有用到的随机数生成器设置随机种子
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def main(args):
    # 设置随机种子
    set_random_seed(args.seed)
    # 创建日志对象
    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name), # env_name = str(args.save_ckpt_log_name)
        config=args.__dict__, # config就是参数字典
        root_dir='prune_log', # log的根目录
        setup_sublogger=True
    )
    # 创建分词器和Llama模型
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        low_cpu_mem_usage=True if args.torch_version >=1.9 else False
    )
    if args.device != "cpu":
        model.half() # 开启半精度，加快运行速度、减少GPU占用
    model.to(args.device)

    if args.test_before_train:
        logger.log("\n==================Generation Results before Pruning================\n")
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                # "pt"表示返回torch.tensor  
                # {'input_ids': tensor([[   1,  306, 4658,  278, 6593,  310, 2834,  338]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]])}
                # 输入的两个句子长度不一致，需要通过短句填充（padding）的方式让其长度一致
                # attention_mask的作用就是用来标记哪些数据是补零的
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.device)

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True, # 生成文本时采用采样策略，而不是仅选择最可能的下一个词
                    top_k=50, # 模型只从概率最高的前k个词中选择下一个词
                    max_length=args.max_seq_len,
                    top_p=args.top_p, # 模型只从累积概率加起来至少为p的最小词集中采样
                    temperature=args.temperature, # 用于控制生成过程的随机性，温度越高，随机性越高
                )
                # 将输出的id解码还原成文本
                result = tokenizer.decode(generation_output[0])
                logger.log(result)
        # 用模型在 wikitext2 和 ptb这两个数据集上计算ppl
        ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.device)
        logger.log("PPL before pruning: {}".format(ppl))

    # 剪枝类型
    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor']
    # 所有参数都需要计算梯度
    for param in model.parameters():
        param.requires_grad_(True)
    # numel()获取tensor中一共包含多少个元素   所以这里是计算剪枝之前的可学习参数总量
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    forward_prompts = torch.tensor([
        [    1,   306,  4658,   278,  6593,   310,  2834,   338],
        [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    ]).to(args.device) # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.

    # 根据剪枝类型获取对应的重要性计算类
    if pruner_type == 'random':
        imp = tp.importance.RandomImportance()
    elif pruner_type == 'l1':
        imp = llama_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        imp = llama_pruner.MagnitudeImportance(p=2)
    elif pruner_type == 'taylor':
        imp = llama_pruner.TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)
    else:
        raise NotImplementedError

    logger.log("Use {} pruner...".format(pruner_type))
    # 按模块进行剪枝
    if args.block_wise:
        # 设置剪枝参数
        kwargs = {
            "importance": imp,
            # 全局修剪选项（bool）：自适应稀疏性将根据其全局重要性等级分配给不同的层。
            #        虽然这种策略可以提供性能优势，但它也有可能过度修剪特定层，导致整体性能大幅下降
            "global_pruning": args.global_pruning, 
            "iterative_steps": args.iterative_steps,
            # 全局通道稀疏度
            "ch_sparsity": args.pruning_ratio, 
            "ignored_layers":[],
            "channel_groups": {
            },
            "consecutive_groups": {
                layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
            },
            # dict，存放定制化模块和剪枝器pair
            "customized_pruners": {
                LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
            },
            # list，可剪枝的模块类型
            "root_module_types": None, 
            "root_instances": [model.model.layers[i].self_attn.q_proj for i in range(args.block_attention_layer_start, args.block_attention_layer_end)] +
                              [model.model.layers[i].mlp.gate_proj for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)]
        }
        logger.log("Pruning Attention Layer = {}".format(list(range(args.block_attention_layer_start, args.block_attention_layer_end))))
        logger.log("Pruning MLP Layer = {}".format(list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))))
        # 创建剪枝模型
        pruner = tp.pruner.MetaPruner(
            model,
            forward_prompts,
            **kwargs
        )
        model.zero_grad()

        logger.log("Start Pruning")
        for i in range(args.iterative_steps):
            # 泰勒展开剪枝
            if pruner_type in ['taylor']:
                # 从数据集中随机获取num_examples个长度为seq_len的token编码后的id
                example_prompts = get_examples('bookcorpus', tokenizer, args.num_examples, seq_len = 64).to(args.device)
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                # param_mix表示泰勒展开第一项第二项都考虑，param_second只考虑第二项
                if args.taylor in ['param_mix', 'param_second']:
                    # 单独计算每一个样例求acc_grad
                    for j in range(args.num_examples):
                        batch_input = example_prompts[j].unsqueeze(0)
                        loss = model(batch_input, labels=batch_input).loss
                        logger.log("Loss = {}".format(loss))
                        loss.backward()

                        for module_param in model.parameters():
                            module_param.grad = module_param.grad * module_param.grad / args.num_examples
                            # 判断是否累加梯度
                            if hasattr(module_param, 'acc_grad'):
                                module_param.acc_grad += module_param.grad
                            else:
                                module_param.acc_grad = copy.deepcopy(module_param.grad)
                        # 这里清空grad  不会清空acc_grad
                        # 如果累加梯度，那么最终结果就是 ([grad_1]^2 + ... + [grad_num]^2) / num
                        # 否则就是 [grad_num]^2 / num
                        model.zero_grad()
                        del loss.grad
                        

                # 计算所有样例的损失和梯度 
                loss = model(example_prompts, labels=example_prompts).loss
                logger.log("Loss = {}".format(loss))
                loss.backward()
             
            # 根据累积的梯度信息进行剪枝
            pruner.step()
            # 统计剪枝后的模型参数量
            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))
        
            # modify inferece-related attributes
            # 修改多头的数目
            for layer in model.model.layers:
                layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None

        del pruner
    
    # 按通道进行剪枝
    elif args.channel_wise:
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            "ignored_layers":[],
            #"round_to": model.config.num_attention_heads * 2,
            "channel_groups": {
                #layer.self_attn: layer.self_attn.num_heads for layer in model.model.layers
            },
            "customized_pruners": {
                LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
                #LlamaAttention: llama_pruner.hf_attention_pruner,
            },
            "root_module_types": [LlamaRMSNorm, LlamaAttention],
        }
        # 用torch_pruning 初始化剪枝器
        pruner = tp.pruner.MetaPruner(
            model,
            forward_prompts,
            **kwargs
        )
        model.zero_grad()
        
        logger.log("Start Pruning")
        for i in range(args.iterative_steps):

            if pruner_type in ['taylor']:
                example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len = 64)
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                loss = model(example_prompts, labels=example_prompts).loss
                logger.log("Loss = {}".format(loss))
                loss.backward()

            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None

        # modify inferece-related attributes
        model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
        model.zero_grad()
        
        del pruner
            
    elif args.layer_wise:
        model.model.layers = model.model.layers[:args.layer]
        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    else:
        raise NotImplementedError
    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))
    
    gc.collect()
    torch.cuda.empty_cache()

    if args.save_model:
        model.half()
        torch.save({
            'model': model, 
            'tokenizer': tokenizer,
        }, logger.best_checkpoint_path)
    
    if args.eval_device != "cpu":
        model.half()
    model.to(args.eval_device)

    model.config.pad_token_id = tokenizer.pad_token_id = 0 
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if args.test_after_train:
        logger.log("\n==================Generation Results After Pruning================\n")
        
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.eval_device)

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                
                result = tokenizer.decode(generation_output[0])
                logger.log(result)
        
        logger.log("\n==================Finish================\n")
    
    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.eval_device)
    logger.log("PPL after pruning: {}".format(ppl))
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # 基础模型路径、日志和模型保存路径名、剪枝稀疏率、剪枝类型
    parser.add_argument('--base_model', type=str, default="/mnt/llms/model/meta-llama/Llama-2-7b-hf", help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')
    parser.add_argument('--pruner_type', type=str, default='l2', help='pruner type')

    # 用于generation的参数
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')

    # 修剪参数
    parser.add_argument('--channel_wise', action='store_true', help='channel wise')
    parser.add_argument('--block_wise', action='store_true', help='block wise')
    parser.add_argument('--layer_wise', action='store_true', help='layer wise')
    parser.add_argument('--layer', type=int, default=12, help='remain the previous n layers')

    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=3)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)

    parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
    parser.add_argument('--grouping_strategy', type=str, default='sum', help='Reduce method for grouping')
    parser.add_argument('--global_pruning', action='store_true', help='whether global pruning')
    parser.add_argument('--taylor', type=str, default='param_first', help='choose from [vectorize, param_second, param_first, param_mix]')
    parser.add_argument('--num_examples', type=int, default=10)

    # 通用参数
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')
    args = parser.parse_args()
    # cuda版本
    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    # print("dict")
    # print(args.__dict__)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~")
    main(args)
