import torch
import torch.nn as nn
import typing
import sys
import logging

from .scheduler import linear_scheduler
from ..import function
from ... import ops, dependency


class MetaPruner:
    """
        Meta pruner for structural pruning. 

        Args:

            # Basic
            * model (nn.Module): A to-be-pruned model
            * example_inputs (torch.Tensor or List): dummy inputs for graph tracing.
            * importance (Callable): importance estimator. 
            * global_pruning (bool): enable global pruning. Default: False.
            * pruning_ratio (float): global channel sparisty. Also known as pruning ratio. Default: 0.5.
            * pruning_ratio_dict (Dict[nn.Module, float]): layer-specific pruning ratio. Will cover pruning_ratio if specified. Default: None.
            * max_pruning_ratio (float): the maximum pruning ratio. Default: 1.0.
            * iterative_steps (int): number of steps for iterative pruning. Default: 1.
            * iterative_pruning_ratio_scheduler (Callable): scheduler for iterative pruning. Default: linear_scheduler.
            * ignored_layers (List[nn.Module | typing.Type]): ignored modules. Default: None.
            * round_to (int): round channels to the nearest multiple of round_to. E.g., round_to=8 means channels will be rounded to 8x. Default: None.
            
            # Advanced
            * in_channel_groups (Dict[nn.Module, int]): The number of channel groups for layer input. Default: dict().
            * out_channel_groups (Dict[nn.Module, int]): The number of channel groups for layer output. Default: dict().
            * num_heads (Dict[nn.Module, int]): The number of heads for multi-head attention. Default: dict().
            * prune_num_heads (bool): remove entire heads in multi-head attention. Default: False.
            * prune_head_dims (bool): remove head dimensions in multi-head attention. Default: True.
            * head_pruning_ratio (float): head pruning ratio. Default: 0.0.
            * head_pruning_ratio_dict (Dict[nn.Module, float]): layer-specific head pruning ratio. Default: None.
            * customized_pruners (dict): a dict containing module-pruner pairs. Default: None.
            * unwrapped_parameters (dict): a dict containing unwrapped parameters & pruning dims. Default: None.
            * root_module_types (list): types of prunable modules. Default: [nn.Conv2d, nn.Linear, nn.LSTM].
            * forward_fn (Callable): A function to execute model.forward. Default: None.
            * output_transform (Callable): A function to transform network outputs. Default: None.

            # Deprecated
            * channel_groups (Dict[nn.Module, int]): output channel grouping. Default: dict().
            * ch_sparsity (float): the same as pruning_ratio. Default: None.
            * ch_sparsity_dict (Dict[nn.Module, float]): the same as pruning_ratio_dict. Default: None.
        """

    def __init__(
        self,
        # Basic
        model: nn.Module,
        example_inputs: torch.Tensor,
        # Callable表示是一个可调用对象类型
        importance: typing.Callable,
        # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning.
        global_pruning: bool = False,
        ch_sparsity: float = 0.5,  # channel/dim sparsity
        ch_sparsity_dict: typing.Dict[nn.Module, float] = None,
        max_ch_sparsity: float = 1.0,
        iterative_steps: int = 1,  # for iterative pruning
        # iterative_sparsity_scheduler(ch_sparsity, steps) 将 ch_sparsity 按照 iterative_steps进行等分
        iterative_sparsity_scheduler: typing.Callable = linear_scheduler,
        ignored_layers: typing.List[nn.Module] = None,

        # Advanced
        round_to: int = None,  # round channels to 8x, 16x, ...
        # for grouped channels.  记录不同模块分组操作（比如分组卷积）的分组数
        channel_groups: typing.Dict[nn.Module, int] = dict(),
        # for consecutive channels. 
        # layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
        consecutive_groups: typing.Dict[nn.Module, int] = dict(),
        # pruners for customized layers
        customized_pruners: typing.Dict[typing.Any,
                                        function.BasePruningFunc] = None,
        # unwrapped nn.Parameters like ViT.pos_emb
        unwrapped_parameters: typing.List[nn.Parameter] = None,
        root_module_types: typing.List = [
            ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],  # root module for each group
        root_instances: typing.List = None,
        forward_fn: typing.Callable = None,
        output_transform: typing.Callable = None,
        enable_index_mapping: bool = False,
    ):
        self.model = model
        self.importance = importance
        self.ch_sparsity = ch_sparsity
        self.ch_sparsity_dict = ch_sparsity_dict if ch_sparsity_dict is not None else {}
        self.max_ch_sparsity = max_ch_sparsity
        self.global_pruning = global_pruning

        self.channel_groups = channel_groups
        self.consecutive_groups = consecutive_groups
        self.root_module_types = root_module_types
        self.root_instances = root_instances
        self.round_to = round_to

        # Build dependency graph
        # 首先构造输入，前向运行一次模型，得到模型对应的依赖图
        self.DG = dependency.DependencyGraph().build_dependency(
            model,
            example_inputs=example_inputs,
            forward_fn=forward_fn,  # forward_fn如果为None，out = model(**example_inputs)
            output_transform=output_transform,
            # unwrapped_parameters可以用于指定未封装在标准nn中的参数的修剪维度
            # 比如Transformers中的cls_token和Conv Next的layer_scale
            unwrapped_parameters=unwrapped_parameters, 
            customized_pruners=customized_pruners,  # 如果有customized_pruners用于特定层的剪枝器，在DG上注册一下就好
        )
        '''
        list1 = ['Google', 'Runoob', 'Taobao']
        list2=list(range(5)) # 创建 0-4 的列表
        list1.extend(list2)  # 扩展列表
        print ("扩展后的列表：", list1)
        扩展后的列表： ['Google', 'Runoob', 'Taobao', 0, 1, 2, 3, 4]
        '''
        # 这里即用扩展的方式将ignored_layers的内容存储到self.ignored_layers
        self.ignored_layers = []
        if ignored_layers:
            for layer in ignored_layers:
                self.ignored_layers.extend(list(layer.modules()))

        self.iterative_steps = iterative_steps
        self.iterative_sparsity_scheduler = iterative_sparsity_scheduler
        self.current_step = 0

        # Record initial status
        # 记录每个层初始的输入和输出维度
        self.layer_init_out_ch = {}
        self.layer_init_in_ch = {}
        # module2node：模块到依赖图结点的字典
        for m in self.DG.module2node.keys():
            # module2type：获取模块对应的类型
            if ops.module2type(m) in self.DG.REGISTERED_PRUNERS:
                self.layer_init_out_ch[m] = self.DG.get_out_channels(m)
                self.layer_init_in_ch[m] = self.DG.get_in_channels(m)

        # global channel sparsity for each iterative step
        # 获取全局稀疏度对应的稀疏度迭代计划
        self.per_step_ch_sparsity = self.iterative_sparsity_scheduler(
            self.ch_sparsity, self.iterative_steps
        )

        # The customized channel sparsity for different layers
        # 针对不同层的定制化稀疏率
        self.ch_sparsity_dict = {}
        if ch_sparsity_dict is not None:
            for module in ch_sparsity_dict:
                sparsity = ch_sparsity_dict[module]
                # 取出每一个模块的子模块
                for submodule in module.modules():
                    # prunable_types是获取所有子模块中支持剪枝的模块类别元组集合
                    prunable_types = tuple([ops.type2class(
                        prunable_type) for prunable_type in self.DG.REGISTERED_PRUNERS.keys()])
                    # 如果子模块是支持剪枝的，那么为其添加定制的迭代稀疏调度
                    if isinstance(submodule, prunable_types):
                        self.ch_sparsity_dict[submodule] = self.iterative_sparsity_scheduler(
                            sparsity, self.iterative_steps
                        )

        # model.modules()迭代遍历模型的所有nn.Module子类
        # detect group convs & group norms
        # 检测分组卷积核分组norm 并且获取它们的分组数
                        
        # 这里要求 m.groups != m.out_channels 的原因应该是如果分组数和输出通道一样
        # 那么每个组只对应一个输出通道，而原本是要求所有组都必须保持相同的大小
        # 这个时候修剪输出通道就相当于直接将该组删除，就不存在要手动处理的额外依赖了 
        for m in self.model.modules():
            if isinstance(m, ops.TORCH_CONV) \
                and m.groups > 1 \
                    and m.groups != m.out_channels:
                self.channel_groups[m] = m.groups
            if isinstance(m, ops.TORCH_GROUPNORM):
                self.channel_groups[m] = m.num_groups
        
        # 启用全局剪枝
        if self.global_pruning: # TODO: Support both ch_groups and consecutive_groups in a single forward
            # 统计所有分组中的可剪枝通道数
            initial_total_channels = 0
            # self.DG.get_all_groups 获取所有的依赖分组
            for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
                # 获取当前组的模块的操作分组数
                ch_groups = self.get_channel_groups(group)
                consecutive_groups = self.get_consecutive_groups(group)
                # utils.count_prunable_out_channels( group[0][0].target.module )

                # 根据情况更新总通道数，其中模块操作分组会划分到同一个依赖组，因此其对应的可剪枝通道数要除以组数
                if ch_groups > 1:
                    initial_total_channels += (self.DG.get_out_channels(
                        group[0][0].target.module) // ch_groups)
                elif consecutive_groups > 1:
                    initial_total_channels += (self.DG.get_out_channels(
                        group[0][0].target.module) // consecutive_groups)
                else:
                    initial_total_channels += self.DG.get_out_channels(group[0][0].target.module) 
                
            self.initial_total_channels = initial_total_channels
        
        # 启用index映射
        if enable_index_mapping:
            for node in self.DG.module2node.values():
                node.enable_index_mapping = True
    

    def pruning_history(self):
        return self.DG.pruning_history()

    def load_pruning_history(self, pruning_history):
        self.DG.load_pruning_history(pruning_history)

    # 获取当前迭代的模块目标稀疏度
    def get_target_sparsity(self, module):
        s = self.ch_sparsity_dict.get(module, self.per_step_ch_sparsity)[
            self.current_step]
        return min(s, self.max_ch_sparsity)

    # 将当前迭代步数设为0
    def reset(self):
        self.current_step = 0

    # 模型正则化方法
    def regularize(self, model, loss):
        """ Model regularizor
        """
        pass
    
    # 执行一步迭代剪枝
    def step(self, interactive=False):
        self.current_step += 1
        # 首先分为全局和局部剪枝
        if self.global_pruning:
            # 然后分为可交互和不可交互剪枝
            if interactive: # 交互模式返回生成器对象，可以在group.prune()之前做一些自己想做的事情
                return self.prune_global()
            else: # 非交互模式直接遍历完生成器并执行 group.prune()
                for group in self.prune_global():
                    group.prune()
        else:
            if interactive:
                return self.prune_local()
            else:
                for group in self.prune_local():
                    group.prune()

    # 调用重要性评估器评估当前组的重要性
    def estimate_importance(self, group, ch_groups=1, consecutive_groups=1):
        return self.importance(group, ch_groups=ch_groups, consecutive_groups=consecutive_groups)

    # 检查group的稀疏度是否还需要剪枝
    def _check_sparsity(self, group):
        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            if dep.target.type == ops.OPTYPE.PARAMETER:
                continue
            if self.DG.is_out_channel_pruning_fn(pruning_fn):
                target_sparsity = self.get_target_sparsity(module)
                layer_out_ch = self.DG.get_out_channels(module)
                if layer_out_ch is None: continue
                if layer_out_ch < self.layer_init_out_ch[module] * (
                    1 - self.max_ch_sparsity
                ) or layer_out_ch == 1:
                    return False

            elif self.DG.is_in_channel_pruning_fn(pruning_fn):
                layer_in_ch = self.DG.get_in_channels(module)
                if layer_in_ch is None: continue
                if layer_in_ch < self.layer_init_in_ch[module] * (
                    1 - self.max_ch_sparsity
                ) or layer_in_ch == 1:
                    return False
        return True

    # 获取当前组的模块的操作分组数
    def get_channel_groups(self, group):
        if isinstance(self.channel_groups, int):
            return self.channel_groups
        for dep, _ in group:
            module = dep.target.module
            if module in self.channel_groups:
                return self.channel_groups[module]
        return 1  # no channel grouping
    
    # 获取当前模块的连续分组数
    def get_consecutive_groups(self, group):
        if isinstance(self.consecutive_groups, int):
            return self.consecutive_groups
        for dep, _ in group:
            module = dep.target.module
            if module in self.consecutive_groups:
                return self.consecutive_groups[module]
        return 1  # no channel grouping

    # 局部剪枝
    def prune_local(self):
        # 超过步数直接退出
        if self.current_step > self.iterative_steps:
            return
        # 从依赖图中枚举所有的剪枝组
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
            # check pruning rate
            if self._check_sparsity(group):
                # 获取组的root依赖
                module = group[0][0].target.module
                pruning_fn = group[0][0].handler

                ch_groups = self.get_channel_groups(group)
                consecutive_groups = self.get_consecutive_groups(group)
                imp = self.estimate_importance(group, ch_groups=ch_groups, consecutive_groups=consecutive_groups)
                # print(imp.shape)
                if imp is None: continue
                current_channels = self.DG.get_out_channels(module)
                target_sparsity = self.get_target_sparsity(module)
                # 计算需要剪枝的通道数
                n_pruned = current_channels - int(
                    self.layer_init_out_ch[module] *
                    (1 - target_sparsity)
                )
                # print(n_pruned, target_sparsity)
                if self.round_to:
                    n_pruned = n_pruned - (n_pruned % self.round_to)
                    
                if n_pruned <= 0:
                    continue
                
                # 如果需要分组，我们看第一个分组的通道即可，因为额外依赖已经考虑
                if ch_groups > 1:
                    imp = imp[:len(imp)//ch_groups]

                if consecutive_groups > 1:
                    imp = imp.view(-1, consecutive_groups).sum(1)

                # 获取将imp从小到大排序后的下标顺序
                imp_argsort = torch.argsort(imp)
                
                if ch_groups > 1:
                    # 通道分组对应的n_pruned也需要减少
                    pruning_idxs = imp_argsort[:(n_pruned//ch_groups)]
                    # 计算通道分组的大小
                    group_size = current_channels//ch_groups
                    # 在每一个通道分组中取对应的剪枝部分
                    pruning_idxs = torch.cat(
                        [pruning_idxs+group_size*i for i in range(ch_groups)], 0)
                elif consecutive_groups > 1:
                    pruning_groups = imp_argsort[:(n_pruned//consecutive_groups)]
                    # 和ch_groups，consecutive_groups直接描述每一个连续通道组的大小
                    group_size = consecutive_groups
                    pruning_idxs = torch.cat(
                        [torch.tensor([j+group_size*i for j in range(group_size)])
                        for i in pruning_groups], 0)
                else:
                    pruning_idxs = imp_argsort[:n_pruned]

                # 获取pruning_fn对应的剪枝组
                group = self.DG.get_pruning_group(
                    module, pruning_fn, pruning_idxs.tolist())
                # 检查组以避免过度修剪。如果有足够的可prunable元素，则返回True。
                if self.DG.check_pruning_group(group):
                    yield group

    # 全局剪枝
    def prune_global(self):
        # 如果当前步数已经大于迭代步数，直接退出
        if self.current_step > self.iterative_steps:
            return
        global_importance = []
        # 枚举依赖图的每个组
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
            # 检查稀疏度，看是否还需要剪枝
            if self._check_sparsity(group):
                ch_groups = self.get_channel_groups(group)
                consecutive_groups = self.get_consecutive_groups(group)
                # 评估当前组的重要性
                imp = self.estimate_importance(group, ch_groups=ch_groups, consecutive_groups=consecutive_groups)
                if imp is None: continue
                if ch_groups > 1:
                    imp = imp[:len(imp)//ch_groups]
                if consecutive_groups > 1:
                    imp = imp.view(-1, consecutive_groups).sum(1)
                # 将评估结果添加到global_importance列表中
                global_importance.append((group, ch_groups, consecutive_groups, imp))

        # 取出重要性得分并cat
        imp = torch.cat([local_imp[-1]
                        for local_imp in global_importance], dim=0)
        # print("meta-Pruner-img.shape:")
        print(imp.shape, len(global_importance))
        # logging.info("meta-Pruner-img.shape:", imp.shape, len(global_importance))
        # 当前目标稀疏度
        target_sparsity = self.per_step_ch_sparsity[self.current_step]
        # 需要剪枝的通道数 即 总通道数减去要保留的通道数
        n_pruned = len(imp) - int(
            self.initial_total_channels *
            (1 - target_sparsity)
        )
        # print("meta-Pruner-n_pruned:")
        print(n_pruned, target_sparsity, self.initial_total_channels)
        # logging.info("meta-Pruner-n_pruned:", n_pruned, target_sparsity, self.initial_total_channels)
        if n_pruned <= 0:
            return
        # 找出重要性最低的n_pruned个通道，返回值和下标
        topk_imp, _ = torch.topk(imp, k=n_pruned, largest=False)
        
        # global pruning through thresholding
        # 需要剪掉的最大重要性
        thres = topk_imp[-1]
        for group, ch_groups, consecutive_groups, imp in global_importance:
            module = group[0][0].target.module
            pruning_fn = group[0][0].handler
            # 需要剪枝的下标
            pruning_indices = (imp <= thres).nonzero().view(-1)
            
            if pruning_indices.size(-1) == 0:
                continue
            if ch_groups > 1:
                group_size = self.DG.get_out_channels(module)//ch_groups
                pruning_indices = torch.cat(
                    [pruning_indices+group_size*i for i in range(ch_groups)], 0)
            if consecutive_groups > 1:
                group_size = consecutive_groups
                pruning_indices = torch.cat(
                    [torch.tensor([j+group_size*i for j in range(group_size)])
                    for i in pruning_indices], 0)
            # 将剪枝下标向下取整成 round_to 的倍数
            if self.round_to:
                n_pruned = len(pruning_indices)
                n_pruned = n_pruned - (n_pruned % self.round_to)
                pruning_indices = pruning_indices[:n_pruned]
            # 获取pruning_fn对应的剪枝组
            group = self.DG.get_pruning_group(
                module, pruning_fn, pruning_indices.tolist())
            # 检查组以避免过度修剪。如果有足够的可prunable元素，则返回True。
            if self.DG.check_pruning_group(group):
                yield group