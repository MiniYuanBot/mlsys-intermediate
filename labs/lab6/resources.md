# Lab 6 资源与改编对应

检索/阅读日期 2026-10-17；优先官方教程和原项目。实验为独立教学实现，不是对上游库的行为兼容承诺。

| 资源                                                                                                                       | 阅读位置                                               | 对应任务                                     | 省略与差异                                                                    |
| -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | -------------------------------------------- | ----------------------------------------------------------------------------- |
| [PEFT LoRA layer](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py)                             | `Linear.forward`、merge/unmerge、base layer 与低秩分支 | A1–A4 forward、幂等合并、嵌套注入            | 只有单 adapter；无 DoRA、量化基座、多 adapter；unmerge 用精确备份简化数值语义 |
| [PEFT checkpoint 指南](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/checkpoint.md)           | adapter_model 与 adapter_config 的职责                 | A5/A6 只保存适配器和 scale，验证后加载       | 自定义内存字典，不提供 safetensors/PEFT 文件互操作                            |
| [PyTorch checkpoint API](https://docs.pytorch.org/docs/stable/checkpoint.html)                                             | non-reentrant 与 preserve_rng_state                    | B1、C3 checkpoint 与 dropout 梯度对照        | 按块处理，不做算子级图切分                                                    |
| [官方 saved-tensor hooks 教程](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html) | pack/unpack 与 saving tensors to CPU                   | B3 storage 去重，C1/C2 拷贝所有权及反传恢复  | 同步 CPU 拷贝，无多 stream overlap；累积 bytes 不代表峰值                     |
| [Activation Checkpointing Techniques](https://pytorch.org/blog/activation-checkpointing-techniques/)                       | selective checkpoint 与 memory-budget 视角             | B4/B5 多选规划、实测候选→计划→C5 全步验证    | 自己实现块级 DP；不是 PyTorch 图级 memory-budget API 的复现                   |
| [Stanford CS336 Assignment 2](https://github.com/stanford-cs336/assignment2-systems)                                       | 系统测量、独立正确性对照的课程组织                     | 测量和正确性分离、warmup、相同输入、多 shape | 使用 LoRA 微型残差网络，训练吞吐不可外推完整 Transformer                      |

建议先结合[讲义 §5](../../course/modules/06-memory-efficient-training.md)的子集枚举与预算阶梯图理解保留/重算取舍，再用当前实验实际输出作图；官方 checkpoint 图解与内存查看器见[可视化阅读索引](../../course/readings/visualizations.md)。D6 的 [QLoRA](https://arxiv.org/abs/2305.14314)、[FlexLLM](https://github.com/FlexLLM/artifact) 用于认识更大系统问题；本 lab 没有 NF4、double quantization 或推理训练共服务，不把它们当作已完成的复现。
