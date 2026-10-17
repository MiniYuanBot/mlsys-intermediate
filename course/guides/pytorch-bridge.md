---
title: PyTorch 的 6 小时桥接
---

# PyTorch 的 6 小时桥接

你已有自写训练框架的基础，不必重新学习 Python 或神经网络入门。本节帮助你读懂研究代码，并正确测量实验。

这 6h 用于按需熟悉 PyTorch API，不包含完成 lab3/lab6 全部 TODO 的时间。两周扫描的 2h 只取前两项。每项采用 15min 官方文档、35min 小程序与观察、10min 记录解释；已经掌握的项用最小验收确认后跳过。

| 时间 | 内容与官方材料                                                                                                                       | 动手验收                                                                       |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| 1h   | [Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)：leaf、requires_grad、no_grad/inference_mode | 比较 `x.grad`、冻结参数与可训练 adapter；解释冻结权重为何仍可传 input gradient |
| 1h   | [nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)：parameter/buffer/state_dict/train/eval             | 保存再恢复同一微型模型，固定输入输出相等；区别 eval 与关闭梯度                 |
| 1h   | [Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 与 CUDA 同步                                     | 输出 trace；比较同步/不同步计时的差异                                          |
| 1h   | [CUDA memory](https://docs.pytorch.org/docs/stable/notes/cuda.html)：allocated/reserved、stream、allocator                           | 用 lab6 记录 peak；解释 empty_cache 不能释放活 tensor                          |
| 1h   | [torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)                                         | lab3 eager/static/dynamic 每种跑一遍，区分首次调用与稳态                       |
| 1h   | [checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html)                                                                   | lab6 dropout 开启后比较梯度；解释 preserve_rng_state 和 non-reentrant          |

这一阶段用到 lab3/lab6 时，可以运行明确标记的 reference 来观察 API 行为，但不能算作个人实现通过。完整的梯度与性能验收留到后续 lab 单独完成。无需同时安装所有训练框架；上游 artifact 的依赖应放在独立环境中。

## 两小时扫描的最小程序

在 `labs/` 下新建自己的草稿，例如 `results/autograd_probe.py`，从下面的输入开始补全实验。这段程序只用于分别理解冻结、输入梯度和状态保存，不是 lab6 的答案。

```python
import torch
from torch import nn

torch.manual_seed(7)
layer = nn.Linear(3, 2, bias=False)
layer.weight.requires_grad_(False)
x = torch.tensor([[1.0, 2.0, -1.0]], requires_grad=True)
y = layer(x)
y.sum().backward()
print("weight grad:", layer.weight.grad)
print("input grad:", x.grad)
print("manual input grad:", layer.weight.sum(dim=0))
```

第一小时的验收：

1. 先手算 `sum(y)` 对 x 的导数，再检查打印结果；解释为什么 `weight.grad` 为 None 而 x.grad 存在。
2. 在独立副本中给前向加 `torch.no_grad()`，记录为何 backward 不再具有同一语义；不要用这一写法实现冻结基座上的 LoRA。
3. 不清零 x.grad，重新执行一次前向并对新输出 backward，观察梯度累积；说清训练循环为什么需要 zero_grad。

第二小时的验收：

1. 写一个含 `nn.Parameter` 和 `register_buffer` 的小 Module，打印 `named_parameters()`、`named_buffers()` 和 `state_dict()`，解释各自用途。
2. 保存并加载 state_dict 到新 Module，固定同一输入比较输出；记录权重恢复和随机状态恢复的区别。
3. 加入 dropout，比较 train/eval 输出；同时检查 eval 模式下是否仍可 backward。用结果区分“评测行为”和“关闭梯度”。

## 后四项如何留下证据

- Profiler：为一个矩阵乘导出 Chrome trace，标记 CPU enqueue 与 GPU kernel；另用 CUDA Event 和同步 wall clock 计时，解释二者测量范围。
- Memory：创建、保留、删除两个 tensor，分别记录 allocated/reserved；指出哪一步仍有 Python 引用，不能靠 empty_cache 释放它。
- Compile：从 `labs/` 运行 `python tools/record.py --label bridge-eager -- python lab3/torch_bench.py --mode eager`，再观察 static/dynamic；这检查提供的 workload，不要求此时实现 DP 或 CUDA fusion。
- Checkpoint：阅读 lab6 的 dropout 梯度测试，在临时 reference 副本中观察普通/重算路径；能解释 keep、RNG 和梯度比较的作用后再进入正式学生任务。

请分别保存这些产物，不要重复计为主方向 lab 的新增实验或原创研究贡献。

## 资料汇总

[环境设置](../../labs/setup.md) · [D3](../modules/03-dynamic-compilation.md) · [D6](../modules/06-memory-efficient-training.md)
