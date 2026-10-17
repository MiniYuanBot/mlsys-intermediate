---
tags: [mlsys, lab, lora, checkpoint, offload]
---

# Lab 6：可训练适配器、激活规划与真实 CPU offload

[课程 D6](../../course/modules/06-memory-efficient-training.md) · [实验索引](../index.md) · [评分](grade.md) · [资源改编](resources.md) · [参考导读](../../solutions/lab6/README.md)

> [!important] 三阶段都要写实质代码
> A 实现小型 LoRA 模块的生命周期；B 实现重计算、saved-tensor 统计和两类预算规划；C 实现 autograd offload、混合策略，以及不等大小微批的训练。17 个 TODO 都在注释中说明了契约，并配有完整参考实现和可运行的验证。最后通过实际执行 optimizer step 绘制显存曲线。

前置：foundations、基本 PyTorch autograd、矩阵求导；欠缺接口知识先做 [PyTorch 桥接](../../course/guides/pytorch-bridge.md)。预计 **26h**：A 6h、B 8h、C 9h、报告 3h，D6 课程另 5h。首次安装 CUDA 环境的时间单独记录，不计入算法学习时间。**命令均在 `mlsys-intermediate/labs/` 下运行**。

CPU 可以完成全部算法正确性检查和实际训练；host/device 搬运及 allocator 峰值需要在 CUDA 上验证。Mac 的 CPU 路径不生成虚构的 H2D/D2H 数据。默认网络使用小型 LayerNorm→LoRA→GELU→LoRA→dropout 残差块，不包含 attention 和词表，因此本实验不属于完整的 QLoRA 复现。

## 文件与任务清单

| 文件                   | 责任                                                  |
| ---------------------- | ----------------------------------------------------- |
| `src/train_student.py` | A1–A7、B1–B5、C1–C5；所有实现区都有 BEGIN/END TODO    |
| `model.py`             | 提供 Block/Model，保留 dropout 用于随机性检查         |
| `test.py`              | 分阶段 14 个测试方法；内含穷举预算、梯度和设备 oracle |
| `bench.py`             | 3 seeds×2 widths×4 policies=24 条训练测量             |
| `exercise.py`          | 测试和测量统一入口                                    |

有效填空代码统计见[工作量审计](../docs/workload.md)。代码量只用于衡量实现规模，测试难度、推导和调试需要另行考虑；不要用空白、注释或重复分支凑行数。

```bash
python tools/run.py --lab 6 --impl student --mode test --part A --device cpu
python tools/run.py --lab 6 --impl student --mode test --part B --device cpu
python tools/run.py --lab 6 --impl student --mode test --part C --device cpu
python tools/run.py --lab 6 --impl student --mode test --device cuda
```

B/C 依赖前面的阶段，请先通过前置阶段的测试再运行。初始骨架尚未实现，测试会失败。运行参考答案时，runner 只覆盖临时副本，保留原始学生文件。

## A 基础：让 LoRA 从公式变成可维护模块（6h）

### A0 形状、梯度和内存手算（0.5h）

设 W=[out,in]、A=[r,in]、B=[out,r]。写出 `Y=XWᵀ+s(XAᵀ)Bᵀ+bias`，推导 X/A/B 梯度，说明 B 初始为零时 A 的首次梯度为何为零。算 in=out=1024、r=8 的基座与可训练参数 bytes。

### A1 `forward`（0.75h）

保留任意前导维度，bias 恰好加一次，merged 状态跳过低秩分支。冻结 W 只是不求 W 的参数梯度；不能用 `no_grad()` 切断 X→基座分支的输入梯度。

测试使用非零 B 和给定的上游梯度，通过独立矩阵式 `GW+sGBA` 检查反传，避免只验证输出而漏掉梯度错误。新增三维输入测试时，应通过合并矩阵计算预期值，不能调用自己的 forward 作为 oracle。

### A2/A3 `merge`、`unmerge`（1h）

merge 仅允许 eval 模式，重复调用幂等；将 delta 加进 W，保存原 W 的精确备份。unmerge 用备份恢复，释放备份，不靠多次浮点减法近似恢复。

解释备份为什么增加峰值内存，以及为何这不是生产推理最省内存的保存方式。测试连续三轮 merge/unmerge，要求原 W bitwise 相等、输出在浮点容差内一致。训练前必须 unmerge；本教学接口没有自动覆盖 `train()` 做恢复。

### A4 `inject_lora`（1.5h）

接收精确 module 名称列表，替换嵌套 `nn.Linear`，冻结原模型，只让新 A/B 可训练。保留 weight、bias、device、dtype 和 train/eval 状态。先保存 named_modules 快照，验证所有目标后再替换。

考虑列表前一项合法、后一项不存在的情况：接口必须在修改模型前拒绝请求，不能只完成一部分替换。安排参数初始化与 `.to()` 的顺序时，要考虑原模型可能已在 CUDA 上。

### A5/A6 `adapter_state`、`load_adapters`（1.5h）

保存的是 cloned CPU A/B 与 scale，不包含 W。模型随后变化不能修改已经保存的对象。load 必须检查完整名称集合、字段、shape、finite 和 scale，并先暂存转换到目标 device/dtype 的张量。源 float64 仍有限的值可能在转 FP32 时溢出；转换后的所有张量也必须保持 finite，全部验证后才 copy。最后一项失败时前面模块仍不能改变。

测试会将最后一个 adapter 的 shape 设错，同时修改前面一个 adapter 的值，检查加载失败后整个模型是否保持原样。merged 模型不能直接导出或加载适配器。这是用于教学的内存字典格式，不提供 PEFT checkpoint 文件格式兼容层。

### A7 `parameter_bytes`（0.75h）

分列 trainable/frozen，避免共享 Parameter 重复计数。说明为什么这个数没有 optimizer state、grad、activation、temporary、allocator reserve；不能用参数量直接预测训练峰值。

阶段验收：`ATests` 的 5 个方法，包含 A2/A3 和 A5/A6 的联合状态测试。对应原理和源码入口见[资源说明](resources.md)。

## B 核心：测量激活并构造预算计划（8h）

### B1 `run_blocks`（1h）

完整检查 keep 长度和 bool 类型；保留块正常调用，重算块显式使用 `use_reentrant=False,preserve_rng_state=True`。不得 detach 边界或忽略列表尾部。

同一模型、输入、seed 下比较 retain/checkpoint/mixed 的全部 trainable 梯度。测试保留 dropout；临时关闭 RNG 保存应能触发失败，记录最小失败后恢复。只有 loss 相同不足以证明反传正确。

### B2 `plan_keep`（1.5h）

实现 0/1 knapsack：sizes 为正整数，cost 是非负有限的重算代价，预算非负整数。输出 mask 最大化保留收益。每轮只扩展上轮状态，防止重复选一个 block。

提供 oracle 穷举四块的全部 16 种选择，扫描 18 个预算。再手算贪心反例 `[10,20,30] / [60,100,120] / budget50`，说明收益密度不保证最优。

### B3 `SavedInventory.pack`（1h）

利用 `saved_tensors_hooks` 统计一个 forward 为反传保存的非参数 storage。views 共用 storage，按 device+storage pointer 去重；参数的 view 也要排除。返回 detached tensor，防止 packed 对象额外保留输入 autograd graph。

这份清单记录整个 forward 遇到的存储，并已去重，**不代表随时间变化的峰值活跃内存**。view 的 storage 大于 view.numel 时，按 storage bytes 统计。阅读 PyTorch hooks 教程理解 pack/unpack 必须还原的内容。

### B4 `plan_modes`（2h）

升级为多选背包：每层候选 retain/checkpoint/offload 各有 `(retained_units, measured_ms)`，必须选择一种，在预算下最小化成本。需要处理预算不可行的情况，并实施 Pareto 剪枝：更小且更便宜的状态支配更大、更昂贵的状态。

测试会独立穷举三块的全部候选组合，检查结果是否满足预算、总成本是否最优、路径长度是否正确，不要求返回某个固定 mask。这个近似加和模型省略了张量的跨层寿命和 workspace，因此还需要在 C 阶段测量完整训练步，验证规划结果。

### B5 `calibrate_blocks`（2.5h）

为每块实际测 retain/checkpoint forward+backward 时间和 saved storage。输入逐块推进并 detach；每次清梯度，先 warmup 一次，再测若干次取中位数。CUDA 计时前后同步，不能只测异步 launch。

默认 benchmark 的 planner 只使用实际校准的 retain/checkpoint 候选；offload 的整步成本单独测量，不能编造 PCIe 代价来让 planner 优先选择它。把测得 bytes 转为 KiB 时向上取整，并在报告写出单位。

校准中临时值、参数梯度、重计算中短寿命张量未全部进入 saved inventory。请解释“预测用量小于 budget”为什么不保证实际 allocator 峰值小于同样数值。

## C 前沿：把内存策略接进真实训练（9h）

### C0 阅读与模型边界（1h）

阅读 [saved tensors hooks 教程](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html) 的 CPU 保存示例与 [activation checkpointing 技术说明](https://pytorch.org/blog/activation-checkpointing-techniques/)。你需要亲手实现 pack/unpack 生命周期，再与官方接口对照。仅调用 API 不足以算作完整训练系统的复现。

### C1/C2 `CpuOffload.pack/unpack`（2h）

参数 storage、CPU tensor、小于阈值的 tensor 走 detached identity。其余 CUDA tensor 同步复制到独立 CPU 存储，payload 保留原 device。unpack 允许多次调用，不能销毁 host 内容；每次真正恢复都计 H2D copy。

实现并解释 counters：packed bytes 是累计传输 payload，不是 host 峰值；D2H/H2D copies 是次数；restores 包含 identity。输入在 pack 后改变时，已经复制的 host payload 应保持原值。

本实验先用同步 copy 验证正确性，不据此声称已隐藏传输时间。异步实现列在挑战中，需要补充 stream/event、所有权和生命周期测试。

### C3 `run_policy`（1.5h）

每块可以 retain、checkpoint 或 offload。进入 offload context 只控制 forward 保存哪些张量；退出 context 后，backward 仍需能调用当时的 unpack hook。没有 offload owner 时必须提前报错。

参考测试比较四种策略的输入梯度和全部 trainable 参数梯度，保留 dropout 并恢复相同 RNG。CUDA 下另外要求实际拷贝发生。CPU 上的正确性测试只能验证 identity fallback。

### C4 `train_accumulated`（2h）

执行一次完整 optimizer step，微批大小可以不同。每个微批 MSE sum 除以全部 target.numel，再 backward；只 zero_grad 一次、step 一次。可选 clip 在累积完成后执行，不能每微批独立裁剪。

测试将大小 5 的 batch 分成 2+3，与完整 batch 做 SGD 更新比较。这里关闭 dropout，因为不同分块的随机数消耗并不保证与拼接 batch 一致；C3 已独立验证 dropout 重算语义。

增加自己的三微批不均匀例子。若使用 mean loss，写出正确的权重；简单平均微批 loss 会偏重小 batch。

### C5 `measure_policy` 与校准闭环（2.5h）

实现真实 AdamW step 测量：先 warmup 初始化 optimizer state，清梯度，重置 CUDA 峰值，再重复 fw/bw/step。学习率设为 0 以便策略比较时权重不变，但优化器计算和状态分配仍真实发生。

默认 `bench.py` 覆盖 width64/128、4 blocks、3 seeds，以及 retain/checkpoint/offload/calibrated 四策略，共 24 行 JSON。输入分成 2+1 微批，测三步平均。每行包含 loss 列表、时间、D2H bytes/copies、H2D copies、allocator allocated/reserved、calibration 与规划路径。

```bash
python tools/run.py --lab 6 --impl reference --mode test --device cuda
python tools/run.py --lab 6 --impl reference --mode bench --device cuda --output results/lab6/reference.json
python tools/extract.py results/lab6/reference.json
python tools/extract.py results/lab6/reference.json --table 0 --out results/lab6/reference.csv
```

CPU bench 也实际训练，但峰值字段为 null，传输为零。CUDA allocator 数包含该进程仍持有的 base 模型/输入及当前测量对象，不能解释成单一激活张量大小。模型构造和校准在稳态计时外，报告要另列准备开销。

## 报告（3h）

提交实现、独立反例、24 行原始结果、运行环境和 `reports/lab6.md`。两图分别为 peak allocated→step time、预测 retained KiB→实测峰值；按 width 分面并保留 seed。检查同一输入/策略间 loss 一致，再解释性能。

回答：LoRA/checkpoint/offload 各自改变哪项内存？saved storage 与 peak 的偏差来自什么？同步 offload 是否可能既慢又不省峰值？哪个指标不能从 CPU 推到 GPU？哪些结果支持进一步做真实 Transformer 实验？

## 可选挑战

以下挑战另计 8–20h，可选择一项继续：

- 实现异步 pinned-host offload，加入 events 与主机峰值账本，验证传输依赖和实际重叠。
- 用 PyTorch selective checkpoint 构造操作级保存策略，与当前每块决策对照。
- 在 0.6B 真实模型上做短上下文 adapter 训练，重新检查梯度、质量和峰值。
- 将激活生命周期纳入规划，比较预测内存与实测峰值的偏差。

这些扩展用于研究单个机制，不等于完整论文复现。

## 资料汇总

[PEFT adapter checkpoint](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/checkpoint.md) · [PEFT LoRA layer](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py) · [PyTorch checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html) · [hooks 教程](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html) · [改编对应](resources.md) · [评分](grade.md) · [实测记录](../docs/validation.md)
