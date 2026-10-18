---
tags: [mlsys, lab, lora, checkpoint, offload, autograd]
---

# Lab 6：LoRA 生命周期、激活规划、saved-tensor offload 与自定义 autograd.Function

[课程 D6](../../course/modules/06-memory-efficient-training.md) · [实验索引](../index.md) · [评分标准](grade.md) · [资源与改编说明](resources.md) · [参考导读](../../solutions/lab6/README.md)

> [!important] 完成定义
> A 实现一个能注入、合并、保存和加载的 LoRA 线性层，并按 storage 统计参数字节。B 实现非重入 checkpoint、saved-tensor 清单、两种预算背包和逐块实测校准，并与 `torch.compile` 的 `activation_memory_budget` 对比。C 实现 saved-tensor hook 的同步 CPU offload、混合策略执行和不等大小微批的一次 optimizer step。R 写三个比 eager autograd 存得更少的 `autograd.Function`（LoRA-FA、重算 GELU 的融合 LoRA MLP、只存种子的 LoRA dropout），用 float64 gradcheck、手算字节数和 CUDA 峰值证明它们真的省了，再在 Qwen3-0.6B 的宽度上测显存与时间的权衡。每个 TODO 都有独立 oracle 和至少一个错误注入。

前置：foundations 的 autograd 和矩阵求导，D6 讲义 §1–§7；接口不熟先看 [PyTorch 桥接](../../course/guides/pytorch-bridge.md)。**下列命令都从 `mlsys-intermediate/labs/` 执行。** 预计 **30h**：A 5h、B 7h、C 7h、R 8h、报告 3h，另加 D6 讲义 5h。参考实现几秒钟就能跑完，不能拿来估计自己的进度。

## 0. 环境

本 lab 的依赖和其他 lab 互不累加。先安装、再检查：

```bash
bash tools/setup_env.sh --lab 6 && bash tools/check_env.sh --lab 6
```

Windows 原生（PowerShell）只做 CPU 部分：

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1 -Lab 6
powershell -ExecutionPolicy Bypass -File tools\check_env.ps1 -Lab 6
```

| 阶段 | 需要                             | WSL2 / Linux | Windows 原生 | macOS       |
| ---- | -------------------------------- | ------------ | ------------ | ----------- |
| A    | PyTorch；GPU 部分另需 NVIDIA GPU | 可做         | 仅 CPU 部分  | 仅 CPU 部分 |
| B    | PyTorch；GPU 部分另需 NVIDIA GPU | 可做         | 仅 CPU 部分  | 仅 CPU 部分 |
| C    | PyTorch；GPU 部分另需 NVIDIA GPU | 可做         | 仅 CPU 部分  | 仅 CPU 部分 |
| R    | PyTorch；GPU 部分另需 NVIDIA GPU | 可做         | 仅 CPU 部分  | 仅 CPU 部分 |

CPU 能完成全部正确性检查：梯度、保存集字节、offload 的拷贝路径（测试在 CPU 上让 `CpuOffload` 把 CPU 张量拷到新的 host 存储，计数和 dtype 往返都会被检查）。只有三类证据需要 `--device cuda`：设备显存在前向后确实被释放、`max_memory_allocated` 峰值、CUDA event 计时。它们只能在 WSL2 或 Linux 上做。

macOS 上可以用 `--mode bench --part R --device mps` 采样 MPS 显存：MPS 没有峰值接口，也不支持 float64 gradcheck，所以只报告前向后的 `current_allocated_memory` 增量，峰值一栏为 null；`--mode test --device mps` 会以退出码 3 提示改用 `--device cpu`。缺少依赖时，runner 只打印一行 `MISSING ...` 和 `fix:`，退出码 3；在 Windows 上请求 CUDA，`fix:` 给出一条可以直接复制的 WSL 命令。各平台细节见[环境配置](../setup.md)。

## 文件和实施顺序

| 文件                     | 学生负责            | 已提供                                                                                       |
| ------------------------ | ------------------- | -------------------------------------------------------------------------------------------- |
| `src/train_student.py`   | A1–A7、B1–B5、C1–C5 | `storage_key`、`SavedInventory.unpack/context`、`CpuOffload.context`                         |
| `src/fused_student.py`   | R1–R4               | `dropout_mask`、三个包装函数 `lora_fa_linear`、`lora_mlp`、`lora_dropout_linear`             |
| `model.py`               | 不需要修改          | A–C 的 `Block`/`Model`；R 的 `MLPBlock`（7 种执行方式）和手算保存集 `predicted_saved_floats` |
| `test.py`、`test_r.py`   | 可以追加自己的反例  | 独立 oracle，逐项 PASS/FAIL                                                                  |
| `checks.py`              | 不需要修改          | PASS/FAIL 报告器                                                                             |
| `bench.py`、`bench_r.py` | 可以增加变量        | A–C 策略矩阵与 `torch.compile` 预算基线；R 变体乘序列长度                                    |
| `exercise.py`            | 不需要修改          | 阶段与设备选择                                                                               |

每对 `BEGIN TODO` 和 `END TODO` 之间是待实现区域，前面的 docstring 或注释写了输入、输出、不变量和边界。参考实现在 `solutions/lab6/src/` 的同名文件里。不要为了通过而删改原测试。有效代码量见[自动统计](../docs/workload.md)。

```bash
python tools/run.py --lab 6 --impl student --mode test --part A
python tools/run.py --lab 6 --impl student --mode test --part B
python tools/run.py --lab 6 --impl student --mode test --part C
python tools/run.py --lab 6 --impl student --mode test --part R
python tools/run.py --lab 6 --impl student --mode test --part all --device cuda   # WSL
python tools/run.py --lab 6 --impl student --mode bench --part all --device cuda  # WSL
python tools/run.py --lab 6 --impl student --mode bench --part R --device mps     # macOS，可选
```

测试逐项打印 `PASS <名称>` 或 `FAIL <名称>: expected ... vs actual ...`，一个检查失败不会中断后面的检查；未完成的 TODO 抛出 `NotImplementedError("A1")`，被记为所在测试函数的一条 `FAIL ...: TODO A1 not implemented`。每个阶段末尾有 `SUMMARY`，有任何 FAIL 就以非零码退出。B 依赖 A1、C 依赖 B1 和 B3，R 依赖 A1 的 `LoRALinear` 和 B3 的 `SavedInventory`，请按顺序完成。runner 用临时副本，不覆盖你的文件。`--device cuda` 的运行会自动加 `/tmp/mlsys-gpu.lock`。

## A 基础：LoRA 从公式到可维护的模块（5h）

### A0 手算（0.5h）

$W$ 是 $[d_{\mathrm{out}},d_{\mathrm{in}}]$、$A$ 是 $[r,d_{\mathrm{in}}]$、$B$ 是 $[d_{\mathrm{out}},r]$。写出 $Y=XW^\top+s\,(XA^\top)B^\top+b$，推出 $dX$、$dA$、$dB$（讲义 §4.1），说明 $B=0$ 时 $A$ 的首次梯度为什么为零。算 $d_{\mathrm{in}}=d_{\mathrm{out}}=1024$、$r=8$ 的基座和可训练字节，再按讲义 §1.3 算 Qwen3-0.6B 的两种模型状态。

### A1 `LoRALinear.forward`（0.75h）

保留任意前导维度，bias 只加一次，merged 状态跳过低秩分支。冻结 $W$ 只表示不求 $W$ 的梯度，不能用 `no_grad()` 切断 $X$ 经过基座的梯度。

测试：$B=0$ 时输出等于基座、$A$ 的梯度为零而 $B$ 的不为零；再令 $B\ne0$、bias 非零，对 $2\times3\times7$ 的三维输入和给定的上游 $G$，用测试自己写的稠密公式核对输出、$dX=GW+s\,GBA$、$dB=s\,G^\top(XA^\top)$、$dA=s\,B^\top G^\top X$。基座放进 `no_grad`、bias 加两次、漏掉 $s$，都会失败。

### A2/A3 `merge`、`unmerge`（0.75h）

merge 只允许在 eval 模式下调用（train 模式抛 `ValueError` 且不改状态），重复调用幂等：把 $sBA$ 加进 $W$，并保存原 $W$ 的精确备份。unmerge 用备份恢复后释放备份，重复调用安全。

测试把 $B$ 设成标准差 10 的大值（这时 $W+\Delta-\Delta$ 按位不等于 $W$），连续三轮 merge 两次、unmerge 两次：merge 后权重等于测试算的 $W+sBA$，unmerge 后 `torch.equal` 原 $W$，备份为 `None`。用减法恢复、不幂等、不释放备份，都会失败。回答：备份为什么增加峰值，生产推理为什么通常不这样保存。

### A4 `inject_lora`（1h）

接收精确的模块名列表，替换嵌套的 `nn.Linear`，冻结整个原模型，只让新的 $A$、$B$ 可训练。保留 weight、bias、device、dtype 和 train/eval 标志。先验证全部目标，再做任何修改。

测试用一个 float64 模型（三个 Linear，只替换其中两个，第一个处于 eval）：不存在的名字、非 Linear 目标、重复名、空列表都要拒绝且模型不变；替换后输出与原模型相同（$B=0$），可训练参数恰好是四个 adapter 张量，没被替换的 Linear 也被冻结，dtype、train 标志和 bias 都保留。

### A5/A6 `adapter_state`、`load_adapters`（1.25h）

保存的是克隆到 CPU 的 $A$、$B$ 和 scale，不含 $W$，之后修改模型不能影响已保存的对象。load 检查完整名称集合、字段、shape、有限性和 scale；把每个张量转换到目标 device/dtype 并暂存，转换后仍须有限（有限的 float64 $10^{100}$ 转 float32 会变成 inf），全部通过后才写入。任何一项失败，所有模块都保持原样。

测试在最后一个 adapter 放错 shape、NaN 或溢出值，同时改动第一个 adapter，检查失败后第一个模块按位不变；另测错误 scale 和缺失模块。在 CPU 上 `.cpu()` 不拷贝，所以 A5 必须显式 `clone()`，测试会抓住别名。

### A7 `parameter_bytes`（0.75h）

按 **storage** 去重，分列 trainable 和 frozen：每个 storage 只算一次、按整个 storage 的字节数算；只要有一个别名需要梯度，这个 storage 就算 trainable。

`model.parameters()` 已经按**对象**去重，所以同一个 Parameter 被两个模块共享的情况不需要你处理。测试真正检查的是：三个不同的 Parameter 对象共享同一块 storage（其中一个是视图），有的可训练有的冻结；以及一个视图 Parameter 让整个 48 字节的 storage 保持存活。按 `id()` 去重、按最后一个别名决定归属、按视图大小计数，都会失败。说明这个数为什么不含梯度、优化器状态、激活和缓存池。

## B 核心：测量激活并构造预算计划（7h）

### B1 `run_blocks`（1h）

先完整检查 keep 的长度和 bool 类型，再执行；保留的块直接调用，重算的块用 `checkpoint(block, x, use_reentrant=False, preserve_rng_state=True)`。

测试保留 dropout，对 retain、全 checkpoint、混合三种策略恢复同一种子，比较 loss 和每个可训练参数的梯度。只比梯度证明不了 checkpoint 真的发生了，所以测试还用 forward pre-hook 数每个块被调用的次数（checkpoint 的块是 2 次），并用测试自己的 saved-tensor hook 数保存字节：全 checkpoint 时恰好只存三个块的输入，retain、混合、全 checkpoint 严格递减。忽略 keep、关闭 RNG 保存、用重入版、不检查长度，都会失败。

### B2 `plan_keep`（0.75h）

0/1 背包：sizes 是正整数，cost 有限且非负，budget 是非负整数。返回 bool 列表，最大化保留块的重算代价之和。每块最多选一次。测试用穷举 16 个子集在 18 个预算上比对，另有讲义里的贪心反例 `[10,20,30] / [60,100,120] / 50` 和非法输入。

### B3 `SavedInventory.pack`（1h）

在 `saved_tensors_hooks` 下统计一次前向为反传保存的非参数 storage：按 `storage_key` 去重、按整个 storage 的字节数计，参数和参数的视图排除，每次调用都计入 `saves`。返回 detach 后的张量，packed 对象不能带 `grad_fn`。它是一次前向的清单，**不是**随时间变化的峰值。

测试对参数、参数视图、叶子张量的两个视图、一个非叶子的视图逐个 pack，检查往返相等、返回值没有 `grad_fn`、字节数等于两块 storage 之和。

### B4 `plan_modes`（1.5h）

多选背包：每块在 `retain`、`checkpoint`、`offload` 里选一个，各有（保留单位，实测 ms），总单位不超过预算，最小化总时间。不可行时抛 `ValueError`。做 Pareto 剪枝：单位更多、代价也不更低的状态被支配。测试穷举三块的全部组合，在预算 0 到 15 上检查可行性、最优代价和返回值自洽；未知模式和空选项要拒绝。

这个加和模型忽略了张量的生命周期和 workspace。`bench.py` 的最后几行把你的块级计划与 `torch.compile` 在 `torch._functorch.config.activation_memory_budget` 为 1.0、0.5、0.0 时的结果放在同一口径下（保存字节、fw+bw 峰值增量、中位数时间），这是研究时必须对比的强基线。没有 C++ 编译器的机器上编译行会被标记为 `skipped`。

### B5 `calibrate_blocks`（2.75h）

对每块实测 retain 和 checkpoint 的 fw+bw 时间（先预热一次、取中位数）和保存字节。第 $i$ 块用前 $i$ 块的 detach 输出做输入；每次清梯度，返回时所有梯度为 `None`；CUDA 计时前后同步。

测试的第一块把宽度从 8 变成 16，忘记推进输入时第二块会直接报错；保存字节要等于测试自己的 hook 对同一输入独立数出来的值。回答：为什么“预测用量小于预算”不保证 allocator 峰值小于同样的数。

## C 前沿：把内存策略接进真实训练（7h）

### C0 阅读（0.5h）

读 [saved tensors hooks 教程](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)的 CPU 保存示例和讲义 §6.1 的五条契约。

### C1/C2 `CpuOffload.pack/unpack`（2h）

`pack` 返回 `(copied, 原设备, 值)`。参数 storage（含视图）、不在 `offload_devices` 里的设备、小于 `min_bytes` 的张量走 detach 后的 identity。其余张量复制到一块和输入不共享任何内存的新 host 存储，payload 里不再引用设备张量。只对真正的拷贝计 `packed_bytes` 和 `d2h_copies`。`unpack` 可以被调用多次，每次返回原设备、原 dtype 的**新**张量（`h2d_copies` 加 1），`restores` 每次调用都加 1。

默认 `offload_devices=("cuda",)`；测试在 CPU 上传 `("cpu",)`，让拷贝路径在没有 GPU 的机器上也被检查：pack 后原地修改输入，unpack 仍返回 pack 时的值；修改第一次 unpack 的结果，第二次仍正确；float64、bool、int64 往返保持 dtype。CUDA 上另测两个宽 1024 的块在 2048 个 token 上：offload 时前向后的 `memory_allocated` 增量必须低于 retain 的 60%。payload 里偷偷留着 CUDA 张量的实现，正确性全过，只有这一项能抓住。

### C3 `run_policy`（1.5h）

每块 retain、checkpoint 或 offload。**执行任何块之前**先验证：长度、每个模式合法、有 offload 模式时必须给 owner，否则抛 `ValueError`。offload 块在 `offload.context()` 里前向，退出 context 后反传仍能调用当时的 unpack。

测试保留 dropout，比较四种策略的输入梯度和全部可训练参数梯度；用 forward hook 数重算次数，用 owner 的计数确认 offload 块确实拷贝且每个拷贝恢复一次。缺 owner、未知模式、长度不对三种调用都要在任何块运行之前、任何 RNG 被消耗之前失败。

### C4 `train_accumulated`（1.5h）

一次完整的 optimizer step：目标是全部微批的 $\sum(\hat y-y)^2$ 除以总元素数（等于全 batch 的 MSE 均值）。只 zero_grad 一次，每个微批 backward，可选的 `clip_grad_norm_` 在**累积完成后**对 blocks 的全部可训练参数做一次，最后 step 一次。返回 step 前的目标值。缺 owner 等非法输入要在清梯度之前拒绝。

测试的 oracle 是测试自己写的全 batch 前向、手动裁剪（$\min(1,c/(\lVert g\rVert+10^{-6}))$）和手动 SGD。它先断言全 batch 梯度范数超过 `max_grad_norm` 的 10 倍，所以“不裁剪”“每微批裁剪”“累积后裁剪”三者给出不同的更新；再测 2+3、1+1+3 和不裁剪的 1+4 三种切分。均值的均值、每微批清梯度、每微批裁剪，都会失败。

### C5 `measure_policy`（1.5h）

真实的 AdamW step，`lr=0` 让各策略之间权重不变，但反传和优化器运算照常执行。先做一次预热 step（创建优化器状态，不计入）；CUDA 上再用同一个 step 让 GPU 忙至少 300 ms（笔记本 GPU 在配置之间会降到低频，慢到 8 倍），然后重置计数和峰值统计，逐步用 CUDA event 计时。返回中位数 `step_ms`、`step_ms_p25/p75`、`stable`（$p_{75}\le1.5\,p_{25}$）、每步 loss、每步传输计数和 CUDA 峰值（CPU 上为 `None`）。

测试检查 loss 和参数在 `lr=0` 下不变、每步 D2H 字节恰好等于被卸载的块在一步里保存的字节（由测试自己的 hook 独立数出），CUDA 上宽 1024 时 checkpoint 的峰值低于 retain。

```bash
python tools/run.py --lab 6 --impl reference --mode bench --part all --device cuda --output results/lab6/reference.json
python tools/extract.py results/lab6/reference.json --table 0 --out results/lab6/reference.csv
```

`bench.py` 在 CUDA 上用宽 256 和 1024、每样本 512 token、4 块、3 个种子、4 种策略输出 24 行 `actual_torch_training`，再输出 `budget_baseline` 行（块级计划与 `torch.compile` 预算的对比）。CPU 上宽度和长度缩小，峰值字段为 null。

## R 真实系统：比 eager autograd 存得更少（8h）

讲义 §3.1 的探针说明：$W$ 冻结时，eager LoRA 本来就只存 $X$ 和 $XA^\top$；$A$ 也冻结时只存 $XA^\top$。所以“只存 $X$ 和 $XA^\top$”的自定义 Function 一个字节也省不了。R 阶段的三个 Function 各针对一件 eager 做不到或已经做到、需要你证明的事。所有 Function 都用合并的 `forward(ctx, ...)` 写法，用 `ctx.save_for_backward` 保存（这样 hook 才数得到），用 `ctx.needs_input_grad` 跳过不需要的梯度。

### R0 阅读（0.5h）

读讲义 §7、[扩展 autograd 文档](https://docs.pytorch.org/docs/2.14/notes/extending.html)和 [Unsloth `fast_lora.py`](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py) 中 `LoRA_MLP` 的 `save_for_backward` 与 `backward`。先在 WSL 里运行讲义 §3.1 的探针，记下三行输出。

### R1 `LoRAFALinearFn`（1.5h）

$A$、$W$、bias 冻结，训练 $B$（和 $x$）。只保存 $Z=xA^\top$ 和参数，不存 $x$。反传 $dB=s\,G^\top Z$、$dx=GW+s\,(GB)A$。

测试在 float64 下：输出和 $dx$、$dB$ 与 eager 参考相差不超过 $10^{-12}$；`gradcheck` 通过；保存字节恰好是 $N r\times8$；再顺带测出 eager 冻结 $A$ 时保存的字节和你的完全一样，$A$ 可训练时是 $N(d_{\mathrm{in}}+r)\times8$。这一项不省显存，目的是亲手写对反传并证明保存集。$dB$ 转置、漏 $s$、$dx$ 只算基座、不返回 $dx$、多存一份 $x$，都会失败。

### R2 `LoRAMLPFn`（2.5h）

up（$h\to f$）、精确 GELU、down（$f\to h$），两个投影都带 LoRA，$A$、$B$ 都可训练。只保存 $x,z_1,u,z_2$，**不存** $g=\mathrm{GELU}(u)$，反传时由 $u$ 重算。$\mathrm{GELU}'(u)=\Phi(u)+u\,\varphi(u)$。

测试：五个梯度与 float64 eager 参考一致（$10^{-10}$），`gradcheck` 通过，`gelu_grad` 等于 erf 版 GELU 的数值导数；保存字节恰好是 $N(h+f+2r)\times8$，而 eager 是 $N(h+2f+2r)\times8$；冻结 $A_1$、$x$ 不需要梯度时其余梯度仍正确。CUDA 上，4 层、$h=1024$、$f=3072$、$r=16$、2048 token 时，前向后的 allocated 增量要比 eager 少 $LNf\times4$ 字节（5% 以内），fw+bw 峰值低于 eager。把 $g$ 也存下来、只用 $\Phi$、反传用 tanh 近似、$dx$ 漏掉 LoRA 路径、$dA_2$ 用了 $u$，都会失败。

### R3 `LoRADropoutFn`（1.5h）

PEFT 式 LoRA dropout：$Y=xW^\top+s\,((x\odot m)A^\top)B^\top$，$m$ 由已提供的 `dropout_mask(shape, p, seed, ...)` 生成（私有 `torch.Generator`，不碰全局 RNG）。只保存 $x$、$Z$、参数，把 $p$ 和整数种子放在 `ctx` 上；反传用同一种子重新生成 $m$，不存 $m$，也不存 $x\odot m$。

测试：用同一 mask 的 float64 稠密参考核对输出和三个梯度，`gradcheck` 通过；保存字节恰好是 $N(d+r)\times8$；作为对照，eager 的 `F.dropout` 版本多存一份 mask（CUDA 上每元素 1 字节，CPU 上是输入 dtype）；`seed=None` 时用全局 CPU RNG 抽种子，在 `torch.manual_seed` 下可复现，反传重新生成的就是前向用的那一个；$p=0$ 退化为普通 LoRA。存 mask、反传换种子、前向用全局 RNG、$dx$ 忘了 $1/(1-p)$、$dA$ 用未 dropout 的 $x$，都会失败。

### R4 `measure_variant`（1.5h）

测一个 fw+bw 变体：预热若干次（不记录）；一次记录的前向在 `SavedInventory(parameters)` 下运行，读前向前后的 allocated（CUDA 用 `memory_allocated`，MPS 用 `current_allocated_memory`，CPU 为 `None`），反传后读 `max_memory_allocated` 减去前向前的值；CUDA 上再让 GPU 忙至少 `warm_ms`；最后 `repeats` 次计时（CUDA event，否则 `perf_counter`），返回中位数、$p_{25}$、$p_{75}$ 和 `stable`。每次 pass 前把参数的 `.grad` 设为 `None`，返回时 `.grad` 恰好是一次 pass 的梯度。

测试用一个已知的玩具：每次 pass 保存一块 4 MiB 的张量和一个参数、内部再临时分配 32 MiB，测量前还有一块 64 MiB 一直存活、一块 128 MiB 已经释放。检查 saved bytes 恰好 4 MiB（参数不计）、`after_forward_bytes` 在 4 到 5 MiB 之间、`peak_delta_bytes` 在 32 到 48 MiB 之间、调用次数、`.grad` 没有累积。没重置峰值、报绝对峰值、把参数算进去、计时循环不清梯度，都会失败。

### R 基准与验收

`bench_r.py` 用 `model.MLPBlock` 的 7 种执行方式（`eager`、`eager_fa`、`fn_fa`、`fused`、`eager_drop`、`fn_drop`、`checkpoint`），在 CUDA 上取 Qwen3-0.6B 的宽度（$h=1024$、$f=3072$、$r=16$，4 层，简化为 LayerNorm 加 GELU、无 attention），序列长度 1024、4096、8192，每行输出手算保存字节（标 `modeled`）、实测保存字节、前向后 allocated 增量、fw+bw 峰值增量和计时。`test_r.py` 的 `R variant` 检查确认 7 种方式的输出和梯度一致（$p=0$），保存字节等于 `predicted_saved_floats` 的手算值。

```bash
python tools/run.py --lab 6 --impl reference --mode bench --part R --device cuda --output results/lab6/r-cuda.json
```

## 报告与验收（3h）

提交实现、至少两个自己的反例、CPU 和 CUDA 的 RECORD JSON 与 CSV、`bash tools/check_env.sh --lab 6` 的输出和 `reports/lab6.md`。必交图表：

1. A–C：峰值 allocated 对 step 时间，按宽度分面，保留三个种子；同一图上标出块级计划与 `activation_memory_budget` 三个取值。
2. R：每种变体在三个长度下的实测保存字节对手算值（应落在对角线上），以及峰值增量对中位数时间的散点（误差棒为 IQR，`stable=false` 的点用空心）。
3. 一张表：每种 R 变体相对 eager 的保存字节减少、峰值减少和时间变化。

回答：LoRA、LoRA-FA、R2、R3、checkpoint、offload 各改变讲义 §1.1 的哪一项？saved bytes 和峰值的差距来自什么？R2 省下的 $LNf\times4$ 字节在峰值里体现了多少，为什么不是全部？R3 保存得更少，峰值为什么可能反而更高？同步 offload 是否可能既慢又不省峰值？你的块级规划比 `torch.compile` 预算差在哪里？哪些数字不能从 CPU 推到 GPU？

自动测试检查算法性质和保存集，报告评的是测量和解释；验收不设固定的加速或省显存门槛。评分见 [grade.md](grade.md)，参考试跑见[验证记录](../docs/validation.md)。

## 可选挑战

挑战另计 8 到 16h。

1. **Triton 融合路径**（WSL 的 venv 有 Triton 3.8）：把 R2 反传里的逐元素部分（GELU 重算、乘以 $\mathrm{GELU}'(u)$、加上 LoRA 的 $dz_2A_2$）融成一个 Triton kernel，GEMM 仍交给 cuBLAS（LoRAFusion 的做法），用 `torch.testing.assert_close` 对 R2 验证，再用 `ncu` 看 `dram__bytes` 的变化。
2. **按位压缩的 mask**：给 R3 加一个保存 1 bit/元素 mask 的变体，和“只存种子”比较保存字节和反传时间。
3. **异步 pinned offload**：用 pinned host 内存、独立 stream 和 event 改写 C1/C2，加上 host 峰值账本和生命周期测试，验证传输是否真的与计算重叠。
4. **真实模型**：把 R2 的思路搬到 Qwen3-0.6B 的 SwiGLU MLP 上（重算 $\mathrm{SiLU}(g)\odot u$），在 2048 token 上测峰值，和讲义研究问题的基线对比。

先保留至少一个失败案例并建立公平的 baseline，再判断是否值得继续。

## 资料汇总

| 资料                                                                                                                   | 类型 | 指定观察                                                           | 时间  |
| ---------------------------------------------------------------------------------------------------------------------- | ---- | ------------------------------------------------------------------ | ----- |
| [课程 D6](../../course/modules/06-memory-efficient-training.md)                                                        | 讲义 | §3.1 的保存规则、§3.2 的保存集表、§7 的三组反传公式                | —     |
| [PEFT `lora/layer.py`](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py)                    | 代码 | `Linear.forward`、`merge`/`unmerge`；对照 A1–A3                    | 20min |
| [PyTorch checkpoint 文档](https://docs.pytorch.org/docs/2.14/checkpoint.html)                                          | 文档 | `use_reentrant`、`preserve_rng_state`、`CheckpointPolicy`；对照 B1 | 15min |
| [saved tensors hooks 教程](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html) | 文档 | 保存到 CPU 的示例；对照 B3、C1/C2                                  | 20min |
| [扩展 autograd 文档](https://docs.pytorch.org/docs/2.14/notes/extending.html)                                          | 文档 | 两种写法、`needs_input_grad`、`gradcheck`；对照 R1–R3              | 20min |
| [Unsloth `fast_lora.py`](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py)                  | 代码 | `LoRA_MLP` 的保存集与激活重算；对照 R2                             | 20min |
| [PyTorch 激活 checkpoint 技术博客](https://pytorch.org/blog/activation-checkpointing-techniques/)                      | 图解 | SAC 与 memory budget 的速度–显存图；对照 B4 的基线                 | 15min |
| [CS336 Assignment 2](https://github.com/stanford-cs336/assignment2-systems)                                            | 课程 | §3 Single-GPU Memory 与本 lab B 阶段重叠，可作额外练习             | 选读  |
| [资源、定位与简化边界](resources.md)                                                                                   | 索引 | 上游具体文件、借鉴内容和未覆盖部分                                 | 15min |
| [参考实现](../../solutions/lab6/README.md)、[实验规范](../../course/guides/experiments.md)                             | 参考 | 独立完成后核对不变量；按规范记录测量和负结果                       | —     |
