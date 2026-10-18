---
title: D6 低显存训练：显存账、autograd 保存集、重计算与 LoRA
lab: lab6
hours: 5
tags: [lecture, training-memory, lora, checkpointing, autograd, offload, research]
updated: 2026-10-18
---

# D6 低显存训练：显存账、autograd 保存集、重计算与 LoRA

[← 课程索引](../index.md) · [配套 lab6](../../labs/lab6/README.md) · [← D5 前缀缓存](05-prefix-cache.md) · [D7 MoE 卸载 →](07-moe-offload.md)

> [!abstract] 这一讲要解决的问题
> foundations 讲过反向传播和 GPU 显存层次，但没有讲训练时显存到底花在哪里。这一讲的约束是一张 8GB 的 RTX 4060 Laptop：Qwen3-0.6B 做全参数混合精度 Adam 训练，光模型状态就要约 9.6 GB，放不下；改成 LoRA 后优化器状态只剩约 160 MB，瓶颈转到激活和 logits 上，2048 个 token 的 FP32 logits 一个张量就有 1.16 GiB。学完后你应该能：推出 16 B/param 和 ZeRO 各阶段的公式；说出 autograd 对每个算子保存什么、为什么冻结 $W$ 后 $X$ 有时被存、有时不被存；推导 $\sqrt{n}$ checkpoint 和预算背包；写出比 eager autograd 存得更少的 `autograd.Function`（LoRA-FA、重算 GELU 的融合 MLP、只存种子的 dropout），并用 float64 gradcheck、saved-tensor hook 和 CUDA 峰值证明它真的省了；最后判断哪些方向已经拥挤，哪些在单张消费级显卡上还有空白。

| 阅读模块                           | 时间  | 必须留下的学习产物                                     | 对应实验 |
| ---------------------------------- | ----- | ------------------------------------------------------ | -------- |
| §1 显存账：五类对象与字节表        | 35min | Qwen3-0.6B 全参数与 LoRA 两种账、logits 张量的字节     | A0、A7   |
| §2 分片与卸载：ZeRO、FSDP、Offload | 25min | ZeRO 三阶段公式与 7.5B 例子                            | 报告     |
| §3 autograd 保存了什么             | 40min | 一张 LoRA 线性层的保存集表，Korthikanti 公式的假设清单 | B3、R    |
| §4 LoRA：梯度、冻结与变体          | 30min | $dX$、$dA$、$dB$ 推导，LoRA-FA/QLoRA/DoRA 改变的对象   | A1–A6    |
| §5 重计算与预算规划                | 40min | $\sqrt{n}$ 推导、背包手算、dropout 反例                | B1/B2/B4 |
| §6 offload、hooks 与微批           | 25min | pack/unpack 契约、计数手算、不均匀微批的权重           | C        |
| §7 自定义 `autograd.Function`      | 40min | R1–R3 的反传公式与保存集预测                           | R1–R3    |
| §8 优化器状态与融合 kernel         | 20min | 一张“改变哪一项”的方法分类卡                           | 报告     |
| §9 测量口径                        | 15min | allocated、reserved、peak、saved 四个数的区别          | B5/C5/R4 |
| §10 机制与代码、前沿对照、研究问题 | 30min | 三篇工作的机制差异卡，一个可证伪的假设                 | R、挑战  |

**共 5h，另做 lab 约 30h。** 5h 包括手算、一个在 WSL 里跑的 20 行探针、PyTorch checkpoint 与 hooks 文档的指定段落和随堂自测；通读论文属于选读。外部图表都是作者在各自模型和硬件上的测量，不是你的结果，下文不再重复。

随堂自测计入上表时间。先在笔记里写答案，再展开核对要点。

## §1 显存账：五类对象与字节表

### 1.1 训练时显存里有什么

某一时刻的显存可以拆成：

$$
M=M_W+M_G+M_O+M_A+M_{\mathrm{tmp}}+M_{\mathrm{frag}}
$$

- $M_W$：参数，包括冻结的基座。
- $M_G$：梯度，只有 `requires_grad=True` 的参数才有。
- $M_O$：优化器状态，Adam 每个可训练参数有一阶矩 $m$ 和二阶矩 $v$；混合精度时还有 FP32 master 权重。
- $M_A$：为 backward 保存的激活（saved tensors），从前向产生一直活到对应的 backward 用完。
- $M_{\mathrm{tmp}}$：算子临时区，如 cuBLAS workspace、反向时重算出来的中间值、optimizer step 的中间张量。
- $M_{\mathrm{frag}}$：分配器碎片和缓存池。PyTorch 的 caching allocator 把释放的块留在池里，`memory_reserved` 大于 `memory_allocated` 的部分大多在这里。

这几类的生命周期不同。参数和优化器状态常驻；激活在前向逐层增长，在反向逐层释放；梯度在反向逐层出现。所以峰值通常出现在“前向刚结束、反向刚开始”的时刻，或者在 optimizer step 时（梯度和状态同时存活，再加上 step 的临时区）。HF 的[训练显存可视化博客](https://huggingface.co/blog/train_memory)给的模型是：峰值约等于参数、优化器状态，再加上“梯度加 step 中间量”与“激活”两者中较大的一个。

### 1.2 16 B/param 从哪里来

全 FP32 加 Adam：参数 4、梯度 4、$m$ 4、$v$ 4，合计 16 B/param。

混合精度加 Adam（ZeRO 论文 §3 的记法）：FP16 参数 2、FP16 梯度 2，再加 $K=12$ 字节的优化器状态（FP32 master 权重 4、$m$ 4、$v$ 4）：

$$
2+2+K=2+2+12=16\ \text{B/param}
$$

两种方案每参数都是 16 字节，但组成不同。ZeRO 的例子：1.5B 参数的 GPT-2 光模型状态就要 $16\times1.5\times10^9=24$ GB。不同配方的字节数不同，所以要列表而不是背一个常数：

| 配方                            | 参数 | 梯度 | 优化器状态            | 合计 B/param |
| ------------------------------- | ---: | ---: | --------------------- | -----------: |
| FP32 + Adam                     |    4 |    4 | $m,v$ 各 4            |           16 |
| FP16/BF16 + FP32 master + Adam  |    2 |    2 | master 4 + $m,v$ 8    |           16 |
| BF16、无 master + Adam          |    2 |    2 | $m,v$ 各 4            |           12 |
| BF16 + FP32 master + 8-bit Adam |    2 |    2 | master 4 + $m,v$ 各 1 |           10 |
| 冻结参数（LoRA 的基座）         |    2 |    0 | 0                     |            2 |

另外两项不在表里：梯度累积若用 FP32 缓冲，每参数再加 2 字节；optimizer step 时 PyTorch 的 foreach 实现会产生和参数同量级的临时张量。

### 1.3 落到 Qwen3-0.6B 上

Qwen3-0.6B 的配置（`~/models/Qwen3-0.6B/config.json`，本机读取）：hidden 1024、intermediate 3072、28 层、16 个 query head、8 个 KV head、head_dim 128、词表 151936，输入输出 embedding 共享。

- **全参数训练**：约 $0.6\times10^9$ 参数乘以 16 B，约 9.6 GB，8GB 卡上还没开始前向就放不下。
- **LoRA，$r=16$，7 个投影都加**：一个 $d_{\mathrm{in}}\to d_{\mathrm{out}}$ 的投影新增 $r(d_{\mathrm{in}}+d_{\mathrm{out}})$ 个参数。q 是 $1024\to2048$，k、v 是 $1024\to1024$，o 是 $2048\to1024$，gate、up 是 $1024\to3072$，down 是 $3072\to1024$。每层

$$
16\times(3072+2048+2048+3072+4096+4096+4096)=360{,}448
$$

28 层合计约 $1.009\times10^7$ 个可训练参数，按 16 B 算约 161 MB。BF16 基座约 1.2 GB。

- **logits**：一条 2048 token 的序列，FP32 logits 是 $2048\times151936\times4\approx1.16$ GiB，它的梯度同样大。

结论：LoRA 在 8GB 卡上把优化器状态压到可以忽略，剩下的大头是激活和 logits。所以本讲的重点放在激活（§3、§5、§7）上，§8 里的优化器状态压缩只在全参数训练时才重要。

> [!question] 随堂自测 D6-1：先把账算对
>
> 1. 忽略 bias，$d_{\mathrm{in}}=d_{\mathrm{out}}=1024$、$r=8$、全部 FP32。基座 $W$、可训练的 $A$ 和 $B$ 各占多少字节？只知道这两个数，能推出训练峰值吗？
> 2. 一个 1.5B 模型用“BF16、无 master、8-bit Adam”（$2+2+2$）训练，模型状态多少 GB？和 16 B/param 相比省了多少？
> 3. 为什么说 LoRA 微调 Qwen3-0.6B 时“换 8-bit Adam”几乎没用？

> [!tip]- 核对要点
>
> 1. $W$ 是 $1024\times1024\times4=4$ MiB；$A$、$B$ 合计 $8\times2048\times4=65{,}536$ 字节。它们不含梯度、优化器状态、激活、临时区和缓存池，推不出峰值。
> 2. $6\times1.5=9$ GB，比 24 GB 少 62.5%。注意这个配方没有 FP32 master，数值稳定性另需验证。
> 3. LoRA 的优化器状态只有约 $10^7\times8$ 字节，即 80 MB；换成 8-bit 后省下的约 60 MB，跟 1.16 GiB 的 logits 和数百 MB 的激活相比可以忽略。

## §2 分片与卸载：ZeRO、FSDP、Offload

单卡 lab 用不到多卡分片，但这套记法是读训练系统论文（包括 Hetu）的基础，也说明了 offload 为什么能让单卡训练更大的模型。

### 2.1 ZeRO 的三个阶段

设参数量为 $\Psi$，数据并行度为 $N_d$，优化器状态每参数 $K$ 字节（混合精度 Adam 时 $K=12$）。普通数据并行在每张卡上都存一份完整的 $(2+2+K)\Psi$。ZeRO 依次把优化器状态、梯度、参数切成 $N_d$ 份：

| 阶段                   | 每卡模型状态            | $\Psi=7.5$B、$N_d=64$ |
| ---------------------- | ----------------------- | --------------------: |
| 普通 DP                | $(2+2+K)\Psi$           |                120 GB |
| ZeRO-1（$P_{os}$）     | $2\Psi+2\Psi+K\Psi/N_d$ |               31.4 GB |
| ZeRO-2（$P_{os+g}$）   | $2\Psi+(2+K)\Psi/N_d$   |               16.6 GB |
| ZeRO-3（$P_{os+g+p}$） | $(2+2+K)\Psi/N_d$       |               1.88 GB |

前两个阶段的通信量和普通 DP 一样，因为 all-reduce 本来就等于 reduce-scatter 加 all-gather；ZeRO-3 每步前向和反向都要重新 all-gather 参数，通信量约为 1.5 倍。数字出自 [ZeRO 论文](https://arxiv.org/abs/1910.02054)图 1 和 §5。

[PyTorch FSDP](https://arxiv.org/abs/2304.11277) 是 ZeRO-3 的工程化：按模块把参数拼成 FlatParameter，前向前 all-gather、用完立刻释放。FSDP2 的 [`fully_shard`](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html) 改成按参数在第 0 维切分的 DTensor，带 `MixedPrecisionPolicy` 和 `CPUOffloadPolicy(pin_memory=True)`。

### 2.2 Offload：用 PCIe 换显存

[ZeRO-Offload](https://www.usenix.org/conference/atc21/presentation/ren-jie) 把 FP32 优化器状态和 optimizer step 本身搬到 CPU，GPU 上只留 FP16 参数和梯度。按 §1.2 的表，GPU 上的模型状态从 16 B/param 降到 4 B/param，论文称能在单张 V100 上训练 13B。[ZeRO-Infinity](https://arxiv.org/abs/2104.07857) 再加上 NVMe 一层。

代价是 PCIe 带宽。每步要把 FP16 梯度搬到 CPU、把更新后的 FP16 参数搬回来，共 $4\Psi$ 字节，0.6B 模型约 2.4 GB；笔记本平台的主机–设备带宽通常只有十几 GB/s 量级（以 lab6 C5 在本机的实测为准），光传输就要上百毫秒。是否值得，要看 GPU 计算能不能把传输盖住。lab6 的 C 阶段做的是激活 offload，原理一样，规模小得多。

> [!question] 随堂自测 D6-2：分片的账
>
> 1. $\Psi=1.5$B、$N_d=8$、$K=12$，算 ZeRO-1、ZeRO-2、ZeRO-3 的每卡模型状态。
> 2. ZeRO-3 把模型状态降到 $1/N_d$，为什么激活不跟着变少？

> [!tip]- 核对要点
>
> 1. ZeRO-1：$4\times1.5+12\times1.5/8=6+2.25=8.25$ GB；ZeRO-2：$3+14\times1.5/8=3+2.625=5.625$ GB；ZeRO-3：$24/8=3$ GB。
> 2. 数据并行下每张卡处理自己的微批，激活和本卡的 token 数成正比，跟参数怎么切无关。减激活要靠 checkpoint、offload、序列并行或少存（§3、§5、§7）。

## §3 autograd 保存了什么

§5 和 §7 都建立在这一节上。不先弄清楚 eager autograd 已经存了什么，就可能写出一个一个字节都没省的自定义 Function。

### 3.1 规则：一个操作数只在“另一个操作数需要梯度”时才被存

对 $Y=XW^\top$，有

$$
\frac{\partial L}{\partial X}=G\,W,\qquad \frac{\partial L}{\partial W}=G^\top X,\qquad G=\frac{\partial L}{\partial Y}
$$

算 $dX$ 需要 $W$，算 $dW$ 需要 $X$。PyTorch 的代码生成器（[`gen_variable_type.py`](https://raw.githubusercontent.com/pytorch/pytorch/main/tools/autograd/gen_variable_type.py) 里的 `guard_for`）在只被一个导数公式用到的保存语句外面包一层 `if (grad_fn->should_compute_output(i))`。`mm` 的两个操作数正是这种情况，所以：

- $W$ 冻结、$X$ 需要梯度：只存 $W$（一个参数的转置视图，不新增显存），**不存 $X$**；
- $W$ 可训练、$X$ 不需要梯度：只存 $X$；
- 两个都需要梯度：两个都存。

下面这段探针在 WSL 的 torch 2.14.1 上实测过（2026-10-18），`saved_tensors_hooks` 的 pack 函数会在每个张量被存下时调用一次：

```python
import torch, torch.nn.functional as F
def saved(fn):
    out = []
    pack = lambda t: out.append(tuple(t.shape)) or t.detach()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn()
    return out
N, di, do, r = 6, 5, 4, 2
X = torch.randn(N, di, requires_grad=True); W = torch.randn(do, di)
A = torch.randn(r, di, requires_grad=True); B = torch.randn(do, r, requires_grad=True)
print(saved(lambda: F.linear(X, W)))                                  # [(5, 4)]  W^T only
print(saved(lambda: F.linear(X, W) + F.linear(F.linear(X, A), B)))    # W^T, A^T, X, B^T, X A^T
A.requires_grad_(False)
print(saved(lambda: F.linear(X, W) + F.linear(F.linear(X, A), B)))    # W^T, A^T, B^T, X A^T
```

输出依次是：冻结的 `F.linear(X, W)` 只存了 $(5,4)$ 的 $W^\top$；LoRA 线性层存了 $X$（$6\times5$）和 $XA^\top$（$6\times2$），外加三个参数视图；把 $A$ 冻结（LoRA-FA）后，$X$ 消失，只剩 $XA^\top$。

这件事的推论很直接：**一个“只保存 $X$ 和 $XA^\top$”的 LoRA `autograd.Function` 不会比 eager 少存任何东西**，因为 eager 已经只存这两个。想省，得改变计算本身（冻结 $A$），或者跨算子重算（§7 的 R2），或者换掉 eager 必须存的东西（R3 的种子代替 mask）。

其他常见算子的保存集（同一台机器实测）：

| 算子                      | 保存什么                  | 备注                                           |
| ------------------------- | ------------------------- | ---------------------------------------------- |
| `F.gelu(u)`               | 输入 $u$                  | 反传用 $u$ 现算 $\mathrm{GELU}'(u)$            |
| `F.layer_norm(x)`         | 输入 $x$、每行均值和 rstd | 均值和 rstd 各 1 个数/token                    |
| `F.rms_norm(x, w)`        | 输入 $x$、rstd 等         | $w$ 可训练时还存归一化后的中间值               |
| `F.dropout(x, p)`         | mask                      | CPU 上是和 $x$ 同 dtype 的张量，CUDA 上是 bool |
| `x.square()`              | $x$                       |                                                |
| 非重入 `checkpoint(f, x)` | 只有 $x$                  | 区域内部在反传时重算                           |

### 3.2 一个 LoRA MLP 块存了多少

lab6 R 阶段的块是 $x+\mathrm{down}(\mathrm{GELU}(\mathrm{up}(\mathrm{LN}(x))))$，up 是 $h\to f$，down 是 $f\to h$，两个都带秩 $r$ 的 LoRA，基座冻结。按上一节逐个算子数每个 token 保存的 FP32 个数：

| 来源         | 保存                        | 每 token 个数 |
| ------------ | --------------------------- | ------------- |
| LayerNorm    | 输入、均值、rstd            | $h+2$         |
| up 的 LoRA   | LN 输出（给 $dA$）、$z_1$   | $h+r$         |
| GELU         | 输入 $u$                    | $f$           |
| down 的 LoRA | $g=\mathrm{GELU}(u)$、$z_2$ | $f+r$         |

合计 $2h+2f+2r+2$。取 Qwen3-0.6B 的 $h=1024$、$f=3072$、$r=16$，每个 token 32,904 字节。三种减法：

- **LoRA-FA**（冻结两个 $A$）：去掉 LN 输出和 $g$，剩 $h+f+2r+2$，16,520 字节，减 50%；
- **R2 融合**（$A$ 仍可训练，反传时由 $u$ 重算 $g$）：去掉 $g$，剩 $2h+f+2r+2$，20,616 字节，减 37%；
- **整块 checkpoint**：只剩块输入 $h$，4096 字节，但反传时整块前向重算一遍。

lab6 的 `test_r.py` 在 4060 上用 4 层、2048 token 验证了第二条：前向结束后融合版比 eager 少持有 $LNf\times4$ 字节，误差在 5% 以内。

### 3.3 Transformer 层的激活公式

[Korthikanti 等](https://arxiv.org/abs/2205.05198)（MLSys 2023）给出每层激活字节

$$
sbh\left(34+\frac{5as}{h}\right)
$$

$s$ 是序列长度，$b$ 是 micro-batch，$h$ 是 hidden，$a$ 是 head 数。前提是 16-bit 激活、1 字节的 dropout mask、GPT 式 $4h$ 的 MLP 加 GeLU、两个 LayerNorm，并且显式保存 attention 分数。拆开：

- attention 部分 $11sbh+5as^2b$：$5as^2b$ 来自 softmax 输出（$2as^2b$）、softmax 后 dropout 的 mask（$as^2b$）和 dropout 输出（$2as^2b$）；
- MLP 部分 $19sbh$；两个 LayerNorm $4sbh$。

张量并行加序列并行时整体除以 $t$。论文的 selective recompute 只重算 attention 分数那一块，去掉 $5as/h$，剩 $34sbh/t$；GPT-3 规模下 $5as/h=80$，所以这一招省了大约 70%。

套到 Qwen3-0.6B 前要先改假设：FlashAttention/SDPA 在 kernel 内部重算分数，$5as^2b$ 这一项本来就不存；Qwen3 用 SwiGLU、RMSNorm、GQA、没有 dropout，$34$ 这个常数也会变。正确做法是用 §3.1 的方法重新数，或者直接用 hook 测。

> [!question] 随堂自测 D6-3：保存集
>
> 1. $s=2048$、$b=1$、$h=1024$、$a=16$，按 Korthikanti 公式算一层的 $34sbh$ 项和 $5as^2b$ 项各多少 MiB。用 FlashAttention 后还剩多少？
> 2. 一个同学写了 `LoRAFn`，在 `save_for_backward` 里只存 $X$ 和 $XA^\top$，宣称“比 PyTorch 省显存”。用 §3.1 的规则说明他错在哪，再给出两种真正能省的改法。
> 3. 冻结的 `F.linear(X, W)` 里，如果 $X$ 不需要梯度，会存什么？

> [!tip]- 核对要点
>
> 1. $34\times2048\times1024=71{,}303{,}168$ 字节，即 68 MiB；$5\times16\times2048^2$ 字节，即 320 MiB。FlashAttention 不存分数矩阵，剩 68 MiB 左右（常数还要按 Qwen3 的结构重算）。
> 2. eager 在 $W$ 冻结时本来就只存 $X$（给 $dA$）和 $XA^\top$（给 $dB$），他的 Function 保存集一模一样。真正能省的：冻结 $A$，只存 $XA^\top$（LoRA-FA）；或者把 up、GELU、down 融成一个 Function，在反传时重算 GELU 输出。
> 3. 什么都不存。$W$ 不需要梯度，所以不存 $X$；$X$ 不需要梯度，所以也不存 $W$。

## §4 LoRA：梯度、冻结与变体

### 4.1 形状与梯度

$X$ 是 $[N,d_{\mathrm{in}}]$，$W$ 是 $[d_{\mathrm{out}},d_{\mathrm{in}}]$，$A$ 是 $[r,d_{\mathrm{in}}]$，$B$ 是 $[d_{\mathrm{out}},r]$，$s=\alpha/r$：

$$
Y=XW^\top+s\,(XA^\top)B^\top
$$

```text
X [N, d_in] --+-- base linear (W, frozen) -----------------------+--> Y [N, d_out]
              |                                                  |
              +-- A --> Z = X A^T [N, r] --> B --> scale (s) ----+
```

令 $G=\partial L/\partial Y$，$Z=XA^\top$：

$$
\frac{\partial L}{\partial B}=s\,G^\top Z,\qquad
\frac{\partial L}{\partial A}=s\,(GB)^\top X,\qquad
\frac{\partial L}{\partial X}=GW+s\,(GB)A
$$

LoRA 把 $B$ 初始化为 0，所以初始输出等于基座，$A$ 的第一次梯度 $s(GB)^\top X$ 也是 0，$B$ 的梯度一般不为零。这是初始化的结果，不是 bug。

### 4.2 冻结 $W$ 不等于关掉基座分支的 autograd

$dX$ 里的 $GW$ 一项来自冻结的基座。如果为了“省事”把 $XW^\top$ 放进 `torch.no_grad()`，这一项就没了。loss 仍会下降，$B$ 仍有梯度，只看这两点发现不了；上游层却拿不到正确的梯度。lab6 A1 的测试把 $B$ 设成非零、给定上游 $G$，用上面三个公式独立核对 $dX$、$dA$、$dB$。

### 4.3 merge 与 unmerge

推理时可以把 $sBA$ 加进 $W$，省掉低秩分支的两次小矩阵乘。训练前必须拆开。用 $W+\Delta-\Delta$ 恢复不是按位精确的：FP32 加减各有一次舍入，$\Delta$ 很大时误差明显。lab 的做法是 merge 时存一份 $W$ 的精确备份，unmerge 时拷回再释放。代价是 merge 期间多占一份 $W$ 的显存。

### 4.4 几个变体分别改了什么

| 方法                                                      | 改了什么                                  | 显存上的效果                                  |
| --------------------------------------------------------- | ----------------------------------------- | --------------------------------------------- |
| [LoRA](https://arxiv.org/abs/2106.09685)（ICLR 2022）     | 冻结 $W$，训练 $BA$                       | 去掉 $W$ 的梯度和优化器状态                   |
| [QLoRA](https://arxiv.org/abs/2305.14314)（NeurIPS 2023） | 基座量化成 NF4，双重量化，paged optimizer | 基座从 2 B/param 降到约 0.5 B/param；激活不变 |
| [LoRA-FA](https://arxiv.org/abs/2308.03303)（arXiv 2023） | 冻结 $A$，只训练 $B$                      | 不再存 $X$，只存 $r$ 维的 $XA^\top$           |
| [DoRA](https://arxiv.org/abs/2402.09353)（ICML 2024）     | 权重拆成幅度和方向，LoRA 作用在方向上     | 多一个按列的范数，激活和计算都略增            |

QLoRA 并不保证任意 7B 模型在 8GB 上能跑任意长度：基座省了，激活和 logits 一点没少。

> [!question] 随堂自测 D6-4：梯度与冻结
>
> 1. $W=\begin{bmatrix}1&2\\3&4\end{bmatrix}$、$B=0$、$X=[5,6]$，loss 是两个输出之和。正确的 $dX$ 是多少？把基座放进 `no_grad()` 后呢？
> 2. LoRA-FA 冻结 $A$ 后，$dX$ 公式里哪一项还在？为什么它仍然需要 $A$？
> 3. 用 $W\gets W+s\,BA$ 再 $W\gets W-s\,BA$ 恢复，三轮之后用 `torch.equal` 比较原 $W$，可能失败吗？

> [!tip]- 核对要点
>
> 1. $G=[1,1]$，$dX=GW=[4,6]$；放进 `no_grad()` 后只剩低秩分支，$B=0$ 时 $dX=[0,0]$，loss 数值不变。
> 2. 两项都在：$dX=GW+s(GB)A$。$A$ 不训练，但仍在前向路径上，$X$ 的梯度要经过它。LoRA-FA 只是不需要 $dA$，所以不必存 $X$。
> 3. 可能。每次加减都要舍入，$\Delta$ 的量级远大于 $W$ 的某些元素时，结果和原值差一个或几个 ulp。精确恢复要靠备份。

## §5 重计算与预算规划

### 5.1 $\sqrt{n}$ checkpoint

[Chen 等](https://arxiv.org/abs/1604.06174)（2016）的推导：一条 $n$ 层的链，每层输出 1 个单位的激活。每隔 $k$ 层存一个检查点，共 $n/k$ 个；反传到某一段时，从该段的检查点重算出段内 $k$ 个激活。峰值约为

$$
M(k)=\frac{n}{k}+k
$$

对 $k$ 求导，令 $1-n/k^2=0$，得 $k=\sqrt{n}$，$M=2\sqrt{n}$。代价是多一次完整前向；反传约为前向两倍时，总计算量大约增加三分之一。递归地对每段再做 checkpoint 可以降到 $O(\log n)$ 显存，代价是 $O(n\log n)$ 的计算。

[Checkmate](https://proceedings.mlsys.org/paper_files/paper/2020/hash/0b816ae8f06f8dd3543dc3d9ef196cab-Abstract.html)（MLSys 2020）把一般计算图上的“存哪些、何时重算”写成混合整数线性规划，求最优调度，也给了 LP 舍入的近似。Korthikanti 的 selective recompute（§3.3）是一个手工挑出来的特例：只重算“存起来很大、重算很便宜”的 attention 分数。

### 5.2 PyTorch 的三套接口

- **非重入 checkpoint**：`torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)`，区域内只存输入，反传时重跑 `fn`。torch 2.14.1 里 `use_reentrant` 的默认值仍是 `None`，务必显式传 `False`。重入版在输入不需要梯度时不会给区域内的参数算梯度，lab 的 fault `b1-reentrant` 就是这个坑。
- **Selective Activation Checkpoint（SAC）**：`context_fn=functools.partial(create_selective_checkpoint_contexts, policy_fn)`，`policy_fn` 对每个算子返回 `CheckpointPolicy`。2.14.1 里实测的成员有 `MUST_SAVE`、`PREFER_SAVE`、`MUST_RECOMPUTE`、`PREFER_RECOMPUTE`、`MUST_CPU_OFFLOAD`、`PREFER_CPU_OFFLOAD`，最后两个直接把某个算子的输出卸载到 CPU。
- **编译期预算**：`torch._functorch.config.activation_memory_budget`，取值 0 到 1。1.0 是 `torch.compile` 默认的“为速度保存”，0.0 相当于对编译区域做完全 checkpoint，中间值让 partitioner 解一个 0/1 背包，求满足预算的最少重算。2.14.1 实测默认求解器是 `"dp"`、代价估计是 `"flops"`。注意 PyTorch 博客里写的 `torch._dynamo.config.activation_memory_budget` 在 2.14.1 里不存在，要用 `_functorch` 的写法；按区域设预算的 `torch.autograd.graph.region_activation_memory_budget` 也存在，但只在编译时有效。

见 [PyTorch checkpoint 文档](https://docs.pytorch.org/docs/2.14/checkpoint.html)和[激活 checkpoint 技术博客](https://pytorch.org/blog/activation-checkpointing-techniques/)。

### 5.3 dropout 让“再算一遍”不平凡

$y=xm/(1-p)$。原前向 $m=0$，重算时若抽到 $m'=1$，反传用的就是另一条随机路径，梯度对不上。`preserve_rng_state=True` 会在进入区域时存下 CPU 和一种设备的 RNG 状态，重算前恢复。比较 retain 和 checkpoint 时也要在每次前向前恢复相同的种子，并逐参数比较梯度：两个向量范数相同，方向可以完全不同。

### 5.4 预算可以抽象成背包

给 block $i$ 一个保存代价 $s_i$（单位：KiB）和不保存时的重算代价 $c_i$（单位：ms）。$x_i=1$ 表示保留激活，预算为 $B$：

$$
\max_{x\in\{0,1\}^n}\ \sum_i c_ix_i\quad \text{s.t.}\quad \sum_i s_ix_i\le B
$$

这是 0/1 背包，目标是“省下最多的重算时间”。写成“最小化保存量”的话最优解就是什么都不存，答不了原问题。lab 的 B4 再推广一步：每块在 retain、checkpoint、offload 里选一个，各有 (单位, ms)，这是多选背包。状态是“前 $i$ 块用了 $u$ 个单位时的最小总代价”，一个状态若单位更多、代价也不更低，就被支配，可以剪掉。

满足 $\sum_i s_ix_i\le B$ 只说明**模型里的预算**可行。这个加和模型漏掉了生命周期重叠、重算时的临时区、梯度和激活同时存活、分配器碎片，所以计划必须用整步实测验证。PyTorch 的 partitioner 解的是同一类问题，而且是在算子级的联合图上，所以研究里它是必须对比的强基线；lab 的 bench 会把块级 DP 的计划和 `activation_memory_budget` 取 1.0、0.5、0.0 的编译结果放在同一张表里。

> [!question] 随堂自测 D6-5：重计算
>
> 1. $n=64$ 层，按 $\sqrt{n}$ 分段，存多少个检查点？峰值是多少个单位？和全存相比呢？
> 2. $\mathrm{sizes}=[2,5,3,7]$、$\mathrm{costs}=[1,9,4,6]$、$B=8$，最优保留集合和收益是多少？贪心按 $c_i/s_i$ 选会得到什么？
> 3. 两个计划的预测保存量都在预算内，一个实际 OOM。先怀疑 DP 还是模型？列出两项模型漏掉的内存。

> [!tip]- 核对要点
>
> 1. $k=8$，存 8 个检查点，峰值约 $8+8=16$ 个单位，全存是 64。
> 2. 保留第 1、2 块（从 0 数），大小 8，收益 13。贪心按密度：$9/5=1.8$、$4/3\approx1.33$、$6/7\approx0.86$、$1/2=0.5$，先选第 1 块（5），再选第 2 块（8），恰好也是 13；换成 $[10,20,30]$、$[60,100,120]$、$B=50$，贪心得 160，最优是 220。
> 3. 先用穷举检查 DP（lab 里已有），再查模型。漏掉的有：重算时区域内的临时张量、反传时梯度与未释放的激活同时存活、optimizer step 的临时区、缓存池碎片。

## §6 offload、hooks 与微批

### 6.1 `saved_tensors_hooks` 的契约

`torch.autograd.graph.saved_tensors_hooks(pack, unpack)` 里，每个被 autograd 存下的张量（内置算子和 `save_for_backward` 都算）都经过 `pack`，返回值替代原张量放进图里；反传用到时调用 `unpack` 还原。[文档](https://docs.pytorch.org/docs/2.14/autograd.html)和[教程](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)写明了几条规则：

- 同一时刻只有一对 hook 生效，嵌套时最内层的赢；
- `pack` 的返回值不能持有输入张量本身的引用（返回 `x.detach()` 或一份拷贝），否则会形成引用环；
- `unpack(pack(t))` 在值、形状、dtype、设备上都要等于 `t`；
- `unpack` 可能被调用多次，payload 不能在第一次 unpack 后失效；
- 对 hook 的输入做原地修改是未定义行为。

`torch.autograd.graph.save_on_cpu(pin_memory=...)` 是官方的 CPU 卸载版。非重入 checkpoint 内部也用这套机制；2.14 新增的 `respect_saved_tensors_hooks` 决定 checkpoint 保存的张量是否还经过外层用户 hook，默认 `None` 在外面有 hook 时会给 FutureWarning。lab 的 C3 把 checkpoint 和 offload 用在不同的块上，避免嵌套。

### 6.2 同步 offload 省的是什么

lab 的 `CpuOffload.pack` 把一个 CUDA 激活复制到新的 host 内存，返回 (copied, 原设备, host 张量)，不再引用设备上的张量；`unpack` 每次新建一个设备张量拷回去。这样前向结束后设备上不再持有这些激活，test 在 4060 上检查前向后的 `memory_allocated` 低于 retain 的 60%。同步拷贝不会和计算重叠，所以通常更慢；省显存和隐藏传输是两个独立的结论。只把 CUDA 张量拷走、却在 payload 里留着原张量引用，正确性测试全过，显存一个字节也没省，lab 有专门的 fault（`c1-keeps-device-reference`）。

### 6.3 不等大小的微批

整体目标是全 batch 的 MSE 均值。两个微批的元素数为 $n_1,n_2$，均值为 $\ell_1,\ell_2$，正确的组合是

$$
\ell=\frac{n_1\ell_1+n_2\ell_2}{n_1+n_2}
$$

实现上每个微批算 $\sum(\hat y-y)^2$，除以**总**元素数再 backward。直接平均两个均值会加重小微批的权重。梯度裁剪必须在累积完成之后做一次：每个微批各自裁剪，等于对一个不同的目标做优化。lab 的 C4 测试会先确认全 batch 梯度范数远大于 `max_grad_norm`，否则“不裁剪”“每微批裁剪”“累积后裁剪”三者无法区分。

> [!question] 随堂自测 D6-6：hooks 与微批
>
> 1. 一个 4 KiB 的 CUDA 激活 pack 一次、unpack 两次，packed bytes、D2H、H2D、restores 各是多少？能否据此说 host 峰值是 4 KiB，或者传输已被隐藏？
> 2. 两个微批的元素数为 2 和 3，均值为 1 和 4。正确的整体目标是多少？简单平均呢？
> 3. 为什么 `pack` 不能直接 `return tensor`？

> [!tip]- 核对要点
>
> 1. 4096、1、2、2。这些是累计计数，不是 host 上同时存活的峰值；同步拷贝也谈不上隐藏传输。
> 2. $(2\times1+3\times4)/5=2.8$；简单平均是 2.5。
> 3. 图持有 payload，payload 持有带 `grad_fn` 的张量，张量又持有图，形成引用环；文档要求返回 detach 后的张量或拷贝。

## §7 自定义 `autograd.Function`：存得比 eager 少

### 7.1 写法与检查

[扩展文档](https://docs.pytorch.org/docs/2.14/notes/extending.html)给了两种写法：`forward(ctx, ...)` 合在一起，或者 `forward(...)` 加 `setup_context(ctx, inputs, output)`。后者和 `torch.func` 变换配合更好，但中间结果（比如 $XA^\top$）只能作为输出返回才能保存；lab 用合在一起的写法。要点：

- 用 `ctx.save_for_backward` 存张量，它们会经过 saved-tensor hook，能被计数和卸载；把张量直接挂在 `ctx.x` 上能用，但绕过了 hook，也绕过了版本检查。
- `ctx.needs_input_grad[i]` 告诉你第 $i$ 个输入要不要梯度，不要的返回 `None`，也就不必为它存东西。
- 用 `torch.autograd.gradcheck` 在 float64 下检查反传，它的默认容差就是为双精度设计的；float32 下有限差分误差太大。
- 保存的不是输入也不是输出的张量时，double backward 可能出错，lab 只要求一阶。

### 7.2 R1：LoRA-FA 线性层

$A$ 冻结，存 $Z=XA^\top$，反传

$$
dB=s\,G^\top Z,\qquad dX=GW+s\,(GB)A
$$

保存集和 eager LoRA-FA 一样（§3.1 的探针）。这个 TODO 的价值在于亲手写出反传、用 gradcheck 和字节计数证明保存集，而不是省显存。

### 7.3 R2：重算 GELU 的融合 LoRA MLP

前向：$z_1=XA_1^\top$，$u=XW_1^\top+s_1z_1B_1^\top$，$g=\mathrm{GELU}(u)$，$z_2=gA_2^\top$，$Y=gW_2^\top+s_2z_2B_2^\top$。eager 会同时存 $u$（给 GELU 的反传）和 $g$（给 down 的 $dA_2$）。融合 Function 只存 $X,z_1,u,z_2$，反传时由 $u$ 重算 $g$。精确 GELU 是 $u\,\Phi(u)$，导数

$$
\mathrm{GELU}'(u)=\Phi(u)+u\,\varphi(u),\qquad \varphi(u)=\frac{1}{\sqrt{2\pi}}e^{-u^2/2}
$$

反传顺序：$dz_2=s_2\,G B_2$，$dB_2=s_2\,G^\top z_2$，$dA_2=dz_2^\top g$；$du=(GW_2+dz_2A_2)\odot\mathrm{GELU}'(u)$；再对 up 重复 R1 的三条公式，$dA_1=dz_1^\top X$。省下的是每 token $f$ 个数，多花的是一次逐元素 GELU。重算必须用和前向同一个 GELU：前向用 erf、反传用 tanh 近似，误差约 $10^{-4}$，float64 的 oracle 能抓住。[Unsloth 的 `LoRA_MLP`](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py) 在 SwiGLU 上做的是同一件事：存 $X$ 和 gate、up 的输出，反传时重算激活。

### 7.4 R3：只存种子的 LoRA dropout

PEFT 式的 LoRA dropout 只对低秩分支的输入做 dropout：$Y=XW^\top+s\,((X\odot m)A^\top)B^\top$。eager 会存 mask（CUDA 上每元素 1 字节）和 $X\odot m$（给 $dA$）。自定义 Function 存 $X$、$Z$ 和一个整数种子，反传时用私有的 `torch.Generator` 按同一种子重新生成 $m$：

$$
dZ=s\,GB,\qquad dA=dZ^\top(X\odot m),\qquad dX=GW+(dZ\,A)\odot m
$$

每 token 省下 $(h+f)$ 字节的 mask；在 CPU 上 mask 是 float32，省得更多。但保存量少不等于峰值低：参考实现在反传里按输入 dtype 重新生成整块 mask，这个临时张量比 bool mask 大 4 倍，4060 上 8192 token 时 R3 的 fw+bw 峰值反而比 eager 高约 5%，时间也慢 13% 到 20%（lab6 R bench 两次实测）。要把省下的保存量变成峰值上的收益，mask 得在反传里按块生成，或者在 kernel 里生成。前向和反传必须用同一个种子、同一个生成函数，不能用全局 RNG（checkpoint 恢复的是全局状态，跟这里无关）。再进一步可以把 mask 按位压成 1/8，或者像 FlashAttention 那样在 kernel 里由 Philox 计数器重新生成。

### 7.5 R4：怎样证明真的省了

三个证据，缺一不可：

1. **saved bytes**：在 `saved_tensors_hooks` 下数非参数 storage 的字节，与 §3.2 的手算公式逐项对上；
2. **前向后的 allocated 增量**：在 CUDA 上 `memory_allocated` 在前向前后的差，说明这些张量确实还活着、确实被释放了；
3. **峰值和时间**：`reset_peak_memory_stats` 后跑一次前向加反传，读 `max_memory_allocated`；时间用 CUDA event，报告中位数和 IQR。

参考实现在 4060 上 8192 token 的实测：R2 少存 384 MiB，fw+bw 峰值只降了约 5%；整块 checkpoint 的保存量是 eager 的 15%，峰值约为一半。两者都更慢，但两次运行之间 eager 的中位数从 185 ms 变到 165 ms，慢多少（R2 约 9% 到 23%，checkpoint 约 27% 到 55%）要靠重复运行和 IQR 才能说清。保存字节、峰值、时间三者要分开报告。

R 的 bench 在 Qwen3-0.6B 的宽度（$h=1024$、$f=3072$、$r=16$，4 层）上扫 1024、4096、8192 个 token，比较 eager、LoRA-FA、R1、R2、eager dropout、R3 和整块 checkpoint，最后画显存与时间的权衡图。

> [!question] 随堂自测 D6-7：自定义 Function
>
> 1. 按 §3.2，$h=1024$、$f=3072$、$r=16$、4 层、8192 token，R2 融合比 eager 少存多少 MiB？
> 2. 为什么 R3 不能在反传里调用 `torch.rand_like(x)` 重新生成 mask？
> 3. 把 $dB$ 写成 $s\,Z^\top G$ 会发生什么？

> [!tip]- 核对要点
>
> 1. $4\times8192\times3072\times4$ 字节，即 384 MiB。
> 2. 全局 RNG 在前向之后已经被别的算子推进了，抽到的是另一个 mask；就算没有被推进，前向也没记下状态。必须由同一种子和同一生成函数确定 mask。
> 3. 形状变成 $[r,d_{\mathrm{out}}]$，和 $B$ 的 $[d_{\mathrm{out}},r]$ 不符，autograd 会直接报错；方阵时不报错但数值错，只有 oracle 能发现。

## §8 优化器状态与融合 kernel

| 方法                                                                                                                      | 改变哪一项      | 机制                                                             |
| ------------------------------------------------------------------------------------------------------------------------- | --------------- | ---------------------------------------------------------------- |
| [8-bit 优化器](https://arxiv.org/abs/2110.02861)（ICLR 2022）                                                             | $M_O$           | 按块动态量化 Adam 的 $m,v$                                       |
| [4-bit 状态](https://arxiv.org/abs/2309.01507)（NeurIPS 2023）                                                            | $M_O$           | 更小的块、rank-1 归一化、无零点的二阶矩量化                      |
| [GaLore](https://arxiv.org/abs/2403.03507)（ICML 2024）                                                                   | $M_O$           | 定期 SVD 把梯度投影到低秩子空间，状态只存在子空间里              |
| [Q-GaLore](https://arxiv.org/abs/2407.08296)（arXiv 2024）                                                                | $M_O$、$M_W$    | INT4 投影、INT8 权重加随机舍入；称 7B 能在 16GB 4060 Ti 上预训练 |
| [APOLLO](https://arxiv.org/abs/2412.05270)（MLSys 2025）                                                                  | $M_O$           | 用随机投影的低秩状态估计按通道的学习率缩放                       |
| [Adam-mini](https://arxiv.org/abs/2406.16793)（ICLR 2025）                                                                | $M_O$           | 每个 Hessian 块只用一个 $v$，状态约减半                          |
| optimizer-in-backward（[教程](https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html)） | $M_G$           | `register_post_accumulate_grad_hook` 里直接 step 并释放梯度      |
| [Liger Kernel](https://arxiv.org/abs/2410.10989)（arXiv 2024）                                                            | $M_A$（logits） | 融合 linear 和交叉熵，分块计算，不生成完整 logits                |
| [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009)（ICLR 2025）                                                        | $M_A$（logits） | 不生成 logits 就算出交叉熵和梯度                                 |

对 LoRA 微调 Qwen3-0.6B 来说，上表前六行只动了约 160 MB 的那一项；后两行处理的是 1.16 GiB 的 logits，影响大得多。全参数训练 0.6B 时情况反过来，GaLore 一类方法才变得关键。

## §9 测量口径

- `torch.cuda.memory_allocated`：此刻被张量占用的字节；`max_memory_allocated`：自上次 `reset_peak_memory_stats` 以来的最大值。
- `memory_reserved`：caching allocator 向驱动要来的总量，包含缓存池。`empty_cache` 只把未用的缓存还给驱动，不会减少活张量。
- saved bytes：hook 数出的为反传保存的 storage 总和，是一次前向的清单，不是随时间变化的峰值。
- 这些数都不含 CUDA context 和不经过 caching allocator 的分配（例如 NCCL 缓冲）。`nvidia-smi` 看到的是整个进程的占用。
- 查看时间线：`torch.cuda.memory._record_memory_history(max_entries=100000)`，跑几步，`torch.cuda.memory._dump_snapshot("snap.pickle")`，再拖进 [memory_viz](https://docs.pytorch.org/memory_viz)。每条记录有几 KB，必须设 `max_entries`；采集会拖慢训练，和计时分开做。
- MPS 只有 `torch.mps.current_allocated_memory()` 和 `driver_allocated_memory()`，没有峰值接口，只能采样。lab 的 R bench 支持 `--device mps`，报告前向后的增量，峰值一栏为空。
- 笔记本 4060 在两次配置之间会降到低功耗状态（显存频率 810 MHz 对 8000 MHz），GPU 操作可能慢到 8 倍，结果呈双峰。计时前用同一个 step 让 GPU 忙至少 300 ms，用 CUDA event 计时，报告中位数和 IQR，并标出 $p_{75}\le1.5\,p_{25}$ 是否成立（lab 里的 `stable` 字段）。峰值显存不受影响。

> [!question] 随堂自测 D6-8：测量
>
> 1. reserved 很高、allocated 每步稳定，能说明内存泄漏吗？
> 2. 一个 offload 实现的 saved bytes 和 retain 一样，`after_forward_bytes` 却小得多。这说明什么？反过来呢？
> 3. 两次 bench 的中位数差 3 倍，其中一次 `stable=false`。下结论前该做什么？

> [!tip]- 核对要点
>
> 1. 不能，那多半是缓存池。泄漏的特征是 allocated 每步增长，常见原因是把带图的 loss 存进了列表。
> 2. 前者说明 hook 看到了同样多的张量，但它们被搬走了，设备上没有持有，offload 生效。反过来（saved bytes 小、allocated 不降）说明有张量绕过了 hook（比如挂在 `ctx` 上）或者 payload 留着设备引用。
> 3. 加长预热、重复多次、记录 `nvidia-smi --query-gpu=clocks.mem --format=csv -lms 100`，排除低频状态后再比较。

## §10 机制与代码的对应

| 机制或公式                           | lab 中的函数                                    | 上游实现位置                                                                                                                       | 本 lab 简化了什么                      |
| ------------------------------------ | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| $Y=XW^\top+s\,XA^\top B^\top$、merge | `LoRALinear`（A1–A3）                           | [PEFT `lora/layer.py`](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py) 的 `Linear.forward`、`merge`   | 单 adapter，unmerge 用精确备份         |
| 注入与 adapter 状态                  | `inject_lora`、`load_adapters`（A4–A6）         | PEFT `inject_adapter`；[checkpoint 指南](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/checkpoint.md) | 内存字典，不兼容 PEFT 文件格式         |
| 按 storage 计参数字节                | `parameter_bytes`（A7）                         | `nn.Module.named_parameters(remove_duplicate=True)` 只按对象去重                                                                   | 只分 trainable/frozen                  |
| 非重入 checkpoint                    | `run_blocks`（B1）                              | [`torch/utils/checkpoint.py`](https://docs.pytorch.org/docs/2.14/checkpoint.html)                                                  | 块级，不做算子级 SAC                   |
| 保存集计数                           | `SavedInventory.pack`（B3）                     | `saved_tensors_hooks` 教程                                                                                                         | 一次前向的清单                         |
| 预算背包                             | `plan_keep`、`plan_modes`（B2/B4）              | `torch/_functorch/partitioners.py` 的 memory-budget 求解（`activation_memory_budget`）                                             | 块级加和模型                           |
| 同步 CPU offload                     | `CpuOffload`（C1/C2）、`run_policy`（C3）       | `torch.autograd.graph.save_on_cpu`                                                                                                 | 同步拷贝，无 pinned 内存与 stream 重叠 |
| LoRA-FA、融合 MLP、种子 dropout      | `fused_student.py`（R1–R3）                     | [Unsloth `fast_lora.py`](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py) 的 `LoRA_MLP`、`LoRA_W`      | GELU 代替 SwiGLU，无 Triton kernel     |
| 显存与时间测量                       | `measure_policy`（C5）、`measure_variant`（R4） | `torch.cuda` 内存统计、[memory_viz](https://docs.pytorch.org/memory_viz)                                                           | 单卡、单进程                           |

上游默认分支在 2026-10-18 可访问，引用前请固定 commit（见[源码地图](../readings/code-map.md)）。

## 前沿与 baseline 对照

核验日期：2026-10-18。会议和年份来自论文页、会议官网或作者主页；标注“arXiv”的没有找到正式会议版本。

| 工作（会议 年）                                                                                                                                                                   | 问题                         | 关键机制                                                                   | 与本 lab 关系（实现/简化/未覆盖）            | 研究时必须对比             |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | -------------------------------------------------------------------------- | -------------------------------------------- | -------------------------- |
| ZeRO（SC 2020）[论文](https://arxiv.org/abs/1910.02054) · [代码](https://github.com/microsoft/DeepSpeed)                                                                          | 数据并行的模型状态冗余       | 依次切分优化器状态、梯度、参数                                             | 未覆盖（单卡）；§2 的公式                    | 每卡模型状态与通信量       |
| ZeRO-Offload（ATC 2021）[论文](https://arxiv.org/abs/2101.06840) · [代码](https://github.com/microsoft/DeepSpeed)                                                                 | 单卡放不下优化器状态         | FP32 状态和 step 搬到 CPU                                                  | 简化：C 阶段只卸载激活                       | PCIe 时间能否被计算盖住    |
| PyTorch FSDP（PVLDB 2023）[论文](https://arxiv.org/abs/2304.11277) · [代码](https://github.com/pytorch/pytorch)                                                                   | ZeRO-3 的工程化              | FlatParameter、预取与重叠；FSDP2 改为按参数 DTensor                        | 未覆盖                                       | 同上                       |
| Checkpointing $\sqrt{n}$（arXiv 2016）[论文](https://arxiv.org/abs/1604.06174) · 无独立代码（在 MXNet 内实现）                                                                    | 激活随层数线性增长           | 每 $\sqrt{n}$ 层存一个检查点                                               | 实现：B1 的块级 checkpoint                   | 全 checkpoint 与固定间隔   |
| Checkmate（MLSys 2020）[论文](https://arxiv.org/abs/1910.02653) · [代码](https://github.com/parasj/checkmate)                                                                     | 一般图上的最优重算           | MILP / LP 舍入                                                             | 简化：B2/B4 是链上的背包                     | 规划时间与最优性           |
| Selective recompute（MLSys 2023）[论文](https://arxiv.org/abs/2205.05198) · [代码](https://github.com/NVIDIA/Megatron-LM)                                                         | attention 分数占大头         | 只重算“大而便宜”的部分；序列并行                                           | 未覆盖；§3.3 的公式                          | 激活公式的假设             |
| PyTorch SAC 与 memory budget（PyTorch 博客，2026-04 更新）[文章](https://pytorch.org/blog/activation-checkpointing-techniques/) · [代码](https://github.com/pytorch/pytorch)      | 算子级保存/重算策略          | `CheckpointPolicy`；partitioner 上的 0/1 背包                              | 对照：bench 的 `activation_memory_budget` 行 | **强基线**：同模型按预算扫 |
| TorchTitan 图级 CPU offload（PyTorch DevLog 2026）[文章](https://docs.pytorch.org/devlogs/distributed/2026-06-23-cpu-offloading/) · [代码](https://github.com/pytorch/torchtitan) | SAC 保存的张量仍太多         | 在预算内按大小卸载 `MUST_SAVE` 张量，预取与 pinned 池                      | 简化：C1/C2 同步、按块                       | 同预算下的吞吐             |
| MEMO（SIGMOD 2025）[论文](https://arxiv.org/abs/2407.12117) · 无公开代码                                                                                                          | 超长上下文的激活             | 一部分 token 卸载、其余重算，比例由 LP 决定；MIP 规划碎片                  | 未覆盖                                       | 按长度变化的 offload 比例  |
| ByteScale / SPPO（arXiv 2025）[论文](https://arxiv.org/abs/2502.21231) · [论文](https://arxiv.org/abs/2503.10377)                                                                 | 变长数据的激活               | 按序列长度自适应卸载比例                                                   | 未覆盖                                       | 同上                       |
| LoRA（ICLR 2022）[论文](https://arxiv.org/abs/2106.09685) · [代码](https://github.com/microsoft/LoRA)                                                                             | 全参数微调太贵               | 冻结 $W$，训练低秩 $BA$                                                    | 实现：A 阶段                                 | 同数据预算的质量           |
| QLoRA（NeurIPS 2023）[论文](https://arxiv.org/abs/2305.14314) · [代码](https://github.com/artidoro/qlora)                                                                         | 基座权重太大                 | NF4 基座、双重量化、paged optimizer                                        | 未覆盖（基座 FP32）                          | 基座精度对质量的影响       |
| LoRA-FA（arXiv 2023）[论文](https://arxiv.org/abs/2308.03303) · 无公开代码                                                                                                        | LoRA 仍要存 $X$              | 冻结 $A$，只存 $XA^\top$                                                   | 实现：R1                                     | 质量是否下降               |
| DoRA（ICML 2024）[论文](https://arxiv.org/abs/2402.09353) · [代码](https://github.com/NVlabs/DoRA)                                                                                | LoRA 与全参数的质量差        | 幅度/方向分解                                                              | 未覆盖                                       | 激活与计算的额外开销       |
| VeLoRA（NeurIPS 2024）[论文](https://arxiv.org/abs/2405.17991) · [代码](https://github.com/roymiles/VeLoRA)                                                                       | 保存的激活太大               | rank-1 子 token 投影压缩激活                                               | 未覆盖（有损）                               | 与精确保存集的质量差       |
| LoRAct（arXiv 2025）[论文](https://arxiv.org/abs/2509.23472) · 无公开代码                                                                                                         | 同上                         | 在线低秩压缩保存的激活                                                     | 未覆盖（有损）                               | 同上                       |
| CARE-LoRA（arXiv 2026）[论文](https://arxiv.org/abs/2607.11940) · 无公开代码                                                                                                      | 不冻结 $A$ 也不存 $X$        | 存 $XA^\top$ 加一个小重建矩阵                                              | 对照：R1 的“只存 $Z$”加上 $dA$ 的修补        | 与 R1、R2 的精确方案比     |
| RunLoRA（ACL 2025 Industry）[论文](https://aclanthology.org/2025.acl-industry.15/) · 无公开代码                                                                                   | LoRA 前后向有多种等价写法    | 按形状和秩选 FLOPs 最少的图，省掉一个中间量                                | 对照：R1 的反传顺序                          | 同形状下的时间             |
| LoRAFusion（EuroSys 2026）[论文](https://arxiv.org/abs/2510.00206) · [代码](https://github.com/CentML/lorafusion)                                                                 | LoRA 的访存型小算子          | 融合访存型算子，GEMM 留给 cuBLAS；多 LoRA 调度                             | 未覆盖；可选挑战的模板                       | kernel 时间与 HBM 流量     |
| Unsloth `fast_lora`（代码）[代码](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py)                                                                    | 消费级卡上的 LoRA            | `LoRA_MLP` 存 $X$ 与 gate/up 输出，反传重算激活                            | 实现：R2（GELU 版）                          | 同卡同形状的峰值与时间     |
| Liger Kernel（arXiv 2024）[论文](https://arxiv.org/abs/2410.10989) · [代码](https://github.com/linkedin/Liger-Kernel)                                                             | logits 和 norm 的显存        | 融合 linear+CE、RMSNorm、SwiGLU                                            | 未覆盖                                       | logits 占比                |
| Cut Cross-Entropy（ICLR 2025）[论文](https://arxiv.org/abs/2411.09009) · [代码](https://github.com/apple/ml-cross-entropy)                                                        | 大词表的 loss 显存           | 不生成 logits 计算 CE                                                      | 未覆盖                                       | 同上                       |
| GaLore（ICML 2024）[论文](https://arxiv.org/abs/2403.03507) · [代码](https://github.com/jiaweizzhao/GaLore)                                                                       | 全参数训练的优化器状态       | 梯度低秩投影                                                               | 未覆盖                                       | 只在全参数训练时有意义     |
| APOLLO（MLSys 2025）[论文](https://arxiv.org/abs/2412.05270) · [代码](https://github.com/zhuhanqing/APOLLO)                                                                       | 同上                         | 随机投影估计学习率缩放                                                     | 未覆盖                                       | 同上                       |
| Adam-mini（ICLR 2025）[论文](https://arxiv.org/abs/2406.16793) · [代码](https://github.com/zyushun/Adam-mini)                                                                     | 同上                         | 按 Hessian 块共享 $v$                                                      | 未覆盖                                       | 同上                       |
| FlexLLM（NSDI 2026）[论文](https://www.usenix.org/conference/nsdi26/presentation/oliaro) · [代码](https://github.com/FlexLLM/artifact)                                            | 推理和 PEFT 微调抢同一组 GPU | token 级微调、混合 token 调度；编译期依赖并行化和图剪枝减少激活            | 未覆盖；R 阶段是它“少存激活”这一半的单卡练习 | 推理 SLO 与微调进度        |
| Loquetier（NeurIPS 2025）[论文](https://arxiv.org/abs/2511.00101) · [代码](https://github.com/NJUDeepEngine/Loquetier)                                                            | 多 LoRA 的训练与服务         | 虚拟化模块，合并训练与推理前向的 kernel                                    | 未覆盖                                       | 同上                       |
| DeltaServe（arXiv 2026）[论文](https://arxiv.org/abs/2607.28848) · 无公开代码                                                                                                     | 共服务要改推理引擎           | LoRA 前向搭推理 prefill 的车，反向在可让出的进程里跑                       | 未覆盖                                       | 同上                       |
| Hetu v2（OSDI 2026）[论文](https://arxiv.org/abs/2504.20490) · [代码](https://github.com/PKU-DAIR/Hetu)                                                                           | 异构集群上的通用分布式训练   | 层次化异构 SPMD 标注、渐进式图特化、动态图切换；支持 ZeRO、offload、重计算 | 未覆盖（多卡）                               | 显存策略如何表示进图       |

**查新结论**：

- **(b) LoRA 的精确或近似“少存”反传已经很拥挤。** LoRA-FA 冻结 $A$；Unsloth 的 `LoRA_MLP` 已经在做重算激活；RunLoRA 按形状选择等价图；CARE-LoRA 只存 $XA^\top$ 再补 $dA$；VeLoRA、LoRAct 做有损压缩；LoRAFusion 做融合 kernel；`torch.compile` 的 partitioner 也会自动重算逐元素算子。R 阶段适合当作一个扎实的练习，不适合当研究题。
- **(d) 优化器状态压缩已经饱和。** 8-bit、4-bit、GaLore、Q-GaLore、APOLLO、Adam-mini 都有，而且 §1.3 说明 LoRA 场景下这一项只占约 2%。除非改做全参数训练，不建议选。
- **(a) 按长度选择 checkpoint/offload 计划，在单张消费级卡上还有空白。** 最近的工作：MEMO 用 LP 决定 token 级的卸载比例，最优比例随长度变化，但面向多卡超长上下文；ByteScale 和 SPPO 按序列长度调整卸载比例，同样是集群场景；PyTorch 的 `activation_memory_budget` 在每个编译图上静态求解，预算是比例而不是字节；TorchTitan 的图级 offload 在一个预算下静态选张量。没有找到“单卡、按长度分桶缓存计划、用实测而非 FLOPs 校准、保证峰值字节而不是保存量”的工作。风险是它离 PyTorch 的动态形状加预算路径很近，基线必须是按桶扫 `activation_memory_budget`。
- **(c) 以显存为中心的推理–微调共服务在 8GB 卡上是空的。** FlexLLM、LLMStation（ATC 2025）、Loquetier、DeltaServe 都在数据中心 GPU 上做，稀缺资源是算力空闲；8GB 卡上稀缺的是显存：微调激活（随在途 token 数增长）和 KV cache 页面争同一块显存。按空闲 KV 块数决定微调的 token 块大小、推理突发时把微调激活卸载出去，这类策略还没人做。FlexLLM 的图剪枝语义要先读 [NSDI 论文 PDF](https://www.usenix.org/system/files/nsdi26-oliaro.pdf) 的第 4 节再声称新颖性。

苗组相关工作（经[苗老师主页](https://hsword.github.io/publications/)核实）：FlexLLM（NSDI 2026，作者含苗旭鹏），它的激活显存优化和本讲 §3、§7 的“只存反传需要的东西”是同一个原理；Hetu v2（OSDI 2026），在一个系统里表示 ZeRO、offload、重计算和混合精度；Quantized Side Tuning（[ACL 2024](https://aclanthology.org/2024.acl-long.1/)，苗旭鹏为作者之一）用侧网络避免对 LLM 本体反传，从根上减少激活。

## 研究问题

- **候选假设**：在 RTX 4060 Laptop 上用 LoRA 微调 Qwen3-0.6B，序列长度在 256 到 4096 之间变化时，按长度分桶、每桶用实测校准选一个计划（每层 retain、R2 融合、checkpoint 或 offload），在峰值不超过 7.5 GB 的约束下，比“能容纳最长桶的单一静态计划”的 token 吞吐高至少 15%。
- **最小实验**：把 lab6 的 B5 校准、B4 规划和 R2 融合 Function 接到 Qwen3-0.6B 的 MLP 上（attention 用 SDPA 不动），三种长度分布（均匀、长尾、真实 SFT 数据集的长度直方图），各 3 个种子。对照组：全 checkpoint、固定间隔、单一静态计划、`torch.compile` 加 `activation_memory_budget` 按桶扫 0 到 1、MEMO 式的固定 token 比例卸载。报告峰值、tokens/s（中位数和 IQR）、校准与重编译开销。
- **继续或停止的证据**：按桶 `activation_memory_budget` 已经拿到 80% 以上的收益，说明块级规划没有增量，停下或改做 (c)；收益只出现在长尾分布、而且被重编译开销吃掉，写成负结果；在三种分布下都比最强基线高 10% 以上，才值得扩展到 1.7B 和 attention 层。
- **与导师讨论的问题**：组里有没有真实的微调长度 trace？(c) 的共服务方向能否接到 FlexLLM 的 artifact 上做单卡缩小版？Hetu v2 的图特化能否直接表示“按长度切换计划”？

## 资料汇总

下表的“指定观察”已计入本讲时间，通读论文或运行上游代码属于选读或 lab 挑战。

| 资料                                                                                                                   | 类型 | 指定观察                                                                                                                          | 时间  |
| ---------------------------------------------------------------------------------------------------------------------- | ---- | --------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [HF：Visualize and understand GPU memory in PyTorch](https://huggingface.co/blog/train_memory)                         | 图解 | 彩色分类的训练显存曲线；bs16 与 bs2 时峰值从前向移到 optimizer step                                                               | 10min |
| [PyTorch：Understanding GPU Memory 1](https://pytorch.org/blog/understanding-gpu-memory-1/)                            | 图解 | 三次迭代的时间线：前向上升、反向下降；跨迭代的带状分配                                                                            | 10min |
| [PyTorch：激活 checkpoint 技术](https://pytorch.org/blog/activation-checkpointing-techniques/)                         | 图解 | 普通前向与 checkpoint 各存几个张量的图；SAC 的红色保留算子；真实 Transformer 的预算扫描图（memory 指保存量，不是峰值）            | 15min |
| [ZeRO 论文图 1](https://arxiv.org/pdf/1910.02054)                                                                      | 论文 | 7.5B、64 卡、$K=12$ 时三个阶段的每卡显存                                                                                          | 10min |
| [Korthikanti 等](https://arxiv.org/pdf/2205.05198)                                                                     | 论文 | 图 3 的 selective recompute 区域；式 1 的推导                                                                                     | 15min |
| [PyTorch checkpoint 文档](https://docs.pytorch.org/docs/2.14/checkpoint.html)                                          | 文档 | `use_reentrant`、`preserve_rng_state`、`CheckpointPolicy` 各成员                                                                  | 10min |
| [saved tensors hooks 教程](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html) | 文档 | 保存到 CPU 的示例；unpack 被调用多次的演示                                                                                        | 10min |
| [扩展 autograd 文档](https://docs.pytorch.org/docs/2.14/notes/extending.html)                                          | 文档 | 合并 forward 与 `setup_context` 两种写法；中间结果的保存限制                                                                      | 10min |
| [Unsloth `fast_lora.py`](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py)                  | 代码 | `LoRA_MLP.forward` 的 `save_for_backward`；`backward` 里重算激活的位置                                                            | 15min |
| [HF Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/)                                       | 交互 | 显存分解小工具：改 $s$、ZeRO 阶段、是否重计算，看各项变化                                                                         | 选读  |
| [Stanford CS336 Spring 2026 Assignment 2](https://github.com/stanford-cs336/assignment2-systems)                       | 课程 | §3 Single-GPU Memory：用 `saved_tensors_hooks` 找 autograd 残差、嵌套 checkpoint 的显存最优问题；和 lab6 B 阶段重叠，可作外部练习 | 选读  |
| [CMU 15-779 Fall 2025](https://www.cs.cmu.edu/~zhihaoj2/15-779/schedule.html)                                          | 课程 | 10/08–10/10 的 ZeRO/FSDP 两讲与 11/05 的 PEFT 一讲（贾志豪，也是 FlexLLM 的作者）                                                 | 选读  |
| [MIT 6.5940 Fall 2026](https://hanlab.mit.edu/courses/2026-fall-65940)                                                 | 课程 | 没有微调 lab（Lab 0–5 是 GPU 基础、剪枝、NAS、量化、笔记本上部署 LLM）；相关的是 L14 LLM Post Training 和 L21 On-Device Training  | 选读  |
| [memory_viz](https://docs.pytorch.org/memory_viz)                                                                      | 工具 | 载入自己的 snapshot，放大一个峰值，悬停看分配栈                                                                                   | 选读  |
| [Lab 6](../../labs/lab6/README.md)、[源码地图](../readings/code-map.md)、[资源索引](../readings/resources.md)          | 实验 | 对照 §10 的函数表                                                                                                                 | —     |

## 自测清单

- [ ] 不看讲义，写出 16 B/param 的两种拆法和 ZeRO 三阶段的公式，算出 7.5B、64 卡的四个数。
- [ ] 算出 Qwen3-0.6B 全参数和 LoRA $r=16$ 两种模型状态，以及 2048 token 的 logits 字节。
- [ ] 说出 `should_compute_output` 规则，预测 eager LoRA、LoRA-FA、冻结线性层各存什么，并用探针验证。
- [ ] 写出 LoRA MLP 块每 token 的保存个数，以及 LoRA-FA、R2、checkpoint 各减掉哪几项。
- [ ] 推导 $\sqrt{n}$ checkpoint，写出 0/1 背包和多选背包的状态转移，并说明“预算可行”为什么不保证不 OOM。
- [ ] 写出 R2 的反传公式和 $\mathrm{GELU}'$，说明 R3 为什么必须用私有生成器。
- [ ] 区分 allocated、reserved、peak、saved bytes，说明笔记本 GPU 计时为什么要预热 300 ms。
- [ ] 对照表中至少三篇工作，各写一句与本 lab 的机制差异和简化边界。
- [ ] 写出一个可证伪的假设、必须对比的 baseline 和停止条件。

[← 课程索引](../index.md) · [配套 lab6](../../labs/lab6/README.md) · [← D5 前缀缓存](05-prefix-cache.md) · [D7 MoE 卸载 →](07-moe-offload.md)
