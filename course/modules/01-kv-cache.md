---
title: D1 KV 缓存：从量化布局到质量预算
lab: lab1
hours: 5
tags: [lecture, kv-cache, quantization, research]
---

# D1 KV 缓存：从量化布局到质量预算

[← 课程索引](../index.md) · [配套 lab1](../../labs/lab1/README.md) · [D2 低比特算子 →](02-low-bit-kernels.md)

> [!abstract] 这一讲要解决的问题
> foundations 已介绍 attention 的 IO 与 paged KV。本讲进一步讨论：缓存预算不足时，应保留哪些内容、采用什么存储格式，又该如何衡量损失。你将实现一种可独立解码的 int4 格式，分别分析表示误差、位置选择误差及其对最终 attention 输出的影响。

| 阅读模块          | 时间  | 必须留下的学习产物            | 对应实验 |
| ----------------- | ----- | ----------------------------- | -------- |
| §1 容量与字节账   | 40min | 两个 batch 的含元数据预算     | A1       |
| §2 仿射量化与布局 | 75min | $3\times3$ 的两种分组编码表   | A1/A2    |
| §3 误差传播与选择 | 60min | 一个 MSE 失效反例、选择不变量 | B        |
| §4 追加与系统接口 | 40min | 残余窗口状态图                | B2–B5    |
| §5 论文与源代码   | 45min | KIVI/PQCache 机制差异卡       | C        |
| §6 实验决策与自测 | 40min | 校准/评测分割与停止条件       | C        |

**共 5h，另做 lab 24h。** 5h 包括手算和少量源码阅读，仅浏览正文不算完成。本讲无需先重新学完 PyTorch：主实验使用 C++17，真实模型诊断列为进阶选项。

随堂自测计入上表的手算与讨论时间。先在笔记中写答案，再展开核对要点；图解只需完成指定观察任务，完整论文与博客留作选读。

## §1 缓存预算：先确定“谁占了字节”

### 1.1 从一个 token 推导整段缓存

每个 decoder 层都保存各 token 的 K 和 V，两者各有 $H_{\mathrm{kv}}\times D$ 个元素。对于配置相同的 $L$ 层、$B$ 条等长序列，所需存储为：

$$
M_{\mathrm{KV}}=2LBTH_{\mathrm{kv}}Db.
$$

这里 $b$ 是每个元素的字节数。对于异长序列，用 $\sum_{b=1}^{B}T_b$ 代替 $BT$；对于各层配置不同的模型，则对层求和。GQA 中一个 KV head 被多个 query head 使用，不应按 query head 数重新存一份 KV。

手算 $L=24,H_{\mathrm{kv}}=8,D=64,T=4096,b=2$：

$$
\begin{aligned}
M_{\text{token, layer}} &= 2\times8\times64\times2 = 2048\,\mathrm{B},\\
M_{\text{token}} &= 24\times2048 = 49152\,\mathrm{B} = 48\,\mathrm{KiB},\\
M_{\text{request}} &= 4096\times48\,\mathrm{KiB} = 192\,\mathrm{MiB},\\
M_{\text{batch}} &= 8\times192\,\mathrm{MiB} = 1536\,\mathrm{MiB}.
\end{aligned}
$$

8GB 显存不能全部用于 KV。计算预算时，还要分别列出权重、暂存激活、CUDA 工作区、页表、allocator 保留空间和系统占用。CPU 内存与 GPU 显存也不能直接相加，按同一层等带宽存储计算。

### 1.2 int4 的有效位宽不是恰好 4bit

lab 对每组 $g$ 个元素保存一个 FP32 scale 和一个 FP32 offset。若组完整且忽略对齐：

$$
b_{\mathrm{eff}}=\frac{4}{8}+\frac{4+4}{g}=0.5+8/g.
$$

$g=64$ 时是 0.625B/元素，相对 FP16 压缩 3.2 倍；$g=16$ 时是 1B/元素，只有 2 倍。用同一公式补算 $g=32$，在笔记中分别列出 payload、metadata 和总字节数；检查只改变 group 时，哪一项开销发生了变化。

精确统计还要处理尾组。对 $[T,D]$ 矩阵：

$$
G_{\mathrm{token}}=T\lceil D/g\rceil,\qquad G_{\mathrm{channel}}=D\lceil T/g\rceil.
$$

本实验每个矩阵的实际编码字节为：

$$
M_{\mathrm{packed}}=\lceil TD/2\rceil+8G.
$$

`Packed::bytes()` 统计的就是这些字节，不包含 C++ vector 对象开销和解码时的临时 FP32 数组。因此，报告中不能将它作为进程 RSS 或 GPU 峰值显存。

### 1.3 三种节省方法改变了不同对象

| 方法          | 保留哪些信息           | 新增成本               | 必须验证什么                |
| ------------- | ---------------------- | ---------------------- | --------------------------- |
| 量化          | 每个位置仍存在，值近似 | scale、解码、误差      | 精度与 fused attention 路径 |
| 淘汰/稀疏访问 | 只访问部分位置         | 选择、索引、潜在漏召回 | 质量及是否真的回收存储      |
| Offload       | 信息留在其他存储层     | 传输、同步、预取       | 总时间及传输是否可隐藏      |

lab 的 `select_tokens` 只返回索引；A3 的 `gather_rows` 才产生物理紧凑数组，C4 将其接入检索。原 K/V 仍由调用方持有，因此 fetch bytes 不是整个系统驻留量。B 的 ResidualCache 持有压缩 chunks 与 FP32 recent，查询只临时解码一块。

> [!example] 看图：分配空间与压缩数值分别改变什么
> 打开 [vLLM 官方 PagedAttention 图解](https://vllm.ai/blog/2023-06-20-vllm)，看 “The Secret Sauce: PagedAttention” 中的分块示意和单请求生成动画。沿一个逻辑块找到物理块，再观察追加 token 时何时需要新块。用两种颜色标出“最后一块空位”和“每个有效 KV 元素占用的字节”：分页和量化分别针对哪一种开销？这些是机制动画，文中的吞吐实验不代表本 lab 的结果。

> [!question] 随堂自测 D1-1：容量账是否完整
>
> 1. $T=5,D=7,g=4$ 时，单个矩阵采用 token/channel 分组各占多少字节？与同尺寸 FP16 比较，哪种反而更大？
> 2. 只将 query head 数从 8 增至 32，KV head 数仍为 8，§1.1 的 KV 预算会增加四倍吗？仅返回一半位置的索引又是否让该预算减半？

> [!tip]- 核对要点
>
> 1. payload 为 18B；token 轴 10 组，共 98B；channel 轴 14 组，共 130B。FP16 为 70B，两种都更大，不能忽略元数据。
> 2. 不会：预算按 KV head 计数。选择索引也不自动释放调用方持有的原 K/V，需要另外证明物理回收。

## §2 仿射 int4：数学定义必须落实到每一个地址

### 2.1 量化和反量化

设组内最小值 $m$、最大值 $M$，4bit 有 16 个码：

$$
s=(M-m)/15,\quad q_i=\operatorname{clip}(\operatorname{round}((x_i-m)/s),0,15),
$$

$$
\hat{x}_i=m+s q_i.
$$

当 $M=m$ 时不能除零。lab 约定 $\mathrm{scale}=1,\mathrm{offset}=m,q=0$，得到精确恢复。对于有限输入、正常数值范围与最近舍入，误差不超过 $s/2$，再留浮点容差；这也是现有测试检查的性质。

例：组 $[-2,-1,1,4]$，理想实数计算得到 $m=-2,s=0.4$。若对精确的半整数采用远离零的舍入规则，码为 $[0,3,8,15]$，恢复为 $[-2,-0.8,1.2,4]$。

实际参考实现先将 scale 存为 FP32，得到的值略大于 0.4；再转为 double 做除法时，中间结果略低于 2.5 和 7.5。因此 `std::lround` 得到的码是 **$[0,2,7,15]$**，反量化约为 **$[-2,-1.2,0.8,4]$**。转成 double 不能消除已存入 scale 的舍入误差。这个例子恰好落在量化分界附近，不能把理想实数手算当作逐位 oracle；检查具体编码时，要对齐存储类型和运算顺序。

这里使用无符号仿射码，码 8 不代表 -8。D2 将使用另一种格式，不能混用解码器。

### 2.2 per-token 和 per-channel 的“轴”

我们把单个 head 的 K/V 写成行优先 $[T,D]$：行是 token，列是 channel。

```text
              channel 0  channel 1  channel 2
 token 0           1          1         30
 token 1           1          2         29
 token 2           2          1         31

per-token:   group each row along D -> 3 groups
per-channel: group each column along T -> 3 groups
```

取 $g=3$。per-token 第一行的 scale 是 $29/15\approx 1.933$，1 附近的差异很容易消失；per-channel 第一列 scale 是 $1/15$，第三列 scale 是 $2/15$，异常大值被限制在自己的列中。

这个例子说明了隔离持续异常的 channel 为什么可能有效，但不能证明所有模型的 K 都有这样的分布。如果异常集中在一个 token 的所有 channel 上，结论可能改变。你应再构造矩阵 $\begin{bmatrix}1&1&1\\30&29&31\\1&2&1\end{bmatrix}$，交换两种轴的角色。

[KIVI](https://arxiv.org/abs/2402.02750) 从真实模型分布研究 K/V 的不同分组，并实现低比特缓存。实验先使用 4bit 来说明布局与误差，不作为论文完整 2bit 结果的复现。

> [!question] 随堂自测 D1-2：分组轴与误差
>
> 1. 在本节 $3\times3$ 矩阵中，分别算出 token 第一组、channel 第一组与第三组的 scale。若把矩阵转置，哪一种分组更容易隔离大值？
> 2. 尾组只有 $[4,4,4]$，补一个零后再算 min/max，会把本 lab 的元数据约定改成什么？即使这三个 4 仍恰好可恢复，为什么也不合格？

> [!tip]- 核对要点
>
> 1. 三个 scale 依次为 $29/15$、$1/15$、$2/15$；转置后大值集中在一行，token 分组更容易将它们与小值隔开。这是构造样例，不是所有模型的分布保证。
> 2. 正确常量组是 $\mathrm{scale}=1,\mathrm{offset}=4,q=0$；补零后变成 $\mathrm{scale}=4/15,\mathrm{offset}=0,q=15$。它违反真实元素分组约定，不能用偶然恢复正确掩盖格式错误。

### 2.3 组索引与码索引是两套映射

给定坐标 `(r,c)`，码的线性索引始终是 $i=r\cdot D+c$，与分组轴无关：

$$
\operatorname{byte\_index}(i)=\left\lfloor\frac{i}{2}\right\rfloor,\qquad
\operatorname{bit\_shift}(i)=4(i\bmod2).
$$

对应的位提取代码为 `(codes[byte_index] >> bit_shift) & 0x0f`。

组索引则依赖轴：

$$
\begin{aligned}
\operatorname{group}_{\mathrm{token}}(r,c)
&=r\left\lceil\frac{D}{g}\right\rceil+\left\lfloor\frac{c}{g}\right\rfloor,\\
\operatorname{group}_{\mathrm{channel}}(r,c)
&=c\left\lceil\frac{T}{g}\right\rceil+\left\lfloor\frac{r}{g}\right\rfloor.
\end{aligned}
$$

因此“按 channel 量化”不等于“把 codes 转置存储”。lab 固定 code 布局，只改变 scale/offset 的索引。反量化时需要区分这两套索引。解码值看似随机、误差很大时，问题可能只出在某个地址公式上。

$T\cdot D$ 为奇数时，最后一个 byte 的高 nibble 没有有效元素，要求初始化为 0。否则序列化输出可能不稳定，即使没有读取它，也会破坏字节级比较。

### 2.4 实现顺序：先正确，再减少遍历

A1 可采用三次遍历：第一遍统计每组 min/max，第二遍生成元数据，第三遍生成 codes。A2 逐元素读取对应组和 nibble，再恢复值。

这套三阶段算法并非最优的 GPU 实现，但便于逐项检查以下不变量：

1. min/max 覆盖且只覆盖本组实际元素，尾组不补入 0。
2. 每个 scale/offset 在编码时和解码时被同一个元素集合使用。
3. 两个 nibble 写入同一 byte 前，该 byte 必须清零；不能覆盖邻居。
4. 输出仅依赖 codes 和元数据，不保留原始 x 作为隐藏答案。

> [!question] 随堂自测 D1-3：能否只凭编码独立恢复
>
> 1. 对 $T=5,D=7,g=4$ 的 $(r,c)=(1,0)$，求 byte index、shift 和两种轴的 group index。为什么换分组轴后前两项不变？
> 2. 最后一个元素 $(4,6)$ 在哪个 nibble？如果反量化函数偷偷读取原始浮点数组，怎样设计一个测试揭露它？

> [!tip]- 核对要点
>
> 1. 线性索引 7，所以 字节索引为 $3$、位移为 $4$；token 组索引为 $2$，channel 组索引为 $0$。codes 始终按行连续存储，只有元数据映射随轴变化。
> 2. 线性索引 34，在 byte 17 的低 nibble；高 nibble 应为 0。编码后销毁或改写原数组，仅从拷贝/序列化后的 codes、scales、offsets 解码，结果应不变。

## §3 从“数值很接近”到“有用信息还在”

### 3.1 K 的误差与 V 的误差进入不同路径

对单个 query：

$$
s=qK^{\top}/\sqrt{D},\quad p=\operatorname{softmax}(s),\quad o=pV.
$$

压缩后 $K'=K+\Delta K,\;V'=V+\Delta V$，则：

$$
\Delta s=q\Delta K^{\top}/\sqrt{D},
$$

$$
\Delta o=(p'-p)V+p'\Delta V.
$$

第一项来自注意力分配变化，第二项来自 value 本身变化。如果同时压缩 K、V，却只观察总 MSE，就无法判断主要误差来自哪条路径。

softmax 的 Jacobian 是 $\operatorname{diag}(p)-pp^{\top}$；局部响应取决于当前概率分布，并非固定倍数。更大的 $q$ 范数、竞争位置接近的分数、不同 value 的方向都会影响输出。不要把“softmax 一定放大量化误差”当普遍定理。

### 3.2 一个选择失败的完整反例

取 $q=[10],\;K=\begin{bmatrix}10\\0\end{bmatrix},\;V=\begin{bmatrix}9\\1\end{bmatrix}$。完整 attention 的 logits 是 $[100,0]$，输出接近 9。若只保留最近位置 1，输出恰为 1；缓存位置数减半，答案却完全改变。

反过来，若 $V=\begin{bmatrix}1\\1\end{bmatrix}$，即使 attention 概率变化很大，输出仍是 1。因此：

- KV 重构误差不等价于 attention 输出误差；
- attention 输出误差不等价于任务准确率；
- 某个层/head 的误差不能直接相加成模型质量下降。

lab 将指标分为三个层次：用元素误差界验证编码，用 relative L2 衡量一次 attention 的输出，最后通过真实模型任务/NLL 验证应用质量。最后一层属于研究扩展，前两层通过后仍需单独验证。

### 3.3 保留集合的硬约束

输入 scores 来自过去可观测信息，保留总预算 $b$。先确定：

$$
S_{\mathrm{must}}=\{0,\ldots,\mathrm{sink}-1\}\cup\{T-\mathrm{recent},\ldots,T-1\}.
$$

这两个集合可能重叠，所以不能把数量简单相加。若 $\lvert S_{\mathrm{must}}\rvert>b$，问题不可行，应报错。余下名额从其他位置按分数降序选；同分时选较小 index，最终结果按位置排序。

例：$\mathrm{scores}=[0,4,4,1,0]$，$\mathrm{budget}=4$，$\mathrm{sink}=1$，$\mathrm{recent}=1$。mandatory 是 $\{0,4\}$；中间选 $\{1,2\}$，最终 $[0,1,2,4]$。

复杂度可以从 $\mathcal{O}(T\log T)$ 排序做起。优化成部分选择或堆之后，仍须维护稳定 tie-break；不能为了速度让相同输入每次留下不同 token。

### 3.4 何时发生未来信息泄漏

如果先计算当前 query 对所有 K 的精确分数，再选择最重要的位置，选择器本身就已完成了原本希望省去的工作。它可以作为明确标注的 oracle，但与历史策略比较时，必须区分成本口径。

新版 bench 的 C 路径用当前 query 查询冻结的 PQ 码本，检索费用计入 query 时间；这是合法的 query-time 检索。若把 B1 选择器用于历史驱逐，分数必须来自过去数据。可以另构造历史/当前 query 不相关的负例，但不能把两种生命周期的成本混在一起。

> [!question] 随堂自测 D1-4：误差与保留集合
>
> 1. $T=5,\mathrm{sink}=3,\mathrm{recent}=3$ 时，$\mathrm{budget}=4$ 是否可行？先写出 mandatory 的并集，再回答；如果把两段长度相加会错在哪里？
> 2. §3.2 中保留最近位置后的输出为 1。将两个 V 都改成 1，是否能据此证明选择器保留了正确位置？应增加什么指标来区分这两件事？

> [!tip]- 核对要点
>
> 1. 并集为 $\{0,1,2,3,4\}$，占 5 个位置，因此不可行；两段在位置 2 重叠，直接相加会误算成 6。
> 2. 不能。相同 V 让不同 attention 分布产生同一输出。可以同时报告受相同 mandatory 约束的精确 top-k recall 和输出误差；二者都不能替代任务准确率。

## §4 从离线矩阵到可追加缓存

### 4.1 为什么残余窗口需要状态机

per-channel 沿时间分组。假设 $g=4$，新 token 依次到达：

```text
t=0..2: residual buffer holds 3 high-precision tokens
t=3:    4 tokens -> per-column min/max -> encode -> packed prefix
t=4:    packed prefix holds 4 tokens; residual buffer holds 1 token
```

若每来一个 token 都重算已有整组的 scale，旧码也必须重新编码；否则同一码在不同 scale 下表示不同数值。残余窗口避免反复重编码，但增加高精度存储。

$$
M=M_{\text{packed prefix}}+M_{\mathrm{residual}}+M_{\mathrm{metadata}}+M_{\mathrm{indices}}.
$$

同时存在量化组、分页块和淘汰块时，它们的粒度未必相同。释放半个量化组是否可行？scale 是否被多个页共享？接入真实 cache manager 前，需要明确这些所有权问题。

### 4.2 计时必须覆盖发生的工作

离线量化可以放在 prefill 结束时，也可分散在 decode 追加时。必须分别报告一次性成本、每步平均/尾部成本，以及跨组边界时的尖峰。

如果每步先把全部 int4 恢复为 FP32，再用普通 attention，packed 容量可能下降，实际工作区和带宽却增加。真正的低比特 attention 应让解码尽量留在寄存器/片上，D2 会给出相同思想的简化算子。

> [!example] 看图：追加、质量与时延要分开观察
> 阅读 [Hugging Face 的 KV 量化图解](https://huggingface.co/blog/kv-cache-quantization) 开头的缓存追加图，再对照 “Comparing performance of fp16 and quantized cache” 中的 perplexity、显存和延迟图。各选一张图，记下横轴、纵轴与固定参数，回答“少占字节”是否足以推出“单步更快”。这些曲线是作者在其模型、后端与参数下的测量；§1 的预算手算来自公式，两者都不是你的实测记录。

> [!question] 随堂自测 D1-5：追加状态与计时边界
>
> 1. 对实际 `ResidualCache(dim=2,block=4,residual=2,group=4)` 连续追加 7 个 token，第 5、6、7 次返回时各有几个 packed token、几个 recent token？第 7 次的 `bytes()` 是多少？
> 2. `attend` 每次只解码一块，是否等于没有反量化成本？应怎样记录跨 flush 边界的追加开销？

> [!tip]- 核对要点
>
> 1. 三个状态为 $(0,5)$、$(4,2)$、$(4,3)$。第一个 K chunk 占 $4+2\times 8=20\,\mathrm{B}$，V chunk 占 $4+4\times 8=36\,\mathrm{B}$；recent 的 K/V 占 $3\times 2\times 2\times 4=48\,\mathrm{B}$，合计 104B，不含容器容量和临时区。§4.1 的示意对应 `residual=0,block=group=4`，实际 flush 门槛是 $\mathrm{residual}+\mathrm{block}$。
> 2. 仍会解码，也有临时 FP32 数组。逐次记录追加时间并标出 flush 事件，分别展示普通步骤和跨边界步骤，不能只给所有步骤的平均值。

## §5 前沿机制：不要把同一主题下的工作当成同一种方法

| 工作                                                       | 核心表示/执行问题                        | 阅读时追问                         | 本 lab 覆盖                        |
| ---------------------------------------------------------- | ---------------------------------------- | ---------------------------------- | ---------------------------------- |
| [KIVI，ICML 2024](https://arxiv.org/abs/2402.02750)        | K/V 非对称低比特表示                     | 何时量化？残余部分何时合并？       | 分组轴、scalar int4、流式 residual |
| [PQCache，SIGMOD 2025](https://arxiv.org/abs/2407.12820)   | 用 product quantization 辅助相关 KV 检索 | 索引建立、候选召回和搬运如何计时？ | C1–C4 实现 PQ 训练、ADC、取回      |
| [FlashInfer，MLSys 2025](https://arxiv.org/abs/2501.01005) | 不同 KV 布局下的高效 attention           | 策略产生的索引能否高效消费？       | 识别接入边界                       |

Scalar quantization 把每个标量映到有限整数；product quantization 把向量分成子向量，用码本表示，并可用查表近似距离。两者虽然都称为量化，却可能分别用于压缩数值和搜索候选，不能混用实验指标。

用 [固定版本源码地图](../readings/code-map.md) 定位 KIVI 的 `models/utils_quant.py`、`quant/csrc/gemv_cuda.cu` 和 PQCache 的 cache manager。45min 阅读只要求画出三条调用关系：写入、压缩/建索引、查询/解码；不要把整个仓库从上到下读一遍。

### 5.1 从论文机制走到三阶段代码

A1–A4 建立可解码的 INT4 格式、物理 gather 和误差度量；B1–B5 建立 append/flush/materialize/online-softmax 状态机；C1–C4 实现子空间 Lloyd、冻结编码、内积查表、选择并取回原 KV。完整接口和时序测试见 [lab1](../../labs/lab1/README.md)。

对于 $M$ 个子空间、$C$ 个中心，查询表构造约 $\mathcal{O}(CD)$，扫描 codes 约 $\mathcal{O}(TM)$，再取回 budget 个原向量。码本准备是一次性成本，追加 token 的编码和重建策略则是扩展问题。代码用 uint16 中心 ID，索引不是原 KV 的替代存储；绘图要同时列索引、原 KV、fetch 和临时区。

> [!question] 随堂自测 D1-6：论文机制落到哪一段代码
>
> 1. 设 $D=8,M=2,C=16,T=100$，按 lab 的 FP32 码本和 uint16 ID，PQ 索引共占多少字节？若仍保留原始 FP32 K/V，二者合计至少多少字节？
> 2. 为什么 PQ 检索不能直接改用 query 到中心的欧氏距离排序？给出一个一维反例。

> [!tip]- 核对要点
>
> 1. 码本 $2\times 16\times 4\times 4=512\,\mathrm{B}$，codes $100\times 2\times 2=400\,\mathrm{B}$，共 912B；原 K/V 为 $2\times 100\times 8\times 4=6400\,\mathrm{B}$，合计至少 7312B，另计工作区。
> 2. attention 使用内积，未保证向量等范数。取 $q=1,k1=1,k2=2$：最近的是 k1，内积更大的是 k2；两种排序优化的目标不同。

## §6 一个可以交给导师评议的实验问题

候选问题：**固定真实字节预算下，用历史校准得到的层/head 敏感度分配 group 或保留数量，能否在分布变化后优于统一分配？** 这只是一个仍需查新的候选问题；adaptive KV 和 mixed precision 已有大量工作。

第一周先建立一个能用实验检验并推翻假设的最小模型：固定 K/V、固定当前 query，分别改变轴、group、保留比例、历史相关性。第二步再使用真实 K/V；训练校准 prompts 与评测 prompts 分开，不能看完评测答案再调预算。

判断继续与否的证据应具体：

- 同字节下是否存在稳定的误差差异，而非更大的元数据换来更小误差？
- 弱选择器的失败来自“分数过时”，还是任何历史方法都缺少信号？
- 量化/选择成本是否吞掉了 attention 局部节省？
- 改善是否只在随机矩阵上成立，真实 K/V 上趋势是否相反？

如果只有 oracle 获得收益，先分析实际策略缺少哪些信息，再考虑是否需要更复杂的模型。如果所有可行预算都明显损害检索任务，收窄到更温和的压缩范围，或停止这一假设。

> [!question] 随堂自测 D1-7：什么结果足以继续研究
>
> 1. 方案 A 使用更小 group，误差下降，但总字节增加 40%。这是否支持“同字节预算下更优”？需要补什么对照？
> 2. 用所有评测 query 调好每层预算，再将同一批 query 的结果称为 held-out，哪里有问题？请写出一次不泄漏的冻结与评测顺序。

> [!tip]- 核对要点
>
> 1. 不支持。应按 payload、元数据和索引的统一口径匹配预算，或给出质量—真实字节曲线，再比较相同预算处的误差与时间。
> 2. 评测信息参与了选参。先在校准数据上确定分配与码本，冻结方案，再用独立 query 评测；若据此修改方案，需要另留最终测试集。

## 资料汇总

下表的“指定观察”已包含在本讲时间中；通读上游仓库、运行模型和论文全套实验属于选读或 lab 挑战。

| 资料                                                                                                                                  | 指定观察与用途                                               |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| [vLLM PagedAttention 官方图解](https://vllm.ai/blog/2023-06-20-vllm)                                                                  | 逻辑块/物理块图与单请求生成动画；区分分页空位与数值精度。    |
| [HF KV 量化图解](https://huggingface.co/blog/kv-cache-quantization)                                                                   | 缓存追加示意及质量、显存、延迟曲线；记录作者测量的适用条件。 |
| [HF Cache 文档](https://huggingface.co/docs/transformers/en/kv_cache)、[MIT 量化课程](https://hanlab.mit.edu/courses/2024-fall-65940) | 选读：查缓存配置与量化基础，接口以安装版本为准。             |
| [KIVI 论文](https://arxiv.org/abs/2402.02750) / [代码](https://github.com/jy-yuan/KIVI)                                               | K/V 分组轴与残余窗口；对照 A/B 的简化边界。                  |
| [PQCache 论文](https://arxiv.org/abs/2407.12820) / [代码](https://github.com/HugoZHL/PQCache)                                         | 索引构建、候选检索和原 KV 取回；对照 C1–C4。                 |
| [FlashInfer 论文](https://arxiv.org/abs/2501.01005) / [代码](https://github.com/flashinfer-ai/flashinfer)                             | 选读：量化或检索产生的布局如何交给 attention kernel。        |
| [Lab 1](../../labs/lab1/README.md)、[源码地图](../readings/code-map.md)、[资源索引](../readings/resources.md)                         | 定位具体接口、固定版本源码和后续阅读。                       |

## 自测清单

- [ ] 不查公式，推导 GQA 缓存预算和两种轴的精确元数据数量。
- [ ] 留下 $5\times7$ 的地址表，画出最后一个 byte，说明两种 group 索引为什么不同。
- [ ] 写出至少一个表示/选择误差与 attention 输出误差不一致的反例。
- [ ] 写出选择集合的不变量，并指出 sink/recent 重叠与不可行预算。
- [ ] 画出实际 $\mathrm{residual}+\mathrm{block}$ flush 状态图，独立核算 104B 例题并说明它不等于 RSS。
- [ ] 区分已实现的残余窗口与块式查询、未实现的 GPU fused decode，以及仍驻留的原 KV。
- [ ] 留下 PQ 索引和原 KV 的分项字节账，以及内积/欧氏排序反例。
- [ ] 为两份外部图解各写一句观察，区分机制演示、作者测量和自己的结果。
- [ ] 写出一个可被实验推翻的假设、同预算对照、校准/评测划分和停止条件。
