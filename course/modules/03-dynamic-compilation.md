---
title: D3 动态形状与编译摊销：guard、缓存、分桶与 CUDA graph
lab: lab3
hours: 5
tags: [lecture, compiler, torch-compile, cuda-graph, dynamic-shape, research]
updated: 2026-10-18
---

# D3 动态形状与编译摊销：guard、缓存、分桶与 CUDA graph

[← 课程索引](../index.md) · [配套 lab3](../../labs/lab3/README.md) · [← D2 低比特 kernel](02-low-bit-kernels.md) · [D4 推测解码 →](04-speculative-slo.md)

> [!abstract] 这一讲要解决的问题
> foundations 讲过一个算子怎样编译得更快。本讲把编译放回服务的整个生命周期：输入形状一直在变，编译和 CUDA graph 捕获都要先付钱，缓存、guard 和 fallback 决定这笔钱付几次。学完后你能推导回本条件，读懂 Dynamo 为什么重编译、何时退回 eager，把分桶写成带预算的 DP，并用它在 4060 上为一个 Qwen3-0.6B 形状的 decode block 挑选 CUDA graph 捕获尺寸。你还会手写一个 warp shuffle 加向量化访存的融合 kernel，与 Inductor 生成的 kernel 用同一套规则对比。

| 阅读模块                          | 时间  | 必须留下的学习产物                        | 对应实验    |
| --------------------------------- | ----- | ----------------------------------------- | ----------- |
| §1 编译成本摊销                   | 30min | 回本推导、两条累计成本曲线                | A2、B5      |
| §2 Dynamo：捕获、guard、符号形状  | 55min | 自己机器上的 recompile 计数表             | C1、C2      |
| §3 缓存怎样改变 $C$               | 30min | 缓存层级表和冷热口径                      | C2、C4      |
| §4 Inductor 代码生成与 CUDA graph | 35min | 一份生成的 Triton kernel 注释、graph 约束 | C、R1、R2   |
| §5 分桶是带预算的优化问题         | 55min | 四种划分手算、DP 证明、capture size 对照  | A3–A5、B、R |
| §6 融合与访存                     | 45min | 28B/24B/16B 账、warp 归约不变量           | C5–C7       |
| §7 机制与代码对应                 | 10min | 对照表                                    | 全部        |
| 前沿与 baseline 对照              | 30min | 三篇工作的机制差异卡                      | C/R         |
| 研究问题                          | 10min | 一个可证伪假设和停止条件                  | R           |

**共 5h，另做 lab 约 34h。** 5h 包括手算、在 WSL 里跑 §2 的十几行演示，以及读 vLLM/SGLang 各一个函数。论文全文、tlparse 的完整报告和 Triton 教程属于选读。

随堂自测计入上表时间。先在笔记里写答案，再展开核对要点。

## §1 编译成本摊销

### 1.1 第一次调用在做什么

`torch.compile` 包装的函数第一次被调用时，要依次经过下面几步。以后的调用如果通过了 guard，就直接执行缓存好的代码：

```text
call f(x)
  |
  v
Dynamo: run guards of cached entries ---- hit ----> run compiled code (fast path)
  | miss
  v
Dynamo: trace Python bytecode -> FX graph (+ guards)      [graph break -> several graphs]
  |
  v
AOTAutograd: functionalize, (backward graph if training)
  |
  v
Inductor: lower -> fuse -> codegen (Triton on GPU, C++/OpenMP on CPU)
  |
  v
compile kernels / autotune  <---- disk caches (FX graph, Triton, autotune) may hit
  |
  v
store entry + guards, run it
```

每一层都可能命中缓存，所以“第一次调用的时间”不等于“编译时间”。lab 里把 $\text{first} - \text{median(steady)}$ 记作 first-call premium（A2），它是当前进程和当前缓存状态下的附加成本，不是冷编译时间。

### 1.2 回本条件

设一次编译的总成本为 $C$（ms），eager 每次调用耗时 $t_e$，编译后每次耗时 $t_c$。调用 $n$ 次的总耗时为

$$
T_e(n)=n\,t_e,\qquad T_c(n)=C+n\,t_c .
$$

只有 $t_e>t_c$ 时编译才可能划算，严格获益的条件是

$$
n>n^\ast=\frac{C}{t_e-t_c}.
$$

$n=n^\ast$ 时两条线相交，只是持平。$t_c\ge t_e$ 时任何 $n$ 都不回本。形状变化引起 $r$ 次重编译时，分子要换成 $\sum_{k=1}^{r}C_k$。

数值例子：$C=800\,\text{ms}$，$t_e=0.12\,\text{ms}$，$t_c=0.08\,\text{ms}$，则 $n^\ast=800/0.04=20000$。$n=5000$ 时 eager 是 600 ms，编译路径是 1200 ms；$n=30000$ 时分别是 3600 ms 和 3200 ms。这是手算，不是某个版本的实测。

测量本身有噪声，所以 B5 的准入规则用保守估计：把 $t_c$ 换成 $t_c+m\cdot\text{MAD}$（$m$ 为裕量倍数），并且要求严格大于：

$$
n\,\bigl(t_e-t_c-m\cdot\text{MAD}\bigr)>C .
$$

$n$ 只能来自历史调用量乘以声明好的 horizon 倍数，不能偷看 held-out 的真实调用数。

### 1.3 优化对象是一个 episode

episode 指一段有限的会话、一批作业或一个预先声明的服务窗口。目标是整个 episode 的完成时间，其中包括校准时编译了却最终没用上的候选（lab 的 `calibration_ms`）。如果 episode 很短、编译路径比 eager 慢，这个结果本身就要写进报告，不能删掉。

> [!question] 随堂自测 D3-1：什么时候真的回本
>
> 1. 沿用 $C=800\,\text{ms}$、$t_e=0.12\,\text{ms}$、$t_c=0.08\,\text{ms}$。20000 次调用是否已经获益？若两个形状各编译一次、总调用仍为 30000 次，结果如何？
> 2. B5 中 $t_e=3$、$t_c=1$、$\text{MAD}=0.5$、$m=2$、$C=10$（单位 ms）。预计 10 次和 11 次调用分别准入吗？$m=4$ 呢？

> [!tip]- 核对要点
>
> 1. 20000 次恰好持平，都是 2400 ms。两次编译时编译路径为 $1600+30000\times0.08=4000\,\text{ms}$，比 eager 的 3600 ms 慢。
> 2. 保守收益为每次 $3-(1+2\times0.5)=1\,\text{ms}$。10 次累计 10 ms，等于 $C$，拒绝；11 次准入。$m=4$ 时保守收益为 $3-3=0$，任何 $n$ 都拒绝。

## §2 Dynamo：捕获、guard 与符号形状

### 2.1 捕获和 guard

Dynamo 在 CPython 执行 frame 之前拦截字节码，符号执行 Python 代码，把张量运算记录成 FX graph，遇到不能追踪的语句就 graph break。每个编译结果都附带一组 guard，例如张量的 dtype、device、维数、每个维度的大小或符号约束、stride，以及用到的全局变量。下次调用先检查已缓存条目的 guard，全部通过才复用。[PyTorch 2 论文](https://docs.pytorch.org/assets/pytorch2-2.pdf)（ASPLOS 2024）§3.3 讲 guard，§3.4 讲符号求值。

guard 失败不会出错，只会触发重编译。所以动态形状的第一个问题是：guard 写得有多“宽”，一个编译结果能覆盖多少形状。

### 2.2 默认静态，第二个新尺寸时自动转动态

你的 torch 2.14.1 上 `torch._dynamo.config` 的相关默认值如下（在 WSL 里逐项打印核实过，请你自己再打印一遍）：

| 配置                          | 本机值 | 含义                                                                |
| ----------------------------- | -----: | ------------------------------------------------------------------- |
| `assume_static_by_default`    |   True | 第一次编译把所有尺寸当常数特化                                      |
| `automatic_dynamic_shapes`    |   True | 某维出现第二个不同尺寸时，把这一维改成符号 `SymInt` 重编译          |
| `recompile_limit`             |      8 | 同一个 code object 最多缓存 8 个编译条目（旧名 `cache_size_limit`） |
| `accumulated_recompile_limit` |    256 | 整个进程累计的上限                                                  |
| `specialize_int`              |  False | Python int 参数默认不按值特化                                       |

用一个只计数、不真正编译的 backend 观察 $f(x)=\text{sum}(2x)$，依次输入长度 $[1,2,3,4,8,1,16]$，每次新增的 graph 数实测为 $[1,1,0,0,0,0,0]$：

- 长度 1：第一次调用，静态特化，得到 graph A（只接受长度 1）。
- 长度 2：guard 失败。automatic dynamic 发现这一维已经见过两个值，于是把它变成符号 $s_0$ 重编译，得到 graph B。
- 长度 3、4、8、16：都满足 graph B 的 guard，不再编译。
- 第二次长度 1：仍然命中 graph A。

graph B 为什么不覆盖长度 1？这就是 **0/1 特化**：尺寸 0 和 1 在广播、contiguity、空张量上语义特殊，Dynamo 给符号尺寸默认加上 $s_0\ge 2$ 的约束，大小为 0 或 1 的输入总是走单独特化的图。用 `torch._dynamo.mark_dynamic(x, 0)` 提前声明动态维也一样：先输入长度 5，再输入 7 和 1，实测共 2 个 graph，长度 1 仍然单独编译。

### 2.3 重编译上限和 fallback

把同一个函数改成 `dynamic=False`，依次输入长度 1 到 12，实测每次新增 graph 数为 $[1,1,1,1,1,1,1,1,0,0,0,0]$。第 9 个尺寸起不再编译，日志打印

```text
torch._dynamo hit config.recompile_limit (8)
   last reason: 0/7: tensor 'x' size mismatch at index 0. expected 8, actual 9
```

此后这个函数静默退回 eager。服务里出现这种情况时，吞吐会突然下降，却没有任何报错。`TORCH_LOGS=recompiles,graph_breaks,dynamic` 能打印每次重编译的原因，[tlparse](https://github.com/pytorch/tlparse) 可以把 `TORCH_TRACE` 目录渲染成按 frame 分组的 HTML 报告。

### 2.4 用 stance 把编译阶段和服务阶段分开

`torch.compiler.set_stance` 改变的是“遇到新形状时怎么办”。本机签名里能看到的取值包括 `default`、`force_eager`、`eager_on_recompile`、`fail_on_recompile`、`eager_then_compile`、`aot_eager_then_compile`。实测：先用长度 3 编译，再在 `fail_on_recompile` 下输入长度 5，抛出 `RuntimeError`；在 `force_eager` 下输入长度 7，不新增 graph。服务里常见的用法是：预热阶段按计划把形状都编译好，上线后切到 `fail_on_recompile`（测试环境）或 `eager_on_recompile`（生产环境），让意外的形状显式暴露出来或安全降级。官方说明见 [set_stance 文档](https://docs.pytorch.org/docs/stable/generated/torch.compiler.set_stance.html)。

> [!question] 随堂自测 D3-2：数一数编译了几次
>
> 1. 默认配置下依次输入长度 $[4,4,6,1,9,1]$，每次新增几个 graph？
> 2. `dynamic=False`，依次输入 20 个不同长度，再输入第 3 个长度一次。一共编译几次？第 21 次调用走哪条路径？
> 3. 为什么 vLLM 要求所有编译都在开始服务之前完成？结合 §1 的回本条件回答。

> [!tip]- 核对要点
>
> 1. $[1,0,1,1,0,0]$：长度 4 静态编译；第二个 4 命中；长度 6 触发 automatic dynamic；长度 1 因 0/1 特化单独编译；9 命中动态图；第二个 1 命中长度 1 的图。
> 2. 8 次，第 9 个长度起退回 eager。第 21 次输入的是第 3 个长度，它在前 8 个之内，有缓存条目，guard 命中，走编译路径。
> 3. 服务中途编译会让当前请求承担完整的 $C$，延迟尖刺可能是稳态的上千倍；同时也说不清 $C$ 何时被摊销完。预先编译把 $C$ 移到启动阶段，代价是启动时间和显存。

## §3 缓存怎样改变 $C$

### 3.1 缓存的层次

| 层次                                                | 本机默认                                                       | 命中时省掉什么                         | 失效条件                          |
| --------------------------------------------------- | -------------------------------------------------------------- | -------------------------------------- | --------------------------------- |
| Dynamo 进程内条目                                   | 随进程                                                         | 全部，只剩 guard 检查                  | guard 失败；进程退出              |
| FX graph cache（本地磁盘）                          | `fx_graph_cache=True`                                          | Inductor lowering 和 codegen           | graph、配置或版本变化             |
| Triton kernel / autotune cache                      | `autotune_local_cache=True`                                    | kernel 编译与 autotune 基准            | 同上，另加 GPU 型号               |
| 远程 FX graph cache                                 | `fx_graph_remote_cache=None`                                   | 跨机器复用                             | 需要配置 Redis 等后端             |
| mega-cache                                          | `torch.compiler.save_cache_artifacts` / `load_cache_artifacts` | 把以上各层打包成字节串，换一台机器预填 | 要求相同的 PyTorch、Triton 和 GPU |
| AOTInductor + `torch.export`                        | 需要显式导出                                                   | 服务时完全不 JIT                       | 只覆盖导出时声明的动态维范围      |
| 区域编译（`nested_compile_region`、逐层 `compile`） | 需要改代码                                                     | 重复的 decoder 层只编译一次            | 层与层结构不同时无效              |

mega-cache 的接口和限制见 [compile 缓存教程](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)；AOTInductor 见 [文档](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html)，`torch.export` 见 [文档](https://docs.pytorch.org/docs/stable/export.html)，区域编译见 [教程](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)。

### 3.2 对回本公式的影响

缓存把 $C$ 拆成两部分：$C=C_{\text{trace}}+C_{\text{backend}}$。Dynamo 追踪 $C_{\text{trace}}$ 每个进程都要付；磁盘缓存命中时 $C_{\text{backend}}$ 接近 0。于是 $n^\ast$ 的冷、热两个口径可能差一个数量级。报告里必须写清楚用的是哪一个：

- **冷**：给 `TORCHINDUCTOR_CACHE_DIR` 指定一个新的空目录，不要删除全局缓存。
- **热**：同一目录跑第二次。
- **进程内**：同一进程里第二次调用同一形状，只剩 guard 检查。

lab 的 C2/C4 默认在“磁盘缓存热、Dynamo 已 reset”的状态下测量，所以记录里标注 `not cold compilation`。

> [!question] 随堂自测 D3-3：选哪一层缓存
>
> 1. 一个服务每天重启一次，GPU 型号固定，torch 版本两周升级一次。mega-cache 和 AOTInductor 各解决了 $C$ 的哪一部分？各自的失效风险是什么？
> 2. 28 层结构相同的 decoder 用区域编译，理想情况下 $C_{\text{backend}}$ 变为原来的多少？为什么 $C_{\text{trace}}$ 不一定同比下降？

> [!tip]- 核对要点
>
> 1. mega-cache 预填各层缓存，$C_{\text{backend}}$ 几乎归零，但每次进程仍要追踪；torch、Triton 或 GPU 一变就失效。AOTInductor 连追踪也省掉，但只支持导出时声明的动态范围，模型或版本变了要重新导出。
> 2. 约 $1/28$，只编译一层再复用。每层仍要被 Dynamo 追踪并生成 guard，Python 侧开销还在；小模型上 Inductor 的固定开销（启动、C++ wrapper 编译）也会冲淡收益，要实测。

## §4 Inductor 代码生成与 CUDA graph

### 4.1 Inductor 生成了什么

Inductor 把 FX graph lowering 成循环级 IR，调度器决定哪些节点融合进同一个 kernel，GPU 上生成 Triton，CPU 上生成 C++/OpenMP。用 `TORCH_LOGS=output_code` 或 `torch._inductor.utils.run_and_get_code` 能拿到源码。lab 的 `inductor_bench.py` 对下面这个函数做了这件事：

$$
z=x+r,\qquad y_{ij}=\frac{z_{ij}}{\sqrt{\frac{1}{D}\sum_{k}z_{ik}^2+\epsilon}}\;\gamma_j .
$$

在 4060 上，它对每个 dtype 和宽度都只生成一个名为 `triton_red_fused_add_mean_mul_pow_rsqrt_0` 的 reduction kernel：每个 program 处理 `XBLOCK` 行，沿行以 `R0_BLOCK` 为步长循环两遍，第一遍求平方和，第二遍写 $y$ 和 $z$。与 §6 的手写 kernel 的数据流相同。

### 4.2 三种 mode 的成本

| mode              | 额外做什么                                        | 对 $C$ 的影响                  | 对 $t_c$ 的影响     |
| ----------------- | ------------------------------------------------- | ------------------------------ | ------------------- |
| `default`         | 启发式选择 kernel 配置                            | 基准                           | 基准                |
| `max-autotune`    | 对 Triton 配置和 GEMM 模板逐个基准测试            | 大幅增加，每个形状都要测       | 可能下降            |
| `reduce-overhead` | 用 CUDA graph 包住编译后的代码（cudagraph trees） | 每个形状额外一次捕获，占显存池 | launch 开销基本消失 |

`reduce-overhead` 等于对每个尺寸再付一次捕获成本，也要多占一块显存，所以它和动态形状天然冲突：动态 graph 覆盖很多尺寸，CUDA graph 却只能按具体尺寸捕获。

### 4.3 CUDA graph 的约束和实测

CUDA graph 把一串 kernel launch 录下来，回放时一次提交，省掉每个 kernel 几微秒的 CPU 开销。代价是：

- 输入输出地址固定，只能把数据先拷进静态缓冲区再回放，结果也写在静态缓冲区里（R2 要处理拷入、补零和拷出）；
- 形状固定，每个 batch size 需要一张图；
- 捕获区间内不能有 CPU 同步，也不能有依赖数据的控制流；
- 每张图有自己的内存池，或者共享一个池。

lab3 R 阶段在 4060 上测了一个 4 层、Qwen3-0.6B 宽度的 FP16 decode step（hidden 1024、16 个 query head、8 个 KV head、head_dim 128、MLP 3072，KV 长度 256）。数据为 2026-10-18 实测，原始记录在 `labs/lab3/real/data/profile_rtx4060.json`：

| batch | eager (ms) | graph replay (ms) | 节省 | 捕获 (ms) | 私有池 (MiB) |
| ----: | ---------: | ----------------: | ---: | --------: | -----------: |
|     1 |      0.964 |             0.618 |  36% |       3.1 |            2 |
|    16 |      1.111 |             1.025 |   8% |       1.3 |            2 |
|    32 |      1.535 |             1.414 |   8% |       1.3 |            2 |
|    64 |      2.253 |             2.149 |   5% |       2.1 |            4 |

batch 小时 step 受 launch 开销限制，graph 省得多；batch 大时 GEMM 和 KV 读取占主导，节省降到 5%。把 8 个尺寸捕获进同一个共享池，池大小等于最大那张图的 4 MiB，所以共享池的显存近似由最大捕获尺寸决定（§5.5 的显存模型）。另一个实测陷阱是：每张图换一个新的捕获 stream，cuBLAS 会给每个 stream 再分配一块约 8 MiB 的 workspace，8 张图就多出 64 MiB。`GraphRunner` 因此只用一个捕获 stream。

### 4.4 vLLM V1 的 piecewise CUDA graph

vLLM V1 用 torch.compile 编译模型，再在 attention 处把 FX graph 切开，其余部分按捕获尺寸录成 CUDA graph（piecewise），attention 则留在 graph 外面，或者另外录成完整 graph。截至 2026-10-12 的[设计文档](https://docs.vllm.ai/en/latest/design/cuda_graphs/)里，`cudagraph_mode` 的取值有 `NONE`、`PIECEWISE`、`FULL`、`FULL_DECODE_ONLY`、`FULL_AND_PIECEWISE`，默认是 `FULL_AND_PIECEWISE`。编译流程见 [torch.compile 集成文档](https://docs.vllm.ai/en/latest/design/torch_compile/) 和 [2025-08-20 的博客](https://vllm.ai/blog/2025-08-20-torch-compile)。以上都没有在学习者的机器上验证：vLLM 没有安装，当前 wheel 需要比 566.24 更新的驱动。

> [!question] 随堂自测 D3-4：graph 回放的正确性
>
> 1. batch 13 回放 batch 16 的图。拷入 13 行后，第 14 到 16 行里还留着上一步 batch 16 的数据。对本讲逐行独立的 decode block 来说结果对不对？为什么 R2 仍然要求补零？
> 2. `run()` 直接返回 `outputs[16][:13]` 这个 view，会出什么错？什么测试能抓住它？

> [!tip]- 核对要点
>
> 1. 每一行都独立，切回前 13 行的结果是对的。补零是为了让补齐行在所有算子下都确定、有限：MoE 路由会把补齐 token 计入专家负载，残留的 inf 或 NaN 也可能经过归一化或通信扩散。
> 2. 下一次回放会覆盖同一块缓冲区，之前返回给调用方的结果被悄悄改写。测试要保留早先几步的输出，等后面的步骤跑完再与 eager 比较。

## §5 分桶是带预算的优化问题

### 5.1 padding 首先是语义问题

本讲的 RMSNorm 输入为 $x,r\in\mathbb{R}^{\text{rows}\times D}$，每一行独立归一化。把 rows 从 17 补到 32 只是多出 15 个独立行，取回前 17 行即可。把 width 从 3 补到 4 却仍然除以 4，会改变每个真实元素：设 $z=[1,2,2]$、$\gamma=\mathbf 1$、$\epsilon\approx0$，真实均方为 3，输出约 $[0.577,1.155,1.155]$；补零后除以 4，均方为 2.25，输出变成 $[0.667,1.333,1.333]$，比值为 $\sqrt{4/3}\approx1.155$。decode 的 batch 维也是独立行，所以 CUDA graph 可以按 batch 补齐；attention 的 token 维不独立，补齐需要 mask。

### 5.2 模型

训练 trace 里不同的长度排成 $n_1<n_2<\dots<n_m$，频次为 $f_1,\dots,f_m$。路由规则是“走能装下它的最小 bucket”（B2），所以每个 bucket 接收的长度一定是连续的一段。设一个 bucket 覆盖 $n_i,\dots,n_j$，上界取 $n_j$，它的成本为

$$
c(i,j)=P(n_j)+t(n_j)\sum_{l=i}^{j}f_l ,
$$

其中 $P(n)$ 是该上界的 first-call premium（编译或捕获，只付一次），$t(n)$ 是实测的稳态耗时。$t$ 直接查实测表，不假设线性。区间频次用前缀和 $F_j=\sum_{l\le j}f_l$，就能 $O(1)$ 算出 $F_j-F_{i-1}$。

为什么上界只取观测到的长度？如果 $t$ 随上界单调不减，把上界从 $n_j$ 放大到 $n_j$ 和 $n_{j+1}$ 之间，接收的请求不变，成本只增不减。实测的 $t$ 不一定单调（不同 batch 会选到不同的 GEMM kernel，见 4060 profile 里 batch 52 比 53 还慢），这时“只取观测值”就不再保证最优，要把候选扩展到所有测过的尺寸。

### 5.3 DP 与最优性

令 $D_{k,j}$ 为用恰好 $k$ 个 bucket 覆盖前 $j$ 个长度的最小成本。初值 $D_{0,0}=0$，其余为正无穷：

$$
D_{k,j}=\min_{k-1\le i<j}\bigl\{D_{k-1,i}+c(i+1,j)\bigr\},
\qquad
\text{answer}=\min_{1\le k\le K}D_{k,m}.
$$

最优子结构：设最优解最后一段从 $i+1$ 开始。如果前 $i$ 个长度用 $k-1$ 段还有更便宜的覆盖，换上它总成本会更低，与最优矛盾。枚举最后一段的所有起点就覆盖了所有连续划分，所以递推不漏解。时间 $O(Km^2)$，空间 $O(Km)$。并列时先选 bucket 数少的，再选最后一段起点最早的，保证结果确定。B1 的测试用组合枚举出所有划分来对照，不调用被测的 DP。

### 5.4 四种划分手算

长度 $[1,4,8]$，频次 $[10,2,1]$，线性成本 $t(n)=0.1n\,\text{ms}$，每个 bucket 的 premium 都是 $P$：

| 连续划分      | 上界      | 执行成本                     | 总成本   |
| ------------- | --------- | ---------------------------- | -------- |
| $[1]/[4]/[8]$ | $[1,4,8]$ | $0.1\times(10+8+8)=2.6$      | $3P+2.6$ |
| $[1]/[4,8]$   | $[1,8]$   | $0.1\times(10+8\times3)=3.4$ | $2P+3.4$ |
| $[1,4]/[8]$   | $[4,8]$   | $0.1\times(4\times12+8)=5.6$ | $2P+5.6$ |
| $[1,4,8]$     | $[8]$     | $0.1\times8\times13=10.4$    | $P+10.4$ |

$P=0.1$ 时选三个 bucket，总成本 2.9；$P=2$ 时选 $[1,8]$，总成本 7.4；$P=10$ 时只用 $[8]$，总成本 20.4。编译越贵，越值得接受 padding。

> [!question] 随堂自测 D3-5：规划器保证什么
>
> 1. 上表取 $P=2$，最多 1 个 bucket 时最优成本是多少？最多 2 个呢？
> 2. 上界 $[8,32]$、`max_padding` 为 2，长度 4、9、16、17、33 各走哪条路径？
> 3. 若实测 $t(16)<t(15)$，但训练 trace 只出现过 15，“上界只取观测值”的最优性还成立吗？要怎样扩展模型？

> [!tip]- 核对要点
>
> 1. 1 个：$[8]$，12.4；2 个：$[1,8]$，7.4，正好也是不限数量的最优值。
> 2. 4 → 8（比值正好 2，允许）；9 → fallback（$32/9>2$）；16 → 32（比值 2，允许）；17 → 32；33 → fallback（超过所有上界）。
> 3. 不成立，证明依赖 $t$ 单调。要把未观测但测过的尺寸（例如对齐到 8 或 16 的值）加入候选，并实测它们的成本；DP 本身不变，只是候选集合变大。

### 5.5 主案例：vLLM 和 SGLang 的 CUDA graph 捕获尺寸

decode 时每一步的 batch size 随请求进出不断变化。服务框架的做法正是分桶：启动时在若干 batch size 上捕获 CUDA graph，运行时把 batch 补到下一个已捕获的尺寸，超过最大尺寸就走 eager。下面的默认值读自两个仓库在指定 commit 的源码，**没有在已安装的版本上验证**（两者都没有安装）：

- **vLLM**（commit `371745eb`）：[`vllm/config/compilation.py`](https://github.com/vllm-project/vllm/blob/371745ebd6c12f314402c6c319de2e7ef82c2981/vllm/config/compilation.py) 中 `max_cudagraph_capture_size` 的说明给出未指定时的规则 `[1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_cudagraph_capture_size + 1, 16))`，上限默认 512（数据中心 Blackwell 为 1024）。运行时 [`vllm/v1/cudagraph_dispatcher.py`](https://github.com/vllm-project/vllm/blob/371745ebd6c12f314402c6c319de2e7ef82c2981/vllm/v1/cudagraph_dispatcher.py) 的 `_compute_bs_to_padded_graph_size` 预先建一张表，把每个 batch size 映射到不小于它的最小捕获尺寸。
- **SGLang**（commit `57db5089`）：[`cuda_graph_hook.py`](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/arg_groups/cuda_graph_hook.py) 的 `generate_decode_cuda_graph_batch_sizes` 默认取 `[1, 2, 4, 8, 12] + list(range(16, 257, 8)) + ...`，推测解码时小尺寸更密，并总是补上 `max_bs`。[`base_cuda_graph_runner.py`](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py) 的 `_pad_to_bucket` 用 `bisect_left` 找上界。

两者都是“手工指定的等差网格”。对照 §5.2：捕获尺寸就是 bucket 上界，捕获时间就是 $P$，replay 延迟就是 $t$，没被覆盖的大 batch 付 eager 成本 $e(b)$，图的内存池就是新增的预算。步长为 $g$ 的网格里，落在同一段的 batch 平均多出约 $(g-1)/2$ 行 padding。网格并不知道 trace 集中在哪里，这正是 trace 驱动的选择可能获益的地方。TensorRT-LLM 有一篇[调 CUDA graph batch size 的技术博客](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.md)，在 GB200 上比较了 128 以上按 2 倍、步长 64、步长 8 增长的三种网格，同时统计吞吐、显存和启动时间，最后把步长 64 定为新默认值。

### 5.6 带显存预算的捕获尺寸选择

设候选尺寸集合为 $S$，按升序排列。trace 中 batch $b$ 出现 $f_b$ 次，记 $\lceil b\rceil_S$ 为 $S$ 中不小于 $b$ 的最小元素：

$$
J(S)=\sum_{s\in S}P(s)+\sum_{b}f_b\,\ell_S(b),\qquad
\ell_S(b)=\begin{cases}t\bigl(\lceil b\rceil_S\bigr) & b\le\max S\\ e(b) & b>\max S\end{cases}
$$

约束为 $\lvert S\rvert\le K$，并且 $M(S)=\text{pool}(\max S)+o\,\lvert S\rvert\le M$。这里 $\text{pool}(s)$ 是单独捕获尺寸 $s$ 时私有池的实测大小，$o$ 是每张图的额外开销。共享池的显存由最大的那张图决定，§4.3 的实测支持这个模型。

化归到 §5.3：枚举最大尺寸 $t$。$t$ 一旦确定，$M(S)$ 只剩 $\lvert S\rvert$ 一个变量；不超过 $t$ 的 batch 构成一个前缀直方图，正好交给 B1 的 DP（容量 $k\le K$）；超过 $t$ 的 batch 一律付 $e(b)$。对所有 $(t,k)$ 取最小，再与全 eager 方案 $S=\varnothing$ 比较。这就是 lab 的 R4，复杂度 $O(mK\cdot Km^2)$，对 $m\le64$ 毫无压力。

用 4060 profile 在合成的 continuous batching trace 上规划（训练和 held-out 都是 4000 步；stable 的平均 batch 约 30，shift 的到达率提高 60%，平均 batch 约 47），modeled 结果如下（regret 相对于看过 held-out 的 hindsight 最优，$K=4$，显存不限）：

| 策略                               | 尺寸                 | stable ms/step | regret | shift ms/step | regret |
| ---------------------------------- | -------------------- | -------------: | -----: | ------------: | -----: |
| DP（训练 trace）                   | $[15,30,33,38]$      |          1.456 |   1.1% |         1.959 |   1.6% |
| vLLM 默认网格（11 张图，超出 $K$） | $[1,2,4,8,\dots,64]$ |          1.494 |   3.8% |         1.975 |   2.5% |
| vLLM 网格前 4 个                   | $[1,2,4,8]$          |          1.502 |   4.4% |         1.967 |   2.0% |
| 均匀 4 段                          | $[16,32,48,64]$      |          1.579 |   9.7% |         2.097 |   8.8% |
| 全 eager                           | 无                   |          1.502 |   4.4% |         1.966 |   2.0% |

这些数字是 modeled（实测 profile 代入公式），不是整条 trace 的实测。`bench_r.py --device cuda` 还会让几种策略把 held-out trace 真正回放一遍，每步结束做一次 device-to-host 同步，模拟服务读回采样结果，并把 measured 和 modeled 的 ms/step 并排输出。有三点需要说明。第一，这个 4 层的小模型主要受 GPU 限制，大 batch 时 graph 只省 5%，比 padding 到下一个格点多出来的计算还少，所以等差网格在中段甚至不如 eager。第二，DP 只在 trace 密集的区间放图，大 batch 直接走 eager。层数换成 28 层后，按层数线性估计 eager 的 launch 开销约增加 7 倍，结论可能完全不同，研究问题一节要回答的就是这一点。

第三，modeled 的优势在实测里没有保住：2026-10-18 的另一次 `bench_r.py --device cuda`（重新测了 profile，$K=4$、显存不限，回放每条 held-out trace 的前 1500 步）中，stable trace 上 DP 实测 1.409 ms/step，vLLM 默认网格 1.411；shift trace 上 DP 1.887，vLLM 默认网格 1.864，反而更快（它用了 11 张图，超出 $K=4$）。两者的差距在 2% 以内，与笔记本 GPU 的波动同一量级，所以这组数据不足以说明 DP 更好，只说明代价模型漏掉了一些东西（例如每步的同步和 batch 之间的 kernel 切换），在下结论前要先修正它。

### 5.7 漂移下的准入

训练和服务的分布不同。A4 用总变差监控漂移：

$$
\text{TV}(p,q)=\frac12\sum_{n}\bigl\lvert p(n)-q(n)\bigr\rvert\in[0,1].
$$

平均长度相同的两个分布，TV 可以等于 1（例如全是 2 与 1、3 各半）。漂移出现后，是否为一个新形状编译或捕获，就是 §1.2 的准入问题，只是 $n$ 要从“最近窗口”预测。这正对应查新结论中的空白 (b)。

> [!question] 随堂自测 D3-6：容量与漂移
>
> 1. 容量为 2 的 artifact cache 依次执行 `put(a)`、`put(b)`、`lease(a)`、`put(c)`、`lease(c)`、`put(d)`。最后 cache 里有哪些键？如果实现“先删一项再检查 pin”，会出什么问题？
> 2. 训练直方图 $\{2:2,4:2\}$，最近窗口 $[2,2,2,4]$，TV 是多少？如果忘了除以 2 呢？

> [!tip]- 核对要点
>
> 1. `put(c)` 淘汰未被 pin 的 b，此后 a、c 都被 pin，`put(d)` 失败，键、值和 LRU 顺序都不变。先删后查会在失败时已经丢掉一项，违反“失败不改状态”的事务约定。
> 2. $p=(0.5,0.5)$，$q=(0.75,0.25)$，TV 为 $\tfrac12(0.25+0.25)=0.25$；忘了除以 2 得 0.5。只测 0 和 1 两种情况的测试抓不住这个错误，所以 A4 加了分数用例和 Fraction oracle。

## §6 融合与访存

### 6.1 按元素记账

decoder 层里的残差加和 RMSNorm 实际上是一个原地更新：$r\gets x+r$，然后 $y=\text{RMSNorm}(r)\odot\gamma$。设每个元素占 $s$ 字节，忽略缓存，按源码级的读写记账：

$$
B_{\text{two}}=\underbrace{3s}_{\text{add: }x,\,r\text{ read},\;r\text{ write}}+\underbrace{4s}_{\text{norm: }r\text{ twice},\;\gamma,\;y}=7s,
\qquad
B_{\text{fused}}=6s,
\qquad
B_{\text{ideal}}=4s+\frac{s}{\text{rows}} .
$$

融合 kernel 第一遍读 $x$、$r$，写回 $r$ 并累加平方和；第二遍读回刚写的 $r$ 和 $\gamma$，写 $y$。所以按源码记账仍有 6 次访问。如果第二遍读到的 $r$ 还在 L1/L2 里，$\gamma$ 也被所有行共享，DRAM 上就只剩 4 次。FP32 下三者分别为 28、24、16 字节每元素。差值 4 字节只是源码级的账，硬件实际流量要看缓存命中，必须测。

### 6.2 warp shuffle 归约

一个 block 256 个线程处理一行。每个线程先串行累加自己负责的元素，然后：

1. warp 内做 butterfly 归约：偏移依次为 16、8、4、2、1，每步 `v += __shfl_xor_sync(0xffffffff, v, offset)`。5 步之后 32 个 lane 都持有整个 warp 的和。掩码必须包含全部 32 个 lane，也不能把 `width` 参数设成 16，否则两个半 warp 永远不交换数据。
2. 每个 warp 的 lane 0 把结果写进 `__shared__ float warp_sums[32]`，`__syncthreads()` 之后，每个 warp 读取前 $\text{blockDim}/32$ 个部分和（其余 lane 取 0），再做一次 butterfly。
3. 同一个 kernel 里可能连续调用两次，所以写 `warp_sums` 之前还要再同步一次，防止上一次调用的读和这一次的写交错。

不变量：所有线程都必须执行每一个 `__syncthreads()`，没分到元素的线程贡献 0 但不能提前返回；分母用真实宽度 $D$，不能用 blockDim 或补齐后的宽度。

向量化：$D$ 是 4 的倍数时，每行起点按 16 字节对齐（`cudaMalloc` 返回 256 字节对齐的地址），可以用 `float4` 一次读写 16 字节；FP16 在 $D$ 为偶数时用 `__half2`。$D$ 为奇数时，行起点不再对齐，退回标量访问。两遍循环里同一个线程要处理同一组元素，这样第二遍读回自己写的 $r$，不需要额外同步。

`width=513`、256 线程的标量路径下，线程 0 处理列 $\{0,256,512\}$，线程 1 处理 $\{1,257\}$，线程 255 处理 $\{255,511\}$。`width=1024` 的 `float4` 路径下，每个线程正好处理一个 `float4`。

### 6.3 实测：手写 kernel、两 kernel 基线与 Inductor

规则见 `labs/docs/authoring.md` §5：每种方法录成一张 20 次调用的 CUDA graph，排除 CPU launch 开销；先连续回放至少 300 ms，再取 25 次，报告中位数，并检查 $p_{75}\le1.5\,p_{25}$。下表数据为 RTX 4060 Laptop、torch 2.14.1、2026-10-18 实测，单位 µs（每次调用），全部行 stable：

| dtype | rows $\times$ $D$ | two-kernel | fused（C6/C7） | Inductor | eager torch |
| ----- | ----------------- | ---------: | -------------: | -------: | ----------: |
| f32   | $16\times4096$    |       4.64 |           2.60 |     2.15 |        12.9 |
| f32   | $128\times4096$   |      12.08 |           7.22 |    15.46 |        24.8 |
| f16   | $128\times4096$   |      10.19 |           4.56 |     4.29 |        30.7 |
| f32   | $1024\times4096$  |      262.3 |          289.5 |    294.7 |       389.2 |
| f16   | $1024\times2048$  |      35.23 |          16.42 |    35.33 |       111.5 |
| f32   | $4096\times4096$  |     1426.8 |         1148.2 |   1142.0 |      2871.5 |

几点解读：

- 小矩阵受 launch 和同步开销限制，一个 kernel 比两个快约 1.7 倍，和字节数关系不大。
- 中等矩阵（几 MiB）能放进 32 MiB 的 L2，按 $B_{\text{ideal}}$ 折算的“带宽”接近 1 TB/s，远超显存带宽，说明第二遍读命中了缓存。
- $4096\times4096$ 的 FP32 矩阵每个张量 64 MiB，超出 L2，fused 和 Inductor 都在约 233 GB/s，基本跑满显存带宽，两者打平。
- 有反例：f32 $1024\times4096$ 时两 kernel 基线反而比 fused 快 10%。16 MiB 的中间结果可能整段留在 L2 里，等于由缓存替两 kernel 方案完成了融合。要用 `ncu` 看 `dram__bytes` 才能确认，这是 lab 的一个挑战题。
- Inductor 在一些尺寸上胜出，在另一些尺寸上落后两倍（f32 $128\times4096$）：它的 `XBLOCK`/`R0_BLOCK` 是启发式选的，没有 autotune。

### 6.4 从一个融合 kernel 到 megakernel

| 层次         | 本 lab 的例子          | 新增的难题                     |
| ------------ | ---------------------- | ------------------------------ |
| 算子内部     | 一行内的 warp 归约     | lane 掩码、同步、向量化对齐    |
| 相邻算子融合 | residual add + RMSNorm | 中间结果物化、缓存重用         |
| 整段程序融合 | 一个 decode step       | 跨任务依赖、资源占用、全局进度 |

[Mirage](https://arxiv.org/abs/2405.05751)（OSDI 2025）在 kernel、block、thread 三层上搜索张量程序并做等价性验证；[MPK](https://arxiv.org/abs/2512.22219)（arXiv 2025）把整段推理编译成一个 persistent megakernel，由运行时在 SM 上调度细粒度任务；[Event Tensor](https://arxiv.org/abs/2604.13327)（MLSys 2026，据 arXiv 页面）用事件张量表达 tile 级任务依赖，让 megakernel 支持动态形状。苗旭鹏是这三篇的合作者之一，当时在 Purdue，这三篇不是 PKU 组内的工作。把几个 kernel 的源码拼进一个 `__global__` 函数，并不等于 megakernel：消费者 block 如果先占满 SM 并自旋等待，生产者可能永远得不到执行资源。

> [!question] 随堂自测 D3-7：从数据流证明融合实现
>
> 1. FP16、$D=1000$ 时有多少个 `__half2`？256 线程中，线程 0 和线程 255 各处理几个？
> 2. 1000 个 FP32 元素，按 §6.1 计算两 kernel、fused 和理想情况的字节数。能把差值直接当成 profiler 里 DRAM 字节的减少量吗？
> 3. 只把偏移从 16 开始改成从 8 开始，一个 warp 的结果会变成什么？

> [!tip]- 核对要点
>
> 1. 500 个 `__half2`；线程 0 处理下标 0、256，共 2 个；线程 255 处理 255，下一个 511 超出 499，只有 1 个。
> 2. 28000、24000、约 16000 字节。不能直接当成 DRAM 减少量，L2 命中和合并访问都会改变实际流量，要用 `ncu` 测。
> 3. 每个 lane 只拿到自己所在半个 warp（16 个 lane）的和，结果约为真实值的一半，并且两个半 warp 的结果不同。

## §7 机制与代码对应

| 机制或公式                        | lab 中的函数                                   | 上游实现位置（固定版本）                                                              | 本 lab 简化了什么                                |
| --------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------ |
| first-call premium、median 和 MAD | `summarize`（A2）、`measure`（C1）             | PyTorch 编译 profiling 文档                                                           | 只用 wall clock，不拆分编译阶段                  |
| 区间成本 $c(i,j)$ 与 DP           | `segment_costs`（A3）、`plan`（B1）            | 无对应上游，vLLM/SGLang 用手工网格                                                    | 只考虑连续区间、观测上界                         |
| 不小于 $b$ 的最小捕获尺寸         | `route`（B2）、`select_graph`（R1）            | vLLM `_compute_bs_to_padded_graph_size`；SGLang `_pad_to_bucket`（见 §5.5 的 commit） | B2 多一个 padding 比例上限                       |
| 静态缓冲区拷入、补零、回放、拷出  | `GraphRunner.run`（R2）                        | SGLang `base_cuda_graph_runner.py`；vLLM `cudagraph_dispatcher.py`                    | 单 stream、无 attention metadata                 |
| 带显存预算的捕获尺寸选择          | `cost_model`（R3）、`plan_capture_sizes`（R4） | 无，上游用固定网格                                                                    | 显存模型为 $\text{pool}(\max S)+o\lvert S\rvert$ |
| guard 失败重编译、准入            | `admit`（B5）、`execute`（C3）                 | Dynamo `recompile_limit`、`set_stance`                                                | 应用层 artifact cache，不控制 Dynamo 缓存        |
| 残差加 + RMSNorm 融合             | `fused_add_rmsnorm_f32/f16`（C6/C7）           | vLLM `fused_add_rms_norm` CUDA kernel（本讲未固定版本，只作背景）                     | 一 block 一行，无 FP8 输出                       |

## 前沿与 baseline 对照

核验日期：2026-10-18（逐条打开论文页和代码仓库，确认会议与年份；“据 arXiv 页面”表示会议信息只出现在 arXiv 页面上）。

| 工作（会议 年）                                                                                                                                                      | 问题                                   | 关键机制                                                                     | 与本 lab 关系（实现/简化/未覆盖）           | 研究时必须对比                     |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------- | ---------------------------------- |
| PyTorch 2（ASPLOS 2024）[论文](https://doi.org/10.1145/3620665.3640366) · [代码](https://github.com/pytorch/pytorch)                                                 | Python 程序的图捕获与编译              | 字节码级 frame 拦截、guard、Inductor 代码生成                                | 实现：C1–C4 直接调用；§2 复现 guard 行为    | 编译模式、缓存状态下的端到端时间   |
| Triton（MAPL 2019）[论文](https://doi.org/10.1145/3315508.3329973) · [代码](https://github.com/triton-lang/triton)                                                   | 让 tile 级 GPU kernel 易写             | 以静态形状的 tile 为单位编程，编译器负责线程映射                             | 简化：读 Inductor 生成的 Triton，不手写     | 同一融合在 Triton 与 CUDA 下的带宽 |
| Ansor（OSDI 2020）[论文](https://www.usenix.org/conference/osdi20/presentation/zheng) · [代码](https://github.com/apache/tvm/tree/v0.14.0/python/tvm/auto_scheduler) | 张量程序自动调度                       | 分层搜索空间加学习的代价模型                                                 | 未覆盖：只作“每个形状都搜索”的代价参照      | 调优时间计入 $C$                   |
| Nimble（MLSys 2021）[论文](https://arxiv.org/abs/2006.03031) · 代码未核验                                                                                            | 动态形状模型的编译与执行               | `Any` 维、形状函数、VM 运行时                                                | 未覆盖                                      | 与分桶 padding 比较尾部延迟        |
| DietCode（MLSys 2022）[论文](https://proceedings.mlsys.org/paper_files/paper/2022/hash/f89b79c9a28d4cae22ef9e557d9fa191-Abstract.html) · 无公开代码                  | 动态形状下的自动调度成本               | 形状无关的搜索空间，自动调度时间最多降 5.88 倍                               | 简化：B1 只选上界，不生成 kernel            | 同一形状集合上的调度时间与性能     |
| Hidet（ASPLOS 2023）[论文](https://arxiv.org/abs/2210.09603) · [代码](https://github.com/hidet-org/hidet)                                                            | 张量程序的可编程调度                   | task mapping，调度与输入尺寸无关                                             | 未覆盖                                      | 每形状编译与形状无关调度的 $C$     |
| BladeDISC（PACMMOD 2023）[论文](https://doi.org/10.1145/3617327) · [代码](https://github.com/alibaba/BladeDISC)                                                      | 动态形状的算子融合                     | 跨层符号形状分析，动态形状下的融合                                           | 未覆盖                                      | 动态融合与分桶静态融合的稳态时间   |
| Relax（ASPLOS 2025）[论文](https://arxiv.org/abs/2311.02103) · [代码](https://github.com/apache/tvm)                                                                 | 端到端动态形状编译                     | 一等公民的符号形状、跨层抽象                                                 | 简化：本 lab 只在 batch 维做分桶            | 符号形状编译与分桶的 $C$、$t_c$    |
| XLA 有界动态性（StableHLO RFC 2023）[RFC](https://github.com/openxla/stablehlo/blob/main/rfcs/20230704-dynamism-101.md) · [代码](https://github.com/openxla/xla)     | 编译期未知但有上界的维度               | `bounds` 标注，区分有界、无界和未知秩                                        | 简化：R 阶段的 graph 也是“有上界、补到上界” | 有界编译与多 bucket 的内存         |
| FlashInfer（MLSys 2025，据 arXiv 页面）[论文](https://arxiv.org/abs/2501.01005) · [代码](https://github.com/flashinfer-ai/flashinfer)                                | 可定制、可放进 CUDA graph 的 attention | 可定制 attention 模板经 JIT 编译；负载均衡调度保持 CUDA graph 要求的静态配置 | 未覆盖：R 阶段用 SDPA                       | graph 内 attention 的 padding 开销 |
| Mirage（OSDI 2025）[论文](https://arxiv.org/abs/2405.05751) · [代码](https://github.com/mirage-project/mirage)                                                       | 多层次张量程序超优化                   | 多层次计算图搜索加概率等价性验证                                             | 未覆盖：只对照“搜索 + 验证”的流程           | 同一子图上的搜索时间与 kernel 时间 |
| MPK（arXiv 2025）[论文](https://arxiv.org/abs/2512.22219) · [代码](https://github.com/mirage-project/mirage)                                                         | 消除 kernel 边界的推理                 | 整段程序编译成一个 persistent megakernel                                     | 未覆盖：§6.4 只讨论调度难点                 | 与 CUDA graph 比较 launch 开销     |
| Event Tensor（MLSys 2026，据 arXiv 页面）[论文](https://arxiv.org/abs/2604.13327) · 无公开代码                                                                       | 动态形状下的 megakernel                | 事件张量编码 tile 任务依赖                                                   | 未覆盖                                      | 动态 batch 下的预热与稳态          |
| Medusa（ASPLOS 2025）[论文](https://doi.org/10.1145/3669940.3707285) · [代码镜像](https://github.com/MachineLearningSystem/25ASPLOS-Medusa)                          | serverless 冷启动                      | 离线物化 CUDA graph 和 KV cache 初始化信息，在线恢复                         | 未覆盖：对应查新 (c)                        | 冷启动时间与捕获全部尺寸的成本     |
| Foundry（arXiv 2026）[论文](https://arxiv.org/abs/2604.06664) · 无公开代码                                                                                           | CUDA graph 捕获拖慢冷启动              | 以模板持久化 graph 拓扑和执行上下文                                          | 未覆盖：对应查新 (c)                        | 每个捕获尺寸的恢复时间与显存       |
| vLLM（系统，commit `371745eb`）[文档](https://docs.vllm.ai/en/latest/design/cuda_graphs/) · [代码](https://github.com/vllm-project/vllm)                             | 生产环境的 decode CUDA graph           | 固定网格捕获尺寸，piecewise 与 full graph                                    | 实现：R 阶段的 `vllm_default` 基线          | 同一 trace 上的 $J(S)$ 与显存      |
| SGLang（系统，commit `57db5089`）[代码](https://github.com/sgl-project/sglang)                                                                                       | 同上                                   | 更密的小尺寸网格，`bisect_left` 补齐                                         | 实现：R 阶段的 `sglang_default` 基线        | 同上                               |

**查新结论**（按研究问题分三项，检索时间 2026-10-18）：

- **(a) 在编译或显存预算下由 trace 驱动选择 bucket 和捕获尺寸。** 最接近的是 vLLM、SGLang 的固定网格（工程默认值，没有优化目标），TensorRT-LLM 关于调 CUDA graph batch size 的技术博客（人工比较三种网格的吞吐、显存和启动时间），以及 DietCode（面向“动态形状集合”的调度，但优化的是 kernel 生成，而不是捕获尺寸的选择）。没有找到把“捕获尺寸 + 显存池 + 捕获时间”写成带约束的优化问题、在真实 trace 上报告 regret 的论文。只在合成 trace 和 4 层小模型上证明 DP 优于网格，会与工程博客撞车。要有贡献，至少要用完整模型、公开 trace（例如 BurstGPT、Azure LLM trace），并给出 piecewise graph 下的显存模型。
- **(b) 形状漂移下的在线重编译准入。** 最接近的是 PyTorch 的 `recompile_limit` 和 `set_stance`（只是机制，没有策略），ezyang 的 [Ways to use torch.compile](https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/)（讨论预热，并指出无法排除重编译），以及 [compile 缓存教程](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)里随 mega-cache 一起保存的 PGO 缓存（跨进程记住哪些维度该当成动态，但不判断值不值得编译）。漂移检测（A4）加回本判断（B5）组成在线策略、并在真实漂移 trace 上与 eager-on-recompile 对比，还没有看到系统的论文。
- **(c) 跨进程、跨机器共享编译产物。** 最接近的是 PyTorch 的 mega-cache 和远程 FX graph cache（要求版本和 GPU 一致）、Medusa（ASPLOS 2025，离线物化 CUDA graph）和 Foundry（arXiv 2026，模板化恢复 graph 上下文）。ServerlessLLM（OSDI 2024）只解决权重加载。还没有回答的是：多租户、多版本共存时，缓存应该按什么粒度、什么键共享，命中率与正确性风险（guard 不完整导致误用）怎样权衡。

苗组相关：Mirage、MPK、Event Tensor 都有苗旭鹏参与，当时他在 Purdue。它们与本讲的关系是“编译期搜索与运行时调度”，不是捕获尺寸选择本身。

## 研究问题

- **候选假设**：在 Qwen3-1.7B 规模的完整 decode step（28 层）上，用历史 trace 加 R4 的 DP 选出的 8 个捕获尺寸，在 shift trace 上的每步延迟比 vLLM 默认网格的前 8 个尺寸低至少 3%，且共享池显存不超过默认网格。
- **最小实验**：R 阶段换成 28 层（权重仍随机，FP16，KV 长度 256；8 GB 显存内要核算权重和 KV）；trace 用 R 阶段的 stable/shift 两种，再加一条公开 trace 的 batch 序列；对照 vLLM 默认网格、SGLang 默认网格、均匀 $K$ 段、全 eager 和 hindsight；每个策略回放 5 次，报告 measured 与 modeled 两列，以及显存池实测。
- **继续或停止的证据**：如果 28 层下 DP 相对网格的 measured 收益小于测量的 IQR，或者 hindsight 与最好的网格只差不到 1%，说明捕获尺寸不是瓶颈，应停下来，转向 (b) 或 (c)。如果 modeled 排序和 measured 排序不一致，先修正代价模型（加入步间同步、piecewise 切分），再谈策略。
- **与导师讨论的问题**：能否拿到真实服务的 batch size 序列；是否值得在 vLLM 上做 piecewise graph 的显存测量（需要更新驱动或另一台机器）；(c) 的缓存共享是否与组内 serving 方向有交集。

## 资料汇总

下表的“指定观察”已计入本讲时间，通读论文或运行上游代码属于选读或 lab 挑战。

| 资料                                                                                                                                                                                                                                                                                                                     | 类型       | 指定观察                                                                          | 时间  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | --------------------------------------------------------------------------------- | ----- |
| [PyTorch 2 论文](https://docs.pytorch.org/assets/pytorch2-2.pdf)                                                                                                                                                                                                                                                         | 论文       | §3.3 guard、§3.4 符号求值；对照 §2 的实测计数                                     | 20min |
| [Dynamic shapes 文档](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html)                                                                                                                                                                                                 | 文档       | automatic dynamic、0/1 特化、`mark_dynamic` 的描述，与本机行为逐条核对            | 15min |
| [torch.compile 排错](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html) 与 [tlparse](https://github.com/pytorch/tlparse)                                                                                                                                                | 文档、工具 | 用 `TORCH_LOGS=recompiles` 复现 §2.3 的日志；选读：tlparse 报告                   | 10min |
| [compile 缓存教程](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)                                                                                                                                                                                                                       | 文档       | mega-cache 的失效条件，填进 §3.1 的表                                             | 10min |
| [Triton fused softmax 教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) 与 [layer norm 教程](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)                                                                                                                | 教程       | 对照 `build/inductor_f32_2048.py`，找出 `XBLOCK` 与 `R0_BLOCK` 的含义             | 15min |
| [vLLM CUDA graph 设计文档](https://docs.vllm.ai/en/latest/design/cuda_graphs/) 与 [torch.compile 博客](https://vllm.ai/blog/2025-08-20-torch-compile)                                                                                                                                                                    | 文档、图解 | piecewise 切分图；记下 `cudagraph_mode` 默认值和文档日期                          | 15min |
| vLLM [`cudagraph_dispatcher.py`](https://github.com/vllm-project/vllm/blob/371745ebd6c12f314402c6c319de2e7ef82c2981/vllm/v1/cudagraph_dispatcher.py)、SGLang [`cuda_graph_hook.py`](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/arg_groups/cuda_graph_hook.py) | 代码       | 各读一个函数，写出它们与 `select_graph` 的对应关系                                | 15min |
| [CMU 15-779](https://www.cs.cmu.edu/~zhihaoj2/15-779/)                                                                                                                                                                                                                                                                   | 课程       | 选读 ML compilation 与自动 kernel 生成两讲                                        | 选读  |
| [Stanford CS336 assignment 2](https://github.com/stanford-cs336/assignment2-systems)                                                                                                                                                                                                                                     | 课程作业   | 选读 handout 中的 benchmark 与 Triton 部分，对照本讲的测量规则                    | 选读  |
| [GPU MODE lectures](https://github.com/gpu-mode/lectures)                                                                                                                                                                                                                                                                | 课程录像   | 选读第 14 讲（Triton 实践）、第 29 讲（Triton internals）、第 79 讲（Mirage MPK） | 选读  |
| [Lab 3](../../labs/lab3/README.md)                                                                                                                                                                                                                                                                                       | 实验       | 对照 §7 的函数表                                                                  | —     |

## 自测清单

- [ ] 不看讲义，推导 $n^\ast=C/(t_e-t_c)$，处理持平、重编译和 $t_c\ge t_e$ 三种情况。
- [ ] 在自己的机器上打印 `recompile_limit` 等配置，复现 $[1,1,0,0,0,0,0]$ 和 recompile limit 退回 eager 的日志。
- [ ] 说出每一层缓存省掉了 $C$ 的哪一部分、在什么条件下失效，以及怎样测冷启动。
- [ ] 写出 DP 递推和最优子结构证明，手算四种划分，说明“上界取观测值”依赖什么假设。
- [ ] 把 vLLM/SGLang 的捕获尺寸映射到 $P$、$t$、$e$、pool，写出带显存预算的 $J(S)$ 和化归方法。
- [ ] 推导 28B、24B、16B 的访存账，画出 warp butterfly 归约，说明三条同步不变量。
- [ ] 对照表中至少三篇工作，各写一句机制差异和本 lab 的简化边界。
- [ ] 写出一个可证伪的假设、必须对比的 baseline 和停止条件。

[← 课程索引](../index.md) · [配套 lab3](../../labs/lab3/README.md) · [← D2 低比特 kernel](02-low-bit-kernels.md) · [D4 推测解码 →](04-speculative-slo.md)
