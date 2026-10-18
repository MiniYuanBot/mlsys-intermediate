---
title: D2 低比特权重 kernel：字节、格式、解码与小 batch 的交叉点
lab: lab2
hours: 5
tags: [lecture, quantization, cuda, ptx, fp8, kernel, research]
updated: 2026-10-18
---

# D2 低比特权重 kernel：字节、格式、解码与小 batch 的交叉点

[← 课程索引](../index.md) · [配套 lab2](../../labs/lab2/README.md) · [← D1 KV 缓存](01-kv-cache.md) · [D3 动态编译 →](03-dynamic-compilation.md)

> [!abstract] 这一讲要解决的问题
> foundations 讲过 GEMM 的分块和 GPU 的访存层次。这一讲的约束更具体：在 RTX 4060 Laptop（256 GB/s 的 GDDR6、32 MiB L2、sm_89）上跑 Qwen3 的 decode，batch 很小，每一步都要把全部线性层权重从显存读一遍。权重压成 4 bit 或 8 bit 能少读字节，但每个权重要多做解包、查 scale、类型转换，batch 一大，省下的字节就不再是瓶颈。学完后你应该能：手算任意格式的每权重字节数和交叉 batch；说清楚 INT4、NF4、FP8、MX、NVFP4、k-quant 各自的元数据和硬件支持；从 PTX 层面写出 `bfe`、`lop3` 加 magic number、`cvt.e4m3x2` 三种寄存器内解码；在 4060 上用冲掉 L2 的基准把自己的 INT4 和 FP8 kernel 和 cuBLAS FP16、cuBLASLt FP8 放进同一张表；最后判断“shape 相关的 kernel 分发”和“Ada 上 FP8 对 INT4”哪个值得做研究。

| 阅读模块                                  | 时间  | 必须留下的学习产物                                | 对应实验     |
| ----------------------------------------- | ----- | ------------------------------------------------- | ------------ |
| §1 字节模型、屋顶线与交叉 batch           | 45min | 五种表示的 $b_w$ 表，4060 上的 modeled 交叉 batch | R5、R 基准   |
| §2 数值格式与量化误差                     | 60min | INT4 与 E4M3 的编码手算、组大小对误差和字节的推导 | A1–A6、R1/R2 |
| §3 寄存器内解码与数据布局                 | 50min | `bfe` 与 `lop3` 每权重指令数，magic number 推导   | B1/B3、R3    |
| §4 kernel 结构：warp、split-K、token 复用 | 40min | 参与集合与 partial 布局的不变量清单               | B2–B6、R4    |
| §5 测量：L2、基线与本机数据               | 35min | 一张 measured 与 modeled 分列的表                 | B/R 基准     |
| §6 量化算法地图                           | 30min | 一张“算法改了什么、kernel 要配合什么”的卡片       | C1–C5        |
| §7 机制与代码的对应                       | 10min | 机制到函数的对照表                                | 全部         |
| 前沿与 baseline 对照                      | 20min | 三篇工作的机制差异卡                              | C/R、挑战    |
| 研究问题                                  | 10min | 一个可证伪的假设和停止条件                        | R、挑战      |

**共 5h，另做 lab 约 35h。** 5h 包括手算、三段源码阅读（Marlin `dequant`、AWQ `_search_module_scale`、cuBLASLt 的 FP8 条件）和随堂自测；通读论文属于选读。标 modeled 的数字由公式或规格表得到，标 measured 的是参考实现在本机上的实测（2026-10-18，驱动 566.24，CUDA 12.6），两者不放在同一列比较。外部论文里的图表是作者在自己硬件上的结果，不是你的结果。

随堂自测计入上表时间。先在笔记里写答案，再展开核对要点。

## §1 字节模型、屋顶线与交叉 batch

### 1.1 decode 的一个线性层读多少字节

decode 时一个线性层做 $Y=XW^\top$：$X$ 是 $[B,K]$ 的激活（$B$ 是这一步同时解码的 token 数），$W$ 是 $[M,K]$ 的权重，$Y$ 是 $[B,M]$。浮点运算量是 $2BMK$。设每个权重平均占 $b_w$ 字节（含 scale 等元数据），激活和输出各占 $b_a$ 字节，那么一次调用至少要从显存读写

$$
\text{bytes}(B)=b_w MK + b_a B(K+M).
$$

Qwen3-1.7B 的 up_proj 是 $M=6144,K=2048$，FP16 权重 24 MiB；$B=1$ 时激活加输出只有 16 KiB，可以忽略。于是算术强度（每字节多少 FLOP）近似为

$$
I(B)\approx\frac{2BMK}{b_w MK}=\frac{2B}{b_w}.
$$

**每权重字节数 $b_w$ 怎么算。** $b$ bit 的码加上每 $g$ 个权重一个 $b_s$ 字节的 scale：

$$
b_w=\frac{b}{8}+\frac{b_s}{g}.
$$

| 表示                                      | $b$ | scale 粒度与类型          | $b_w$（B/权重） | $I(B)$  |
| ----------------------------------------- | --: | ------------------------- | --------------: | ------- |
| FP32                                      |  32 | 无                        |               4 | $0.5B$  |
| FP16 / BF16                               |  16 | 无                        |               2 | $B$     |
| FP8 E4M3，逐行或 $g=128$，FP32 scale      |   8 | $4/128$                   |         1.03125 | $1.94B$ |
| INT4 $g=128$，FP16 scale（GPTQ/AWQ 常见） |   4 | $2/128$                   |         0.51563 | $3.88B$ |
| INT4 $g=128$，FP32 scale（lab2 的布局）   |   4 | $4/128$                   |         0.53125 | $3.76B$ |
| INT4 $g=32$，FP16 scale                   |   4 | $2/32$                    |          0.5625 | $3.56B$ |
| NVFP4：E2M1，每 16 个一个 E4M3 scale      |   4 | $1/16$（另有每张量 FP32） |          0.5625 | $3.56B$ |
| MXFP4：E2M1，每 32 个一个 E8M0 scale      |   4 | $1/32$                    |         0.53125 | $3.76B$ |

lab2 把 scale 存成 FP32 是为了让测试容易核对，生产格式一般用 FP16/BF16（或 MX/NVFP4 的 8 bit scale），元数据只有一半。非对称格式（AWQ、GPTQ 的 uint4 加零点）每组还要再存一个零点，通常是 4 bit 或 FP16。

### 1.2 屋顶线：什么时候字节不再是瓶颈

屋顶线模型说，kernel 的时间至少是 $\max(\text{bytes}/\mathrm{BW},\ \text{FLOP}/P)$，$\mathrm{BW}$ 是显存带宽，$P$ 是所用计算单元的峰值。两者相等的强度叫脊点 $I^\ast=P/\mathrm{BW}$。$I(B)<I^\ast$ 时访存受限，时间几乎不随 $B$ 变；$I(B)>I^\ast$ 后时间随 $B$ 线性增长。

RTX 4060 Laptop 的数字（modeled，由规格推算）：

| 量                          | 值                                   | 来源                                                                                                             |
| --------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| 显存带宽                    | 256 GB/s（128 bit，16 Gbps）         | TechPowerUp、NotebookCheck 规格页                                                                                |
| 本机实测 copy 带宽          | 约 223 GB/s（measured）              | lab2 `gpu_bench.cu` 的 256 MiB 读写                                                                              |
| L2                          | 32 MiB（measured，`cudaDeviceProp`） | 台式 4060 是 24 MiB                                                                                              |
| FP32 CUDA core              | 约 11.6 TFLOPS（最高 boost）         | NotebookCheck                                                                                                    |
| FP16 Tensor Core，FP32 累加 | 约 29 TFLOPS                         | 由 Ada 白皮书 4090 的每 SM 数值按 24 SM、2.37 GHz 换算，GeForce 的 FP32 累加是半速；**推算，不是 NVIDIA 的规格** |
| FP8 Tensor Core，FP32 累加  | 约 58 TFLOPS                         | 同上                                                                                                             |

由此得到三个脊点：FP16 Tensor Core 约 $29\times10^{12}/256\times10^9\approx113$ FLOP/B；FP8 Tensor Core 约 227 FLOP/B；FP32 SIMT 约 45 FLOP/B。

**交叉 batch。** 令 $I(B)=I^\ast$，得 $B^\ast=I^\ast b_w/2$：

| 路径                                   |  $b_w$ | 计算单元       | $B^\ast$（modeled） |
| -------------------------------------- | -----: | -------------- | ------------------: |
| FP16 权重，FP16 Tensor Core            |      2 | FP16 TC        |                 113 |
| W4A16（Marlin 一类），FP16 Tensor Core | 0.5156 | FP16 TC        |                  29 |
| W8A16（FP8 权重升成 FP16），FP16 TC    |  1.031 | FP16 TC        |                  58 |
| W8A8 FP8，FP8 Tensor Core              |  1.031 | FP8 TC         |                 117 |
| lab2 的 W4A32 SIMT                     | 0.5313 | FP32 CUDA core |                  12 |
| lab2 的 W8A32 SIMT（R 阶段）           |  1.031 | FP32 CUDA core |                  23 |

W4A16 的 $B^\ast\approx29$ 与 Marlin README 写的“batch 16–32 以内接近理想的 4 倍加速”一致（Marlin 在 A10 上测得）。这张表有两个用处。第一，batch 1 时所有路径都远在脊点左边，速度只取决于字节，所以 INT4 理论上比 FP16 快 $2/0.5156\approx3.9$ 倍。第二，batch 超过 $B^\ast$ 之后，低比特路径会被它所用的计算单元卡住：W4A32 SIMT 在 $B\approx12$ 就碰到 CUDA core 的屋顶，而 FP16 cuBLAS 要到 113 才碰到 Tensor Core 的屋顶。所以 FP16 cuBLAS 迟早会追上，追上的 batch 取决于低比特 kernel 用什么单元算、解码花多少指令。这些 $B^\ast$ 都是 modeled 值，§5.3 给出本机 measured 的交叉点。

**解码指令不在屋顶线里。** 每个权重的解包、乘 scale、类型转换都占发射槽。粗略估一下（modeled，忽略双发射和整数单元的细节）：4060 每秒约能发 $5.8\times10^{12}$ 条 FMA 级指令，INT4 权重的到达速率是 $256\times10^9\times2=5.12\times10^{11}$ 个/秒，所以每个权重平均不到 **11 条**指令时才不会被指令卡住。batch 为 $B$ 时，每个权重除了解码还要做 $B$ 次 FMA。§3 会看到，朴素的 `bfe` 解码每个权重约 3–4 条，`lop3` 版本约 1 条。

### 1.3 Amdahl：kernel 加速不等于 decode 加速

若线性层占一步 decode 时间的比例为 $p$，它们加速 $s$ 倍，总加速是

$$
S=\frac{1}{(1-p)+p/s}.
$$

batch 1 的 decode 中线性层常占大部分时间，但 attention、norm、采样和 launch 开销都不会变快。$p=0.7$、$s=3.9$ 时 $S\approx2.1$。端到端结论要在真实模型上测，不能从单个 kernel 推。

> [!question] 随堂自测 D2-1：字节和交叉点
>
> 1. Qwen3-1.7B 的 down_proj 是 $M=2048,K=6144$。分别算 FP16、INT4 $g=128$ FP16 scale、FP8 逐行 FP32 scale 的权重字节数，以及按实测 224 GB/s 只读一遍权重的时间下界。
> 2. NVFP4 和 MXFP4 的码都是 4 bit，为什么前者每权重多 0.03 B？哪个格式的 scale 误差更小？
> 3. 某 W4A16 kernel 在 batch 32 时比 FP16 cuBLAS 慢。用 §1.2 的模型给出两种解释，并各写一个能区分它们的测量。

> [!tip]- 核对要点
>
> 1. FP16：$2\times2048\times6144=24\,\mathrm{MiB}$，约 $112\,\mu\mathrm{s}$；INT4：$0.5156\times MK\approx6.2\,\mathrm{MiB}$，约 $29\,\mu\mathrm{s}$；FP8 逐行（$g=K$）：$MK+4M\approx12.0\,\mathrm{MiB}$，约 $56\,\mu\mathrm{s}$。
> 2. NVFP4 每 16 个权重一个 8 bit scale，即 $0.5$ bit/权重；MXFP4 每 32 个一个，$0.25$ bit/权重。NVFP4 的块更小，scale 又是 E4M3（有尾数），而 MX 的 E8M0 只能表示 2 的幂，所以 NVFP4 的量化误差更小。
> 3. 一是已经过了 W4A16 的 $B^\ast\approx29$，Tensor Core 算力成了瓶颈，此时它和 FP16 cuBLAS 做同样的 FLOP 却多了解码；二是解码指令过多，在更小的 batch 就被指令发射卡住。前者看 `ncu` 的 Tensor Core 利用率是否接近峰值，后者看每权重的 SASS 指令数和 `sm__inst_executed`。

## §2 数值格式与量化误差

### 2.1 整数格式：对称、非对称、分组

**对称（lab2 的格式）。** 一组 $g$ 个权重共用 $s=\max\lvert w\rvert/7$，码 $q=\mathrm{clamp}(\mathrm{round}(w/s),-7,7)$，存 $u=q+8\in[1,15]$。解码 $\hat w=(u-8)s$。码 0 永远不出现，16 个码只用了 15 个，换来的是零点固定、解码只需一次减法。

**非对称（AWQ、GPTQ 检查点的常见格式）。** 组内 $s=(\max w-\min w)/15$，零点 $z=\mathrm{round}(-\min w/s)$，存 $u=\mathrm{clamp}(\mathrm{round}(w/s)+z,0,15)$，解码 $\hat w=(u-z)s$。16 个码都用上，对偏斜的分布更友好，但每组多存一个 $z$，解码多一次减法（或者把 $-zs$ 预先算好变成一次 FMA）。

**行步长。** lab2 每行单独按字节对齐：$N$ 为奇数时行尾空出一个高半字节。码在 $r\lceil N/2\rceil+\lfloor c/2\rfloor$，scale 在 $r\lceil N/g\rceil+\lfloor c/g\rfloor$。D1 的 KV 格式是整个矩阵连续打包，两者都合理，但打包和解码必须用同一种。

**组只在一行之内。** “每组一个 scale”里的组是同一输出行上连续的 $g$ 列，因为 GEMV 的一个点积恰好沿 $K$ 走过这些列。按整行取 absmax 也能得到一个合法的格式，但它把一行里最大的值强加给所有组。例子：$W=[7,1,-1,\ 0.7,0.1,0]$、$g=3$，正确的 scale 是 $[1,0.1]$；按行取会得到 $[1,1]$，第二组的三个值全被量化成 $[1,0,0]$。lab2 的 A1 测试就是这个例子，因为“码都在半步之内”这种检查会被按行 absmax 骗过：scale 大了，半步也跟着大了。

### 2.2 量化误差与组大小

对一个 scale 为 $s$ 的均匀量化器，舍入误差近似在 $[-s/2,s/2]$ 上均匀分布，均方误差为 $s^2/12$。组越小，组内最大值越接近典型值，$s$ 越小。对独立同分布的 $N(0,\sigma^2)$ 权重，$g$ 个样本绝对值的最大值约为 $\sigma\sqrt{2\ln g}$（渐近式，$g$ 小时偏大一些），于是

$$
\mathrm{RMSE}(g)\approx\frac{s}{\sqrt{12}}\approx\frac{\sigma\sqrt{2\ln g}}{7\sqrt{12}}.
$$

$g=128$ 时 $\sqrt{2\ln128}\approx3.1$，RMSE 约 $0.13\sigma$；$g=32$ 时约 $0.11\sigma$。**measured**：lab2 R 基准里 INT4 $g=128$ 对 Gaussian 权重的输出相对误差约 0.117，与这个估计一致（随机激活下，输出的相对误差约等于权重的相对误差）。

组变小要付字节：FP16 scale 下 $g=128$ 是 4.125 bit/权重，$g=32$ 是 4.5 bit/权重。同样多出 0.375 bit，也可以花在别处，比如 AWQ 的通道缩放（每通道一个 scale，几乎不占字节）或者 GPTQ 的误差补偿（不占字节，只花校准时间）。lab2 C5 让你在固定的总字节下比较这几种花法。

**clipping。** 把 scale 缩小到 $\rho\cdot\max\lvert w\rvert/7$（$\rho<1$），大值会被截断，其余值的舍入步长变小。lab2 A6 按组内平方误差选 $\rho$。一个能手算的例子：31 个 $\pm0.5$ 加一个 1.0。$\rho=1$ 时 $s=1/7$，$0.5/s=3.5$ 正好落在舍入平局上，按 ties-away 进到 4，每个误差 $1/14$，平方误差和约 0.158；$\rho=0.9$ 时 $0.5/s\approx3.89$，进到 4，误差约 0.014，1.0 则被截到 $7s=0.9$，误差 0.1；平方误差和合计约 0.016，仍是原来的十分之一。

### 2.3 浮点格式：FP8、NF4、MX、NVFP4

**FP8。** [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) 定义了两种 8 bit 浮点：

| 格式 | 符号/指数/尾数 | 偏置 | 最大有限值 | 最小正规数 | 最小 subnormal | 特殊值                           |
| ---- | -------------- | ---: | ---------: | ---------: | -------------: | -------------------------------- |
| E4M3 | 1/4/3          |    7 |        448 |   $2^{-6}$ |       $2^{-9}$ | 只有 NaN（S.1111.111），没有 inf |
| E5M2 | 1/5/2          |   15 |      57344 |  $2^{-14}$ |      $2^{-16}$ | 与 IEEE 相同的 inf 和 NaN        |

E4M3 的值：指数字段 $e\ge1$ 时为 $(1+m/8)\,2^{e-7}$，$e=0$ 时为 $(m/8)\,2^{-6}$。最大有限值是 $e=15,m=6$，即 $1.75\times2^8=448$（$m=7$ 留给了 NaN，所以不是 IEEE 风格的 480）。相邻正规数的相对间距是 $2^{-3}$，舍入的最大相对误差是 $2^{-4}=6.25\%$，与数值大小无关。这和整数格式不同：INT4 的误差是相对组内最大值的固定步长，小值的相对误差很大。lab2 R 基准里 FP8 $g=128$ 的输出相对误差约 0.024（measured），比 INT4 小 5 倍。

**编码时的三个陷阱**（lab2 R1）：

- 舍入是最近值、平局取偶数尾数（RNE），与 `cvt.rn` 一致；
- 超出 448 的值要饱和到 $\pm448$，`cvt.rn.satfinite.e4m3x2.f32` 就这样做。不饱和的话 $480$ 会被舍入到 $e=15,m=7$，也就是 NaN；
- $\lvert x\rvert<2^{-6}$ 进入 subnormal，最小 $2^{-9}$，$2^{-10}$ 恰好是 0 和 $2^{-9}$ 的中点，平局取偶数得 0。

对 FP8 权重，scale 取 $\max\lvert w\rvert/448$，让组内最大值正好落在最大码上。

**NF4。** [QLoRA](https://arxiv.org/abs/2305.14314)（NeurIPS 2023）提出的 4 bit NormalFloat：16 个码对应标准正态分布的分位点，而不是等间距；bitsandbytes 用 `bnb_4bit_quant_type="nf4"` 实现，scale 还可以再量化一次（double quantization）。它在 Gaussian 权重上的误差比等间距 INT4 小，但解码要查表，不能用 §3 的 magic number。

**MX 与 NVFP4。** [OCP Microscaling](https://arxiv.org/abs/2310.10537)（2023）规定 32 个元素共享一个 E8M0 scale（只有指数，即 2 的幂），元素可以是 FP8、FP6、FP4 或 INT8（MXFP8/MXFP6/MXFP4/MXINT8）。[NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)（NVIDIA 博客，2025-06-24）的元素是 E2M1，每 16 个一个 E4M3 scale，再加每张量一个 FP32 scale。E2M1 只有 $\pm\{0,0.5,1,1.5,2,3,4,6\}$ 这些值。这两种格式的硬件支持在 Blackwell 的第五代 Tensor Core 上。

**4060（sm_89）上能用什么。**

- FP8：硬件转换 `cvt.rn.f16x2.e4m3x2`（两个 E4M3 转成 `half2`）和 `cvt.rn.satfinite.e4m3x2.f32`；FP8 Tensor Core `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`，PTX 8.4 起。CUDA 12.4 带 PTX 8.4，12.5 和 12.6 带 PTX 8.5，所以本机 CUDA 12.6 能用的 FP8 MMA 只有 `m16n8k32`、FP32 累加。FP8 的 `m16n8k16` 和 FP16 累加要 PTX 8.7（CUDA 12.8 起）。cuBLASLt 的 FP8 matmul 从 compute capability 8.9 起支持；在 CUDA 12.6 和 Ada 上只有逐张量 scale，分块 scale（MXFP8）要 10.0。
- INT4：没有 INT4 的 MMA 可以直接吃非对称码，生产做法是在寄存器里转成 FP16 再用 FP16 MMA（W4A16）。
- FP4（MXFP4/NVFP4）：没有硬件支持。

### 2.4 本地推理会遇到的格式：GGUF、k-quant、EXL2/EXL3

llama.cpp 的 [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) 是自描述的容器。[k-quants（PR #1684）](https://github.com/ggml-org/llama.cpp/pull/1684) 用 256 个权重的超块，超块内再分子块，子块的 scale 本身也量化成 6 bit 或 8 bit。这样元数据很小，Q4_K 约 4.5 bit/权重，Q2_K 约 2.56，Q6_K 约 6.56。i-quant（IQ2_XXS 等）是 2–3 bit 的格点码本，用 importance matrix 加权，统计量是激活平方的均值，和 AWQ 的思路相近。ExLlamaV2 的 EXL2 按列混合 2–8 bit 来凑一个目标 bpw；ExLlamaV3 的 EXL3 改成精简版 QTIP（Hadamard 加 trellis）。这些格式都要求 kernel 专门实现解码，没有哪一种能被 cuBLAS 直接消费。

> [!question] 随堂自测 D2-2：编码手算
>
> 1. 对称 INT4，$W=[7,2.5,-2.5,0.5,-0.5,1.5]$，$g=6$。写出三个字节（偶数列在低半字节），并说明把 `lround` 换成 `nearbyint` 后哪几个码变了。
> 2. 把 $x=300$、$x=0.0013$、$x=500$ 编成 E4M3，写出码（十六进制）和解码值。
> 3. Gaussian 权重从 $g=128$ 改成 $g=32$，RMSE 大约降多少？同样多花的 0.375 bit/权重若改用 FP8，误差会是多少量级？

> [!tip]- 核对要点
>
> 1. $s=1$，$q=[7,3,-3,1,-1,2]$，$u=[15,11,5,9,7,10]$，字节为 `0xBF 0x95 0xA7`。`nearbyint` 平局取偶：$2.5\to2$、$-2.5\to-2$、$0.5\to0$、$-0.5\to0$，前四个码之外的 7 和 1.5 不变。
> 2. $300=1.171875\times2^8$，尾数 $1.171875\times8=9.375$，取 9，即 $m=1$、$e=15$，码 `0x79`，解码 $1.125\times256=288$（相邻值 288 和 320，300 更近 288）。$0.0013$ 在 subnormal 区，$0.0013/2^{-9}\approx0.67$，舍入为 1，码 `0x01`，值 $2^{-9}\approx0.00195$。$500>448$，饱和为 `0x7E`，值 448。
> 3. $\sqrt{2\ln32}/\sqrt{2\ln128}\approx2.63/3.11$，RMSE 约降 15%。FP8 每权重多 4 bit 而不是 0.375 bit，不能同口径比较；同字节下的比较应该在 INT4 的不同组大小、AWQ 和 GPTQ 之间做。

## §3 寄存器内解码与数据布局

### 3.1 一个权重从字节到 FMA 要几条指令

lab2 B 阶段每个权重的解码是：取字节（或与相邻 lane 共享一次加载）、`bfe.u32` 取出 4 bit、减 8、整数转浮点、乘 scale、与 $x$ 做 FMA。[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 里 `bfe.u32 d, a, b, c` 从第 $b$ 位开始取 $c$ 位并零扩展；`bfe.s32` 则把字段最高位复制到高位（符号扩展），在 lab2 的偏移编码里是错的：$u=15$ 会变成 $-1$。

PTX 不是最终机器码。`bfe` 在 SASS 里可能变成移位加与，也可能和普通的 `(b >> s) & 15` 生成完全相同的指令。所以比较两种写法时先看 SASS（`cuobjdump -sass`），再计时。粗算下来每个权重 3–4 条整数和转换指令，加上 $B$ 条 FMA。

### 3.2 一次加载 16 个 FP8 权重

lab2 R3 每个 lane 用一条 128 位加载读 16 个 E4M3 码（行宽补到 16 字节的倍数才能这样做），然后两两用 `__nv_cvt_fp8x2_to_halfraw2(.., __NV_E4M3)` 转成 `half2`。这在 sm_89 上是一条 `cvt.rn.f16x2.e4m3x2`。每个 E4M3 值都能被 FP16 精确表示，所以这一步没有误差。一组 16 个权重共用一个 scale，可以先算 $\sum_i w_i x_i$ 再乘一次 scale，scale 的乘法就摊到了 16 个权重上。

### 3.3 INT4 的 magic number：一条 `lop3` 解两个权重

FasterTransformer 的 MoE 论文（[Kim et al., 2022](https://arxiv.org/abs/2211.10017)）给出了一个不需要整数转浮点指令的技巧，后来被 AWQ、Marlin、vLLM、TileLang 沿用。FP16 的 `0x6400` 是 1024：指数字段 25，$2^{25-15}=2^{10}$，尾数为 0。在 $[1024,2048)$ 里 FP16 的步长正好是 1，所以对 $0\le v<1024$ 的整数，位模式 `0x6400 | v` 恰好表示 $1024+v$。把一个 32 位寄存器看成两个 FP16：

```text
q        = [ ... v5 v4 | ... v1 v0 ]   8 nibbles in one 32-bit register
q & 0x000F000F           keeps the low nibble of each 16-bit half
  | 0x64006400           -> half2(1024 + v4', 1024 + v0')
hsub2(.., 0x64086408)    -> half2(v4' - 8, v0' - 8)     since 0x6408 = 1032
```

`(q & LO) | EX` 是三输入布尔函数，`lop3.b32` 一条指令就能做，查找表常数 $\mathrm{immLut}=(\mathtt{0xF0}\ \&\ \mathtt{0xCC})\mid\mathtt{0xAA}=\mathtt{0xEA}$（`0xF0`、`0xCC`、`0xAA` 分别代表输入 a、b、c）。高半字节用掩码 `0x00F000F0` 得到 $1024+16v$，再一条 `hfma2` 乘 $1/16$（`0x2C00`）加 $-72$（`0xD480`）：

$$
\frac{1024+16v}{16}-72=64+v-72=v-8.
$$

[Marlin `marlin_cuda_kernel.cu`](https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda_kernel.cu) 的 `dequant` 就是这五个常数：`LO=0x000f000f`、`HI=0x00f000f0`、`EX=0x64006400`、`SUB=0x64086408`、`MUL=0x2c002c00`、`ADD=0xd480d480`。每 4 个权重约 2 条 `lop3`、1 条 `hsub2`、1 条 `hfma2`，平均每个权重约 1 条，而 §3.1 的做法是 3–4 条（这是按源码估的数，SASS 要你自己数）。FasterTransformer 里对应的代码在 [`interleaved_numeric_conversion.h`](https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h)：`I4s_TO_F16s_MAGIC_NUM = 0x64006400` 和同一个 immLut；int8 路径用 `prmt.b32` 做字节重排，减的常数是 1152。

**代价是布局。** 一条 `lop3` 产出的两个值来自寄存器两个半字的低半字节，即原序列里的第 0 和第 4 个码，不是相邻的两个。Marlin 因此离线把每 8 个码重排成 $[0,2,4,6,1,3,5,7]$（`marlin/__init__.py` 的 `_get_perms`），让解出来的 `half2` 正好是 `mma.m16n8k16` 的 B 片段要的一对。lab2 的行主序、偶数列在低半字节的格式不能直接套这段代码，要先重排，并且让 oracle 认新布局（lab2 挑战 2）。vLLM 的 `gptq_marlin_repack.cu`、`awq_marlin_repack.cu` 在加载模型时做这件事，这是一次性的离线成本。

### 3.4 layout 与 Tensor Core 片段

Marlin 的思路是让权重在显存里就按 MMA 片段的顺序存放：一次 16 字节的 `cp.async` 加上共享内存读取之后，每个线程的 32 位寄存器里已经是它在 `mma.m16n8k16` 里要用的 8 个 int4。torchao 的 tinygemm（`TensorCoreTiledLayout`）即使在 $M=1$ 时也按 MMA 的 tile 存，Machete（Hopper 专用）同样预先重排。重排把解码的布局成本挪到了加载时，代价是一份权重只能服务一种 kernel；要在两个 kernel 之间切换，就得存两份或者在线重排。这一点是 §研究问题 (b) 的出发点。

> [!question] 随堂自测 D2-3：解码指令
>
> 1. 验证 `0xD480` 是 FP16 的 $-72$、`0x2C00` 是 $1/16$。
> 2. 一个寄存器 `q = 0x76543210`（8 个码从低到高是 0..7）。`(q & 0x000F000F) | 0x64006400` 得到哪两个 FP16 值？减去 `0x64086408` 后呢？
> 3. 按每权重指令数估算：batch 1 时，`bfe` 版 INT4 kernel 每权重约 5 条（含 FMA 和地址计算），`lop3` 版约 2 条。按 §1.2 的 11 条上限，两者在 batch 多大时会被指令发射卡住？

> [!tip]- 核对要点
>
> 1. `0xD480`：符号 1，指数字段 $\mathtt{0x15}=21$，$2^{21-15}=64$，尾数 $\mathtt{0x080}/1024=0.125$，值 $-64\times1.125=-72$。`0x2C00`：指数 11，$2^{-4}=1/16$。
> 2. 低半字的低半字节是 0，高半字（`0x7654`）的低半字节是 4，得到 $\mathrm{half2}(1028,1024)$（高、低），减 1032 后是 $(-4,-8)$，即码 4 和 0 各减 8。
> 3. 每多一个 token 多 1 条 FMA：`bfe` 版 $4+B\le11$ 给出 $B\le7$，`lop3` 版 $1+B\le11$ 给出 $B\le10$。这只是发射槽的上限，实际还受寄存器、访存延迟和 warp 归约影响。

## §4 kernel 结构：warp、split-K、token 复用

### 4.1 一个 warp 负责一行

第 $r$ 行的输出是

$$
y_r=\sum_{c=0}^{K-1}(u_{r,c}-8)\,s_{r,\lfloor c/g\rfloor}\,x_c.
$$

lab2 B2 让一个 warp 的 32 个 lane 分别处理列 $l,l+32,\dots$，各自累加后用 `__shfl_down_sync` 做五轮（16、8、4、2、1）树形归约，lane 0 得到整行的和。

```text
lane 0: columns 0, 32, 64, ...        each lane: private partial sum
lane 1: columns 1, 33, 65, ...
...
lane31: columns 31, 63, 95, ...
        | shuffle offsets 16 -> 8 -> 4 -> 2 -> 1
        v
lane 0 holds y[row]
```

两条不变量：行越界（$\mathrm{row}\ge M$）时整个 warp 一起返回；有效行内没有列可处理的 lane（$K<32$ 时）不能提前返回，必须以 0 参加归约，否则全掩码的 shuffle 就有不参与的线程。[NVIDIA 的 warp 原语博客](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)开头的归约树图和 Listing 1 画的就是这个过程。

B3 改成每个 lane 读一个字节、处理相邻两列，相邻 lane 不再重复加载同一个字节。两列可能属于不同的组：$g=3$ 时第 2、3 列在同一个字节里，却分属组 0 和组 1。

### 4.2 split-K：行少列多时增加并行度

$M$ 小、$K$ 大时（down_proj 的 $M=2048,K=6144$，24 个 SM 每个只分到不足 100 行），一个 warp 一行的并行度不够。split-K 把 $K$ 切成 $P=\lceil K/\mathrm{chunk}\rceil$ 段，一个 warp 负责（行，段），写 partial$[p\cdot M+r]$；第二个 kernel 按段的顺序把 $P$ 个 partial 加起来。代价是多写读 $2\cdot4MP$ 字节和一次 launch，partial 不归约（或少归约一段）就是 lab2 的 fault `b6-partial-not-reduced`。固定的归约顺序让结果可复现，但它的 FP32 舍入和 B2 不同，所以测试比较的是对 FP64 的误差界，而不是两个 kernel 逐位相等。GemLite 和 FLUTE（Stream-K）都在小 batch 时用某种 K 维切分。

### 4.3 token 复用：batch 变大时少读几遍权重

batch 为 $B$ 时，若 B2 逐 token 启动，权重要读 $B$ 遍。B4 让 `grid.y` 选 4 个 token 一组的 tile，每个解码出的权重在寄存器里给 4 个 token 用，权重只读 $\lceil B/4\rceil$ 遍；R4 对 FP8 用 8 个 token。代价是每个 lane 要 4（或 8）个累加器，batch 很大时还是不如 Tensor Core 的分块。最后一个 tile 不满时，越界 token 不读不写，但所有 lane 仍做完全部归约。

> [!question] 随堂自测 D2-4：参与与布局
>
> 1. $K=65$ 时 lane 0 和 lane 31 各处理哪些列？$K=1$ 时哪些 lane 必须执行 shuffle？
> 2. $M=2048,K=6144$，chunk 512，split-K 的 partial 缓冲区多大？它相对 INT4 权重字节占多少？
> 3. B4 的 tile 起点写成 `blockIdx.y * 3`，batch 7 能发现吗？batch 8 呢？

> [!tip]- 核对要点
>
> 1. lane 0：$\{0,32,64\}$；lane 31：$\{31,63\}$。$K=1$ 时只有 lane 0 有数据，但 32 个 lane 都要参加归约。
> 2. $P=12$，partial $4\times2048\times12=96\,\mathrm{KiB}$，写一遍读一遍共 192 KiB；INT4 权重约 6.2 MiB，额外流量约 3%，主要代价是第二次 launch。
> 3. batch 7 需要 2 个 tile，起点 0 和 3 覆盖 $0..6$，所有 token 都算了，发现不了；batch 8 时 2 个 tile 只覆盖 $0..6$，token 7 的输出没有写，会被发现。lab2 的 B4 测试因此包含 batch 8。

## §5 测量：L2、基线与本机数据

### 5.1 小形状测到的是 L2，不是显存

4060 Laptop 的 L2 有 32 MiB。Qwen3-1.7B 的每个线性层在 INT4 下是 2–6 MiB，FP8 下 4–12 MiB，连 q_proj 的 FP16 权重（8 MiB）都放得下。连续多次调用同一个 kernel，从第二次起权重就在 L2 里，测出来的“带宽”可以超过显存带宽。真实 decode 里，两次用到同一层之间要流过所有其他层的权重，L2 早就被冲掉了。所以基准要么用比 L2 大得多的形状（$8192\times8192$ 的 FP16 权重 128 MiB），要么每次计时前在计时区间外写一块至少两倍 L2 大小的缓冲区。lab2 的 `bench_plan.hpp` 对每个形状和格式做这个检查，关掉冲刷时 R5 测试就失败。

**笔记本 GPU 的频率。** 主机在两组配置之间准备数据时 GPU 会空闲，驱动随即把显存频率从 8000 MHz 降到 810 MHz，访存受限的 kernel 因此会慢好几倍。计时前用同一个 launch 让 GPU 连续忙 300 ms 以上，每行输出 `stable` 列（$p_{75}\le1.5\,p_{25}$），详见 [authoring §5](../../labs/docs/authoring.md#5-测量规则)。

### 5.2 必须有的基线

| 基线                                       | 它回答的问题                                               | lab2 实现                                     |
| ------------------------------------------ | ---------------------------------------------------------- | --------------------------------------------- |
| FP32 cuBLAS（`cublasSgemm`）               | 不压缩、权重常驻时有多快；也是数值参考                     | `fp32_cublas_sgemm`                           |
| FP16 cuBLAS（`cublasGemmEx`，Tensor Core） | 生产里的默认做法；低比特路径必须赢过它才有意义             | `fp16_cublas_gemmex`                          |
| 先解包成 FP16 再 cuBLAS                    | 不写融合 kernel、每次调用临时解包的代价                    | `int4_unpack_fp16_gemmex`                     |
| cuBLASLt FP8（W8A8）                       | 4060 上的 FP8 Tensor Core 路径；激活也被量化，误差要单独报 | `fp8_cublaslt_w8a8`，TN、`CUBLAS_COMPUTE_32F` |
| 成熟的低比特 kernel                        | 你的 kernel 与 Marlin、tinygemm、GemLite 差多少            | 挑战 3                                        |

cuBLASLt 的 FP8 在 Ada 上有几条硬性要求（[cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul-regular-imma-conditions)）：A 必须转置、B 不转置（“TN”），计算类型 `CUBLAS_COMPUTE_32F`、scale 类型 `CUDA_R_32F`，指针和 leading dimension 16 字节对齐。行主序的 $W[M,K]$ 在 cuBLAS 的列主序视角下是 $K\times M$、`ld=K` 的矩阵，所以 $Y^\top=W X^\top$ 正好是 `TRANSA=T`、`TRANSB=N`。

### 5.3 本机实测（measured）

参考实现，RTX 4060 Laptop，2026-10-18，一次完整运行（copy 峰值 223 GB/s），L2 冲刷，300 ms 预热，25 次中位数，所有行 `stable=yes`。`frac` 是逻辑字节除以时间再除以 copy 峰值。

**batch 1，Qwen3-1.7B up_proj（$6144\times2048$）与 $8192\times8192$：**

| 路径                         | up_proj µs | frac | $8192^2$ µs | frac | 输出相对误差（up_proj / $8192^2$） |
| ---------------------------- | ---------: | ---: | ----------: | ---: | ---------------------------------: |
| INT4 B2 warp（SIMT）         |        102 | 0.29 |         569 | 0.28 |                      0.116 / 0.116 |
| INT4 B3 byte-pair（SIMT）    |         83 | 0.36 |         418 | 0.38 |                      0.116 / 0.116 |
| INT4 B4 4-token tile（SIMT） |        141 | 0.21 |         762 | 0.21 |                      0.116 / 0.116 |
| FP8 R3 GEMV（SIMT）          |         87 | 0.67 |         384 | 0.81 |                      0.022 / 0.024 |
| FP16 cuBLAS（Tensor Core）   |        139 | 0.81 |         574 | 1.05 |                    0.0002 / 0.0003 |
| FP32 cuBLAS                  |        288 | 0.78 |        1191 | 1.01 |                         $<10^{-6}$ |
| INT4 先解包再 FP16 cuBLAS    |        374 | 0.68 |        1956 | 0.70 |                      0.116 / 0.116 |
| cuBLASLt FP8 W8A8            |         77 | 0.74 |         391 | 0.77 |                      0.034 / 0.037 |

纯读的 kernel 可以略超过 1，因为 copy 是读写混合，比纯读慢一点。

**batch 扫描，$8192\times8192$（µs）：**

| batch | INT4 B4 | FP8 R4 | FP16 cuBLAS | cuBLASLt FP8 W8A8 |
| ----: | ------: | -----: | ----------: | ----------------: |
|     1 |     762 |    390 |         574 |               391 |
|     2 |     782 |    434 |         639 |               371 |
|     4 |     858 |    698 |         640 |               372 |
|     8 |    1749 |   1241 |         646 |               372 |
|    16 |    3299 |   2370 |         648 |               373 |
|    64 |   13756 |   9268 |         677 |               376 |

这些数说明了四件事。

1. **FP16 cuBLAS 到 batch 64 都几乎不变**，符合 §1.2 的 FP16 交叉点 $B^\ast\approx113$（modeled）。
2. **本 lab 的 SIMT kernel 远没有到它们的屋顶。** INT4 B4 在 batch 64 时只有约 0.6 TFLOPS，FP8 R4 约 0.9 TFLOPS，都不到 FP32 CUDA core 峰值的 10%。所以 INT4 kernel 只在 batch 1 时胜过 FP16 cuBLAS（B3 最快，B4 在 batch 4 时已经慢于 FP16），FP8 R4 的交叉点在 $8192^2$ 上是 batch 2 到 4、在 up_proj 上是 4 到 8，都比 §1.2 按屋顶线算的 12 和 23 小得多。差距来自每个权重的解码指令、每个 token 一个累加器和 warp 归约，而不是 FLOP。
3. **INT4 在 batch 1 没有赢 FP8。** 字节少一半，可 INT4 SIMT kernel 只用了约三分之一的带宽，FP8 kernel 用了约 0.7–0.8。这是一个教学 kernel 的结果（`bfe` 逐个解码，没有向量化加载和 `lop3`），不能推广到 Marlin，但它正是研究问题 (a) 要问的：在这张带宽很窄的卡上，解码成本能把 INT4 的字节优势吃掉多少。
4. **先解包再算比 FP16 还慢**，因为它多写读了一整份 FP16 矩阵（逻辑字节 $B_4+2\cdot2MK$）。只有在权重会被很多次调用复用、解包能摊销时，这种做法才划算。

> [!question] 随堂自测 D2-5：读一张基准表
>
> 1. 某同学的 INT4 kernel 在 $2048\times2048$ 上测到 400 GB/s。列出至少两种可能的原因，并说明哪个检查能排除它们。
> 2. 从上表估算 FP8 R4 相对 FP16 cuBLAS 的 measured 交叉 batch，并与 §1.2 的 modeled 值比较。差距该归到哪里？
> 3. cuBLASLt FP8 W8A8 在 batch 1 最快，能不能据此说“FP8 权重量化在 4060 上最好”？

> [!tip]- 核对要点
>
> 1. 权重留在 L2 里（形状小于 32 MiB 又没有冲刷），或者字节口径算多了（比如把逻辑字节按 FP16 算）。看形状对 L2 的比例、打开冲刷重测；按格式重算字节。
> 2. $8192^2$ 上 batch 2 时 FP8 R4 是 434 µs，FP16 是 639 µs；batch 4 时 698 对 640，交叉在 2 到 4 之间。modeled 的 W8A32 SIMT 交叉点约 23。差距来自 kernel 远没到 CUDA core 的峰值：每权重的转换和 scale、每 token 的累加和归约都占指令，而屋顶线只算 FMA。
> 3. 不能。W8A8 还量化了激活，输出误差 0.037 比 FP8 weight-only 的 0.024 大，模型质量要单独评测；它用的是 Tensor Core，和 SIMT 的 R3 不是同一种实现。结论只能是“在这些形状和 batch 上，cuBLASLt 的 W8A8 延迟最低”。

## §6 量化算法地图：算法改了什么，kernel 要配合什么

### 6.1 只量化权重的 PTQ

| 方法                                                           | 机制                                                                                                                                              | 产出的格式                  | kernel 要配合什么                                                 |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- | ----------------------------------------------------------------- |
| RTN                                                            | 组内 absmax 或 min/max 直接舍入                                                                                                                   | INT4 对称或非对称           | 无                                                                |
| [GPTQ](https://arxiv.org/abs/2210.17323)（ICLR 2023）          | 按列量化，用逆 Hessian 把每列的误差推给还没量化的列；可按激活排序（act-order）                                                                    | 非对称 INT4 加零点，$g=128$ | act-order 打乱了组和列的对应，kernel 要额外读一张“列到组”的索引表 |
| [AWQ](https://arxiv.org/abs/2306.00978)（MLSys 2024 最佳论文） | 用激活幅度找出约 1% 的重要输入通道，等价缩放 $W\,\mathrm{diag}(s)$、$X\,\mathrm{diag}(s)^{-1}$，$s=\overline{\lvert x\rvert}^{\alpha}$ 在网格上搜 | 非对称 INT4                 | 缩放折进前一层，kernel 不变（TinyChat 自带 W4A16 kernel）         |
| [HQQ](https://dropbox.github.io/hqq_blog/)（博客 2023）        | 不要校准数据，用半二次求解器在 $\ell_p$（$p<1$）损失下优化零点                                                                                    | 非对称，可到 1–4 bit        | 无                                                                |
| [OmniQuant](https://arxiv.org/abs/2308.13137)（ICLR 2024）     | 可学习的权重裁剪加可学习的等价变换，逐块优化                                                                                                      | INT4/INT3/INT2              | 无                                                                |
| [AdaDim](https://arxiv.org/abs/2309.15531)（ICLR 2024）        | 沿输入通道而不是输出通道分组，把激活离群的列隔进一个组                                                                                            | 按层选轴                    | 组的方向变了，GEMV 的 scale 访问模式也变                          |

lab2 C1–C5 实现的就是 AWQ 的核心：统计量 $m_k=\frac1T\sum_t\lvert x_{t,k}\rvert$，$s_k=m_k^\alpha$ 再按 $\sqrt{\max s\cdot\min s}$ 归一化，在 $\alpha\in\{0,0.25,0.5,0.75,1\}$ 上按原始浮点输出的 MSE 穷举，$\alpha=0$ 就是 RTN。为什么有效：激活大的通道 $k$，权重误差 $\Delta w_{r,k}$ 对输出的贡献是 $\Delta w_{r,k}x_k$，被 $x_k$ 放大。把 $w_{\cdot,k}$ 乘以 $s_k>1$，它在组内占的相对份额变大，舍入误差相对它自己变小；输入端除以 $s_k$ 把数值还原。代价是同一组里其他通道的步长变大，所以 $\alpha$ 要搜。lab2 的校准表里，四个宽度上同分布的 held-out 误差都比 RTN 低；放大的通道一换位置，其中三个宽度的误差反而比 RTN 高 2.7 到 14 倍：缩放是针对校准分布定的。

### 6.2 权重和激活都量化

[LLM.int8()](https://arxiv.org/abs/2208.07339)（NeurIPS 2022）把激活里少数离群的特征维拆出来用 FP16 算，其余用 INT8。[SmoothQuant](https://arxiv.org/abs/2211.10438)（ICML 2023）用和 AWQ 一样的等价缩放，但方向相反：把激活的离群值“搬”进权重，让 W8A8 两边都好量化。[QServe](https://arxiv.org/abs/2405.04532)（MLSys 2025）做 W4A8KV4：权重先按通道量化到 INT8，再按组量化到 INT4（progressive），反量化到 INT8 时先乘后减以免溢出，配合 SmoothAttention 和寄存器级并行；论文在 A100 上给出 W4A16 与 W8A8 的交叉点约在 batch 78。[Atom](https://arxiv.org/abs/2310.19102)（MLSys 2024）做 W4A4，把离群通道重排到一起用 INT8。

### 6.3 旋转：把离群值摊平

[QuaRot](https://arxiv.org/abs/2404.00456)（NeurIPS 2024）在残差流上乘随机 Hadamard 矩阵 $H$（$H^\top H=I$），把它折进相邻的权重，计算不变；旋转后激活和权重的离群值被摊到所有通道，W4A4KV4 才可行。[SpinQuant](https://arxiv.org/abs/2405.16406)（ICLR 2025）把旋转当参数，在 Stiefel 流形上用 Cayley SGD 学。在线的那部分 Hadamard 变换要进 kernel（或单独一个快速 Hadamard kernel）。2026 年的综述 [Transforms for LLM Quantization](https://arxiv.org/abs/2608.25188) 指出，在逐组 absmax 下，“能量集中”的 KLT 式变换和“组内摊平”的 Hadamard 是相反的处方，MXFP4 和 NVFP4 的组大小又会改变哪一个更好。

### 6.4 向量量化与 trellis

[QuIP#](https://arxiv.org/abs/2402.04396)（ICML 2024）先做随机 Hadamard 的非相干处理，再用 E8 格点码本把 8 个权重一起量化；[AQLM](https://arxiv.org/abs/2401.06118)（ICML 2024）学多个加性码本；[QTIP](https://arxiv.org/abs/2406.11235)（NeurIPS 2024）用 trellis 编码，在不存大码本的情况下做高维向量量化。它们在 2 bit 附近比标量格式好得多，但解码是查表或算码，和 §3 的位运算完全不同，kernel 往往成为瓶颈。

### 6.5 查表 kernel：不反量化

[LUT-GEMM](https://arxiv.org/abs/2206.09557)（ICLR 2024）用二值编码量化，预先算好激活子向量的所有部分和，GEMM 变成查表加法；[T-MAC](https://arxiv.org/abs/2407.00088)（EuroSys 2025）在 CPU 上用 `tbl`/`pshufb` 做按位的查表 mpGEMM；[FLUTE](https://arxiv.org/abs/2407.10960)（EMNLP 2024 Findings）在 GPU 上把查找表复制进共享内存，支持 NF4 这类任意码表；[Ladder/BitBLAS](https://github.com/microsoft/BitBLAS)（OSDI 2024）用 tType 抽象和硬件感知的布局变换，为自定义低比特类型生成 kernel；[Tilus](https://arxiv.org/abs/2504.12984)（ASPLOS 2026）是面向任意 1–8 bit 的 tile 级 GPU 编程语言。

### 6.6 生产后端

| 后端                                                                                                       | 格式             | 关键实现                                                             | 4060 上                                                  |
| ---------------------------------------------------------------------------------------------------------- | ---------------- | -------------------------------------------------------------------- | -------------------------------------------------------- |
| [Marlin](https://github.com/IST-DASLab/marlin)（PPoPP 2025）                                               | W4A16，$g=128$   | 离线重排、`cp.async` 多级流水、`lop3` 解码、条带式 SM 划分           | 支持 sm_80+（Ampere、Ada）                               |
| Machete（vLLM，Neural Magic 2024）                                                                         | W4A16 等混合输入 | wgmma、TMA、warp 特化                                                | 只支持 SM90，不能在 4060 上跑                            |
| [vLLM FP8 Marlin](https://github.com/vllm-project/vllm/pull/5975)                                          | W8A16            | FP8 打包成 Marlin 的 int32 格式，位运算升成 FP16                     | sm_89 上默认走 W8A8，需要 `VLLM_TEST_FORCE_FP8_MARLIN=1` |
| [torchao](https://github.com/pytorch/ao) tinygemm                                                          | W4A16            | `_weight_int4pack_mm`，权重按 MMA tile 存                            | sm_80+；batch 1 最好，batch 变大退化                     |
| [GemLite](https://github.com/dropbox/gemlite)                                                              | 1–8 bit          | Triton；按 batch 和位宽选 GEMV / split-K GEMV / GEMM，自动调 SPLIT_K | 支持                                                     |
| llama.cpp                                                                                                  | k-quant、i-quant | 每种格式一个 CUDA 解码与点积 kernel                                  | 支持，本地推理最常用                                     |
| [ExLlamaV2](https://github.com/turboderp-org/exllamav2) / [V3](https://github.com/turboderp-org/exllamav3) | EXL2、EXL3       | 混合位宽；EXL3 为 `mma.m16n8k16` 组织的 $16\times16$ tile            | 支持                                                     |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)                                    | NF4、INT8        | 查表解码；LLM.int8() 的离群拆分                                      | 支持                                                     |

> [!question] 随堂自测 D2-6：等价缩放
>
> 1. $W=[2,3]$、$x=[4,5]$、$s=[2,0.5]$，算 $W'$、$x'$ 和变换前后的点积。为什么等价之后还要搜 $s$？
> 2. AWQ 和 SmoothQuant 都做 $W\,\mathrm{diag}(s)$、$X\,\mathrm{diag}(s)^{-1}$。各自希望 $s$ 在激活大的通道上大还是小？为什么？
> 3. 搜索候选里包含 $\alpha=0$，能保证 held-out 误差不比 RTN 差吗？

> [!tip]- 核对要点
>
> 1. $W'=[4,1.5]$，$x'=[2,10]$，两个点积都是 23。等价只在浮点下成立，量化后各通道的舍入误差随 $s$ 改变，要实际比较。
> 2. AWQ 只量化权重，希望激活大的通道的权重更精确，$s$ 取大（权重放大、占组内更大份额）；SmoothQuant 要让激活也好量化，在激活大的通道上 $s$ 也取大，但目的是把激活除小，把难度转给权重。两者的 $s$ 公式不同：AWQ 是 $\overline{\lvert x\rvert}^\alpha$，SmoothQuant 是 $\max\lvert x\rvert^\alpha/\max\lvert w\rvert^{1-\alpha}$。
> 3. 不能。$\alpha=0$ 只保证校准集上不比 RTN 差，lab2 的校准表里分布一偏移，多数宽度上就比 RTN 差好几倍。

## §7 机制与代码的对应

| 机制或公式                                        | lab2 中的函数                                | 上游实现位置                                                                                  | 本 lab 简化了什么                          |
| ------------------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------ |
| 组内 $s=\max\lvert w\rvert/7$，$u=q+8$            | `calibrate`、`pack`（A1/A2）                 | llm-awq `awq/quantize/quantizer.py`（固定版本见[代码地图](../readings/code-map.md)）          | 对称、FP32 scale，无零点                   |
| 逐组 clipping                                     | `clip_pack`（A6）                            | llm-awq `auto_clip.py`                                                                        | 只按权重 SSE，不按输出                     |
| `bfe.u32` 解包、warp 归约                         | `nibble`、`fused_gemv`（B1/B2）              | llm-awq `awq/kernels/csrc/quantization/gemv_cuda.cu` 的 `gemv_kernel_g128`                    | FP32 激活，逐个解包                        |
| `lop3` 加 `0x64006400`                            | 挑战 2                                       | Marlin `marlin_cuda_kernel.cu::dequant`；FasterTransformer `interleaved_numeric_conversion.h` | 本 lab 布局不交织，不能直接套用            |
| split-K                                           | `split_partial`、`merge_partial`（B5/B6）    | GemLite `gemlite/core.py` 的 GEMV split-K 分支                                                | 固定 chunk，不自动调                       |
| 解码后跨 token 复用                               | `tiled_batch`（B4）、`fp8_tiled_batch`（R4） | Marlin 的 token 维分块                                                                        | 寄存器里 4 或 8 个 token，没有 Tensor Core |
| $s_k=\overline{\lvert x\rvert}^\alpha$ 与网格搜索 | `channel_scales`、`search`（C1/C3）          | llm-awq `auto_scale.py::_search_module_scale`                                                 | 单层，不折进前一层                         |
| E4M3 RNE 加饱和                                   | `encode_e4m3`（R1）                          | `cuda_fp8.h::__nv_cvt_float_to_fp8`                                                           | 主机上实现，GPU 指令作参照                 |
| 128 位加载加 `cvt.e4m3x2`                         | `fp8_gemv`（R3）                             | vLLM `marlin_utils_fp8.py`（W8A16 打包）                                                      | SIMT，W8A32                                |
| cuBLASLt FP8 TN                                   | `baselines.cuh::Fp8Lt`                       | cuBLASLt 文档的 FP8 条件；NVIDIA CUDALibrarySamples `LtFp8Matmul`                             | 逐张量 scale                               |

## 前沿与 baseline 对照

核验日期：2026-10-18（逐条打开 arXiv 页面和代码仓库，会议信息来自论文页、会议官网或 dblp）。

| 工作（会议 年）                                                                                                                                                                                                                    | 问题                                 | 关键机制                                                               | 与本 lab 关系（实现/简化/未覆盖）               | 研究时必须对比                  |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | ---------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------- |
| GPTQ（ICLR 2023）[论文](https://arxiv.org/abs/2210.17323) · [代码](https://github.com/IST-DASLab/gptq)                                                                                                                             | 一次性 4 bit PTQ                     | 逆 Hessian 误差补偿，逐列量化                                          | 未覆盖                                          | 同字节下与 AWQ、小组 RTN 的质量 |
| AWQ（MLSys 2024）[论文](https://arxiv.org/abs/2306.00978) · [代码](https://github.com/mit-han-lab/llm-awq)                                                                                                                         | 保护重要权重而不混合精度             | 激活幅度驱动的等价通道缩放，网格搜 $\alpha$                            | 实现：C1–C5（单层）                             | held-out 与分布偏移下的误差     |
| SmoothQuant（ICML 2023）[论文](https://arxiv.org/abs/2211.10438) · [代码](https://github.com/mit-han-lab/smoothquant)                                                                                                              | W8A8 的激活离群值                    | 把激活难度迁给权重                                                     | 未覆盖                                          | W8A8 与 W4A16 的交叉 batch      |
| QuaRot（NeurIPS 2024）[论文](https://arxiv.org/abs/2404.00456) · [代码](https://github.com/spcl/QuaRot)                                                                                                                            | 4 bit 激活的离群值                   | 随机 Hadamard 旋转，计算不变                                           | 未覆盖                                          | 在线 Hadamard 的 kernel 开销    |
| SpinQuant（ICLR 2025）[论文](https://arxiv.org/abs/2405.16406) · [代码](https://github.com/facebookresearch/SpinQuant)                                                                                                             | 随机旋转不稳定                       | 学出来的旋转（Cayley SGD）                                             | 未覆盖                                          | 同上                            |
| QuIP#（ICML 2024）[论文](https://arxiv.org/abs/2402.04396) · [代码](https://github.com/Cornell-RelaxML/quip-sharp)；QTIP（NeurIPS 2024）[论文](https://arxiv.org/abs/2406.11235) · [代码](https://github.com/Cornell-RelaxML/qtip) | 2 bit 下的质量                       | Hadamard 非相干处理加 E8 格点或 trellis 码                             | 未覆盖                                          | 解码 kernel 的带宽利用率        |
| QServe（MLSys 2025）[论文](https://arxiv.org/abs/2405.04532) · [代码](https://github.com/mit-han-lab/omniserve)                                                                                                                    | W4A8KV4 的服务吞吐                   | 两级量化、先乘后减、寄存器级并行                                       | 简化：B 只做 W4A32                              | 每权重解码指令数                |
| Marlin（PPoPP 2025）[论文](https://arxiv.org/abs/2408.11743) · [代码](https://github.com/IST-DASLab/marlin)                                                                                                                        | W4A16 在 batch 16–32 内保持 4 倍加速 | 离线重排、`lop3` 解码、`cp.async` 流水、条带划分                       | 简化：B 的 SIMT 版；挑战 2/3                    | 4060 上的成熟 INT4 基线         |
| FLUTE（EMNLP 2024 Findings）[论文](https://arxiv.org/abs/2407.10960) · [代码](https://github.com/HanGuo97/flute)                                                                                                                   | 任意码表（NF4 等）的 GPU GEMM        | 共享内存里复制的 LUT、Stream-K                                         | 未覆盖                                          | 非均匀码与均匀 INT4 的速度      |
| T-MAC（EuroSys 2025）[论文](https://arxiv.org/abs/2407.00088) · [代码](https://github.com/microsoft/T-MAC)                                                                                                                         | CPU 上的低比特 mpGEMM                | 按位查表，无反量化                                                     | 未覆盖                                          | —                               |
| Ladder/BitBLAS（OSDI 2024）[代码](https://github.com/microsoft/BitBLAS)                                                                                                                                                            | 自定义低比特类型的 kernel 生成       | tType 与硬件感知的布局变换                                             | 未覆盖                                          | 按形状调优后的 kernel           |
| Tilus（ASPLOS 2026）[论文](https://arxiv.org/abs/2504.12984) · 代码未核实                                                                                                                                                          | 任意位宽的 GPU 编程                  | tile 级 VM、代数布局                                                   | 未覆盖                                          | 编译生成与手写 kernel           |
| LiquidGEMM（SC 2025）[论文](https://arxiv.org/abs/2509.01229) · 代码未核实                                                                                                                                                         | W4A8 GEMM 的解码开销                 | 每 4 个元素 2 条指令的防溢出解码、隐式流水                             | 未覆盖（Hopper 流水），解码指令计数的论证可移植 | 每权重指令数                    |
| MicroMix（ICLR 2026）[论文](https://arxiv.org/abs/2508.02343) · [代码](https://github.com/lwy2020/MicroMix)                                                                                                                        | MX 格式的混合精度                    | 按误差阈值逐通道分配 MXFP4/6/8                                         | 未覆盖（Blackwell）                             | —                               |
| Quartet（NeurIPS 2025）[论文](https://arxiv.org/abs/2505.14669) · [代码](https://github.com/IST-DASLab/Quartet)                                                                                                                    | 原生 FP4 训练                        | 全部 matmul 用 MXFP4                                                   | 未覆盖（Blackwell）                             | —                               |
| “Give Me BF16 or Give Me Death”?（arXiv 2024）[论文](https://arxiv.org/abs/2411.02355) · 无公开代码                                                                                                                                | 量化格式的精度与性能取舍             | vLLM 上大规模端到端评测：同步低延迟时 W4A16 最好，异步吞吐时 W8A8 最好 | 未覆盖；R 基准是其 kernel 级缩影                | 没有测 Ada                      |
| Win Fast or Lose Slow（arXiv 2025）[论文](https://arxiv.org/abs/2505.19481) · 无公开代码                                                                                                                                           | 延迟敏感任务里速度与精度的平衡       | RTX 5090 上 Qwen-7B 的 FP8 比 W4A16 快                                 | 未覆盖                                          | 消费级 GPU 上 INT4 不一定赢     |
| BitDecoding（arXiv 2025）[论文](https://arxiv.org/abs/2503.18773) · [代码](https://github.com/OpenBitSys/BitDecoding)                                                                                                              | 低比特 KV 用上 Tensor Core           | TC 友好布局、warp 并行解码                                             | 未覆盖（见 D1 R）                               | 融合权重与 KV 解码              |
| Transforms for LLM Quantization（arXiv 2026）[论文](https://arxiv.org/abs/2608.25188) · 无公开代码                                                                                                                                 | 43 种变换方法怎么选                  | 能量集中与组内摊平在逐组 absmax 下相反                                 | 未覆盖                                          | 变换与组大小、格式的交互        |
| GemLite（库）[代码](https://github.com/dropbox/gemlite)                                                                                                                                                                            | 不同 batch 用不同低比特 kernel       | 按 batch 与位宽选 GEMV / split-K / GEMM，自动调 SPLIT_K                | 简化：B 的四条路径，不自动选                    | 分发器的选择是否接近每形状最优  |

**查新结论**：按“对苗老师组的契合度、8 GB 显存上的可行性、能否复用 lab2 代码”排序，四个候选题的顺序是 (b) > (a) > (d) > (c)。

- **(b) 低比特 kernel 的形状相关分发与自动调优，最值得做。** 最近的工作：GemLite 按 batch 和位宽选 GEMV、split-K GEMV 或 GEMM，并自动调 SPLIT_K；vLLM 的 `choose_mp_linear_kernel` 按硬件能力和整除条件从一个固定优先级列表（CutlassW4A8、Machete、AllSpark、Marlin……Exllama）里挑第一个能用的，不看 batch；DietCode（MLSys 2022）给动态形状做形状通用的搜索空间；BitBLAS/Ladder 和 TileLang 按形状编译低比特 kernel。我没有找到同时按格式（组大小、INT4 还是 FP8）、batch 和层形状三者分发的工作，也没有找到把“同一份权重要不要存两种布局”的显存和重排成本放进代价模型的工作，而这在 8 GB 的卡上正是约束。§5.3 的表已经显示：在 4060 上，最快的路径随 batch 从 FP8 SIMT 换到 FP16 cuBLAS，交叉点只有 2–4。只做“按 batch 选 kernel”会和 GemLite 撞车。
- **(a) Ada 上小 batch 的 FP8 对 INT4，是最便宜的第一步。** “Give Me BF16 or Give Me Death”? 做了端到端评测，但没有 Ada 卡也没有 kernel 级拆解；QServe 给出 A100 上 W4A16 与 W8A8 的交叉点约 78，并指出 KV4 解码在 A100 上受算力限制，在 L40S（Ada）上不受；vLLM 同时有 FP8 Marlin 和 cuBLASLt FP8，但没有公开的 Ada 上对 INT4 Marlin 的比较；Win Fast or Lose Slow 在 RTX 5090 上看到 FP8 比 W4A16 快。空白是带宽只有 4090 约四分之一的消费级 Ada 卡上的 kernel 级交叉点，以及一个能用公开规格预测其他卡的模型（交叉 batch 作为带宽、Tensor Core 速率、每字节解码指令数的函数）。它同时是 (b) 所需的测量工具。
- **(d) 组大小与激活感知缩放的取舍**：AWQ 的消融只用 $g=128$，在那里 RTN 到 AWQ 的提升已经很小；OmniQuant 报告了 $g=64$ 和 $g=128$；AdaDim 换了分组的轴；2026 年的 [The Structure of Quantization Damage in LLMs](https://arxiv.org/abs/2609.01587)（workshop 投稿）认为同预算下全模型更细的粒度胜过针对性的逐层修复。在同总字节下比较 $g=32$ RTN、$g=128$ AWQ、$g=64$ AWQ 的工作我没有找到，lab2 C5 就是这个实验的雏形；但它偏算法，和组里的方向契合度较弱。
- **(c) 低比特权重 GEMV 与低比特 KV attention 融合成一个持久 kernel**：QServe 两者是分开的 kernel；BitDecoding 只做 KV；MPK（OSDI 2026，苗旭鹏为作者之一）和 Hazy Research 的 Llama-1B megakernel 把整个 decode 编成一个持久 kernel，但都是 FP16/BF16。空白真实存在，但工程量最大，还要证明收益超出 CUDA Graph 已经给的部分。

苗组相关工作：苗旭鹏老师[主页](https://hsword.github.io/publications/)上没有低比特 kernel 或权重量化格式的工作，最接近的量化论文是 Quantized Side Tuning（ACL 2024，量化大模型的高效微调），与本讲的推理 kernel 关系不大。组里的 kernel 工作集中在编译与超优化：Mirage（OSDI 2025）、MPK（OSDI 2026，[arXiv](https://arxiv.org/abs/2512.22219)）、Event Tensor（MLSys 2026）。所以 (b) 最好表述成“低比特 kernel 作为编译和分发对象”：例如在 megakernel 里放一个混合输入的算子，或对“解码加 GEMV”的子图做超优化。这是我的判断，不是组里公开说过的方向。

## 研究问题

- **候选假设**（对应 (a)，并为 (b) 打基础）：在 RTX 4060 Laptop 上，对 Qwen3-0.6B/1.7B 的全部线性层形状，FP8 weight-only 路径相对 INT4 weight-only 路径的速度比随 batch 的变化，能被“带宽、计算单元速率、每权重解码指令数”三参数模型预测到 15% 以内；并且存在一段 batch 区间（预期在 1–16），其中 FP8 比 INT4 快。
- **最小实验**：从 lab2 R 基准出发，加入一个用 `lop3` 和向量化加载的 INT4 kernel（挑战 2），以及 Marlin 或 torch `_weight_int4pack_mm`（挑战 3）作为成熟 INT4 基线，再加 FP8 Tensor Core 的 W8A8（挑战 1）和 FP16 cuBLAS。形状取 Qwen3 两个模型的 q/k/v/o/gate/up/down，batch 1–64，每个配置 5 次独立运行，冲掉 L2，记录 `stable` 和时钟。用 `cuobjdump -sass` 数每权重指令数作为模型输入，拟合两张卡（4060 和一张公开数据足够的卡）后检验预测。
- **继续或停止的证据**：如果成熟的 INT4 kernel（Marlin）在 4060 上所有 batch 都胜过 FP8 路径，说明本 lab 看到的 FP8 优势只来自教学 kernel 写得差，(a) 应停止，把测量工具转给 (b)。如果交叉点随形状明显移动、模型能预测，就继续做 (b) 的分发器：在真实 decode 的形状分布上，以“每层选最快的路径”为上界，以“一律 Marlin”和 GemLite 为基线，计入两份布局的显存成本。若分发器的收益小于 5% 或被重排成本抵消，就停下，整理成负结果。
- **与导师讨论的问题**：组里的 Mirage/MPK 能否表示混合输入的算子，还是需要扩展；是否有比 4060 更合适的 Ada 卡（L40S、4090）可以借来验证模型；分发器应该以单 kernel 时间为目标，还是以 megakernel 里的端到端 decode 时间为目标。

## 资料汇总

下表的“指定观察”已计入本讲时间，通读论文或运行上游代码属于选读或 lab 挑战。

| 资料                                                                                                                                                                                                               | 类型 | 指定观察                                                                                                                         | 时间  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---- | -------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [Marlin `marlin_cuda_kernel.cu`](https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda_kernel.cu)                                                                                                    | 代码 | `dequant` 的六个常数，用 §3.3 的推导逐个核对；数每 4 个权重的指令                                                                | 20min |
| [Marlin README](https://github.com/IST-DASLab/marlin)                                                                                                                                                              | 图解 | `assets/peak.png`：不同 batch 下与其他 4-bit kernel 的比较（A10，$g=128$）；读出接近 4 倍的 batch 范围，与 §1.2 的 $B^\ast$ 对照 | 10min |
| [FasterTransformer `interleaved_numeric_conversion.h`](https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h) | 代码 | `I4s_TO_F16s_MAGIC_NUM` 与 int8 的 `prmt` 路径                                                                                   | 10min |
| [AWQ `auto_scale.py`](https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py)                                                                                                                 | 代码 | `_search_module_scale`：统计量、网格、loss 对哪个输出算                                                                          | 15min |
| [cuBLAS 文档，FP8 条件](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul-regular-imma-conditions)                                                                                                     | 文档 | Ada 上的 TN 要求、计算类型、对齐                                                                                                 | 10min |
| [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)                                                                                                                                       | 文档 | `bfe`、`lop3`、`prmt` 的定义；§9.7.16.5.10 `m16n8k32` 的 FP8 片段图                                                              | 15min |
| [Transformer Engine FP8 入门](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.13/user-guide/examples/fp8_primer.html)                                                                   | 图解 | 图 1：0.3952 在 FP16/BF16/E4M3/E5M2 下的编码；图 6 E8M0、图 8 E2M1                                                               | 10min |
| [NVFP4 博客](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)                                                                                              | 图解 | 图 2 两级 scale；图 5 MXFP4 的 32 元素块与 NVFP4 的 16 元素块                                                                    | 10min |
| [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)                                                                                                      | 图解 | “Symmetric/Asymmetric Quantization”与“Range Mapping and Clipping”两节；GGUF 的超块图                                             | 15min |
| [HF：4-bit 与 QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)                                                                                                                                   | 图解 | “FP8 format”一节的 E4M3/E5M2 图；NF4 与嵌套量化                                                                                  | 10min |
| [NVIDIA warp 原语](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)                                                                                                                            | 图解 | 开头的归约树和 Listing 1；对照 B2                                                                                                | 10min |
| [Modal GPU Glossary：roofline](https://modal.com/gpu-glossary/perf/roofline-model)                                                                                                                                 | 图解 | 脊点的定义和 A100、H100 的数值；自己补上 4060 一行                                                                               | 5min  |
| [MIT 6.5940 Fall 2026](https://hanlab.mit.edu/courses/2026-fall-65940)                                                                                                                                             | 课程 | Lecture 5/6 Quantization、Lecture 13 LLM Quantization（选读）                                                                    | —     |
| [GPU MODE lectures](https://github.com/gpu-mode/lectures)                                                                                                                                                          | 课程 | L7 Advanced Quantization、L34 Low Bit Triton Kernels（GemLite）（选读）                                                          | —     |
| [Lab 2](../../labs/lab2/README.md)、[资源与改编说明](../../labs/lab2/resources.md)                                                                                                                                 | 实验 | 对照 §7 的函数表                                                                                                                 | —     |

## 自测清单

- [ ] 不看讲义，写出 $b_w=b/8+b_s/g$ 和 $B^\ast=I^\ast b_w/2$，算出 FP16、W4A16、W8A8 在 4060 上的 modeled 交叉 batch。
- [ ] 手算一个 INT4 组（含 ties-away 平局）和三个 E4M3 码（含饱和与 subnormal）。
- [ ] 推导 `0x6400` magic number 和 Marlin 的 `SUB`、`MUL`、`ADD` 三个常数，说明为什么需要 $[0,2,4,6,1,3,5,7]$ 的交织。
- [ ] 说明 warp 归约的参与规则、split-K 的 partial 布局和 token tile 的越界规则，各指出由 lab2 哪个测试检查。
- [ ] 解释为什么小形状的基准会测到 L2，说出至少两种避免的办法。
- [ ] 读 §5.3 的表，说出 measured 交叉 batch 和 modeled 值的差距来自哪里。
- [ ] 对照前沿表中至少三篇工作，各写一句机制差异和本 lab 的简化边界。
- [ ] 写出研究问题 (a) 或 (b) 的一个可证伪假设、必须对比的 baseline 和停止条件。

[← 课程索引](../index.md) · [配套 lab2](../../labs/lab2/README.md) · [← D1 KV 缓存](01-kv-cache.md) · [D3 动态编译 →](03-dynamic-compilation.md)
