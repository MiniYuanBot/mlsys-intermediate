---
tags: [mlsys, lab, quantization, cuda, ptx, fp8]
---

# Lab 2：从 int4 格式到 FP8 kernel，以及和 cuBLAS 的同口径对比

[课程 D2](../../course/modules/02-low-bit-kernels.md) · [实验索引](../index.md) · [评分标准](grade.md) · [资源与改编说明](resources.md) · [参考导读](../../solutions/lab2/README.md)

> [!important] 完成定义
> A 和 C 用 C++17 在 CPU 上完成：一个能逐字节核对的分组对称 int4 格式、逐组 clipping、AWQ 式的激活感知通道缩放和字节预算搜索。B 和 R 用 CUDA 在 WSL 的 RTX 4060 Laptop（sm_89）上完成：四个 W4A32 SIMT kernel（PTX `bfe` 解包、warp 归约、byte-pair、4 token 复用、split-K），以及 FP8 E4M3 权重的编码器、打包和寄存器内转换的 GEMV/小 batch kernel。基准把它们和 FP32 cuBLAS、FP16 cuBLAS（Tensor Core）、“先解包成 FP16 再 cuBLAS”、cuBLASLt FP8 W8A8 放在同一张表里，形状大于 L2 或每次计时前冲掉 L2，batch 从 1 扫到 64。每个 TODO 都有独立 oracle 和至少一个错误注入。

需要先掌握 foundations 的 C++、CUDA 线程层次、warp shuffle 和基准测量。**下列命令都从 `mlsys-intermediate/labs/` 执行。** 预计 **35h**：A 6h、B 9h、C 7h、R 10h、报告 3h，另加 D2 讲义 5h。估时按独立编写和调试计算，未经学生试做验证；首次写 CUDA 的同学再加 4–8h。

## 0. 环境

本 lab 的依赖和其他 lab 互不累加。先安装、再检查：

```bash
bash tools/setup_env.sh --lab 2 && bash tools/check_env.sh --lab 2
```

Windows 原生（PowerShell）只做 CPU 部分：

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1 -Lab 2
powershell -ExecutionPolicy Bypass -File tools\check_env.ps1 -Lab 2
```

| 阶段 | 需要                                  | WSL2 / Linux | Windows 原生   | macOS       |
| ---- | ------------------------------------- | ------------ | -------------- | ----------- |
| A    | C++17                                 | 可做         | 可做           | 可做        |
| B    | C++17 + nvcc + NVIDIA GPU             | 可做         | 不支持，转 WSL | 不支持      |
| C    | C++17；GPU 部分另需 nvcc + NVIDIA GPU | 可做         | 仅 CPU 部分    | 仅 CPU 部分 |
| R    | C++17 + nvcc + NVIDIA GPU             | 可做         | 不支持，转 WSL | 不支持      |

A 和 C 的主机算法在任何平台都能跑。B、R 和 C 的 GPU 集成只在 WSL2/Linux 上跑，基准额外链接 CUDA Toolkit 自带的 `libcublas` 和 `libcublasLt`，不需要另装。缺依赖时 runner 只打印一行 `MISSING ...` 和 `fix:`，退出码 3；在 Windows 上请求 B/R，`fix:` 是一条可以直接复制的 WSL 命令。FP8 的硬件转换和 cuBLASLt FP8 需要 sm_89 及以上（Ada、Hopper）；更早的卡上 `cuda_fp8.h` 改用软件模拟，R 的测试仍能跑，但计时没有参考意义，cuBLASLt 那一行会标为不支持。各平台细节见[环境配置](../setup.md)。

## 文件和实施顺序

| 文件                                              | 学生负责                                    | 已提供                                                                   |
| ------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------ |
| `src/quant_student.hpp`                           | A1–A6 格式与 clipping，C1–C5 通道缩放与搜索 | `Packed`、`Calibrated` 等数据结构和接口契约                              |
| `src/gemv_student.cuh`                            | B1 解包，B2–B6 四条 W4A32 kernel            | launcher                                                                 |
| `src/fp8_student.hpp`                             | R1 E4M3 编码，R2 FP8 打包，R5 字节模型      | `Fp8Packed`、`Format`                                                    |
| `src/fp8_gemv_student.cuh`                        | R3 FP8 GEMV，R4 8 token 小 batch kernel     | launcher、布局说明                                                       |
| `oracle.hpp`、`check.hpp`                         | 可以追加                                    | 独立 oracle（FP64 解码、自己的打包器、穷举的 AWQ 搜索、E4M3 暴力最近值） |
| `quant_test.cpp`                                  | 可以追加反例                                | A/C 测试，逐项 PASS/FAIL；`bench` 模式输出校准表                         |
| `gpu_test.cu`                                     | 可以追加反例                                | B/C/R 测试，每组单独进程                                                 |
| `gpu_bench.cu`、`baselines.cuh`、`bench_plan.hpp` | 可以增加形状和路径                          | 计时、cuBLAS/cuBLASLt 基线、解包 kernel、形状和 L2 冲刷计划              |
| `exercise.py`                                     | 不需要修改                                  | 编译、阶段选择、GPU 分组运行                                             |

每对 `BEGIN TODO` 和 `END TODO` 之间是待实现区域，前面的注释写了输入、输出、不变量和边界。参考实现在 `solutions/lab2/src/` 的同名文件里。测试用的 kernel 输入由 `oracle.hpp` 自己打包，输出和同一份字节的 FP64 解码比较，所以 B 和 R3/R4 的正确性不依赖你的 A2、R2。不要为了通过而改测试。

```bash
python tools/run.py --lab 2 --impl student --mode test --part A
python tools/run.py --lab 2 --impl student --mode test --part C                   # 主机部分
python tools/run.py --lab 2 --impl student --mode test --part B --device cuda     # WSL
python tools/run.py --lab 2 --impl student --mode test --part C --device cuda     # 含 GPU 集成
python tools/run.py --lab 2 --impl student --mode test --part R --device cuda
python tools/run.py --lab 2 --impl student --mode bench --part C                  # 校准表
python tools/run.py --lab 2 --impl student --mode bench --part B --device cuda    # int4 kernel 与基线
python tools/run.py --lab 2 --impl student --mode bench --part R --device cuda    # batch 扫描
```

每个检查打印一行 `PASS` 或 `FAIL <名称>: <期望> vs <实际>`，失败不会中断后面的检查。GPU 测试分组跑（B1…B5、C、R1…R5），一个 kernel 因 `asm("trap;")` 停下只影响自己那一组。`--impl reference` 跑参考答案，不会覆盖你的文件。

## A 基础：一个能逐字节核对的 int4 格式（6h）

格式：$W$ 是行主序 $[M,N]$。每行按 $g$ 个连续列分组，组 $j$ 的 scale 为 $s_{r,j}=\max_{c\in j}\lvert W_{r,c}\rvert/7$（全零组取 1）。码 $q=\mathrm{clamp}(\mathrm{lround}(w/s),-7,7)$，存 $u=q+8$；列 $c$ 在字节 $r\lceil N/2\rceil+\lfloor c/2\rfloor$，偶数列放低半字节，奇数宽度行尾的高半字节为 0。

### A0 手算（0.5h）

1. $M=2,N=5,g=3$：画出 6 个字节、4 个 scale，标出每组起止和两个无效高半字节，算出相对 FP32 的字节比。
2. $W=[7,2.5,-2.5,0.5,-0.5,1.5]$、$g=6$：手算三个字节（测试里的 `BF 95 A7`），说明为什么 `std::nearbyint` 会得到不同结果。
3. 一组 31 个 $\pm0.5$ 加一个 1.0：比较 clipping 比例 1 和 0.9 的平方误差和（讲义 §2.4）。

### A1 `calibrate`（1h）

输入 $W$、$M$、$N$、$g$，输出 $M\lceil N/g\rceil$ 个 FP32 scale。不变量：组只在一行之内，尾组在 $N$ 处截止；非法形状、非有限值抛 `std::invalid_argument`。测试：`[7,1,-1 | 0.7,0.1,0]` 在 $g=3$ 时应得 `[1, 0.1]`（按行取 absmax 会得到 `[1,1]`）；两行不同最大值加全零组；尾组后面紧跟一个 70 的反例；36 种随机形状对 FP64 oracle。

### A2 `pack`（1h）

舍入是 ties-away（`std::lround`），在 FP32 里做 $w/s$。测试：手算字节 `BF 95 A7`、15 个合法码、奇数宽度 `8F 01 FF 0F`、随机形状里每个码等于 oracle 舍入（离舍入边界 $10^{-4}$ 以内的元素允许两种结果）。

### A3 `unpack`（0.75h）

只读字节和 scale 还原 $(u-8)s$，校验长度和 scale（0、负数、NaN 都拒绝）。测试用一个手写的 `Packed`，你的 A1/A2 不参与。注意 $u=15$ 表示 $q=7$，不要当成有符号 int4 的 $-1$。

### A4 `packed_linear`（0.75h）

$Y=X\hat W^\top$，$X$ 为 $[B,N]$，输出 $[B,M]$，直接从字节解码，用 double 累加，不调用 `unpack`。测试：手写 artifact 的两 token 例子，和测试自己的解码器在 27 种形状乘 3 种 batch 上比较。

### A5 `certificate`（0.75h）

返回 $\max\lvert W-\hat W\rvert/(s/2)$、MSE 和实际字节数（码加每个 scale 4 B）。不 clipping 的 absmax 格式这个比例不超过 1。测试按同样定义在测试里重算。

### A6 `clip_pack`（1.25h）

每组枚举 $s=\rho\cdot\mathrm{absmax}/7$，按**权重**平方误差选第一个最小者，并用选中的 scale 重新量化这一组的码。测试有一个逐组穷举的 oracle 和两个专门设计的组（31 个 $\pm0.5$ 加一个最大值，比例 0.9 一定胜出）；只换 scale 不重写码、按最大误差选，都会失败。clipping 后 A5 的半步上界不再成立，这是预期行为。

## B 核心：解码、复用与并行度（9h）

### B0 阅读（0.5h）

讲义 §3 和 [NVIDIA warp 原语](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)的归约树。画出 $K=65$ 时每个 lane 拿到哪些列。

### B1 `nibble`（0.5h）

用内联 PTX `bfe.u32` 取出 4 位无符号字段。测试用一个探针 kernel 遍历 256 个字节和两个 shift。`bfe.s32` 会把 8–15 符号扩展成负数，在这个偏移编码里是错的。

### B2 `fused_gemv`（2h）

一个 warp 一行，lane $l$ 处理列 $l,l+32,\dots$，`__shfl_down_sync` 全 warp 归约。行越界时整个 warp 返回；没有列可处理的 lane 必须照常参加归约。测试覆盖行数 1/7/33、列数 1/31/65/257、组大小 3/32/128。

### B3 `pair_gemv`（1.5h）

每个 lane 读一个字节、处理两列。两列可能属于不同的组（$g=3$ 时第 2、3 列就是），各用各的 scale；奇数尾只有一列。

### B4 `tiled_batch`（1.5h）

`grid.y` 选 4 个 token 的 tile，每个解码出的权重在寄存器里给 4 个 token 复用。最后一个 tile 不满时，越界 token 不读不写，但每个 lane 仍做完 4 次归约。测试 batch 1/3/4/7/8（batch 8 能发现 tile 步长写错）。

### B5 `split_partial` 与 B6 `merge_partial`（2h）

把 $K$ 切成长 `chunk` 的段，一个 warp 负责（行，段），partial 布局 $[\mathrm{parts},\mathrm{rows}]$；B6 按段的顺序把所有 partial 加起来。测试用 chunk 17（含 $N<17$），并直接读回 partial 缓冲区逐段核对。

### B 基准（1h）

`--mode bench --part B` 在 `bench_plan.hpp` 的 4 个形状（Qwen3-1.7B 的 q_proj $2048\times2048$、up_proj $6144\times2048$、down_proj $2048\times6144$，以及 $8192\times8192$）上，batch 1 和 4，对比：

| 路径                      | 做什么                                                       | 逻辑字节                               |
| ------------------------- | ------------------------------------------------------------ | -------------------------------------- |
| `int4_b2_warp` 等四条     | 你的 B2/B3/B5（batch>1 时逐 token 启动）、B4（4 token tile） | 权重读几遍算几遍，加 $X$、$Y$、partial |
| `fp32_cublas_sgemm`       | 常驻 FP32 原始权重，`cublasSgemm`                            | $4MN$                                  |
| `fp16_cublas_gemmex`      | 常驻 FP16 权重和激活，`cublasGemmEx`，Tensor Core，FP32 累加 | $2MN$                                  |
| `int4_unpack_fp16_gemmex` | 先把 int4 解成 FP16 矩阵，再 `cublasGemmEx`                  | $B_4+2\cdot2MN$                        |

计时规则见 [authoring §5](../docs/authoring.md#5-测量规则)：同一 launch 先连续跑满 300 ms 让笔记本 GPU 升频，然后 30 次计时，每次之前在计时区间外 memset 64 MiB 冲掉 L2，去掉前 5 次，报告中位数和 IQR，`stable` 列表示 $p_{75}\le1.5\,p_{25}$。`frac_peak` 是相对同一次运行里 256 MiB copy kernel 的实测带宽；纯读的 cuBLAS 可能略超过 1，因为 copy 的读写混合比纯读慢。`floor_us_modeled` 是“权重读一遍加 $X$、$Y$”除以 copy 峰值，标注为 modeled。`rel_err` 是相对**原始 FP32 权重**的 FP64 输出的相对 L2 误差（256 个抽样行），所以包含格式本身的量化误差；超出该路径的上界时整行标 `WRONG_OUTPUT`，基准以非零码退出。

## C 前沿：AWQ 机制的可控缩小版（7h）

### C0 阅读（0.75h）

[AWQ 论文](https://arxiv.org/abs/2306.00978) §3.2 与 [`auto_scale.py`](https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py) 的 `_search_module_scale`。写下：统计量是什么，网格怎么取，loss 对哪个输出算。

### C1 `channel_scales`（1h）

$$
m_k=\frac1T\sum_t\lvert x_{t,k}\rvert,\qquad s_k=\frac{\max(m_k,10^{-4})^\alpha}{\sqrt{\max_j s'_j\cdot\min_j s'_j}},\quad s'_j=\max(m_j,10^{-4})^\alpha .
$$

$\alpha=0$ 时全为 1。测试：手算的 $2\times3$ 例子 $X=[[1,-2,4],[3,2,-4]]$，均值 $[2,2,4]$，$\alpha=0.5$ 时应得 $2^{[-1/4,-1/4,1/4]}$（用最大值统计会得到别的数）；全零通道在 $\alpha=0.5$ 时应得 $[0.1,10]$（先 clamp 再求幂）；随机数据对 FP64 oracle。

### C2 `transform`（0.5h）

$W'=W\,\mathrm{diag}(s)$，$X'=X\,\mathrm{diag}(s)^{-1}$，返回新副本，在精确算术下 $X'W'^\top=XW^\top$。测试：讲义里的手算例子和随机等价性。

### C3 `search`（1.5h）

对每个 $\alpha$：C1、C2、A2 量化 $W'$、A4 在 $X'$ 上算输出，与**原始浮点** $W$ 的输出比较 MSE，取第一个最小者；候选必须含 $\alpha=0$。测试里有一份不调用你任何函数的穷举搜索（`oracle::search`），比较选中的 $\alpha$、MSE 和通道 scale；另有一个两条通道放大 30 倍的例子，oracle 的最优 $\alpha>0$ 且比 $\alpha=0$ 的 MSE 低 20% 以上，你的搜索也必须选到它。

### C4 `execute`（0.5h）

冻结 artifact 在新输入上执行：$X$ 除以保存的通道 scale 一次，再走 A4，不重新估计 scale。测试用一个手写 artifact（通道 scale $[2,0.5,4]$）和测试自己的解码器。

### C5 `search_budget`（1h）

在字节预算内联合选组大小和 $\alpha$：码、每组 4 B、每通道 4 B（即使 $\alpha=0$ 也计入）。测试用比 $g=8$ 的总字节少 1 B 的预算，必须退到 $g=32$；没有可行解时抛 `std::runtime_error`。

### C 实验（1.75h）

`--mode bench --part C` 输出校准表：校准集通道 0 放大 12 倍；`heldout` 是同分布的新样本，`shifted` 把放大通道换到通道 1；同时给出 RTN（$\alpha=0$）和 clipping-only 的对照。参考实现上，四个宽度中同分布 held-out 的 MSE 都低于 RTN（低 12% 到 2.8 倍）；放大通道一换位置，四个宽度里有三个反而比 RTN 差 2.7 到 14 倍。`--part C --device cuda` 还会把 C3 选出的 artifact 交给 B 的四条 kernel，与 C4 和 FP64 核对。本实验只复现“按校准激活搜索等价通道缩放”这一点，不含 Transformer 块内的 scale 折叠、完整模型困惑度或 W4A16 kernel。

## R 真实系统：sm_89 上的 FP8 权重 GEMV（10h）

RTX 4060 Laptop 的 Ada 架构在硬件上支持 FP8（E4M3、E5M2）：标量转换指令 `cvt.rn.f16x2.e4m3x2` 和 FP8 Tensor Core（`mma.sync.m16n8k32`，PTX 8.4 起；CUDA 12.6 带 PTX 8.5）。它没有 FP4。R 回答讲义研究问题 (a) 的一个缩小版：在这张卡上，INT4 省下的字节能不能敌过 FP8 更便宜的解码？

### R0 阅读（1h）

讲义 §2.3 的 E4M3 编码表；[cuda_fp8.h 文档](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html)的 `__nv_cvt_fp8x2_to_halfraw2`；[vLLM `marlin_utils_fp8.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py) 怎么把 FP8 权重给 W8A16 kernel 用。

### R1 `encode_e4m3`（2h）

FP32 到最近的 E4M3 码，平局取偶数尾数：NaN 得 `0x7F`；$\lvert x\rvert\ge448$ 和 $\pm\infty$ 饱和到 $\pm448$（`0x7E`/`0xFE`），这是 `cvt.rn.satfinite` 的行为；保留 subnormal（最小 $2^{-9}$），$\lvert x\rvert\le2^{-10}$ 舍入为带符号的 0。测试的 oracle 是对 127 个非负有限码的暴力最近值搜索，它本身先和 GPU 上的 `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)` 在全部测试输入上逐一比对。输入包括每个精确码、每对相邻码的中点和中点两侧、饱和区、subnormal 区和 2 万个对数均匀分布的随机数。

### R2 `fp8_pack`（0.75h）

逐（行，组）scale $=\mathrm{absmax}/448$，码 $=$ R1$(w/s)$，行宽向上取整到 16 字节（kernel 才能用 128 位加载），填充字节为 0，组大小必须是 16 的倍数（一个 16 字节块不会跨组）。测试：两组 absmax 分别为 448 和 4.48 的手算例子应得 scale $[1,0.01]$ 和码 `7E 38 C0 … 7E B8 … 6E`；随机形状对 FP64 scale 和暴力最近码。

### R3 `fp8_gemv`（2.5h）

一个 warp 一行，每个 lane 每次用一条 128 位加载读 16 个码，在寄存器里两两转成 `half2`（`__nv_cvt_fp8x2_to_halfraw2(.., __NV_E4M3)`），乘以 $x$ 和该块的组 scale，再 warp 归约。测试在填充码里放 `0x7F`（NaN），在 $x$ 末尾后面也放 NaN，所以读过 `cols` 或用了填充码都会得到 NaN。参考实现在 `cols` 是 4 的倍数时用 `float4` 读 $x$；这一项改动让 8192 方阵上的带宽利用率从约 0.55 升到 0.81。

### R4 `fp8_tiled_batch`（1.5h）

和 R3 一样，但 `grid.y` 选 8 个 token 的 tile，转换后的权重给 8 个 token 复用。测试 batch 1/3/8/9/17。

### R5 `weight_bytes`（0.5h）

讲义 §1 的字节模型：INT4 $M\lceil N/2\rceil+4M\lceil N/g\rceil$，FP8 $M\cdot\mathrm{roundup}_{16}(N)+4M\lceil N/g\rceil$，FP16 $2MN$，FP32 $4MN$。基准用它算 GB/s；测试还用它检查 `bench_plan.hpp`：每个形状、每种格式在计时时都必须不在 L2 里（冲刷量至少是 L2 的两倍，或者权重本身至少是 L2 的两倍）。把冲刷关掉，这个检查就会失败，因为 Qwen3-1.7B 的每个 INT4/FP8 线性层都小于 32 MiB。

### R 基准（1.25h）

`--mode bench --part R`：在 q_proj、down_proj 上跑 batch 1，在 up_proj 和 $8192^2$ 上 batch 1、2、4、8、16、32、64 扫描，路径为 `int4_b4_tile`、`fp8_r3_gemv`（仅 batch 1）、`fp8_r4_tile`、`fp16_cublas_gemmex`、`fp32_cublas_sgemm`、`int4_unpack_fp16_gemmex`，以及 `fp8_cublaslt_w8a8`：权重和激活都量化成 E4M3，逐张量 scale，`cublasLtMatmul` 的 TN 布局、`CUBLAS_COMPUTE_32F`。它是 W8A8，激活也有量化误差，`rel_err` 要单独看；Ada 上的 cuBLASLt 只支持逐张量 scale，分块 scale 要 sm_100。

参考实现在 RTX 4060 Laptop 上（2026-10-18，copy 峰值 223 GB/s）的部分结果：up_proj、batch 1 时，最快的 INT4 SIMT kernel（B3）83 µs，只用了 0.36 的峰值；FP8 R3 87 µs，用了 0.67；FP16 cuBLAS 139 µs，cuBLASLt FP8 W8A8 77 µs。FP16 cuBLAS 从 batch 1 到 64 几乎不变（139 到 153 µs），FP8 R4 在 batch 4 时还比它快，到 batch 8 就慢了。完整表格和解释在[参考导读](../../solutions/lab2/README.md)。

## 报告与验收（3h）

提交实现、至少两个自己的反例、A–C 的测试记录、B/R 基准的 RECORD JSON 与 CSV、C 的校准表，以及 `reports/lab2.md`。必交图表：

1. batch 1 时各路径的 µs 和 `frac_peak`（四个形状），标出 copy 峰值和 `floor_us_modeled`。
2. up_proj 和 $8192^2$ 上 µs 随 batch 的曲线（中位数加 IQR 误差棒，对数坐标），叠加讲义 §1 的 modeled 曲线，标出各路径相对 FP16 cuBLAS 的交叉 batch。
3. `rel_err` 对字节数的散点：INT4 g128、FP8 g128、FP8 W8A8、FP16、FP32。
4. C 的校准表：同分布与分布偏移下 AWQ 相对 RTN 的变化。

回答：INT4 kernel 为什么只用了三分之一的带宽（用 `cuobjdump -sass` 数每个权重的指令，或 `ncu` 的 `sm__inst_executed` 与 `dram__throughput`，至少一条证据）？FP8 的交叉 batch 为什么比讲义 §1 的 FLOP 屋顶线预测的小得多？`int4_unpack_fp16_gemmex` 为什么比直接用 FP16 还慢？`stable` 为 `NO` 的行怎样处理？哪些结论只对这张笔记本卡成立？

自动测试检查的是算法性质，报告评的是测量和解释，不设固定的加速倍数门槛，也不要求超过 cuBLAS。评分见 [grade.md](grade.md)，参考试跑见[验证记录](../docs/validation.md)。

## 可选挑战

挑战另计 8–16h。

1. **FP8 Tensor Core**（6–10h）：用 `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`（PTX 8.4，sm_89）写一个 W8A8 小 batch GEMM，batch 维补到 8。片段布局见 [PTX ISA §9.7.16.5.10](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16832)。注意 vLLM `marlin_mma.h` 里的 `m16n8k16 … e4m3` 需要 PTX 8.7（CUDA 12.8 起），CUDA 12.6 编不过。和 `fp8_cublaslt_w8a8` 同形状比较。
2. **magic number 解包**：把 B3 换成 Marlin 的 `lop3`/`0x64006400` 解包（讲义 §3.3），需要先离线把每 8 个码重排成 $[0,2,4,6,1,3,5,7]$，并让 oracle 也认这个布局。比较每个权重的 SASS 指令数和 µs。
3. **W4A16 Tensor Core**：在 4060 上编译运行 [Marlin](https://github.com/IST-DASLab/marlin)（需要 PyTorch 扩展）或 torch 的 `_weight_int4pack_mm`，作为成熟的 INT4 基线放进 R 的扫描，检验讲义推导的交叉 batch 约 29。
4. 在 Qwen3-0.6B 的一个真实线性层上采集激活，重复 C 的校准与 held-out 实验，换三类文本。

先留一个失败案例并建立公平的 baseline，再判断是否值得继续。

## 资料汇总

| 资料                                                                                                                                                                                                               | 类型 | 指定观察                                                 | 时间  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---- | -------------------------------------------------------- | ----- |
| [课程 D2](../../course/modules/02-low-bit-kernels.md)                                                                                                                                                              | 讲义 | §1 字节模型与交叉 batch、§2 格式表、§3 寄存器内解包      | —     |
| [AWQ `auto_scale.py`](https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py)                                                                                                                 | 代码 | `_search_module_scale` 的统计量、网格和 loss；对照 C1–C3 | 30min |
| [Marlin kernel](https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda_kernel.cu)                                                                                                                     | 代码 | `dequant` 的 `lop3` 与 `0x64086408`；对照 B1/B3          | 30min |
| [FasterTransformer `interleaved_numeric_conversion.h`](https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h) | 代码 | `I4s_TO_F16s_MAGIC_NUM` 与 `prmt` 的 int8 路径           | 20min |
| [vLLM FP8 文档](https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/fp8/)                                                                                                                          | 文档 | 哪些 GPU 走 W8A8、哪些走 FP8 Marlin（W8A16）；对照 R     | 10min |
| [cuBLASLt FP8 条件](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul-regular-imma-conditions)                                                                                                         | 文档 | Ada 上的 TN 要求和 16 字节对齐；对照 `baselines.cuh`     | 15min |
| [资源、定位与简化边界](resources.md)                                                                                                                                                                               | 索引 | 上游具体文件、借鉴内容和未覆盖部分                       | 15min |
| [参考实现](../../solutions/lab2/README.md)、[实验规范](../../course/guides/experiments.md)                                                                                                                         | 参考 | 独立完成后核对不变量；按规范记录测量和负结果             | —     |
