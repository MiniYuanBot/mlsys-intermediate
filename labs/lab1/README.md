---
tags: [mlsys, lab, kv-cache, quantization, retrieval, cuda]
---

# Lab 1：INT4 KV 布局、流式缓存、PQ 检索与融合 decode kernel

[课程 D1](../../course/modules/01-kv-cache.md) · [实验索引](../index.md) · [评分标准](grade.md) · [资源与改编说明](resources.md) · [参考导读](../../solutions/lab1/README.md)

> [!important] 完成定义
> A/B/C 用 C++17 在 CPU 上完成：能独立解码的 int4 格式、带残余窗口的流式缓存、PQ 检索。R 用 CUDA 在 WSL 的 RTX 4060 上完成：读 A/B 布局的 int4 KV、按 Qwen3-1.7B 形状（16 个 query head、8 个 KV head、head_dim 128）做一步 decode attention 的融合 kernel，与 FP64 oracle 和“先反量化再算”的基线对比，并测带宽利用率。每个 TODO 都有独立 oracle 和至少一个错误注入。

需要先掌握 foundations 中的 C++、内存层次、attention、CUDA 线程层次和基准测量。**下列命令都从 `mlsys-intermediate/labs/` 执行。** 预计 **30h**：A 5h、B 7h、C 7h、R 8h、报告 3h，另加 D1 讲义 5h。估时包括独立编写和调试；参考实现几秒就能跑完，不能拿来估计自己的进度。

## 0. 环境

本 lab 的依赖和其他 lab 互不累加。先安装、再检查：

```bash
bash tools/setup_env.sh --lab 1 && bash tools/check_env.sh --lab 1
```

Windows 原生（PowerShell）只做 CPU 部分：

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1 -Lab 1
powershell -ExecutionPolicy Bypass -File tools\check_env.ps1 -Lab 1
```

| 阶段 | 需要                                           | WSL2 / Linux | Windows 原生   | macOS  |
| ---- | ---------------------------------------------- | ------------ | -------------- | ------ |
| A    | C++17                                          | 可做         | 可做           | 可做   |
| B    | C++17                                          | 可做         | 可做           | 可做   |
| C    | C++17                                          | 可做         | 可做           | 可做   |
| R    | C++17 + nvcc + NVIDIA GPU；可选模型 Qwen3-0.6B | 可做         | 不支持，转 WSL | 不支持 |

A/B/C 只需要 C++17 编译器。R 需要 nvcc 和 GPU，`setup_env` 会装 CUDA Toolkit，缺少时 `check_env` 只报 `[WARN]`，不挡 A/B/C。缺依赖时 runner 只打印一行 `MISSING ...` 和 `fix:`，退出码 3；在 Windows 上请求 R，`fix:` 给出一条可以直接复制的 WSL 命令。可选挑战 `axis_probe.py` 另需 torch、transformers 和 `~/models/Qwen3-0.6B`。各平台细节见[环境配置](../setup.md)。

## 文件和实施顺序

| 文件                              | 学生负责                            | 已提供                                                         |
| --------------------------------- | ----------------------------------- | -------------------------------------------------------------- |
| `src/kv_student.hpp`              | A1/A2 编解码、B1 选择               | 分组索引 `group_index`、FP64 累加的 attention 参考函数         |
| `src/cache_student.hpp`           | A3 收集、A4 误差、B2–B5 流式缓存    | 状态结构和接口                                                 |
| `src/pq_student.hpp`              | C1–C4 训练、编码、打分、取回        | 码本布局、结果类型                                             |
| `src/decode_student.cuh`          | R1 解包、R2 分段 attention、R3 合并 | 布局说明、FP16 缓存、残余窗口读取、反量化基线 kernel、launcher |
| `test.cpp`                        | 可以追加自己的反例                  | A/B/C 的独立 oracle，逐项 PASS/FAIL                            |
| `decode_test.cu`、`r_fixture.cuh` | 可以追加                            | R 的主机端量化器、FP64 解码器与 attention、误差界              |
| `bench.cpp`、`decode_bench.cu`    | 可以增加变量                        | CPU 三 seed 矩阵；GPU 三条路径、长度与 split 扫描              |
| `axis_probe.py`                   | 可选挑战                            | Qwen3 真实 cache 的轴选择探针                                  |
| `exercise.py`                     | 不需要修改                          | 编译、阶段选择、R 的分组运行                                   |

每对 `BEGIN TODO` 和 `END TODO` 之间是待实现区域，前面的注释写了输入、输出、不变量和边界。参考实现在 `solutions/lab1/src/` 的同名文件里。不要为了通过而删改原测试。有效代码量见[自动统计](../docs/workload.md)。

```bash
python tools/run.py --lab 1 --impl student --mode test --part A
python tools/run.py --lab 1 --impl student --mode test --part B
python tools/run.py --lab 1 --impl student --mode test --part C
python tools/run.py --lab 1 --impl student --mode bench                      # CPU 矩阵
python tools/run.py --lab 1 --impl student --mode test  --part R --device cuda   # WSL
python tools/run.py --lab 1 --impl student --mode bench --part R --device cuda   # WSL
```

测试逐项打印 `PASS <名称>` 或 `FAIL <名称>: <期望> vs <实际>`，一个检查失败不会中断后面的检查；未完成的 TODO 抛出 `Complete A1` 这类异常，被记为该组的一条 FAIL。每个阶段末尾有 `SUMMARY`，有任何 FAIL 就以非零码退出。R 的四组检查（R3、R2、R1、fused）分别在独立进程里跑，一个 kernel 执行到 `asm("trap;")` 只会让它所在的组失败。runner 用临时副本，不覆盖你的文件，并记录编译命令、源码哈希和输出。`--part all --device cuda` 依次跑 A/B/C 和 R。

## A 基础：能独立解码的低位宽表示（5h）

### A0 手算和布局检查（0.5h）

画出 $T=5$、$D=7$、$g=4$ 时的 token 分组和 channel 分组；写出 $(0,6)$、$(1,0)$、$(4,6)$ 的组号、byte 位置和半字节位移。区分 $\lceil TD/2\rceil$ 和 $T\lceil D/2\rceil$：本格式连续打包，行与行之间不补齐。算出两种布局的元数据字节（FP32 scale 和 offset，每组 8 B），说明为什么不能只报告“每元素 4 bit”。

### A1 `quantize`（1.5h）

输入是行优先的 FP32 矩阵，输出无符号 0 到 15 的码。先检查形状、组大小和有限性，再统计每组**真实元素**的 min/max，得到 $s=(M-m)/15$ 和 offset $m$，最后打包。偶数线性索引放低半字节，奇数尾元素所在 byte 的高半字节必须是 0。常量组用有限的非零 scale（约定 1），要求精确恢复。尾组只统计真实元素，不能补零。范围差和量化运算都用 double，避免相反符号的 FP32 极值相减溢出；scale 下溢时取最小正值。

测试做法：对非对称数据（$5\times7$，每个通道有自己的偏移，行内、列内都有变化），用 FP64 独立重算每组 min/max 和 scale，与你的元数据逐组比对；再用**测试自己算的 scale**（不是你存的 scale）检查每个码离原值不超过半步。另有全正尾组 $[4,4,4]$、常量矩阵、FP32 极值和 subnormal。只要 scale 用了全局 min/max、补零尾组、半字节顺序写反、截断而非舍入，都会在这里失败。

> [!note] FP32 scale 的舍入
> 组 $[-2,-1,1,4]$ 的理想 scale 是 0.4，码是 $[0,3,8,15]$。实际存进 FP32 的 scale 略大于 0.4，再用 double 做除法时中间值略低于 2.5 和 7.5，`std::lround` 给出 $[0,2,7,15]$。这不是错误：测试按“误差不超过半步”验收，不按理想实数逐位比对。

### A2 `dequantize`（0.75h）

只读 `Packed` 的码和元数据。码 8 到 15 不是负数。测试直接手写一个 `Packed`（含 8 到 15 的码、奇数尾元素、每组不同的 scale 和 offset），和测试里按公式 $\hat x=o+s\,q$ 算出的值比对，所以解码器不可能借用原始输入。

### A3 `gather_rows`（1h）

返回新的连续数组，不只是索引。ID 必须严格递增且在范围内，空选择合法；先校验全部 ID 再开始拷贝。手算 $[1,2;3,4;5,6]$ 取第 0、2 行。解释为什么调用方仍持有原 K/V 时，这个函数会增加瞬时内存。

### A4 `errors`（1.25h）

实现 relative L2（分母是参考向量 a 的范数，下限 $10^{-20}$）、max absolute error、cosine，累加用 double。两个零向量 cosine 为 1，只有一个为零时为 0；拒绝 Inf 和长度不等。测试用手算例子 $a=[1,2,2]$、$b=[1,2,3]$（relative L2 为 $1/3$，cosine 为 $11/(3\sqrt{14})$）和 $[-1,1]$ 对 $[1,1]$（max abs 为 2）。回答：接近零的 attention 输出为什么会让 relative L2 很大？为什么还要同时看 max abs 和任务指标？

## B 核心：有状态的增量压缩缓存（7h）

### B1 `select_tokens`（1h）

先取 sink 和 recent 的并集，再按分数降序填满预算，**分数相同时选编号小的**，最后按位置排序。不可行预算显式拒绝，重叠的 mandatory 只算一次。测试除手算例子外，还有一个“预算边界上的三方并列”，以及 300 组随机、大量并列的小规模输入，与暴力枚举所有子集的结果逐个比对。

这个函数不规定分数从哪来。用于历史淘汰时不能偷看未来 query；用于 C 的当前 query 检索时，查询和扫描的成本要计入。

### B2 `ResidualCache::append`（2h）

状态是已压缩的 chunks 和高精度的 recent K/V。每次追加一行；recent 行数达到 `residual+block` 时，把**最早**的一个完整块压缩：K 按通道分组（`Axis::Channel`），V 按 token 分组（`Axis::Token`），组大小都是 `group`。需要维护：

1. `chunks.size()*block + recent_rows == tokens`；
2. 返回后 `recent_rows < residual+block`；
3. 已压缩块的元数据不随新 token 改变；
4. 非法输入在修改状态前被拒绝。

本 lab 让 K/V 在同一时刻 flush，KIVI 原实现两者的条件不同，见[改编说明](resources.md)；报告里要说明这个简化。先在 residual=0、block=4 上画 1 到 9 次追加的状态，再调 residual=3 和 8。

测试在三种 residual 下各追加 37 行，每一步都：从**原始追加的 K/V** 独立算出应有的 chunk 数；断言每个 chunk 的 K 轴是 Channel、V 轴是 Token、行数是 block；用 FP64 对**原始数据中最早那一块**重算每组 min/max 和 scale，与 chunk 元数据比对；断言 recent 正好是原始数据的尾部。flush 时把块清零、量化最新块而删掉最早块、K/V 轴互换、晚一步 flush，都会被抓到。

### B3 `materialize`（1h）

调试用的 oracle：按时间顺序解码每个 chunk，再接上 recent。测试用自己的解码器把你的 chunk 解出来再拼接，与你的输出比对；同时检查解出的历史和原始历史相差不超过半步，块的顺序反了会失败。生产查询路径 B5 不许调用它。

### B4 `bytes`（0.5h）

统计逻辑 payload：码、scale、offset 和 FP32 的 recent。测试用闭式公式 `chunks*(2*ceil(block*dim/2) + 8*(Gk+Gv)) + 8*recent*dim` 核对。它不含 vector 容量、分配器开销和查询临时数组，不是 RSS。报告里同时列出 FP32 全缓存 $8TD$ 和 FP16 理论基线 $4TD$。

### B5 `attend`（2.5h）

维护运行最大值 $m$、分母 $\ell$ 和加权 V 累加；新分数到来时，用新最大值同时缩放旧分母和旧累加。每次只解码一个块，最后处理 recent。测试两层：一是对**缓存里实际存的值**（用测试自己的解码器）做 FP64 attention，要求 $\max\lvert\Delta\rvert<2\times10^{-5}$；二是对**原始 K/V** 做 FP64 attention，要求误差在讲义 §4.2 推导的量化误差界之内。另有一个 logit 约 7000 的溢出用例。

查询额外空间约为 $O(\mathrm{block}\cdot D)$，状态 $O(D)$；量化 payload 仍要持有全部历史。CPU 实验不证明 GPU 上能提速，那是 R 的事。

## C 前沿：PQCache 启发的检索（7h）

### C0 阅读与问题拆解（1h）

读 [PQCache](https://arxiv.org/abs/2407.12820) 的码本、检索和 CPU/GPU 数据流，再读[源码映射](resources.md)。区分“用低位宽表示所有值”和“用压缩索引挑出要取回的精确值”。本阶段实现 PQ 和精确 KV 收集，不实现原系统的多头异步 offload。

### C1 `ProductIndex::fit`（2h）

维度均分成 $M$ 个子空间，每个子空间独立做 Lloyd。确定性初始化用第 $\lfloor cN/C\rfloor$ 行；最近中心同距离时选编号小的；空簇保留旧中心；每轮**更新中心之后**记录总 SSE。再次 fit 必须清空旧 codes。测试里有一份独立实现的 Lloyd（同样的规则），你的码本和每轮 SSE 必须与它一致；随机数据上第一轮更新必须让 SSE 严格下降。不更新中心、换了初始化行、refit 后保留旧码，都会失败。

### C2 `encode`（1h）

对冻结的码本，每个子空间取欧氏最近的中心，存 uint16 编号；先验证全部 key，再发布结果，编码不重新训练。测试用暴力最近中心在 held-out 数据上比对，并有一个等距并列用例。索引大小是 $MC(D/M)\cdot4+TM\cdot2$ 字节，原始 KV 和查询临时表不在其中。

### C3 `scores`（1h）

为当前 $q$ 建表 $\mathrm{lookup}[p,c]=q_p\cdot\mu_{p,c}$，再按码求和。排序用内积，不是欧氏距离。测试用 16 个没参加训练的 query，与“重建 key 再做 FP64 点积”的结果逐项比对，并检查打分不修改索引。复杂度 $O(CD+TM)$；分析小 $T$ 时建表为什么可能不划算。

### C4 `retrieve` 与完整实验（2h）

依次算 ADC 分数、在 mandatory 约束下选预算内的候选、物理收集原 K/V、做精确 attention，返回 ID、输出和这次收集的字节。测试在有簇结构的数据上用 8 个 query 检查：返回的 ID 是否恰好是 mandatory 加上 ADC 分数的 top-k（并列取小编号）；输出是否等于对这些原始行做的 FP64 attention；字节是否为 $2\cdot\mathrm{budget}\cdot D\cdot4$；与精确内积 top-k 的 recall 是否至少是随机选择的 3 倍。预算覆盖全部时必须还原完整 attention。

CPU 基准（`--mode bench`）跑 3 seeds、2 个长度、3 个预算共 18 行，包括 append、流式查询、PQ 准备、PQ 查询加选择加取回、recall、误差和分项字节。recall 的参照是同样 mandatory 约束下的精确内积 top-k，不是任务准确率；`fetch_bytes` 是 FP32 物理收集量，不是 PCIe 实测流量。

```bash
python tools/run.py --lab 1 --impl reference --mode bench --output results/lab1/reference.json
python tools/extract.py results/lab1/reference.json --table 0 --out results/lab1/reference.csv
```

## R 真实系统：int4 KV 融合 decode-attention kernel（8h）

R 在 WSL 的 GPU 上跑，读 A/B 产生的布局（K 按块逐通道量化，V 逐 token 每 `vgroup` 个通道一组，最后接 FP32 残余窗口），对 16 个 query head 做一步 decode attention。布局细节写在 `src/decode_student.cuh` 开头的注释里。

### R0 阅读（1h）

读讲义 §5.3、§5.4，[Flash-Decoding 博客](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)的动画，[llama.cpp `fattn-common.cuh`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-common.cuh) 的 `vec_dot_fattn_vec_KQ_q4_0`，和 [vLLM `triton_decode_attention.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py) 的 `_fwd_grouped_kernel_stage1` 与 `_fwd_kernel_stage2`。写下：每个 lane 管哪几个通道，K 和 V 的元数据分别按什么索引，GQA 下 query head $h$ 读哪个 KV head。

### R1 `Int4Cache::load4`（2h）

输入 KV head `kh`、是 K 还是 V、token `t`（只在 int4 前缀里）、lane；输出这个 lane 的 4 个通道 $[4\,\mathrm{lane},4\,\mathrm{lane}+4)$ 的 FP32 值。每个 lane 只读一个 16 bit 的字（4 个码，偶数通道在低半字节）和对应的元数据：K 的 scale/offset 按（块，通道）索引，V 的按（token，通道组）索引。函数里不要写依赖数据的分支：参考实现的初版这样写，32K 时慢了约 1.8 倍。

测试：一个探针 kernel 对每个 token 调用你的函数，与主机端独立的 FP64 解码器比对（两种 `vgroup`），并检查解码值离**原始**K/V 不超过半步。半字节顺序反了、K 的 scale 当成整组共用、V 的 scale 取成块首 token 的，都会失败。

### R2 `split_partial`（3h）

一个 warp 负责一个（KV head，split），同时服务这个 KV head 对应的 $G=2$ 个 query head。对 split 内每个 token：用 R1 读 K/V，warp 内规约出 $q\cdot k/\sqrt D$，按在线 softmax 更新每个 query head 的 $(m,\ell,a)$。split $s$ 覆盖 $[s\cdot\mathrm{split},\min(T,(s+1)\cdot\mathrm{split}))$，可能跨过 int4 前缀和残余窗口的边界；前缀用 `load4`，残余窗口用已提供的 `load4_tail`。输出未归一化的 `part_m`、`part_l`、`part_acc`。

测试用 FP16 稠密缓存（不经过 R1），对每个（query head，split）在 FP64 里重算这三个量；覆盖 split 整除和不整除、单 split、split 大于 T 的情况。split 末尾少一个 token、query head 映射成 $h\bmod H_{\mathrm{kv}}$ 而不是 $\lfloor h/G\rfloor$、漏掉累加量的重缩放、少了 $1/\sqrt D$，都会失败。

### R3 `reduce_splits`（1h）

每个 query head 一个线程块，线程 $c$ 管通道 $c$，按

$$
o_c=\frac{\sum_s e^{m_s-M}\,a_{s,c}}{\sum_s e^{m_s-M}\,\ell_s},\qquad M=\max_s m_s
$$

合并。空 split（$m_s=-\infty$，$\ell_s=0$）权重为 0。测试构造局部最大值相差 100 以上的 split 和一个空 split，与 FP64 合并比对。不乘 $e^{m_s-M}$、分母不加权、不减 $M$（FP32 指数溢出），都会失败。

### R 端到端检查与基准（1h）

`fused` 组把 R1、R2、R3 接起来，在三种 split 下检查：16 个 query head 的输出等于对**同一份 int4 值**做的 FP64 attention（误差 $<10^{-4}$）；与**未量化**的原始 K/V 相比，误差在讲义 §4.2 的界内。

`--mode bench --part R` 测一层、batch 1、残余窗口 64 行，序列长度 1K 到 32K，三条路径：

| 路径      | 做什么                                             | 搬运的逻辑字节                |
| --------- | -------------------------------------------------- | ----------------------------- |
| `fp16`    | KV 以 FP16 存放，同一个 split-K kernel             | $B_{16}$                      |
| `dequant` | 先用 `dequant_to_fp16` 写出 FP16 缓存，再走 `fp16` | $B_4+2B_{16}$                 |
| `fused`   | 你的 R1+R2+R3 直接读 int4                          | $B_4$（码、元数据、残余窗口） |

计时用 cudaEvent，每次计时前在计时区间外 memset 64 MiB 冲掉 L2，5 次预热、25 次计时，报告中位数和 IQR。峰值带宽用同一程序里的 256 MiB copy kernel 实测。还会在 32K 上扫描 split 大小。参考实现在 RTX 4060 Laptop 上的结果见[参考导读](../../solutions/lab1/README.md)，32K 时融合路径约 0.38 ms，FP16 约 0.74 ms，“先反量化”约 1.55 ms。

## 报告与验收（3h）

提交实现、至少两个自己的反例、CPU 和 GPU 的 RECORD JSON 与 CSV，以及 `reports/lab1.md`。必交图表：

1. 预算对 recall 和误差；预算对查询时间和取回字节（CPU，区分三个 seed）。
2. 序列长度对三条 GPU 路径的延迟（中位数加 IQR 误差棒）和达到的 GB/s，画出实测 copy 峰值的水平线。
3. 32K 时 split 大小对融合路径延迟的扫描，解释最优 split，并说明 `stable` 列为 `NO` 的行。benchmark 已在计时前预热 300 ms：笔记本 GPU 空闲时会掉到低 P-state，显存频率可从 8000 MHz 降到 810 MHz，访存受限的 kernel 会因此慢到 8 倍。若仍看到不稳定，运行时记录 `nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv -lms 50`，并确认没有别的进程在用 GPU。

回答：融合 kernel 达到峰值的多少？剩下的差距归到哪几项（用 `ncu` 看 `dram__throughput` 和指令数，至少给一个证据）？为什么“先反量化再算”比 FP16 还慢？元数据在什么尺寸下让 int4 不划算？PQ 索引字节有没有算原 KV？若 query 分布改变，哪些保证仍然成立？

自动测试检查算法性质，报告评的是测量和解释；验收不设固定的加速倍数门槛。评分见 [grade.md](grade.md)，参考试跑见[验证记录](../docs/validation.md)。

## 可选挑战

挑战另计 8 到 16h。

1. **真实 K/V 上的轴选择**（R4，约 3h）：在 WSL 里运行

   ```bash
   git show 0332b83:course/modules/01-kv-cache.md > /tmp/d1-text.md
   python lab1/axis_probe.py --model ~/models/Qwen3-0.6B --tokens 2048 --text /tmp/d1-text.md \
     --out results/lab1/axis-0.6B.json
   ```

   它对一整份 prefill cache 做伪量化，比较 K/V 各自用哪个轴时后 32 个 token 的 NLL。用上面这段固定文本，参考机上 K per-token $g=128$ 让 NLL 从 5.27 升到 8.36，per-channel $g=64$ 只到 5.29（讲义 §2.2）；不加 `--text` 时默认读当前的讲义，数字会随讲义修改而变。在此基础上加入讲义「研究问题」里的“折叠 $\gamma$”配置，换三类文本各跑 5 段，再用 [kvpress](https://github.com/NVIDIA/kvpress) 跑 RULER-4K。也可以先用 [`tools/capture_model.py`](../tools/capture_model.py) 和 `tools/analyze_kv.py` 看单层单头的热力图。

2. **per-token K 的融合 kernel**：给 R 加一个 K 也按 token 分组、没有残余窗口的变体，在同一形状下比较 µs 和 GB/s，回答讲义查新结论 (c) 的问题。
3. 把 selective gather 迁到 pinned host 加 CUDA stream，测真实传输；为 decode 新 token 加增量 PQ 编码和重建策略。

先保留至少一个失败案例并建立公平的 baseline，再判断是否值得继续。[prompts.txt](prompts.txt) 里有四条中英文示例，可以先用它跑通 `capture_model.py`。

## 资料汇总

| 资料                                                                                                                                 | 类型 | 指定观察                                                | 时间  |
| ------------------------------------------------------------------------------------------------------------------------------------ | ---- | ------------------------------------------------------- | ----- |
| [课程 D1](../../course/modules/01-kv-cache.md)                                                                                       | 讲义 | §3.4 的字节表、§4.2 的误差界、§5.3 的合并公式           | —     |
| [KIVI](https://github.com/jy-yuan/KIVI)                                                                                              | 代码 | `models/llama_kivi.py` 的 residual 分支；对照 B2 的简化 | 30min |
| [PQCache](https://github.com/HugoZHL/PQCache)                                                                                        | 代码 | `cache_manager.py` 的检索与取回；对照 C4                | 30min |
| [llama.cpp `fattn-common.cuh`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-common.cuh)                | 代码 | q4_0 点积的解包与零点修正；对照 R1                      | 30min |
| [vLLM `triton_decode_attention.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py) | 代码 | 两阶段 split-K 与合并；对照 R2/R3                       | 30min |
| [Flash-Decoding 博客](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)                                                       | 图解 | `parallelization_kv` 动画                               | 10min |
| [资源、定位与简化边界](resources.md)                                                                                                 | 索引 | 上游具体文件、借鉴内容和未覆盖部分                      | 15min |
| [参考实现](../../solutions/lab1/README.md)、[实验规范](../../course/guides/experiments.md)                                           | 参考 | 独立完成后核对不变量；按规范记录测量和负结果            | —     |
