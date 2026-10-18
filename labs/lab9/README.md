---
tags: [mlsys, lab, serving, heterogeneous, max-flow, cost-model, planning]
---

# Lab 9：异构与多模型 serving 的代价模型与规划

[课程 D9](../../course/modules/09-hetero-serving.md) · [实验索引](../index.md) · [评分标准](grade.md) · [资源与改编说明](resources.md) · [参考导读](../../solutions/lab9/README.md)

> [!important] 完成定义
> A/B/C 用 C++17 在 CPU 上完成：先写一个能拟合、能报误差的 roofline 代价模型，再写 Helix 式的 max-flow 层放置，最后在 SLO 和价格表下做多模型共置的最小成本规划，并用离散事件仿真检验。R 阶段在 RTX 4060 上用 CUDA 实测带宽、GEMV 和 GEMM，重新拟合效率系数 $\eta$，重新规划，并做 $\eta$ 的敏感度扫描。除了 R1 到 R4 在本机 GPU 上的计时，所有数字都是 **modeled, not measured**，报告里必须这样标注。

需要先读完 [D9 讲义](../../course/modules/09-hetero-serving.md) 的 §2 到 §8。下列命令都从 `mlsys-intermediate/labs/` 执行。预计 **25h**：A 5.5h、B 7.5h、C 6h、R 4h、报告 2h，另加 D9 讲义 5h。估时包含独立编写和调试；参考实现跑完只要几十秒，不能用它估计学习时间。

## 0. 环境

本 lab 的依赖和其他 lab 互不累加。在 `labs/` 下先安装、再检查：

```bash
bash tools/setup_env.sh --lab 9 && bash tools/check_env.sh --lab 9
```

Windows 原生（PowerShell）只做 CPU 部分：

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1 -Lab 9
powershell -ExecutionPolicy Bypass -File tools\check_env.ps1 -Lab 9
```

| 阶段 | 需要                                  | WSL2 / Linux | Windows 原生 | macOS       |
| ---- | ------------------------------------- | ------------ | ------------ | ----------- |
| A    | C++17                                 | 可做         | 可做         | 可做        |
| B    | C++17                                 | 可做         | 可做         | 可做        |
| C    | C++17                                 | 可做         | 可做         | 可做        |
| R    | C++17；GPU 部分另需 nvcc + NVIDIA GPU | 可做         | 仅 CPU 部分  | 仅 CPU 部分 |

A/B/C 只需要 C++17 编译器。R 阶段分两半：R1 到 R4 是 CUDA 计时与 kernel，只能在 WSL2 或 Linux 上跑；R5、R6 是重新拟合与重新规划，在 CPU 上读一份 profile CSV 即可。没有 GPU 的同学用随附的 `data/rtx4060_reference_profile.csv`（作者在 RTX 4060 Laptop、驱动 566.24、WSL2 上于 2026-10-18 实测）。`setup_env` 会装 CUDA Toolkit；缺少时 `check_env` 只对 R 报 `[WARN]`。缺依赖时 runner 只打印一行 `MISSING ...` 和 `fix:`，退出码 3；在原生 Windows 上请求 `--device cuda`，`fix:` 给出可以直接复制的 WSL 命令。

## 文件和实施顺序

| 文件                        | 学生负责                           | 已提供                                                  |
| --------------------------- | ---------------------------------- | ------------------------------------------------------- |
| `src/types.hpp`             | 不需要修改                         | 全部数据结构、模型配置、CSV 读取、确定性随机数          |
| `src/cost_student.hpp`      | A1 到 A5                           | `layer_params`、`lm_head_cost`、`serving_model`         |
| `src/flow_student.hpp`      | B1 到 B7                           | `Dinic::add_edge`、`make_dinic`、`placement_throughput` |
| `src/coloc_student.hpp`     | C1 到 C4                           | 模板生成、trace 生成、部署与路由、汇总 `simulate`       |
| `src/replan_student.hpp`    | R5、R6                             | `SweepRow`                                              |
| `src/profile_student.cuh`   | R1 到 R4                           | kernel 的 launch 包装、`HS_CUDA`                        |
| `test.cpp`                  | 可追加自己的反例                   | A/B/C/R 的独立 oracle：暴力计数、穷举、手算、排队论     |
| `profile.cu`                | 不需要修改                         | R1 到 R4 的 GPU 测试，`bench` 模式写 profile CSV        |
| `bench.cpp`、`scenario.hpp` | 可增加场景                         | 模拟集群、四模型场景、基线、敏感度扫描                  |
| `data/gpus.csv`             | 可追加 GPU，但必须带来源和访问日期 | dense FP16 规格、显存、带宽、价格，每行有来源 URL       |
| `exercise.py`               | 不需要修改                         | 编译、阶段选择、CUDA 门控                               |

每个 `BEGIN TODO` 到 `END TODO` 之间是待实现区域，前面的注释写了输入、输出、不变量和边界。初始骨架抛出 `Complete A1` 这类异常，CUDA kernel 的骨架是 `asm("trap;")`。测试逐项打印 `PASS`/`FAIL`，一个检查失败不会中止后面的检查。

```bash
python tools/run.py --lab 9 --impl student --mode test --part A
python tools/run.py --lab 9 --impl student --mode test --part B
python tools/run.py --lab 9 --impl student --mode test --part C
python tools/run.py --lab 9 --impl student --mode test --part R                 # R5/R6 on CPU
python tools/run.py --lab 9 --impl student --mode test --part R --device cuda   # + R1-R4 (WSL)
python tools/run.py --lab 9 --impl student --mode bench --part all
python tools/run.py --lab 9 --impl student --mode bench --part R --device cuda  # measure, refit, re-plan
```

先完成 A，再做 B、C；B 和 C 的 bench 用到 A 的代价模型，C 的仿真用到 B7 的 `Iwrr`。runner 在临时副本里编译运行，不覆盖你的文件，并把命令、输出和源码哈希写进 `results/` 下的记录。

## A 基础：代价模型（5.5h）

所有时间单位是秒，FLOPs 按每次乘加 2 次计。`Batch{R, q, c}` 表示 $R$ 个请求、每个请求本轮算 $q$ 个新 token、已有 $c$ 个缓存 token：prefill 是 `{R, s, 0}`，decode 是 `{R, 1, c}`。

### A0 手算（0.5h）

不写代码，先在笔记里完成：Qwen3-1.7B（$L=28$、$h=2048$、$H=16$、$H_{kv}=8$、$d=128$、$I=6144$）单层参数个数（应为 50,336,000，含 norm）；decode `{1, 1, 1024}` 的 FLOPs、KV 读字节（4 MiB）和算术强度；RTX 4060 Laptop（29.1 TFLOPS 推算值、256 GB/s）的 ridge point。这些数在测试里都有。

### A1 `layer_cost`（1.5h）

- **输入**：`ModelConfig`、`Batch`。
- **输出**：`LayerCost`，含 FLOPs、权重字节、KV 读、KV 写、激活读写五项。
- **不变量**：第 $i$ 个新 token（从 0 数）attend 到 $c+i+1$ 个位置；attention FLOPs 乘的是 query 头数 $H$，不是 $H_{kv}$；KV 读是 $R\,c\cdot2\,kv_{dim}\,b$，KV 写是 $n\cdot2\,kv_{dim}\,b$。
- **边界**：$R\ge1$、$q\ge1$、$c\ge0$，否则抛 `std::invalid_argument`。
- **测试**：`tiny_model()`（$H\ne H_{kv}$、$Hd\ne h$）上用逐元素循环暴力计数，对照 6 组 batch；Qwen3-1.7B 的两个手算值；decode 强度低于 2、长 prefill 高于 4060 ridge。

### A2 `roofline_time`（0.5h）

$t=t_0+\max\left(F/(\eta_cP),\,B/(\eta_mW)\right)$，$P$、$W$ 取 `GpuSpec::peak_flops()` 和 `peak_bw()`。$\eta\le0$ 或负的 $F$、$B$ 要拒绝。测试有算力受限、带宽受限、零工作量、恰在 ridge 上四个手算值。

### A3 `fit_efficiency`（2h）

- **输入**：样本 $(F_i,B_i,t_i)$ 和规格。
- **输出**：$(\eta_c,\eta_m,t_0)$。
- **方法**：参数 $\theta=(t_0,1/\eta_c,1/\eta_m)$，最小化相对误差平方和 $\sum_i((\hat t_i-t_i)/t_i)^2$，即权重 $1/t_i^2$ 的加权最小二乘；自己写 $3\times3$ 正规方程的高斯消元（部分主元）。样本归属（算力侧或带宽侧）用当前估计标注，重新拟合、重新标注，直到标签不变（最多 50 轮）。推导见讲义 §3.2。
- **边界**：少于 3 个样本、$t_i\le0$、某一侧没有样本、矩阵奇异、拟合出非正的 $1/\eta$，都抛 `std::invalid_argument`。
- **测试**：无噪声合成数据恢复已知系数（其中 5 个样本的强度位于真实 ridge 和规格 ridge 之间，不重新标注就恢复不出来）；3% 噪声下误差小于 5%；在返回点上对**测试自己写的**加权目标做中心差分，梯度接近零。

### A4 `memory_fit`（0.75h）

权重（可选含 embedding，共享权重只算一份）、KV、激活峰值 $T_{act}(2h+2I)b$ 加保留量与容量比较，并给出最多能缓存多少 KV token。测试用 3 个模型对照暴力计数，Qwen3-1.7B 整模型权重 3,441,149,952 字节，容量恰好装下 12345 个 token 时返回 12345、少 1 字节时返回 12344，权重都放不下时返回 $-1$。

### A5 `heldout_error`（0.25h）

只用训练集拟合，在测试集上报相对误差的均值、中位数（偶数个时取中间两个的平均）和最大值。测试集的误差是构造出来的，期望值按 $\lvert\delta/(1+\delta)\rvert$ 独立算出。

## B 核心：Helix 式放置（7.5h）

集群里每个节点是一台单 GPU 机器。`Node::rate_per_layer` 是它只放一层时的 token/s，放 $k$ 层时为 $r/k$；`max_layers` 是显存允许的最大层数。链路容量单位是 token/s。顶点编号固定：0 是 $S$，1 是 $T$，$2+2i$ 和 $3+2i$ 是 GPU $i$ 拆出的入点和出点。

### B1 `build_flow_graph`（1h）

按讲义 §5.1 建图：计算边 $i_{in}\to i_{out}$ 容量 $r_i/k_i$；区间从 0 开始的节点连源，到 $L$ 结束的连汇；$e_i=s_j$ 且链路容量为正时连 $i_{out}\to j_{in}$。区间越界、超过 `max_layers`、placement 长度不对，都要拒绝。测试对一个 3 卡手算例子比较边的多重集，再对 40 个随机放置和测试自己的建图规则比较。

### B2 `Dinic::max_flow`（1.5h）

BFS 分层加当前弧 DFS 推阻塞流；每推 $f$，正向残量减 $f$、反向残量加 $f$。测试：60 个随机图上和**枚举全部 $S$-$T$ 割**（$n\le12$）的最小割比较；容量约束和中间点守恒；平行边与反平行边的手算图；一个必须沿反向弧"撤销"流量才能达到 2 的图。

### B3 `min_cut`（0.75h）

在残量图上从 $S$ 做 BFS（正向剩余容量为正，或反向已有流量），可达集到不可达集的原图边就是割边。测试独立重算可达集，检查割值等于 max-flow 和枚举最小割，割边全部满流；两张快卡加一条慢链路的例子里，割必须正好是那条链路。

### B4 `partition_chain`（1h）

一条给定顺序的链，每个节点至少一层，按讲义 §6 的 DP 求最大瓶颈（含源、汇和链路），并回溯出每个节点的层数。测试对 150 条随机链枚举 $L$ 的所有切分。

### B5 `place_layers`（2h）

确定性蛇形初始解加 24 个随机初始解，从最好的 6 个出发做最优改进局部搜索：单卡移动无改进时再试成对移动（副本或相邻区间）。测试：6 个固定小集群上穷举全部放置（$\le5$ 卡、$\le8$ 层），每个不低于最优的 90%、平均不低于 0.97，结果对单卡移动局部最优（测试自己枚举验证），以及一个必须用 3 个整模型副本才能达到 450 token/s 的例子。这是启发式，不保证最优：参考实现在 60 个随机小集群上有 7 个低于最优的 97%，最差约 87%，报告里要说明。

### B6 `decompose_flow`（0.75h）

把一个可行流拆成若干条 pipeline（GPU 序列加速率）：从 $S$ 沿编号最小的正流量边走到 $T$，速率取路径最小值，沿路减去。测试检查速率和等于流值、按边加回去等于原流、每条 pipeline 从第 0 层恰好衔接到第 $L$ 层、路径数不超过正流量边数；流不守恒时要抛 `std::runtime_error`。

### B7 `Iwrr::next`（0.5h）

nginx 式平滑加权轮询（讲义 §7）。测试：权重 $(5,1,1)$ 的手算序列 a a b a c a a；20 组随机权重下任意前缀的计数偏差小于 1；零权重从不被选；整数权重 $(3,2)$ 的序列以 5 为周期。

## C 前沿：SLO 下的多模型共置（6h）

C 是 Mélange、Coral、AlpaServe 的可执行缩小版：模板代替 Coral 的离线放置 ILP，自写 B&B 代替商业求解器，时间份额等分代替 MuxServe 的空间复用，迭代级仿真代替真实引擎。

### C1 `slo_capacity`（1.5h）

讲义 §8.1 的公式：份额 $\alpha$ 下所有时间除以 $\alpha$，$\lambda(b)=1/(T_p'+(o-1)\text{TPOT}'(b)/b)$，观察到的 TPOT 为 $\text{TPOT}'(b)/(1-\lambda T_p')$，加 TTFT 与 KV 显存约束，在 $b=1..b_{max}$ 里取 $\lambda$ 最大者（并列取小 $b$）。测试用闭式延迟函数对 6 种情况（含半份额、显存受限、TTFT 不可行、$o=1$、TPOT 很紧）暴力扫 $b$。

### C2 `min_cost_assignment`（1.5h）

整数覆盖问题 $\min\sum_t\pi_tn_t$，$\sum_tn_tc_{t,m}\ge\lambda_m$。按模板顺序分支，下界取 $\max_m\rho_m\min_{t'}\pi_{t'}/c_{t',m}$；**对模型求和不是有效下界**（讲义 §8.2）。测试：一个必须选共置模板的手算例子（求和下界会把它剪掉），12 个随机实例对照暴力枚举，返回的计数确实覆盖需求、成本自洽。

### C3 `simulate_instance`（2h）

单实例的迭代级离散事件仿真，规则写在函数注释里：每个模型一个 FIFO 队列；在有活的模型间轮转；能接纳就先 prefill，否则做一步 decode；到达只在迭代边界进入队列，但系统内请求数的积分要包含迭代中途到达的请求。

手算例子（测试里的第一组）：prefill 恒为 1 s，一步 decode 恒为 0.5 s，batch 上限 2。请求 r0（到达 0，输出 3）、r1（0.2，输出 2）、r2（0.3，输出 1）：

| 时刻 | 动作                                | 结束时刻 | 系统内请求数             | 面积累计 |
| ---- | ----------------------------------- | -------- | ------------------------ | -------- |
| 0    | prefill r0，r0 的首 token 在 1.0    | 1.0      | 1，加上中途到达的 r1、r2 | 2.5      |
| 1.0  | prefill r1，首 token 在 2.0         | 2.0      | 3                        | 5.5      |
| 2.0  | batch 已满，decode r0、r1；r1 完成  | 2.5      | 3                        | 7.0      |
| 2.5  | prefill r2，它只要 1 个 token，完成 | 3.5      | 2                        | 9.0      |
| 3.5  | decode r0，完成                     | 4.0      | 1                        | 9.5      |

其他 oracle：M/D/1（单模型、batch 1、输出 1、Poisson 到达，10 万个请求）的平均排队时间对照 Pollaczek-Khinchine 公式 $W_q=\rho D/(2(1-\rho))$，容差 5%；Little 定律（面积等于逗留时间之和）；过载时完成速率不超过 $1/(T_p+(o-1)\,\text{step}(b,p)/b)$ 且不低于它的 80%；两模型轮转顺序。

### C4 `baseline_plans`（1h）

三个必须对比的基线：贪心（每个模型独立选单价最低的单模型模板，向上取整）、不共置（只用单模型模板求最优）、同构（每种 GPU 单独求最优，取最便宜的一种）。计数一律用完整的模板下标。测试独立重算贪心，用暴力法求两个受限问题，并检查三者都不比全局最优便宜。

## R 真实系统：在 RTX 4060 上标定 $\eta$（4h）

R1 到 R4 只在 WSL2 或 Linux 上运行。`profile.cu` 的 bench 在每组配置计时前，先用同一个 launch 让 GPU 连续忙至少 300 ms：笔记本 GPU 在两组配置之间一空闲就降到低 P-state，RTX 4060 Laptop 上显存频率会从 8000 MHz 掉到 810 MHz，几次 warmup 拉不回来。每行还输出 `stable` 列（$p_{75}\le1.5\,p_{25}$），不稳定的行自动重测最多两次，仍不稳定就保留 0 并在报告里说明。手动跑 GPU 程序时请包一层 `flock /tmp/mlsys-gpu.lock`（runner 和 `fault_inject` 会自己加锁的除外）。

### R1 `time_cuda`（0.75h）

先跑 `warmup` 次并同步，再对每次重复用一对 `cudaEvent` 包住 `launch()`，`cudaEventSynchronize` 后取 `cudaEventElapsedTime`；统计只用正式重复，分位数用线性插值 $h=(n-1)p$。测试用 `globaltimer` 自旋 kernel：400 µs 的中位数；前 4 次慢 20 倍的 warmup 不能计入；20 个打乱顺序的 1 到 20 ms 自旋，期望分位数由测试自己逐个计时后插值得到（最近秩法会差 0.25 ms 以上）。

### R2 `copy_vec4_kernel`（0.5h）

`float4` 向量化加 grid-stride 循环，尾部逐元素处理。launcher 会限制 grid 大小，一次只覆盖一遍的写法过不了 400 万元素的测试。测试还在输出后面放了一段哨兵，检查没有越界写。

### R3 `gemv_fp16_kernel`（0.75h）

一个 warp 算一行，每个 lane 一次读 16 字节（8 个 half），FP32 累加，`__shfl_down_sync` 规约。测试对照 FP64 参考（相对 $\sum\lvert w_kx_k\rvert$ 的误差小于 $10^{-4}$），并检查第 $M$ 行之后不被写。

### R4 `gemm_wmma_kernel`（1h）

$64\times64$ 的 block tile、K 步长 32、4 个 warp 各算 $32\times32$，`nvcuda::wmma` 的 $16\times16\times16$ 形状，共享内存行尾加 8 个 half 的 padding。测试对照 FP64 参考，输出先填成 NaN，漏写的 tile 会被发现。

### R5 `profile_to_samples` 与 R6 `sensitivity_sweep`（1h）

R5 把 profile CSV 的每一行换成 $(F,B,t)$ 样本（copy：$B=2m$；GEMV：$F=2mk$，$B=2mk+2k+4m$；GEMM：$F=2mnk$，$B=2(mk+kn)+4mn$）。这是"每个操作数读一次"的必要流量，是真实 DRAM 流量的下界。R6 对每个 $\eta$ 缩放档位给方案排名，报告排名第一是否变化和相对名义排名的 Kendall 距离。测试：手算字节数；随附 profile 的拟合结果落在合理区间；一组已知交叉点的成本函数上，排名和不一致对数对照暴力计数。

`--mode bench --part R --device cuda` 会：(1) 用 R1 到 R4 实测，写 `results/lab9/rtx4060_profile.csv`（含设备、驱动、日期）；(2) 用 R5 加 A3 拟合 $\eta$，逐样本打印误差，并做"copy 加 GEMM 训练、GEMV 测试"的 held-out；(3) 用实测 $\eta$ 重跑 B 的放置；(4) 对 C 的四模型场景做三组扫描（所有 GPU、只缩 $\eta_m$、只缩名义最优方案最依赖的那种 GPU），$\eta\times\{0.5,0.625,0.75,0.875,1.0\}$。没有 GPU 时去掉 `--device cuda`，(2) 到 (4) 读随附 profile。

## 报告与验收（2h）

报告用 Markdown，附上全部原始 `RECORD` JSON，必须包含：

1. **代价模型表**：Qwen3-1.7B 在 prefill 128/512/2048 和 decode batch 1/8/32/128 下的 FLOPs、字节、强度、瓶颈侧，至少 3 种 GPU（bench A1A2 行，modeled, not measured）。
2. **实测与拟合**：R2 的 copy、GEMV、GEMM 表（中位数加 IQR，measured），拟合的 $\eta_c$、$\eta_m$、$t_0$，逐样本误差和 held-out 误差。指出至少一处模型系统性偏差的来源（例如 copy 与 GEMV 的带宽差、大 GEMM 的功耗墙、skinny GEMM 的 tile 重读）。
3. **放置**：B bench 两个网络场景下四种规划器的 max-flow、USD/1M tokens、min-cut 瓶颈，以及 IWRR 的前 1000 个请求分布。讨论吞吐最高和单位成本最低为什么不是同一个方案。
4. **共置**：C bench 五个方案的 USD/h、USD/1M tokens，以及 Poisson 与 CV 为 3 两种到达下的 SLO 达成率、p99 TTFT、Little 定律两列。解释为什么成本最低的方案在仿真里 SLO 达成率最低，并给出一个负结果（例如 30% 余量仍然不够）。
5. **敏感度**：R6 的三组扫描表，标出排名第一翻转的最小缩放档位；说明这些都是 modeled, not measured。
6. **局限**：至少列出讲义 §3.3 中三条代价模型抓不住的因素，并说明它们会让哪个结论变化。

measured 和 modeled 的数字不放在同一列比较；所有"其他 GPU"的数字都要带 modeled, not measured 标签。

## 可选挑战

- **P/D 拆分**：在 B1 的图上加"prefill 组到 decode 组"的 KV 传输边，容量为带宽除以每请求 KV 字节，复现讲义 §10 的估算。
- **鲁棒规划**：给 C2 加一个 $\Gamma$-robust 版本（最坏的 $\Gamma$ 个模板容量打折），比较鲁棒性代价与 R6 的后悔值。
- **真实 trace**：用 Azure LLM Inference 2023/2024 或 BurstGPT 的到达时间和长度替换 `make_trace`，注意许可证要求的引用（见资料汇总）。
- **价格表**：换成 RunPod 价格（`data/gpus.csv` 的 `price_runpod_hr` 列），看最优组合是否改变。
- **区间重叠**：放宽 B1 的"恰好衔接"规则，允许下一跳持有第 $e_i$ 层即可（Helix 的做法），比较 max-flow。
- **cuBLAS 差距**：分析 R4 的 skinny GEMM 为什么只有 cuBLAS 的 60% 到 80%（随附 profile），改进 tile 形状。

## 资料汇总

| 资料                                                                                                                                                                | 类型        | 指定观察                                                       | 时间  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | -------------------------------------------------------------- | ----- |
| [D9 讲义](../../course/modules/09-hetero-serving.md)                                                                                                                | 讲义        | §2 到 §8 的公式与 §11 函数表                                   | 5h    |
| [Helix 论文 HTML](https://arxiv.org/html/2406.01566) · [Helix-ASPLOS25](https://github.com/Thesys-lab/Helix-ASPLOS25)                                               | 论文 / 代码 | 集群图与 MILP 的变量；README 的 simulator tutorial             | 30min |
| [Coral](https://arxiv.org/abs/2605.04357)                                                                                                                           | 论文        | 模板枚举与在线 ILP，对照 C2                                    | 15min |
| [Mélange](https://arxiv.org/abs/2404.14527) · [melange-release](https://github.com/tyler-griggs/melange-release)                                                    | 论文 / 代码 | 成本感知装箱与 SLO 约束                                        | 15min |
| [AlpaServe](https://arxiv.org/abs/2302.11665)                                                                                                                       | 论文        | 突发度与统计复用的实验图，对照 C3 的 CV 扫描                   | 10min |
| [VisuAlgo Network Flow](https://visualgo.net/en/maxflow)                                                                                                            | 图解        | 单步运行 Dinic，观察层次图和反向边                             | 10min |
| [CUDA C++ Programming Guide 12.6](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-programming-guide/index.html)                                                  | 文档        | fragment 的 ldm 与对齐要求，对照 R4                            | 10min |
| [Azure LLM Inference 2023](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md) · [BurstGPT](https://github.com/HPMLL/BurstGPT) | trace       | 字段与许可证（CC-BY，分别要求引用 Splitwise 和 BurstGPT 论文） | 选读  |
| [资源与改编说明](resources.md)                                                                                                                                      | 说明        | 上游代码做了什么、本 lab 简化到哪里                            | 15min |
