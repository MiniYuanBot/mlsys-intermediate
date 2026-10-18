# Lab 9 参考实现导读

[任务说明](../../labs/lab9/README.md) · [评分标准](../../labs/lab9/grade.md) · [资源与改编说明](../../labs/lab9/resources.md) · [D9 讲义](../../course/modules/09-hetero-serving.md)

参考实现只放在 `src/` 下与学生骨架同名的文件里，`run.py --impl reference` 把它们覆盖到临时副本上运行。下面按阶段说明实现要点、为什么这样做、容易错在哪里，最后是 oracle 与 fault 的对照表。

## A 代价模型

**A1**。FLOPs 只计矩阵乘法：七个投影各 $2n\cdot\text{in}\cdot\text{out}$，attention 的 $QK^\top$ 和 $PV$ 合计 $4HdP$，其中 $P=R(qc+q(q+1)/2)$ 是因果掩码下的 (query, key) 对数。容易错的三处：GQA 只减少 KV 字节，不减少 attention FLOPs（乘 $H$ 而不是 $H_{kv}$）；第 $i$ 个新 token 看到 $c+i+1$ 个位置，含自己；decode 的字节必须含 KV 读，长上下文时它会超过权重本身。oracle 是 `tiny_model()` 上的显式循环，它不知道任何闭式，只对每个乘加计数；tiny 配置特意取 $H\ne H_{kv}$、$Hd\ne h$、$I$ 不是 $h$ 的倍数，让轴写错的公式产生不同的数。

**A2**。`max` 而不是求和：roofline 假设计算和访存完全重叠。$\eta$ 作用在分母上（达到的速率是 $\eta$ 乘峰值）。

**A3**。把 $\max$ 拆成两段线性模型后，参数 $\theta=(t_0,1/\eta_c,1/\eta_m)$ 是线性的，加权正规方程 $(X^\top WX)\theta=X^\top Wt$ 可以闭式求解。权重 $1/t_i^2$ 让目标变成相对误差平方和，否则毫秒级的大 GEMM 会完全压过微秒级样本。交替标注是必要的：$\eta_c<\eta_m$ 时真实 ridge 比规格 ridge 低，强度落在两者之间的样本按规格会被标成带宽受限。测试的合成数据专门放了 5 个这样的样本，所以"只标一次"的实现在无噪声数据上也恢复不出真值。收敛时每个样本的标签与 $\max$ 的较大项一致，解是原分段目标的驻点；测试据此对**测试自己写的**目标函数做中心差分，而不是对另一份闭式梯度。实测数据上，一个 $\eta_m$ 同时拟合拷贝（读写各一次）和 GEMV（几乎只读），两者实测带宽差约 20%，拟合只能折中。

**A4**。浮点向下取整在"恰好装满"时会差 1：$(C-\text{fixed})/m$ 理论上是整数，实际可能是 12344.999999。参考实现加 $10^{-9}$ 再取整，测试用容量恰好等于 12345 个 token 的需求来检查。

**A5**。测试集绝不进入拟合。偶数个误差的中位数取中间两个的平均。

## B Helix 式放置

**B1**。节点容量只能放在拆点的内部边上。放在入边上时，中间段 GPU 没有来自 $S$ 的边，容量就丢了；放在出边（链路）上时，一张 GPU 的计算容量会被每条出边各算一遍。"恰好衔接"让图无环，B6 的分解因此不会遇到环流。

**B2**。Dinic 的正确性依赖反向残量。测试里的"撤销图"先插入 a 到 d 的边，使第一条增广路是 S-a-d-T；最大流 2 需要第二条路 S-b-d-a-c-T 反向走 a 到 d。没有反向残量的实现在这里得 1。随机图上的 oracle 是枚举全部 $2^{n-2}$ 个 $S$-$T$ 割，取最小值，与任何 max-flow 实现都无关。

**B3**。最小割必须在**残量图**上找可达集：正向有剩余容量，或反向已有流量。只走正向边会漏掉通过"退流"可达的顶点；把两个方向的跨割边都算进容量会得到比流值大的数。

**B4**。DP 的状态 $f(j,\ell)$ 是前 $j$ 个节点恰好覆盖前 $\ell$ 层时的最大瓶颈。层数循环上界是 $\min(K_j,\ell)$，少一个就让单节点链永远不可行。并列时取较小的 $k$（只在严格变大时替换），回溯结果确定，测试还检查回溯出的层数确实实现了报告的瓶颈。

**B5**。为什么不用"排序后切成 $p$ 条链"就结束：最优放置里常有两张卡持有同一段层（副本）并行喂给下游，或一张卡的输出分流给两张卡，这是 DAG 上的流，不是不相交的链。单卡移动到不了这些解，因为只改一张卡时中间状态的流是 0。成对移动（副本或相邻区间）正好补上这一步。即便如此，这仍是启发式：在 60 个随机 5 卡以内的小集群上，参考实现有 7 个低于最优的 97%，最差约 0.87。所以测试的门槛是每例 90%、平均 0.97，再加一个与实现无关的性质：结果对所有单卡移动局部最优，由测试自己枚举验证。时间上，bench 的 14 卡、64 层模型按 4 层一块放置，单次规划约 2 到 7 秒。

**B6**。每次从 $S$ 沿编号最小的正流边走到 $T$，取路径最小流量后沿路减去；每轮至少把一条边减到零，所以路径数不超过正流边数。只有计算边贡献 GPU，否则一条 pipeline 里会出现重复的 GPU。

**B7**。平滑加权轮询：所有当前值加各自权重，选最大的，再把**总权重**从被选者身上减掉。可以证明任意前缀的计数偏差小于 1，所以测试用"偏差小于 1"作 oracle。减自己的权重或把权重写反，偏差都会无界增长。

## C 多模型共置

**C1**。份额 $\alpha$ 下所有 kernel 时间除以 $\alpha$，所以 $\lambda'T_p'=\lambda T_p$、观察到的 TPOT 放大 $1/\alpha$。首个输出 token 来自 prefill，decode 只有 $o-1$ 步。prefill 打断 decode 的比例是 $\lambda T_p'$，忽略它会高估 TPOT 很紧时的容量。

**C2**。下界 $\max_m\rho_m\min_t\pi_t/c_{t,m}$ 有效，因为每个模型的剩余需求都必须被覆盖，而单位容量的单价有下限；对模型求和则把同时服务多个模型的模板的价格算了多次，会剪掉共置最优解。分支上限取覆盖所有正剩余需求所需台数的最大值（向上取整），再往下枚举到 0。在 bench 的 42 个模板上，B&B 访问约 1.4 万个节点、用时约 1 ms。

**C3**。实现要点是面积 $\int N(t)\,dt$ 的精确累计：迭代期间系统内的请求（含排队的）乘 $dt$，加上迭代中途到达的请求从到达时刻到迭代结束的部分。Little 定律的检查因此不是恒等式：漏掉排队请求或中途到达，面积就和逗留时间之和对不上。M/D/1 的检查把整套"到达时刻、FIFO、服务开始"串起来，对照 Pollaczek-Khinchine 公式。

**C4 与 bench 的发现**（modeled, not measured）。四模型场景中，B&B 最优（两张 A10G 各共置两个模型，2.01 USD/h）在仿真里 Poisson 到达下 SLO 达成率只有约 73%，突发时约 49%；不共置方案贵 18%，Poisson 下达成率 100%；贪心方案（独立选单价最低的单模型模板）要 6.59 USD/h，因为它忽略了整数取整。加 30% 余量重新规划后，达成率反而更低（约 44%）。新方案把 Qwen3-0.6B 和 Qwen3-8B 放在同一张 A10G 上，可能是小模型的大量请求拖慢了大模型；这一点没有单独验证，报告里可以作为待查的负结果。这说明 C1 的容量是 SLO 的边界，不是工作点；把仿真放进规划回路，或给每个实例设利用率上限，是这个 lab 留给研究的问题。

## R 真实系统

**R1**。统计只用正式重复；分位数用线性插值 $h=(n-1)p$。`globaltimer` 自旋 kernel 在这台笔记本上会比设定时长多出 0 到约 0.1 ms，所以分位数测试的期望值由测试自己逐个计时得到，容差 0.12 ms，仍然小于最近秩法造成的 0.25 ms 偏差。

**R2 到 R4**。拷贝必须用 grid-stride（launcher 把 grid 限制在 $8\times$SM 数个 block）。GEMV 一个 warp 一行、16 字节向量读、shuffle 规约。WMMA GEMM 的共享内存行尾加 8 个 half，`load_matrix_sync` 的 ldm 必须用带 padding 的行宽。bench 在每组配置计时前先让 GPU 连续忙 300 ms（`keep_busy`），否则 GPU 在配置之间降到低 P-state，访存受限的 kernel 会慢很多；每行输出 `stable`（$p_{75}\le1.5\,p_{25}$），不稳定就重测最多两次。随附 profile（RTX 4060 Laptop，驱动 566.24，WSL2，2026-10-18，measured，30 行全部 stable）：64 到 512 MiB 拷贝 227 到 231 GB/s，GEMV 235 到 247 GB/s，WMMA GEMM 在 2048 方阵上 22.1 TFLOPS（cuBLAS 25.4）、4096 方阵上 18.2 TFLOPS（cuBLAS 25.7）；skinny GEMM 只有 cuBLAS 的 60% 到 80%，可能是 tile 沿 M 只有 64、权重被重复读，没有用 profiler 确认。拟合 $\eta_c\approx0.67$、$\eta_m\approx0.91$、$t_0\approx4.3\,\mu s$；copy 加 GEMM 训练、GEMV 测试的 held-out 平均误差约 9.5%。加 300 ms 预热前的一次运行里拷贝只有 197 GB/s，而 2048 GEMM 反而达到 29.2 TFLOPS：短时突发和持续负载的性能不同，profile 要写明测量方法。

**R5/R6**。字节数是"每个操作数读一次"的必要流量，是真实 DRAM 流量的下界；skinny GEMM 实际流量更大，所以在模型里看起来"比带宽允许的还快"或"比预测慢"都有可能，报告要指出来。扫描里成本是"按当前 $\eta$ 把方案等比放大到刚好覆盖需求"的每小时成本，某个模型容量变成 0 时为无穷大。参考运行中，全部 GPU 的 $\eta$ 缩到 0.75 时名义最优方案就被同构 A10G 方案超过（Kendall 距离 3），只缩 $\eta_m$ 时同样在 0.75 翻转。

## 运行命令

```bash
python tools/run.py --lab 9 --impl reference --mode test --part all
python tools/run.py --lab 9 --impl reference --mode bench --part all
python tools/run.py --lab 9 --impl reference --mode test --part R --device cuda    # WSL/Linux
python tools/run.py --lab 9 --impl reference --mode bench --part R --device cuda   # WSL/Linux
python tools/fault_inject.py --lab 9                 # CPU faults
python tools/fault_inject.py --lab 9 --device cuda   # CUDA faults (WSL/Linux)
```

## oracle 与 fault 对照表

`fault_inject` 在 WSL 下的结果（2026-10-18）：`--device cpu` 50/50 CAUGHT，`--device cuda` 58/58 CAUGHT（含 8 个只在 GPU 上运行的 R1 到 R4 fault），0 STALE。状态列取自 `results/faults-lab9-cuda.json`。CPU 运行加 `--require-coverage` 时会因为 R1 到 R4 只有 CUDA fault 而报覆盖不足，这是设备过滤造成的，CUDA 运行覆盖全部 22 个 TODO。

| TODO | 独立 oracle                                 | fault id                               | 注入的错误                                                                                                                                    | 在哪个 part 失败 | 状态   |
| ---- | ------------------------------------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ------ |
| A1   | tiny 模型逐元素暴力计数 + Qwen3-1.7B 手算   | `a1-decode-missing-kv-read`            | decode cost without reading the cached K/V: the brute-force byte counter (and the 4 MiB hand value) rejects it                                | A                | CAUGHT |
| A1   | tiny 模型逐元素暴力计数 + Qwen3-1.7B 手算   | `a1-gqa-attention-flops-kv-heads`      | GQA shares K/V but every query head computes its own QK^T and PV; the explicit MAC loop over query heads catches it                           | A                | CAUGHT |
| A1   | tiny 模型逐元素暴力计数 + Qwen3-1.7B 手算   | `a1-causal-pairs-off-by-one`           | token i of a prefill attends to i + 1 positions (itself included); the brute-force loop counts context + i + 1                                | A                | CAUGHT |
| A1   | tiny 模型逐元素暴力计数 + Qwen3-1.7B 手算   | `a1-kv-write-only-k`                   | writes only K rows for new tokens; V must be written too                                                                                      | A                | CAUGHT |
| A2   | 四个手算时间（含恰在 ridge 上）             | `a2-roofline-sum-not-max`              | adds compute and memory time instead of overlapping them; the at-ridge hand value doubles                                                     | A                | CAUGHT |
| A2   | 四个手算时间（含恰在 ridge 上）             | `a2-eta-multiplies-time`               | efficiency applied in the wrong direction (achieved = eta * peak, so time divides by eta)                                                     | A                | CAUGHT |
| A3   | 无噪声已知系数恢复 + 测试自写目标的有限差分 | `a3-unweighted-least-squares`          | absolute instead of relative error: still exact on noise-free data, but the finite-difference gradient of the weighted objective is not zero  | A                | CAUGHT |
| A3   | 无噪声已知系数恢复 + 测试自写目标的有限差分 | `a3-no-regime-relabel`                 | labels regimes once with the spec ridge; samples between the true and the spec ridge stay mislabeled, so known coefficients are not recovered | A                | CAUGHT |
| A3   | 无噪声已知系数恢复 + 测试自写目标的有限差分 | `a3-swap-eta`                          | compute and memory coefficients swapped when unpacking the solution vector                                                                    | A                | CAUGHT |
| A4   | 暴力计数 + 恰好装满 12345/12344 边界        | `a4-tied-embedding-counted-twice`      | tied embeddings share one matrix with the LM head; the hand total for Qwen3-1.7B (3,441,149,952 B) rejects it                                 | A                | CAUGHT |
| A4   | 暴力计数 + 恰好装满 12345/12344 边界        | `a4-exact-fit-floor`                   | floating-point floor at an exact fit: capacity = need(12345) must return 12345, not 12344                                                     | A                | CAUGHT |
| A4   | 暴力计数 + 恰好装满 12345/12344 边界        | `a4-kv-only-k`                         | KV memory counts only K                                                                                                                       | A                | CAUGHT |
| A5   | 构造误差的独立计算 + 只用训练集             | `a5-median-even-n`                     | median of an even-sized set must average the two middle values                                                                                | A                | CAUGHT |
| A5   | 构造误差的独立计算 + 只用训练集             | `a5-fit-on-test-set`                   | held-out error computed after fitting on the test set (leakage); the fit-uses-train check catches it                                          | A                | CAUGHT |
| B1   | 手算边多重集 + 测试自写建图规则             | `b1-compute-cap-not-divided-by-layers` | a node holding k layers is k times slower; the hand edge multiset (600/600/300) rejects the undivided capacity                                | B                | CAUGHT |
| B1   | 手算边多重集 + 测试自写建图规则             | `b1-no-node-splitting`                 | node capacity lost (GPU modeled as an unconstrained vertex): max-flow no longer equals the enumerated min-cut of the true graph               | B                | CAUGHT |
| B1   | 手算边多重集 + 测试自写建图规则             | `b1-link-cap-from-compute`             | node-split capacity put on the outgoing link edge instead of the link bandwidth                                                               | B                | CAUGHT |
| B1   | 手算边多重集 + 测试自写建图规则             | `b1-overlapping-ranges-connected`      | connects overlapping ranges; a pipeline must hand off exactly at the layer boundary                                                           | B                | CAUGHT |
| B2   | 枚举全部 S-T 割 + 撤销图手算                | `b2-no-reverse-residual`               | Dinic without reverse residual arcs cannot cancel flow; the cancellation graph stops at 1 instead of 2                                        | B                | CAUGHT |
| B2   | 枚举全部 S-T 割 + 撤销图手算                | `b2-single-phase`                      | stops after the first blocking flow; random graphs disagree with the enumerated min-cut                                                       | B                | CAUGHT |
| B2   | 枚举全部 S-T 割 + 撤销图手算                | `b2-push-ignores-bottleneck`           | pushes more than an earlier arc can carry: capacity check and max-flow = min-cut both fail                                                    | B                | CAUGHT |
| B3   | 测试重算残量可达集 + 枚举最小割             | `b3-forward-arcs-only`                 | residual reachability must follow backward arcs that carry flow; the test recomputes the reachable set                                        | B                | CAUGHT |
| B3   | 测试重算残量可达集 + 枚举最小割             | `b3-cut-counts-both-directions`        | edges from the T side back to the S side are not part of the cut capacity                                                                     | B                | CAUGHT |
| B4   | 枚举 L 的全部切分                           | `b4-dp-layer-range-off-by-one`         | DP never gives a node its full allowed range; composition enumeration finds better or feasible plans                                          | B                | CAUGHT |
| B4   | 枚举 L 的全部切分                           | `b4-ignores-link`                      | chain bottleneck must include the link into each node                                                                                         | B                | CAUGHT |
| B4   | 枚举 L 的全部切分                           | `b4-ignores-sink`                      | the last node's link to the coordinator also bounds the chain                                                                                 | B                | CAUGHT |
| B5   | 穷举放置（5 卡 8 层以内）+ 单卡移动局部最优 | `b5-no-local-search`                   | seeds only: single-node moves still improve the result, which the local-optimality oracle detects                                             | B                | CAUGHT |
| B5   | 穷举放置（5 卡 8 层以内）+ 单卡移动局部最优 | `b5-ignores-memory`                    | local search proposes a range one layer larger than the memory allows                                                                         | B                | CAUGHT |
| B5   | 穷举放置（5 卡 8 层以内）+ 单卡移动局部最优 | `b5-single-seed`                       | local search only from the best seed; on the replica-heavy fixed cases the mean ratio to the optimum drops below 0.97                         | B                | CAUGHT |
| B6   | 按边重组流 + 扇出手算例子                   | `b6-path-rate-first-edge`              | path rate must be the bottleneck of the walk; otherwise per-edge recomposition and conservation break                                         | B                | CAUGHT |
| B6   | 按边重组流 + 扇出手算例子                   | `b6-gpus-from-every-edge`              | pipeline lists GPUs once per compute edge only; link/sink edges must not add stages                                                           | B                | CAUGHT |
| B7   | 手算序列 + 前缀偏差小于 1                   | `b7-iwrr-weight-inversion`             | IWRR weight inversion: light pipelines get heavy traffic; prefix deviation and the hand sequence fail                                         | B                | CAUGHT |
| B7   | 手算序列 + 前缀偏差小于 1                   | `b7-subtract-own-weight`               | smooth WRR subtracts the TOTAL weight from the picked entry; subtracting its own weight breaks the proportional schedule                      | B                | CAUGHT |
| C1   | 闭式延迟函数暴力扫描 b                      | `c1-ignores-share`                     | a model with half the GPU sees prefill twice as long; the half-share case disagrees                                                           | C                | CAUGHT |
| C1   | 闭式延迟函数暴力扫描 b                      | `c1-no-prefill-interruption`           | decode pauses while other requests prefill; ignoring it overstates capacity under a tight TPOT                                                | C                | CAUGHT |
| C1   | 闭式延迟函数暴力扫描 b                      | `c1-output-steps-not-minus-one`        | the first output token comes from prefill: only output - 1 decode steps                                                                       | C                | CAUGHT |
| C1   | 闭式延迟函数暴力扫描 b                      | `c1-memory-limit-doubled`              | KV budget check too loose; the memory-bound case picks an infeasible batch                                                                    | C                | CAUGHT |
| C2   | 暴力枚举计数向量 + 共置手算例子             | `c2-sum-lower-bound`                   | B&B pruning bound wrong: summing per-model bounds overestimates when one instance serves several models and prunes the co-located optimum     | C                | CAUGHT |
| C2   | 暴力枚举计数向量 + 共置手算例子             | `c2-branch-upper-floor`                | branching upper limit rounded down: the count that covers a fractional demand is never tried                                                  | C                | CAUGHT |
| C2   | 暴力枚举计数向量 + 共置手算例子             | `c2-over-pruning`                      | heuristic pruning margin makes B&B miss optima that brute force finds                                                                         | C                | CAUGHT |
| C3   | 手算 trace + M/D/1 + Little 定律 + 吞吐上界 | `c3-area-ignores-queue`                | simulator not counting queueing: Little's law (area = sum of sojourn) and the hand area 9.5 fail                                              | C                | CAUGHT |
| C3   | 手算 trace + M/D/1 + Little 定律 + 吞吐上界 | `c3-area-misses-midstep-arrivals`      | requests that arrive during an iteration are in the system before they are admitted                                                           | C                | CAUGHT |
| C3   | 手算 trace + M/D/1 + Little 定律 + 吞吐上界 | `c3-decode-before-prefill`             | prefill-first rule replaced by decode-first; the hand trace's first-token times change                                                        | C                | CAUGHT |
| C3   | 手算 trace + M/D/1 + Little 定律 + 吞吐上界 | `c3-no-round-robin`                    | always restarting at slot 0 starves later models; the two-model round-robin check fails                                                       | C                | CAUGHT |
| C4   | 独立重算贪心 + 受限问题暴力解               | `c4-greedy-floor`                      | greedy baseline under-provisions; recomputed counts disagree                                                                                  | C                | CAUGHT |
| C4   | 独立重算贪心 + 受限问题暴力解               | `c4-homogeneous-first-type`            | homogeneous baseline must be the best single GPU type, not the first feasible one                                                             | C                | CAUGHT |
| R5   | 手算字节数                                  | `r5-copy-read-only`                    | a copy reads and writes every byte                                                                                                            | R                | CAUGHT |
| R5   | 手算字节数                                  | `r5-gemm-output-fp16`                  | the GEMM output is FP32 (4 bytes per element)                                                                                                 | R                | CAUGHT |
| R6   | 暴力计数不一致对 + 已知交叉点               | `r6-wrong-nominal-row`                 | rankings must be compared with the scale-1.0 row, not with the first row                                                                      | R                | CAUGHT |
| R6   | 暴力计数不一致对 + 已知交叉点               | `r6-rank-descending`                   | cheapest plan ranks first                                                                                                                     | R                | CAUGHT |
| R1   | globaltimer 自旋 + 测试自测分位数           | `r1-no-warmup`                         | skipping warmup times the slow first launches; the warmup-exclusion check sees a 4 ms max                                                     | R (cuda)         | CAUGHT |
| R1   | globaltimer 自旋 + 测试自测分位数           | `r1-nearest-rank-quantile`             | nearest-rank quantiles differ from linear interpolation by >= 0.25 ms on the 1..20 ms pattern                                                 | R (cuda)         | CAUGHT |
| R2   | 逐元素比较 + 越界哨兵                       | `r2-no-grid-stride`                    | one pass per thread with a capped grid leaves most of a 4M-element copy unwritten                                                             | R (cuda)         | CAUGHT |
| R2   | 逐元素比较 + 越界哨兵                       | `r2-tail-overrun`                      | the scalar tail writes past n; the sentinel guard band catches it                                                                             | R (cuda)         | CAUGHT |
| R3   | FP64 参考 + 行外哨兵                        | `r3-reduction-stops-early`             | warp reduction misses the last shuffle step                                                                                                   | R (cuda)         | CAUGHT |
| R3   | FP64 参考 + 行外哨兵                        | `r3-row-index`                         | one warp per row needs the block offset times warps per block                                                                                 | R (cuda)         | CAUGHT |
| R4   | FP64 参考 + NaN 预填                        | `r4-ldm-without-padding`               | leading dimension must include the shared-memory padding                                                                                      | R (cuda)         | CAUGHT |
| R4   | FP64 参考 + NaN 预填                        | `r4-half-k-step`                       | only the first 16 of each 32-deep K tile are multiplied                                                                                       | R (cuda)         | CAUGHT |
