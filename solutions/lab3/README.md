# Lab 3 参考实现导读

[任务](../../labs/lab3/README.md) · [评分](../../labs/lab3/grade.md) · [来源](../../labs/lab3/resources.md)

参考答案在 [pipeline_student.py](src/pipeline_student.py)（A1–A5、B1–B5）、[execution_student.py](src/execution_student.py)（C1–C4）、[fusion_student.cuh](src/fusion_student.cuh)（C5–C7）和 [graphs_student.py](src/graphs_student.py)（R1–R4）。文件名与学生侧一致，`run.py --impl reference` 用它们覆盖学生文件；测试、bench 和 `real/` 只在学生侧。

## A：trace 与计时统计

- **A1** 先切分再统计。先排序再切分会把 held-out 里常见的长度带进训练直方图，规划看起来更准，其实偷看了未来。
- **A2** 用中位数和 MAD 而不用均值和标准差，是因为笔记本 GPU 的计时有长尾（P-state 切换、后台进程）。一个 200 ms 的离群值会让均值失去意义，但它仍然被保留在原始样本里。premium 截断为非负：磁盘缓存命中时第一次调用可能比 steady 还快。
- **A3** 区间 $[i,j]$ 的成本是 $P(n_j)+t(n_j)(F_j-F_{i-1})$。premium 每个 bucket 只付一次，执行成本按最大长度计价，不做线性拟合。
- **A4** TV 在两个支撑集的并集上求和后除以 2，取值在 $[0,1]$ 内。旧测试只有 0 和 1 两个结果，而且两边样本数相等，所以“最近窗口用训练总数归一化”这类错误完全看不出来。新测试加了 0.25、0.5、样本数不等的用例和随机用例，oracle 用 `Fraction` 精确计算；漏除以 2 会在分数用例上得到两倍的结果。
- **A5** fallback 请求按真实行数执行。padding 比例按频次加权：$\text{waste}/\text{executed}$，而不是各形状比例的平均。

## B：有预算的分桶 DP 与运行时状态

- **B1** 状态是“已覆盖的形状数 $j$”与“已用的 bucket 数 $k$”的组合。证明见讲义 §5.3：最后一段的起点固定后，前缀必须最优，否则可以替换。复杂度 $O(Km^2)$。并列规则要写进代码（`<` 而不是 `<=`，从少到多枚举 $k$），否则结果依赖浮点噪声，测试也无法复现。
- **B2** `bisect_left` 找不小于长度的最小上界。比值恰好等于 `max_padding` 时允许补齐：讲义的约定是“最多补到两倍”，边界属于允许的一侧。
- **B3/B4** 插入只有在找到合法的淘汰对象后才修改状态，这样失败的插入不会留下半截状态。pin 计数保护正在执行的 artifact；lease 会改变 LRU 顺序，因为“刚用过”的项最不该被淘汰。
- **B5** 保守收益 $t_e-(t_c+m\cdot\text{MAD})$，累计收益必须严格大于 premium。等号拒绝是有意为之：持平时编译只增加风险（显存、尾延迟），没有收益。

## C：真实编译与融合 kernel

- **C1** 每次计时调用后都要同步。GPU 上的 kernel 是异步的，不同步只能量到 enqueue 时间。测试用 `torch.cuda._sleep` 构造一个已知时长的 GPU 任务来检查这一点。
- **C2** `calibration_ms` 包住整个候选循环。候选中可能有最终不被选中的，它们的编译时间同样是这次实验的成本。
- **C3** 两个输入要一起补齐（Inductor 为固定形状编译，输入形状不一致会触发 guard 失败并重编译），输出切回真实行数，`finally` 里释放 lease。
- **C4** 旧测试只检查了上界集合和一个事件类型，从没比较 held-out 的输出。只要路径对，输出即使与请求错位、甚至算的是另一组输入，测试也会通过（见 fault `c4-outputs-reordered` 和 `c4-heldout-seed-drift`）。新测试按约定的 seed 独立重建每个请求的输入，逐个与 FP64 oracle 比较，并检查 `replay_ms` 不包含 `calibration_ms`。
- **C5** 两级归约：warp 内 butterfly 用 `__shfl_xor_sync`，5 步后 32 个 lane 都持有总和，不需要再广播；warp 间经 shared memory 再做一次。写 `warp_sums` 之前的那次 `__syncthreads()` 不能省：同一个 kernel 第二次调用时，别的 warp 可能还在读上一次的结果。测试用小整数输入，浮点和是精确的，所以可以要求逐位相等。
- **C6/C7** 两遍循环中每个线程处理同一组下标，第二遍读回的是自己写的 residual，不需要额外同步。`float4` 和 `__half2` 只在宽度整除时使用，这时每行起点都对齐；奇数宽度退回标量。C7 的平方和必须用 FP32：测试里的大幅值行让每个线程的部分和都超过 65504。
- **测量**：`bench.cu` 和 `inductor_bench.py` 都把 20 次调用录成一张 CUDA graph，排除 launch 开销，先连续跑 300 ms 再计时。这样三种实现的差别只来自 kernel 本身。实测里 fused 和 Inductor 在大矩阵上都达到约 233 GB/s，基本是显存带宽上限；f32 $1024\times4096$ 时两 kernel 方案反而更快，原因可能是中间结果留在 L2 里，需要 `ncu` 确认。

## R：CUDA graph 分桶

- **R1** 与 vLLM 的查表、SGLang 的 `bisect_left` 规则相同：只往上补。
- **R2** 三个容易漏的点：拷入输入；清零补齐行（防止上一步较大 batch 的残留数据参与计算）；返回拷贝（静态输出缓冲区会被下一次回放覆盖）。测试先跑大 batch 再跑小 batch，并保留早先的输出，最后统一比较。
- **R3** 与 A2 同口径：replay 取中位数，premium 取捕获时间。
- **R4** 化归：枚举最大尺寸 $t$ 后，显存约束只剩 $\lvert S\rvert$ 一个变量，前缀直方图正好是 B1 的输入，超过 $t$ 的 batch 付 eager 成本。共享内存池的显存由最大的那张图决定（实测：64 个尺寸各自单独捕获时池为 2–4 MiB，8 个尺寸共享一个池时为 4 MiB），所以 $M(S)=\text{pool}(\max S)+o\lvert S\rvert$ 是合理的一阶模型。完整模型下池会大得多，挑战题要求重新测。
- **负结果**：2026-10-18 的参考 bench 中，modeled 排序是 DP 最好，但真实回放（每步一次同步）时 DP 与 vLLM 默认网格相差不到 2%，shift trace 上网格反而略快。报告应把它当作代价模型不完整的证据，而不是 DP 的胜利。
- **实测要点**：捕获时每张图都换一个新 stream，cuBLAS 会给每个 stream 再分配约 8 MiB 的 workspace，8 张图就多出 64 MiB。`GraphRunner` 因此用同一个捕获 stream，量池大小前先 `empty_cache`。

## 运行命令

```bash
# cwd: mlsys-intermediate/labs
python tools/run.py --lab 3 --impl reference --mode test --device cpu --part all
python tools/run.py --lab 3 --impl reference --mode test --device cuda --part all
python tools/run.py --lab 3 --impl reference --mode bench --device cuda --part all
python tools/fault_inject.py --lab 3 --device cpu
python tools/fault_inject.py --lab 3 --device cuda
```

2026-10-18 在 WSL（RTX 4060 Laptop，torch 2.14.1+cu126，CUDA 12.6）上：CPU test 为 A 43/43、B 124/124、C1–C4 38/38、R 463/463；CUDA test 另有 C1–C4 39/39、C5 6/6、C6 40/40、C7 32/32、R 485/485。fault_inject：cpu 49/49、cuda 12/12 CAUGHT，0 STALE。

## oracle 与 fault 对照表

| TODO | 独立 oracle                           | fault id                          | 注入的错误                      | 在哪个 part 失败 | 状态   |
| ---- | ------------------------------------- | --------------------------------- | ------------------------------- | ---------------- | ------ |
| A1   | `Counter(trace[:k])` 重算与手算 trace | `a1-sort-before-split`            | 先排序再切分，泄漏未来流行度    | A（cpu）         | CAUGHT |
| A1   | `Counter(trace[:k])` 重算与手算 trace | `a1-histogram-all-lengths`        | 直方图统计了 held-out           | A（cpu）         | CAUGHT |
| A2   | `statistics` 手算中位数和 MAD         | `a2-mean-not-median`              | 用均值代替中位数                | A（cpu）         | CAUGHT |
| A2   | `statistics` 手算中位数和 MAD         | `a2-premium-not-clamped`          | premium 未截断为非负            | A（cpu）         | CAUGHT |
| A2   | `statistics` 手算中位数和 MAD         | `a2-mad-around-mean`              | MAD 绕均值计算                  | A（cpu）         | CAUGHT |
| A3   | 逐区间手算                            | `a3-premium-per-call`             | premium 每次调用都付            | A（cpu）         | CAUGHT |
| A3   | 逐区间手算                            | `a3-smallest-shape-cost`          | 区间按最小长度计价              | A（cpu）         | CAUGHT |
| A4   | `Fraction` 精确 TV                    | `a4-tv-missing-half`              | TV 漏除以 2                     | A（cpu）         | CAUGHT |
| A4   | `Fraction` 精确 TV                    | `a4-ignore-new-shapes`            | 只在训练支撑上求和              | A（cpu）         | CAUGHT |
| A4   | `Fraction` 精确 TV                    | `a4-wrong-normalizer`             | 最近窗口用训练总数归一化        | A（cpu）         | CAUGHT |
| A5   | 测试内 first-fit 重算加权行数         | `a5-fallback-padded`              | fallback 也按补齐行计           | A（cpu）         | CAUGHT |
| A5   | 测试内 first-fit 重算加权行数         | `a5-unweighted-fallback`          | fallback 按形状而非调用计数     | A（cpu）         | CAUGHT |
| B1   | 枚举全部连续划分                      | `b1-backtrack-off-by-one`         | 回溯上界差一位                  | B（cpu）         | CAUGHT |
| B1   | 枚举全部连续划分                      | `b1-first-split-skipped`          | 分割点起点晚一位                | B（cpu）         | CAUGHT |
| B1   | 枚举全部连续划分                      | `b1-tie-more-buckets`             | 并列时偏向更多 bucket           | B（cpu）         | CAUGHT |
| B1   | 枚举全部连续划分                      | `b1-capacity-ignored`             | 忽略容量                        | B（cpu）         | CAUGHT |
| B2   | 线性 first-fit 扫描                   | `b2-ratio-boundary-strict`        | 比值恰为 2 时拒绝               | B（cpu）         | CAUGHT |
| B2   | 线性 first-fit 扫描                   | `b2-bisect-right`                 | 等于上界时补到下一个            | B（cpu）         | CAUGHT |
| B2   | 线性 first-fit 扫描                   | `b2-pad-to-smaller`               | 补到更小的 bucket               | B（cpu）         | CAUGHT |
| B3   | 手工追踪 put/lease 序列               | `b3-pin-ignored`                  | LRU 淘汰时忽略 pin              | B（cpu）         | CAUGHT |
| B3   | 手工追踪 put/lease 序列               | `b3-pinned-replace`               | 替换被 pin 的项                 | B（cpu）         | CAUGHT |
| B3   | 手工追踪 put/lease 序列               | `b3-refresh-no-touch`             | 刷新不移到 MRU                  | B（cpu）         | CAUGHT |
| B4   | 手工追踪 lease 计数与 LRU             | `b4-release-unchecked`            | 无 lease 也能 release           | B（cpu）         | CAUGHT |
| B4   | 手工追踪 lease 计数与 LRU             | `b4-no-lru-touch`                 | lease 不更新 LRU                | B（cpu）         | CAUGHT |
| B5   | 闭式回本不等式                        | `b5-threshold-inverted`           | 准入不等号写反                  | B（cpu）         | CAUGHT |
| B5   | 闭式回本不等式                        | `b5-equality-admits`              | 恰好回本也准入                  | B（cpu）         | CAUGHT |
| B5   | 闭式回本不等式                        | `b5-mad-ignored`                  | 忽略 MAD 裕量                   | B（cpu）         | CAUGHT |
| C1   | 调用计数、`time.sleep`、GPU `_sleep`  | `c1-warmup-short`                 | warmup 少一次                   | C（cpu）         | CAUGHT |
| C1   | 调用计数、`time.sleep`、GPU `_sleep`  | `c1-first-in-samples`             | 首次调用混进 steady             | C（cpu）         | CAUGHT |
| C1   | 调用计数、`time.sleep`、GPU `_sleep`  | `c1-no-sync`                      | 计时不同步                      | C（cuda）        | CAUGHT |
| C2   | FP64 oracle 与 Dynamo 包装检查        | `c2-eager-artifact`               | 保存 eager 函数当 artifact      | C（cpu）         | CAUGHT |
| C2   | FP64 oracle 与 Dynamo 包装检查        | `c2-calibration-last-only`        | 只计最后一个候选的校准时间      | C（cpu）         | CAUGHT |
| C3   | 逐请求 FP64 oracle 与 pin 计数        | `c3-pad-x-only`                   | 只补 x 不补 residual            | C（cpu）         | CAUGHT |
| C3   | 逐请求 FP64 oracle 与 pin 计数        | `c3-no-slice`                     | 不切回真实行                    | C（cpu）         | CAUGHT |
| C3   | 逐请求 FP64 oracle 与 pin 计数        | `c3-lease-leak`                   | lease 不释放                    | C（cpu）         | CAUGHT |
| C3   | 逐请求 FP64 oracle 与 pin 计数        | `c3-admission-ignored`            | 跳过准入                        | C（cpu）         | CAUGHT |
| C4   | 按 seed 重建输入后逐请求 FP64 oracle  | `c4-plan-on-holdout`              | 规划看到 held-out               | C（cpu）         | CAUGHT |
| C4   | 按 seed 重建输入后逐请求 FP64 oracle  | `c4-calibration-in-replay`        | 校准时间算进 replay             | C（cpu）         | CAUGHT |
| C4   | 按 seed 重建输入后逐请求 FP64 oracle  | `c4-outputs-reordered`            | 输出顺序与请求错位              | C（cpu）         | CAUGHT |
| C4   | 按 seed 重建输入后逐请求 FP64 oracle  | `c4-heldout-seed-drift`           | held-out 输入不是约定的请求     | C（cpu）         | CAUGHT |
| C5   | 小整数精确求和                        | `c5-shfl-offset-8`                | butterfly 从 8 开始             | C（cuda）        | CAUGHT |
| C5   | 小整数精确求和                        | `c5-shfl-width-16`                | shuffle width 设成 16           | C（cuda）        | CAUGHT |
| C5   | 小整数精确求和                        | `c5-only-first-warp`              | 第二级只读 warp 0               | C（cuda）        | CAUGHT |
| C6   | host FP64 oracle 与逐位 residual      | `c6-no-residual-writeback-vec`    | float4 路径漏写 residual        | C（cuda）        | CAUGHT |
| C6   | host FP64 oracle 与逐位 residual      | `c6-no-residual-writeback-scalar` | 标量路径漏写 residual           | C（cuda）        | CAUGHT |
| C6   | host FP64 oracle 与逐位 residual      | `c6-divide-by-blockdim`           | 均值除以 blockDim               | C（cuda）        | CAUGHT |
| C6   | host FP64 oracle 与逐位 residual      | `c6-gamma-vector-index`           | 向量路径 gamma 下标错           | C（cuda）        | CAUGHT |
| C7   | host FP64 oracle 与大幅值行           | `c7-half-accumulator`             | FP16 累加平方和                 | C（cuda）        | CAUGHT |
| C7   | host FP64 oracle 与大幅值行           | `c7-no-residual-writeback`        | half2 路径漏写 residual         | C（cuda）        | CAUGHT |
| C7   | host FP64 oracle 与大幅值行           | `c7-odd-tail-dropped`             | 奇数宽度丢最后一列              | C（cuda）        | CAUGHT |
| R1   | 线性 first-fit                        | `r1-pad-to-smaller`               | 补到更小的 graph                | R（cpu）         | CAUGHT |
| R1   | 线性 first-fit                        | `r1-bisect-right`                 | 等于捕获尺寸时补到下一个        | R（cpu）         | CAUGHT |
| R2   | 逐步对照 eager 并保留旧输出           | `r2-output-alias`                 | 返回静态缓冲区的 view           | R（cpu）         | CAUGHT |
| R2   | 逐步对照 eager 并保留旧输出           | `r2-output-alias-cuda`            | 同上（cuda 设备运行）           | R（cuda）        | CAUGHT |
| R2   | 逐步对照 eager 并保留旧输出           | `r2-pad-not-zeroed`               | 补齐行不清零                    | R（cpu）         | CAUGHT |
| R2   | 逐步对照 eager 并保留旧输出           | `r2-input-not-copied`             | 不拷入输入                      | R（cpu）         | CAUGHT |
| R3   | `statistics` 手算                     | `r3-mean-not-median`              | replay 用均值                   | R（cpu）         | CAUGHT |
| R3   | `statistics` 手算                     | `r3-capture-ignored`              | 丢掉捕获成本                    | R（cpu）         | CAUGHT |
| R4   | 全部子集暴力枚举                      | `r4-memory-ignored`               | 忽略显存预算                    | R（cpu）         | CAUGHT |
| R4   | 全部子集暴力枚举                      | `r4-graph-count-off-by-one`       | 图数量上限多一                  | R（cpu）         | CAUGHT |
| R4   | 全部子集暴力枚举                      | `r4-eager-tail-free`              | 超出最大尺寸的 batch 记为零成本 | R（cpu）         | CAUGHT |
