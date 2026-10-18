# Lab 7 参考答案导读

[任务与全部接口](../../labs/lab7/README.md) · [评分](../../labs/lab7/grade.md) · [来源](../../labs/lab7/resources.md)

参考实现覆盖 25 个 TODO：A1–A6、B1–B8、C1–C6、R1–R5。文件是 [moe_student.hpp](src/moe_student.hpp)（A/B 与 CPU 侧的 C）、[pipeline_student.cuh](src/pipeline_student.cuh)（C3/C4/C6）、[r_pipeline.cuh](src/r_pipeline.cuh)（R1–R5）。测试、基准和 driver 在 `labs/lab7/`，参考模式运行时用这里的同名文件覆盖学生骨架。错误注入清单在 [faults.json](faults.json)。

## A：选择与加权必须分开

**做法。** A1 先算两组量：`base`（`SigmoidRenorm` 用 sigmoid，其余用原 logit）和 `select`（`base` 加上 bias）。排序只看 `select`，权重只看 `base`，这就把“selection-only bias”直接编码进了数据流，而不是靠注释提醒。三个分支的分母各不相同：

- `TopkSoftmax`：分母是选中 $k$ 项 $\exp$ 之和；
- `SoftmaxNoRenorm`：分母是**全部** experts 的 $\exp$ 之和，所以权重和小于 1；
- `SigmoidRenorm`：分母是选中项 sigmoid 之和。

最后统一乘 `spec.scale`。排序用 `stable_sort` 加严格 `>`，并列时自然落到较小的 ID。

**为什么测试要这样写。** 旧版测试只有两行 logits：`{1000,1000,0}` 和 `{-10,-9,-8}`。第一行在“先 top-k 再 softmax”和“softmax 全部不重归一化”下都给 0.5/0.5，因为 $\exp(-1000)\approx 0$；第二行只断言 `routes[2].expert == 2`，根本没看权重。于是整整一类归一化错误可以通过。

新测试用 `{2,1,0,-1}`、$k=2$，三种约定各有一条精确断言（0.7310585786 / 0.6439142599 / 1.3661227579），并且同时比对一个 long double 写的独立 oracle。bias 用 $z=[0.1,0.2,0.3]$、$b=[0.9,0.5,0]$：bias 改变了选中集合（选 0 和 1 而不是 1 和 2），权重却必须来自无偏 sigmoid（0.48843761）。`{1000,1000,0}` 保留下来专门测溢出和并列。

**常见错误。** 不减最大值直接 `exp`（$\{1000,\ldots\}$ 变 nan）；把 bias 带进权重；用 `std::sort` 配 `>=` 比较器（违反严格弱序，并列顺序未定义）；`SoftmaxNoRenorm` 分母只算选中项（那就变成了 Mixtral）。

**A2–A6。** A2 是 counts → **排他**前缀和 → cursor 散射，桶内保持输入顺序。A3 每输出行用 $\max\lvert w\rvert/127$，全零行 scale 取 1，让 CPU 与 CUDA 共用同一套 INT8 布局。A4 是**加法** scatter，重复 route 刻意保留。A5 把每个专家补到 tile 整数倍，补出来的行 `route_index = -1`；C2 跳过这些行，绝不能向负索引散射。A6 的 `overflow` 只是诊断，不驱动丢 token，所以结果与 dropless oracle 可比。

A2 的测试用一个独立的 $O(RE)$ 扫描重建整个布局（而不是复用前缀和），因此“inclusive 前缀和”和“反向散射”都会被逐桶逐项地指出来。A6 的测试有两组熵：4 条 route 4 个专家（0.5623351446）和 6 条 route 3 个专家（1.0114042647）——**只有后者能区分“除以 route 总数”和“除以专家数”**，这是旧版漏掉的。

## B：时间、所有权和字节，三件事分开记

**做法。** B1 先把所有“ready 且 unpinned”的 entry 按策略排成候选序列，累加它们的字节直到够用，**确认够用之后**才真正逐出。失败路径一个 entry 也不动。新载入的完成时间是 $\max(\mathrm{now},\mathrm{link\_free})+\mathrm{bytes}/B$，从被接纳那一刻就占容量。

策略由 `Policy` 选：排序键是 `(uses_if_LFU, stamp, key)` 的三元组，LRU 时第一项恒为 0，于是同一段代码既是 LRU 也是 LFU。demand 访问已有 entry 时刷新 `stamp`、`++uses`、清 `prefetched`；**prefetch 访问已有 entry 什么都不做**——否则预测器就能靠猜把自己猜中的专家保护起来，`wasted` 也永远记不到东西。

B5 对整张转移表统一衰减再累加；B6 按置信度排序后逐个检查字节预算，放不下的**跳过**而不是终止扫描。B7 把 demand、prefetch、compute 和完成事件放在同一条时间线上：miss 时等到真实 `ready` 才 pin，pin 之后才 observe，然后才用 $\mathrm{compute}\times B$ 字节的预算做预测。

**为什么测试要这样写。** 旧版 B 测试只有 5 个命名检查，而且 `b_admission` 没有区分“先规划后提交”和“边逐出边检查”——两者在成功路径上完全一样。新测试在失败路径上断言 `c.entries.count(1) == 1`，即**失败的 10 B 请求之后 8 B 的专家必须还在**。

`b_policy` 是新增的：3 个 10 B 专家填满 30 B 缓存，再依次 demand `0,0,1,2`，此时使用次数是 $0{:}3,1{:}2,2{:}2$ 而最近使用顺序是 $0,1,2$。接纳第 4 个专家时 LRU 淘汰 0、LFU 淘汰 1，两条断言互为对照，LFU 一旦退化成 LRU 立刻失败。

`b_predictor` 的相变用例特意做成“30 次旧热点 + 11 次新热点”：不衰减时 $p(\text{旧})=30/41=0.73$ 会选错，衰减后新热点占优。旧版的 10 + 30 两种答案都是新热点，所以测不出“根本没有遗忘”。超大候选那一条同理：必须让**置信度最高的那个**放不下，才能区分 `continue` 和 `break`。

`b_replay` 除了手算的 finish 24，还检查三个字节口径自洽：`wasted + stranded <= prefetched <= transferred`，并且在一条 7 专家 2 槽的相变 trace 上要求 `wasted > 0 && stranded > 0`——不记账的实现会在这里暴露。

B8 是隔离的 Belady 诊断下界，只适用于等大小专家，允许使用未来信息。它不参与 B7，也不能把 trace 交给 `Predictor`。测试除了手算的 4，还在 12 个 (trace, slots) 组合上与一个独立写的 Belady 模拟逐一比对，并检查它确实不超过在线 LRU 的 miss 数。

**常见错误。** 把 `Loaded` 当成数据已就绪；失败的准入先逐出再报错；忘记 $\max(\mathrm{now},\mathrm{link\_free})$（等于假设有无限条 copy engine）；LFU 的并列不看 `stamp` 导致结果依赖 map 的遍历顺序；B7 里 predict 跑在 observe 之前（预测的是当前这个已驻留的专家）。

## C：确实算了，而且算的是正确的那个专家

**做法。** C1 实现量化两层 MLP，C2 用 packed 布局执行并还原，C5 串起真实的 dense gate 与 residual。CPU 测试用独立的 FP64 累加当 oracle，不拿同一个函数的另一条路径互相印证。

C6 把 up/down 和各自的 scale 放进连续的 pinned 记录（`qn` 数 INT8 字节，`sn` 数 float **元素**，这是最容易写错的一行）；C3 用同样的 offset 从 GPU 槽里读。C4 每次挑一个槽，在 copy stream 上等该槽上次的 `done`，复制参数后记录 `ready`；compute stream 等 `ready`，连续执行 up/down，最后记录 `done`。不同槽独立推进，同一个槽在上次 compute 读完之前不得覆盖。

每条 route 独立写输出，最后由 CPU 做加权合并，所以 GPU 端不需要 atomic。这个设计便于验证，不追求吞吐。

**为什么测试要这样写。** 旧版只比了输出数值，而 5 个 token、8 个专家的 kernel 只有几微秒，槽复用的竞态几乎打不中。新增的 `c_cuda_slot_reuse` 用 512 个 token、宽 65、8 个专家、重复 8 次，让每个槽被反复覆盖且 kernel 足够长，缺了任一方向的依赖都会变成数值错误。`c_block` 新增了一个**非方阵** gate（3 专家 $\times$ 宽 2）：方阵单位阵转置之后还是单位阵，只有非方阵才能抓住把 gate 读成 $[\mathrm{width},\mathrm{experts}]$ 的错误。

**这里不测重叠。** 专家记录只有 17–67 KB，在 12.6 GB/s 下是 1.3–5.3 µs，比 WSL2 的 kernel launch 延迟还短；墙钟又包含 pinning、`cudaMalloc`、stream/event 创建和 CPU combine；而且这些事件是用 `cudaEventDisableTiming` 建的，根本不能计时。定量的重叠测量在 R 阶段。

## R：真实流水线，用数据而不是时间来判对错

**做法。** R1 一次性建好 pinned 专家池、每槽一块清零的显存、一条 copy stream 加每槽一条 compute stream、每步 4 个启用计时的事件和一对 wall 事件，以及 device 侧的校验和与 `bad` 计数器。**计时区间里不允许有任何分配。**

R2 建立三条依赖：拷贝等该槽上次 `comp_end`（写后读）、计算等自己的 `copy_end`（读后写）、循环结束后 copy stream 等每个槽最后一次 `comp_end`。第三条最容易漏：compute stream 建成 `cudaStreamNonBlocking` 之后不会自动与默认流汇合，少了它 `cudaEventElapsedTime` 返回 `cudaErrorNotReady`——参考实现第一次跑就卡在这里。

R3 的 $\rho$ 把“被隐藏的时间”除以**较短的那一边**。分母写成 $T_{\mathrm{wall}}$ 看似合理，但在拷贝远长于计算时会永远接近 0，看不出流水线其实已经满了。R4 用 `rfix::spin_warmup` 先让 GPU 以同一个拷贝连续忙 300 ms，再取 9 次中位数，并用 `cudaPointerGetAttributes` 确认用的确实是 page-locked 内存。R5 的判定式里，**CPU 步不搬权重所以不改变驻留集合**，这条不变量最容易写错。

**为什么能确定性地抓住缺失的依赖。** 笔记本的时钟波动比效应还大，光比时间不可靠。`rfix::expert_consume` 先把整个槽读一遍求和，可选地自旋若干周期，再重读若干遍；任何一遍与第一遍不符就在 device 上 `atomicAdd` 到 `bad`。`r_race` 把自旋设成 $6 imes 10^{7}$ 周期（几十 ms，远长于一次 16 MiB 拷贝的 1.3 ms），缺了写后读依赖时覆盖必然落在 kernel 执行期间。缺读后写依赖则更直接：槽在 R1 里被清零，kernel 会把零求和，和 host 端按 `rfix::pattern` 算的校验和对不上。

**本机实测（RTX 4060 Laptop，WSL2，CUDA 12.6，12 步，9 次重复取中位数）。** `asyncEngineCount` 为 1；pinned H2D 12.6–12.8 GB/s，pageable 11.1–11.6 GB/s（1 MiB 时 12.0 对 9.0）。16 MiB 专家、`reads=2` 时：单槽 $\rho$ 精确为 0，双槽 0.876，三槽 0.889。把 `reads` 提到 16 让 $\sum T_{\mathrm{compute}}$ 从 1.9 ms 涨到 14.1 ms，墙钟只从 15.99 ms 涨到 17.48 ms——多出来的计算几乎全被藏进了拷贝的影子里。标量 CPU 专家在 16 MiB、`reads=2` 下要 7.74 ms，而 $T_{\mathrm{fetch}}+T_{\mathrm{gpu}}$ 只有 1.51 ms，所以**这台机器上纯标量的 CPU 专家从不划算**；bench 因此扫描一个标注为 modeled 的 `cpu_speedup` 列来定位 crossover。

## 如何复核

在 `labs/` 执行：

```bash
python tools/run.py --lab 7 --impl reference --part A --mode test
python tools/run.py --lab 7 --impl reference --part B --mode test
python tools/run.py --lab 7 --impl reference --part all --mode bench
# WSL / Linux
python tools/run.py --lab 7 --impl reference --part C --mode test --device cuda
python tools/run.py --lab 7 --impl reference --part R --mode test --device cuda
python tools/run.py --lab 7 --impl reference --part R --mode bench --device cuda
python tools/fault_inject.py --lab 7 --device cpu
python tools/fault_inject.py --lab 7 --device cuda
```

CPU 路径有 92 个命名检查，C 的 GPU 路径有 15 个，R 的六组共 50 个。B 的模拟时间不能用来证明 R 的加速，R 的 $\rho$ 也不能反过来当作 B 的事件模型已被验证。

对照答案时建议只看当前卡住的那个 TODO：先写出不变量、构造最小反例，再看实现。除了调用顺序，更要能解释为什么 `Blocked` 必须满足事务语义、为什么拷贝要等上一次 compute、以及为什么 $\rho$ 的分母不能是墙钟。

## oracle 与 fault 对照表

有三个注入一开始没有被抓住，原因值得写下来，因为它们都属于“错误实现仍然自洽”这一类：

- 把槽轮转整体偏移一位（`s = (i+1) % slots`）之后，写后读依赖仍然落在同一个被偏移的槽上，流水线照常正确，`Segment` 里的槽号又是在 TODO 之外算的。补救办法是**回读每个槽的末态**：槽 $s$ 必须装着它最后一步用的那个专家。
- 把 CPU 判定写成只比传输（漏掉 fetch 之后的 GPU 计算）时，原有的几个用例都落在两个判据的同一侧。补一个 $T_{\mathrm{fetch}}\le T_{\mathrm{cpu}}<T_{\mathrm{fetch}}+T_{\mathrm{gpu}}$ 的用例才能分开。
- 命中时不刷新 recency，在 `0,1,0,2,3,0,1,2` 上 LRU 与 FIFO 的命中数恰好相同。换成 `0,1,0,2,1` 才会分歧（LRU 1 次命中 4 次搬运，FIFO 2 次命中 3 次搬运）。

另有两个注入被证明**无法**从公开接口观察到，已替换成等价的真实错误：对整张转移表衰减与只对当前行衰减，在 `predict` 的行内比值下完全等价（改为“根本不衰减”）；`Pending` 的 demand 在本 replay 中不可达，因为预取预算就是 $\mathrm{compute} imes B$，被接纳的预取必然在计算窗口内完成（改为“不等传输完成就计算”）。

共 62 个 fault（`python tools/fault_inject.py --lab 7 --list` 可查覆盖数），每个 TODO 至少 1 个，核心 TODO（A1、B1、B7、C4、R2、R5）至少 3 个。

| TODO | 独立 oracle                                                                                    | fault id                            | 注入的错误                                    | 在哪个 part 失败 | 状态   |
| ---- | ---------------------------------------------------------------------------------------------- | ----------------------------------- | --------------------------------------------- | ---------------- | ------ |
| A1   | long double 重写三种约定 + 手算权重                                                            | `a1-topk-before-softmax`            | 不重归一化的分母只算选中项                    | A                | CAUGHT |
| A1   | 同上                                                                                           | `a1-renormalise-no-renorm`          | 给 `norm_topk_prob=false` 的约定加了重归一化  | A                | CAUGHT |
| A1   | 同上                                                                                           | `a1-bias-enters-weights`            | selection-only bias 进了权重                  | A                | CAUGHT |
| A1   | 同上                                                                                           | `a1-no-max-subtraction`             | `exp` 前不减最大值                            | A                | CAUGHT |
| A1   | 同上                                                                                           | `a1-unstable-tie-break`             | `sort` 配 `>=`，并列顺序未定义                | A                | CAUGHT |
| A2   | 独立 $O(RE)$ 布局扫描                                                                          | `a2-inclusive-prefix-sum`           | inclusive 前缀和                              | A                | CAUGHT |
| A2   | 同上                                                                                           | `a2-unstable-scatter`               | 反向散射打破桶内顺序                          | A                | CAUGHT |
| A3   | FP64 逐行 rowmax/127 元数据                                                                    | `a3-scale-denominator-128`          | 分母写成 128                                  | A                | CAUGHT |
| A3   | 同上                                                                                           | `a3-global-scale`                   | 整个矩阵共用一个 scale                        | A                | CAUGHT |
| A4   | 手算 $[2,4,6,8]\to[3,14]$                                                                      | `a4-overwrite-instead-of-add`       | `=` 代替 `+=`                                 | A                | CAUGHT |
| A4   | 同上                                                                                           | `a4-drop-route-weight`              | 合并时漏乘 gate 权重                          | A                | CAUGHT |
| A5   | 手写期望的 `route_index`/`values`                                                              | `a5-padding-marked-as-route-zero`   | padding 标成 route 0                          | A                | CAUGHT |
| A5   | 同上                                                                                           | `a5-tile-rounding-down`             | tile 向下取整丢掉末尾 route                   | A                | CAUGHT |
| A6   | 两组手算熵（4 route/4 专家、6 route/3 专家）                                                   | `a6-overflow-counts-tiles`          | overflow 按 padded 行算                       | A                | CAUGHT |
| A6   | 同上                                                                                           | `a6-entropy-over-tokens`            | 熵的分母用专家数                              | A                | CAUGHT |
| B1   | 手算准入时刻表 + LRU/LFU 分歧轨迹                                                              | `b1-evict-before-planning`          | 先逐出再检查                                  | B                | CAUGHT |
| B1   | 同上                                                                                           | `b1-evict-pinned`                   | 把 pinned entry 当受害者                      | B                | CAUGHT |
| B1   | 同上                                                                                           | `b1-copy-engine-not-serial`         | 忽略 `link_free` 排队                         | B                | CAUGHT |
| B1   | 同上                                                                                           | `b1-prefetch-promotes-entry`        | 预取触碰刷新 recency 与使用次数               | B                | CAUGHT |
| B1   | 同上                                                                                           | `b1-lfu-ignores-use-count`          | LFU 退化成 LRU                                | B                | CAUGHT |
| B1   | 同上                                                                                           | `b1-waste-not-counted`              | 不记未用即逐出的预取字节                      | B                | CAUGHT |
| B2   | 手算完成时刻与排序                                                                             | `b2-advance-reports-everything`     | 重复报告早已完成的加载                        | B                | CAUGHT |
| B3   | 状态机不变量                                                                                   | `b3-pin-while-loading`              | pin 在途加载                                  | B                | CAUGHT |
| B4   | 同上                                                                                           | `b4-release-without-lease`          | 释放不存在的 lease                            | B                | CAUGHT |
| B5   | 长陈旧相位后的预测                                                                             | `b5-no-forgetting`                  | 衰减因子改成 1.0                              | B                | CAUGHT |
| B5   | 同上                                                                                           | `b5-previous-not-updated`           | 不更新 `previous`                             | B                | CAUGHT |
| B6   | 手算置信度与预算                                                                               | `b6-stop-at-oversized-candidate`    | 遇到放不下的候选就 `break`                    | B                | CAUGHT |
| B6   | 同上                                                                                           | `b6-threshold-ignored`              | 忽略置信度阈值                                | B                | CAUGHT |
| B7   | 手算 finish 24 + 字节守恒                                                                      | `b7-prefetch-is-free`               | 只记预取字节不真发起                          | B                | CAUGHT |
| B7   | 同上                                                                                           | `b7-predict-before-observe`         | 从不 observe                                  | B                | CAUGHT |
| B7   | 同上                                                                                           | `b7-no-wait-for-ready`              | 不等传输完成就计算                            | B                | CAUGHT |
| B8   | 独立 Belady 模拟（12 个用例）                                                                  | `b8-evict-nearest-use`              | Belady 反向                                   | B                | CAUGHT |
| B8   | 同上                                                                                           | `b8-never-used-again-is-nearest`    | “永不再用”当成最近                            | B                | CAUGHT |
| C1   | FP64 反量化 MLP                                                                                | `c1-relu-on-the-output`             | 去掉隐藏层 ReLU                               | C                | CAUGHT |
| C1   | 同上                                                                                           | `c1-scale-applied-once`             | 漏乘第二层 row scale                          | C                | CAUGHT |
| C2   | 逐 route 的 FP64 oracle                                                                        | `c2-padding-rows-executed`          | padding 行也执行                              | C                | CAUGHT |
| C5   | 手算 $[3,5]$ + 非方阵 gate 的 $[6,3]$                                                          | `c5-gate-transposed`                | gate 读成 $[\mathrm{width},\mathrm{experts}]$ | C                | CAUGHT |
| C5   | 同上                                                                                           | `c5-residual-missing`               | 丢掉 residual                                 | C                | CAUGHT |
| C3   | CPU `forward`（本身有 FP64 oracle）                                                            | `c3-channel-index-swapped`          | 线程到 (token, channel) 的映射转置            | C (cuda)         | CAUGHT |
| C3   | 同上                                                                                           | `c3-missing-row-scale`              | 不反量化                                      | C (cuda)         | CAUGHT |
| C6   | 同上（奇数宽度 + 槽复用）                                                                      | `c6-scale-offset-in-bytes`          | float 元素当字节                              | C (cuda)         | CAUGHT |
| C4   | 同上 + 512 token 槽复用压力                                                                    | `c4-copy-does-not-wait-for-compute` | 去掉写后读依赖                                | C (cuda)         | CAUGHT |
| C4   | 同上                                                                                           | `c4-compute-does-not-wait-for-copy` | 去掉读后写依赖                                | C (cuda)         | CAUGHT |
| C4   | 同上                                                                                           | `c4-ignores-slot-limit`             | 写死两个槽，不理会 `slots` 参数               | C (cuda)         | CAUGHT |
| R1   | `cudaPointerGetAttributes` + 回读槽                                                            | `r1-pageable-pool`                  | 池子不是 pinned                               | R                | CAUGHT |
| R1   | 同上                                                                                           | `r1-slots-not-zeroed`               | 槽不清零                                      | R                | CAUGHT |
| R1   | 同上                                                                                           | `r1-one-buffer-for-all-slots`       | 所有槽别名到同一块显存                        | R                | CAUGHT |
| R1   | 同上                                                                                           | `r1-events-without-timing`          | 事件用 `cudaEventDisableTiming`               | R                | CAUGHT |
| R2   | host 端 `rfix::expected_sum` + device 端重读检查                                               | `r2-copy-does-not-wait-for-compute` | 去掉写后读依赖                                | R                | CAUGHT |
| R2   | 同上                                                                                           | `r2-compute-does-not-wait-for-copy` | 去掉读后写依赖                                | R                | CAUGHT |
| R2   | 同上 + 实测 $\rho$                                                                             | `r2-single-stream`                  | 拷贝与计算同流                                | R                | CAUGHT |
| R2   | 同上 + 回读每个槽的末态                                                                        | `r2-wrong-slot-index`               | 槽轮转偏移一位                                | R                | CAUGHT |
| R2   | 同上 + 实测 $\rho$                                                                             | `r2-sync-inside-the-loop`           | 循环内每步同步                                | R                | CAUGHT |
| R2   | 同上                                                                                           | `r2-expert-offset-ignored`          | 按步号而不是专家号索引池子                    | R                | CAUGHT |
| R3   | 手算 0 / 1 / 0.5 三个 $\rho$                                                                   | `r3-overlap-divided-by-wall`        | 分母用墙钟                                    | R                | CAUGHT |
| R3   | 同上                                                                                           | `r3-overlap-sign-flipped`           | 重叠公式写反                                  | R                | CAUGHT |
| R4   | 测试自己用事件重测同一个拷贝                                                                   | `r4-bandwidth-in-gibibytes`         | GiB/s 冒充 GB/s                               | R                | CAUGHT |
| R4   | 同上（`cudaPointerGetAttributes`）                                                             | `r4-pageable-control-is-pinned`     | pageable 对照组也用 pinned                    | R                | CAUGHT |
| R5   | 独立的 LRU 走查 + 两端极值                                                                     | `r5-cpu-step-warms-the-cache`       | CPU 步也让专家驻留                            | R                | CAUGHT |
| R5   | 同上 + $T_{\mathrm{fetch}}\le T_{\mathrm{cpu}}<T_{\mathrm{fetch}}+T_{\mathrm{gpu}}$ 的边界用例 | `r5-fetch-cost-ignores-compute`     | 只比传输不比计算                              | R                | CAUGHT |
| R5   | 同上                                                                                           | `r5-fetch-time-unit`                | 秒和毫秒混用                                  | R                | CAUGHT |
| R5   | 同上 + 轨迹 0,1,0,2,1（LRU 与 FIFO 必然分歧）                                                  | `r5-lru-not-refreshed-on-hit`       | 命中不刷新 recency（LRU 变 FIFO）             | R                | CAUGHT |
