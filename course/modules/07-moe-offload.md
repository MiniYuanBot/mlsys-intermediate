---
title: D7 MoE 卸载：路由布局、专家缓存与异步预取
lab: lab7
hours: 5
tags: [lecture, moe, offload, cuda, cpp]
---

# D7 MoE 卸载：路由布局、专家缓存与异步预取

[← 课程索引](../index.md) · [配套 lab7](../../labs/lab7/README.md) · [源码阅读地图](../readings/code-map.md)

> [!abstract] 这一讲要解决的三个问题
>
> 1. 每个 token 只激活少数专家，为什么模型仍可能放不进显存？
> 2. 如何按专家重排 token，同时保留加权输出的精确语义？
> 3. 为什么预取能减少 demand miss，却增加总流量甚至拖慢执行？

本讲接续 foundations 的 CUDA、带宽测量和异步执行，逐步分析 route→dispatch→load→compute→combine 流程，不要求训练大型 MoE。学习时需要同时考虑数据布局、缓存生命周期和预测策略，理解它们如何相互制约。

**5h 学习安排**：§1 模型与资源账 0.75h；§2–§3 路由手算 1.25h；§4–§5 缓存/预测推演 1h；§6 CUDA 时间线 1h；§7–§8 论文与假设 1h。另配 26h 实验：基础6h、核心7h、前沿9h、报告4h。实验要求实现量化专家的实际计算和双槽 CUDA 依赖关系，仅运行传输校准还不够。

随堂题计入上面的手算与阅读时间。每节先独立写出计算、反例或依赖关系，再展开核对要点；图解先看指定位置，整篇论文与完整文档留作选读。文末清单用于检查是否能进入实验，不增加必交作业。

| 节    | 本讲产物                                  | lab 对应 |
| ----- | ----------------------------------------- | -------- |
| §1    | 参数与传输下界估算                        | A0       |
| §2–§3 | offsets、ordered routes、weighted outputs | A1–A6    |
| §4–§5 | 缓存事件表和总流量账                      | B1–B8    |
| §6    | 双 buffer 的 happens-before 图            | C3/C4/C6 |
| §7–§8 | 计算布局与卸载机制对照                    | C1/C2/C5 |

## §1 稀疏计算不等于稀疏存储

### 1.1 一个 MoE 层的最小形式

router 为 token $x$ 产生专家权重，选择集合 $\mathcal{K}(x)$，输出：

$$
y(x)=\sum_{e\in\mathcal{K}(x)}g_e(x)f_e(x).
$$

实验 A1 从 $[\mathrm{tokens},\mathrm{experts}]$ logits 中选择 top-k：相同 logit 优先选择编号小的专家，再只对选中的 $k$ 个值做 softmax，得到 `Route{token,expert,weight}`。C5 则先计算 dense gate，再调用这条路由路径，并在专家输出上加 residual。本实验实现这些前向计算，不训练 router。

dispatch 和 combine 接收的是已经确定的 routes，因此 combine 不能再次 softmax，也不能把重复路由的累加改为覆盖。A1 的 top-k 不会为同一 token 重复选择同一专家；A2/A4 仍用手工构造的重复路由检查布局与累加契约。

若有 $E$ 个专家，每个占 $S$ 字节，常驻存储仍需要近似 $ES$ 字节，而一个 token 的专家计算只涉及 $k$ 个专家。批次中的不同 token 可能选不同专家，使本轮需要的专家数远大于 $k$。

> [!example] 看图：沿着一个 token 找到专家
>
> 打开 [Switch Transformers 原论文图 2（PDF 第 4 页）](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf#page=4)，追踪两个 token 从 router 到不同 FFN 的路径，再在纸上补出“只选两位专家、仍保存全部专家”的 top-2 情况。这是网络结构示意，不能读成显存或传输实测。
>
> 读图时注意权重定义：论文的 Switch 路由先对全部专家做 softmax，再选 top-1，图中的 gate 值因此不一定为 1；本 lab 只对选中的 top-k 重归一化。两者不能直接共用一份数值 oracle。

### 1.2 三种“大小”应该分别报告

总参数量决定权重存储规模；active 参数量帮助估算每 token 算术量；本轮实际需要加载的专家工作集决定传输压力。它们不是同一个数。

假设 16 个专家，每个 64MiB，所有专家合计 1GiB；若显存缓存只能放 4 个，即 256MiB，top-2 并不免除其余 12 个专家的存储需求。它们仍需放在主机内存或其他存储层中，未命中时再载入。

### 1.3 传输下界

如果一次真实加载 $S$ 字节、有效带宽为 $\mathrm{BW}$，则：

$$
T_{\mathrm{copy}}\ge\frac{S}{\mathrm{BW}}.
$$

64MiB 专家、假设 12GiB/s 有效带宽，理想传输时间约为 $64/(12\times1024)$ 秒，即 5.2ms。这是手算，不是 RTX 4060 的实测。小传输启动开销、PCIe 拓扑、主机内存和争用都可能让实际值更大。

如果计算只需 1ms，仅提前一层发起传输，就很难完全隐藏 5.2ms 的传输时间。更早预取则需要更多缓存空间，也面临更高的误预测风险，因此需要权衡预取距离。

主机 RAM、pageable buffer、pinned buffer、SSD 是不同层。16GB 机器上不能只检查 GPU 预算；下载模型前应算总权重、进程常驻内存和加载时副本峰值。

> [!question] 随堂自测 D7-1：路由与资源预算
>
> 1. logits 为 $[2,2,0]$、$k=2$ 时，A1 应选哪两个专家、各自权重是多少？若改成 $k=1$，能否认为只需保存一个专家？
> 2. 把 §1.3 的专家大小减半为 32MiB，带宽仍为 12GiB/s，且只有 1ms 计算可以覆盖传输。理想传输下界和未被覆盖的等待各是多少？

> [!tip]- 核对要点
>
> 1. 选 expert0/1，权重都是 0.5；$k=1$ 选 expert0、权重为 1。一次选择不改变其他输入可能访问的专家集合，也不改变总权重存储需求。
> 2. 约 2.60ms 和 1.60ms。这里假设独占带宽且传输及时开始；启动和排队只会增加等待，数值不是硬件测量。

## §2 Dispatch：用计数排序得到专家连续布局

### 2.1 为什么需要重排

路由通常按 token 顺序生成，但专家计算希望同专家的 token 连续，以提高矩阵运算效率。dispatch 本质上是从 token-major 转成 expert-major，同时保存回到原 token 的映射。

用以下五条 route：

```text
r0: token0 -> expert2, weight=.75
r1: token0 -> expert0, weight=.25
r2: token1 -> expert2, weight=.50
r3: token1 -> expert2, weight=.50   # duplicate route kept intentionally
r4: token2 -> expert1, weight=1.0
```

有 4 个专家。$\mathrm{count}=[1,1,3,0]$，排他边界 $\mathrm{offsets}=[0,1,2,5,5]$。专家 e 占据 `ordered[offsets[e]:offsets[e+1]]`。最后一个 offset 必须等于 route 总数 R。

### 2.2 三步算法

第一遍校验 token/expert index，并统计每专家 route 数；第二遍对 count 做前缀和；第三遍按原 route 顺序，用每专家写指针 scatter。这里以有限权重为输入前提；当前 `dispatch` 只检查索引，不负责检测 NaN/Inf。

```text
expert0: r1
expert1: r4
expert2: r0, r2, r3
expert3: empty
```

因为第三遍按原输入顺序扫描，同一个 expert 内的 route 次序稳定。稳定次序并非得到正确数学输出的必要条件，但能简化测试、结果重现和后续布局分析。

算法复杂度为 $O(R+E)$，额外存储 $O(R+E)$。不要改为让每个专家都扫描全部 routes 的 $O(RE)$ 实现，否则无法练习这里的布局算法；后者可以保留为小规模独立 oracle。

### 2.3 不变量与边界

必须满足 offsets 长度 E+1、非递减、首项 0、末项 R；每个原 route 出现一次且仅一次；每个分段内 expert 字段正确。空专家区间长度为 0，空路由在 `tokens=0` 时合法。

不能忽略非法 expert index，否则得到的加速只是少做了工作。容量不足时，也不能静默截断 routes。本实验选择不丢 token 的语义；若研究训练中的 capacity/drop policy，那属于另一个会影响模型质量的变量。

> [!question] 随堂自测 D7-2：分桶与守恒
>
> 1. 在 §2 的五条 route 末尾追加 `r5: token2 -> expert0, weight=.2`。写出新的 offsets 和 ordered route 序列；空 expert3 的区间如何表示？
> 2. 若把 token1 的两条 expert2 route 去重，只保留一条且权重不变，哪些不变量被破坏？仅核对输出 shape 能否发现？

> [!tip]- 核对要点
>
> 1. offsets 为 $[0,2,3,6,6]$，顺序为 `r1,r5,r4,r0,r2,r3`；expert3 为 $[6,6)$。同专家内仍保持原输入次序。
> 2. route 数量与逐条保留契约被破坏，后续贡献会减半；shape 仍可能完全正确。需要检查 route 覆盖和数值累加，不能只检查维度。

## §3 Combine：逆映射不是简单拷贝

### 3.1 数据布局约定

`expert_outputs` 按 ordered route 顺序存放，每条有 width 个元素；它的长度必须是 $R\cdot\mathrm{width}$。输出是 $[\mathrm{tokens},\mathrm{width}]$，初始全零。

对 ordered 第 $i$ 条 route $r$，执行：

$$
\mathrm{out}[t_i,j]\leftarrow\mathrm{out}[t_i,j]+w_i\,\mathrm{expert\_outputs}[i,j].
$$

这里 $t_i$ 和 $w_i$ 分别对应第 $i$ 条 route 的 `token` 和 `weight`。把 `+=` 写成 `=` 会覆盖之前专家的贡献。CUDA 实现中若多个线程块同时累加同 token，还必须考虑原子操作或先局部归约；CPU 的顺序实现没有这个竞态。

### 3.2 一个完整数值例子

设 expert $e$ 对任一输入返回 $[e+1,2(e+1)]$。上例中 token0 为：

$$
0.75[3,6]+0.25[1,2]=[2.5,5].
$$

token1 为 $.5[3,6]+.5[3,6]=[3,6]$；token2 为 $[2,4]$。最终展平输出是 $[2.5,5,3,6,2,4]$。这是本节五条 route 的手算例子。

配套 [test.cpp](../../labs/lab7/test.cpp) 的 `a_dispatch/a_combine` 使用另一组更小的输入：$[(0,2,.5),(0,0,.5),(1,2,1),(1,2,1)]$。其 offsets 为 $[0,1,1,4,4]$；按分桶后的顺序给定单通道输出 $[2,4,6,8]$，合并结果应为 $[3,14]$。先算清这两个例子，再运行测试，避免混用 route 顺序。

这个简单的专家函数用于单独检验映射正确性，避免与专家网络的计算混在一起。通过这项测试，并不能证明真实 MoE MLP 或 grouped GEMM 已经正确实现。

### 3.3 GPU 化时新增的问题

dispatch 需要计数、前缀和与 gather/scatter；专家侧可能需要按大小不同的组做 GEMM；combine 要处理写冲突和数值顺序。小批次下，布局转换本身可能比专家算术更显著。

因此测完整系统至少分出 dispatch、H2D、expert compute、combine 和 wall time。由于不同阶段可能重叠，各部分耗时之和可能大于 wall time，不能按串行过程直接相加。

> [!question] 随堂自测 D7-3：合并语义与计时
>
> 1. 在 §3.2 的五条 route 例子中，只将 token1 第二条 route 的权重从 0.5 改为 0.25，其他输入不变。token1 的输出是多少？为什么不能在 combine 中再归一化？
> 2. 一份报告给出 H2D 4ms、compute 6ms、完整墙钟 7ms。能否直接判定计时错误？还需要检查什么？

> [!tip]- 核对要点
>
> 1. 输出为 $[2.25,4.5]$；权重和为 0.75 是输入契约的一部分，再归一化会改写结果。
> 2. 不能；复制与计算可能重叠。需核对测量窗口、同步边界和 profiler 时间线，确认各段是否属于同一次运行，不能单凭相加推翻墙钟。

## §4 专家缓存：区分命中、等待、发起加载与阻塞

### 4.1 Cache 接口的含义

当前 [Cache 接口](../../labs/lab7/src/moe_student.hpp) 是 `request(id, bytes, time, prefetch=false)`。容量以 bytes 计，时间以模拟 us 计，返回值有四种：

| 返回值    | 含义                             | 调用方下一步                              |
| --------- | -------------------------------- | ----------------------------------------- |
| `Hit`     | 专家已驻留，且 `ready <= time`   | 可以 `pin(id)` 后计算                     |
| `Pending` | 同一专家已有尚未完成的加载       | 等到该 entry 的 `ready`，不能重复提交复制 |
| `Loaded`  | 已为新专家预留空间并发起模拟加载 | 等待完成；此时还不能计算                  |
| `Blocked` | 专家过大，或没有足够的可淘汰空间 | 等待资源释放；超容量请求不能靠等待解决    |

特别注意名字容易误导的 `Loaded`：它表示加载已被接纳，不表示数据已就绪。模型只有一个串行 copy engine，新加载的完成时刻为 $\max(\mathrm{time},\mathrm{link\_free})+\mathrm{bytes}/\mathrm{bandwidth}$，并据此更新 `link_free`。loading entry 从接纳时就占容量，完成前不能 pin 或淘汰。

`advance(time)` 将时钟向前推进，返回这段时间内完成的专家，按完成时刻和 ID 排序。实验用 `ready` 时间和 `pins` 引用计数表示 absent→loading→ready→in use→ready 的生命周期；这些是 CPU 上的事件模型，实际 CUDA 依赖留到 C 阶段验证。

### 4.2 pin 优先于 LRU

`pin(id)` 只允许获取已经加载完成的专家，每次调用增加一个引用；`release(id)` 释放一个引用。不存在的专家、未完成加载的 pin、没有引用时的 release 都应报错。LRU 只在已就绪且 `pins == 0` 的专家中选择受害者。

按 `b_admission/b_lifetime` 手算：容量 20B、带宽 2B/us，0us 先请求 expert0 的 12B，再请求 expert1 的 8B。串行链路使它们分别在 6us、10us 完成。6us pin expert0；到 10us，请求 expert2 的 10B 仍应 `Blocked`：唯一可淘汰的 expert1 只能提供 8B。

这个失败请求不能先逐出 expert1 再发现空间不够。B1 要先规划全部受害者，确认能容纳新专家后再提交淘汰。释放 expert0 后，请求才能被接纳，并在 15us 就绪。若两个消费者各 pin 一次，一次 release 后引用数仍是 1，不能提前淘汰。这些约束分别检查容量原子性与资源生命周期。

### 4.3 预取也会改变 LRU

提供实现的 demand 和 prefetch 使用同一个 `request`。访问已有 entry 会更新 `stamp`；接纳预取会占用 bytes 和复制链路。demand 访问预取 entry 时清除其 `prefetched` 标记，无论此时已就绪还是仍在 loading。因此预取同时影响传输排队和未来淘汰次序。

画一条容量 2 的轨迹：当前热专家 1/2 交替；错误预取 3 驱逐 2；下一次 demand2 又驱逐 1；继续 demand1 需要再次加载。即便预测偶尔正确，总流量也可能高于 demand-only。

> [!question] 随堂自测 D7-4：缓存状态与原子性
>
> 1. 沿用容量 20B、带宽 2B/us 的例子：0us 请求 12B 的 expert0，1us 再请求它，应分别返回什么？调用 `advance(5)` 后能否 pin？随后 `advance(6)`，两次 pin 后只 release 一次，是否可淘汰？
> 2. expert0 的 12B 已被 pin、expert1 的 8B 已 ready 且无引用时，请求 10B 的 expert2 为什么必须保留 expert1？写出若“边淘汰边检查”会造成的状态变化。

> [!tip]- 核对要点
>
> 1. 先 `Loaded`，后 `Pending`；5us 不能 pin。6us 已 ready，但两次 pin 后一次 release 仍有一个引用，不能淘汰。
> 2. 最多只可释放 8B，接纳不可能成功。错误实现会先删掉 expert1 再返回 `Blocked`，失败请求也污染缓存；正确实现先规划足够的受害者再提交。

## §5 在线预测与流量账

### 5.1 历史转移表

实验预测器保存带遗忘的转移权重 $C[a,b]$。每次 `observe(e)` 先把整张表乘以 0.9，再给 previous→e 加 1，最后令 `previous=e`。较早的历史因此逐渐衰减。

`predict(bytes, budget, min_probability)` 取当前 previous 对应的一行，用 $p(b\mid a)=C[a,b]/\sum_j C[a,j]$ 作为预测置信度。先筛掉低于阈值的后继，再按置信度降序、expert ID 升序选择；只有放得进剩余 byte budget 的专家才能进入结果。冷启动或没有合格候选时返回空列表，不使用“累计两次才预测”的规则。这是经验转移概率，不是统计置信区间。

例如先观察 $0,1,0$，当前专家 0 唯一已知后继为 1。若 expert1 大小为 10B、阈值为 0.3，预算 10B 时返回 $[1]$，预算 9B 时返回空列表。随后持续观察 $2,0$，新的 0→2 转移会逐渐压过旧的 0→1；`b_predictor` 正是用热点变化和预算不足检查这种行为。

B7 在当前 demand 就绪并 pin 后调用 `observe`，再以 $\mathrm{compute}\cdot\mathrm{bandwidth}$ 为预取预算、0.3 为阈值选择候选。这个预算只估计计算窗口能覆盖多少传输；能否真正载入，还要经过缓存容量和同一复制链路的约束。

### 5.2 严格的信息边界

预测器只能看到已经 observe 的专家；完整测试 trace 只能供外部 evaluator 计算结果。基于下一条真实路由预取是 oracle，可作为诊断上界，不能混入在线策略。

校准时，不能在同一测试 trace 上反复选择最优阈值，再用该结果声称具有泛化能力。lab 要求保留原轨迹，并增加无局部性和新热点轨迹，用于区分开发与评价。

### 5.3 一张必须平衡的账

若每个专家都按 $S$ 字节加载，模拟总流量为：

$$
B_{\mathrm{total}}=S(N_{\text{demand load}}+N_{\text{prefetch load}}).
$$

例如 demand-only 有 100 次加载；预取后 demand 仅 40 次，但 prefetch 加载了 80 次，则总加载从 100 变成 120，增加 20%。如果只报告 demand miss 降低 60%，就会掩盖总流量增加的事实。

B benchmark 的专家大小固定为 2048B，容量为 4096/8192/12288B，带宽为 128B/us，每次专家计算为 20us；每种容量分别关闭和开启预取，共 6 行。120 次访问的前半段在 3 个专家间循环，后半段换成 4 个新专家。这些数值是事件模型的参数，不是硬件校准结果。

接口支持按专家传入不同 bytes，账本应逐次累加实际接纳的加载大小。C 阶段则实际使用 INT8 权重：一个 up $[H,D]$、down $[D,H]$ 专家的参数记录占 $2HD+4(H+D)$ bytes，其中包含 FP32 row scales；不能把 B 的 2048B 直接当作 C 的记录大小。

### 5.4 未使用预取与收益

要判断预取是否有效，还应记录“预取后在驱逐前是否被 demand 使用”、等待被隐藏的时间、预测 CPU 开销。B benchmark 提供 `finish_us/transferred_bytes/wasted_bytes/misses/blocked/oracle_misses`。`wasted_bytes` 只计未被 demand 访问就被逐出的预取，不含结束时仍驻留的预取；`transferred_bytes` 在加载被接纳时计入，因此可能含结束时尚未完成的预取。`blocked` 统计 demand 等待容量的重试次数，不包含被拒绝的预取候选。

`misses` 统计 demand 没有立即取得 ready 专家的次数，包含 `Pending` 和新加载。`oracle_misses` 则来自独立的等大小 Belady 换入模型：它知道未来，用来给 demand-only 的缓存未命中数作下界。预取可能把等待提前隐藏，因此开启预取后的 `misses` 可以低于 `oracle_misses`；这不表示突破了最少换入次数。比较时应同时看总加载 bytes 与完成时间。

总流量与临界路径时间也需要分别评价。提前多搬运少量数据，有时能减少等待，却会增加带宽压力。是否值得，需要结合真实 timeline 判断，不能仅凭 miss 推断整体性能。

> [!question] 随堂自测 D7-5：预测与指标口径
>
> 1. 历史只观察到 $0,1,0$，expert1 为 10B、阈值 0.3，预算 9B/10B 时分别预测什么？如果 evaluator 知道下一条是 expert2，能否把这个信息交给在线预测器？
> 2. demand-only 加载 100 次；预取方案 demand 加载 40 次、prefetch 加载 80 次，专家等大。总 bytes 如何变化？若后者 misses 小于 Belady 的 oracle_misses，是否说明突破了缓存最优下界？

> [!tip]- 核对要点
>
> 1. 分别为空列表、$[1]$。未来 expert2 只能用于隔离的 oracle；把它交给在线策略会泄漏未来。
> 2. 总 bytes 增加 20%。预取 misses 衡量未立即取得 ready 专家的 demand，Belady 衡量等大小 demand-only 的最少换入；两个事件计数不同，不能据此宣称突破下界。

## §6 CUDA 双 buffer：异步 API 不是重叠证据

### 6.1 从传输校准走到真实专家执行

`transfer_bench.cu` 只保留作可选传输校准。必修 [cuda_exercise.cu](../../labs/lab7/cuda_exercise.cu) 从 pinned host 载入 INT8 up/down 专家参数，执行 `down(ReLU(up(x)))`，将每条 route 的输出拷回 CPU 后加权合并。1/2 槽都必须与 CPU 数值 oracle 一致，奇数 D/H 与多次槽复用用于暴露偏移和生命周期错误。

C3 的 `expert_matvec` 让每个线程计算一个 route 的一个输出通道：INT8 权重参与内积，乘该行 scale 后得到浮点结果。它计算的是真实专家 MLP，但尚未采用 Tensor Core 或 grouped GEMM 优化。C5 的 dense gate 与 residual 在 CPU block 路径检查；CUDA 驱动接收预先生成的 routes，不能把其计时当作完整 block 的 GPU 延迟。

每个槽分别有一条 copy stream 和一条 compute stream。copy 等待该槽上一次 compute 的 done；本次复制完成后记录 ready；compute 等待 ready，执行 up/down，最后记录 done。下面只画每个槽的依赖链，双槽实际使用四条 stream：

```text
slot0: H2D0 -> ready0 -> use0 -> done0 -> H2D2 -> use2 ...
slot1:         H2D1 -> ready1 -> use1 -> done1 -> H2D3 -> use3 ...
```

host buffer 只读且在全部工作结束前保持有效。若主机在异步传输完成前改写源数据，同样会破坏语义。

### 6.2 缺少跨流依赖会怎样

如果把两个 stream 都改成写同一个 device buffer，stream1 的 copy 可能覆盖 stream0 尚未消费的数据。一次测试通过不足以保证安全；必须通过 event/wait 或分离 buffer 建立所需的 happens-before 关系。

pinning 是支持异步传输的必要条件之一；是否实际重叠还受 copy engine、带宽、kernel 资源和排队影响。调用 `cudaMemcpyAsync` 并不能证明传输时间已被完全隐藏。

> [!example] 看图：同样的异步调用，为什么时间线不同
>
> [NVIDIA 的传输重叠教程](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)给出 C1060 与 C2050 的执行时间线。只看这两组图：按 H2D、kernel、D2H 着色，圈出第一种提交顺序中的等待，再比较改变提交顺序后的临界路径。随后把本实验每槽的 ready/done 依赖画在旁边。
>
> 教程图是解释历史设备的示意，文中时间属于那些设备；不能把其中的引擎配置或加速比套到 RTX 4060。当前机器是否重叠，仍要看自己的 profiler 轨道。

### 6.3 应如何观察

用 [Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)检查 CUDA API、H2D 和 kernel 轨道。对比单槽和双槽的时间线，标出 overlap 区间和仍处于串行的区间。工具不可用时，保留 wall-time 结果，但 overlap 是否发生仍应标为未验证。

CUDA test 使用 3 个 seed、D=7/33、slots=1/2，共 12 组；bench 使用 D=64/128，同样为 12 组，每配置预热 1 次、计时 5 次取中位数。两者都要求 `max_abs_error <= 2e-4`。test 的单次时间包含冷启动影响，性能比较应使用 bench。

bench 的主机墙钟覆盖整个 `offload_forward`，包括 pinned buffer 准备、分配、输入与参数传输、kernel、输出回传、资源回收和 CPU combine。B 的缓存/预测策略尚未接入这条 GPU 路径，C 也没有跨调用保留专家缓存。因此 B 能解释策略的流量与等待，C 能验证实际专家卸载的数值与耗时，不能把两张表拼成完整 MoE 引擎的 tok/s。双槽是否更快仍取决于当前输入规模与实际重叠。

> [!question] 随堂自测 D7-6：异步执行证据
>
> 1. 若删掉“copy 等待该槽上次 compute done”的依赖，但保留“compute 等待本次 ready”，能否保证正确？写出一次会覆盖旧权重的交错。
> 2. 双槽 benchmark 更快，但没有 profiler；能否说已经证明 copy/compute 重叠？这次计时是否包含 GPU gate 和 residual，能否直接报告完整 MoE block 的 GPU 延迟？

> [!tip]- 核对要点
>
> 1. 不能：旧 compute 尚在读取槽0，新 H2D 已覆盖槽0，随后才完成 ready。新 compute 等 ready 并不能保护旧 compute；两个方向的生命周期依赖都需要保留。
> 2. 只能报告该输入下墙钟更短，重叠仍待验证。CUDA 路径接收给定 routes，gate/residual 属于 CPU block 路径；计时还包含准备、传输、回收及 CPU combine，不能改名为完整 block 的 GPU 延迟。

## §7 论文中“计算稀疏”和“存储卸载”是不同路线

[MoE-Infinity](https://arxiv.org/abs/2401.14361)研究个人设备上的专家卸载；[官方实现](https://github.com/EfficientMoE/MoE-Infinity)与[固定源码地图](../readings/code-map.md)提供 prefetch 生命周期及 I/O microbench 入口。当前工程版与原论文版本有差异，应先固定 commit 再阅读。

阅读 `ArcherPrefetchHandle` 时，用本讲四态模型标出 acquire、prefetch、ready、release 的对应位置；检查 buffer 是谁分配、哪个线程/stream 完成传输、取消预测是否回收资源。阅读时应重点找出本实验省略的并发状态，仅逐行翻译还不够。

[MegaBlocks（MLSys 2023）](https://github.com/stanford-futuredata/megablocks)侧重 MoE 稀疏计算布局。它帮助你看到，改善路由后的 GPU 计算与减少 PCIe 载入不是同一优化。即使 LRU 实现得很完善，也未必涵盖论文的贡献。

| 层次     | 可观察瓶颈             | 合适 baseline                  |
| -------- | ---------------------- | ------------------------------ |
| 路由布局 | scatter/稀疏 GEMM 开销 | 逐专家计算、已有 grouped 实现  |
| 专家缓存 | 工作集超过容量         | demand LRU、固定策略           |
| 预取管线 | 加载处于临界路径       | 无预取、总是预取、关闭 overlap |
| 模型执行 | 综合资源与质量         | 同模型、同路由、同生成请求     |

> [!question] 随堂自测 D7-7：论文机制与基线
>
> 1. 一种改动降低路由后的稀疏 GEMM 时间，另一种减少专家 H2D bytes。分别应与 §7 哪一层的 baseline 比较？只跑 LRU 命中率能评价前者吗？
> 2. 在 MoE-Infinity 阅读卡上，“本实验实现了预取”这句话还缺什么证据？至少补出两个生命周期问题，并指出应去哪里找答案。

> [!tip]- 核对要点
>
> 1. 前者需要逐专家或 grouped 计算基线，后者需要同容量、同路由的 demand-only/固定缓存策略。命中率没有测量稀疏 GEMM，不能评价计算布局收益。
> 2. 例如谁拥有传输 buffer、何时 ready、使用中能否淘汰、取消预取怎样回收。沿固定源码地图检查 handle、提交和释放入口，注明本实验的 CPU 事件模型与真实并发机制的差别。

## §8 一个可以逐层验证的研究问题

候选假设：预测置信度与缓存污染成本联合决定是否预取，在热点变化时比总是预取更稳。已有预取/缓存研究很多，首先应检查 MoE-Infinity 与相关工作，明确新增场景和约束。

第一步在 B 的 trace 上固定容量与路由，比较 demand-only、always-prefetch、选择性预取；看 demand misses、总 bytes 和未使用预取。第二步用 C 的真实专家执行器测量单槽/双槽耗时，并通过 profiler 分析复制和计算。第三步将 B 的缓存准入与预测接入 C，让专家跨调用驻留，再确认端到端净收益。每一步都可能推翻假设，相应的负结果也应保留。

获得组内服务器后再考虑真实 MoE 模型；先依据总权重和主机内存预算选择规模。活跃参数少并不保证下载/加载阶段能放下所有权重。

停止条件：收益只在周期性合成轨迹成立；未使用预取造成更大流量；GPU 已有计算瓶颈而非传输瓶颈；或优化通过丢 route 改变了输出。

> [!question] 随堂自测 D7-8：设计一个可推翻的实验
>
> 1. 若选择性预取只在周期 trace 上有效，进入热点突变后总 bytes 增加、finish_us 也变差，下一步应先申请更大 GPU，还是先调整假设？请给出一个能区分容量不足与预测失效的对照。
> 2. B 得到更少等待、C 的双槽也更快，是否足以宣称获得完整 MoE 推理加速？在进入真实模型前，当前还缺哪条必须实际运行的路径？

> [!tip]- 核对要点
>
> 1. 先检验假设。保持 trace 与带宽不变，分别扫描缓存容量和预测开关；容量充足时预测仍恶化、或错误预取激增，可帮助定位预测问题。保留负结果，不能只挑周期 trace。
> 2. 还不够；必须把 B 的准入/预测接入 C 的真实执行器并实现跨调用驻留，在同路由、同输出和同预算下测端到端时间。两张独立表不能拼成系统加速比。

## §9 本版实验与代码映射

A1–A6 负责稳定 top-k、分段、INT8 布局、scatter、tile 与路由统计；B1–B8 负责字节容量、pin、完成事件、历史预测、带宽竞争及独立 Belady 下界；C1/C2/C5 在 CPU 上检查 gate、两层专家与 residual，C3/C4/C6 则检查有限槽 CUDA 专家卸载。共 20 个 TODO，参考有效实现约 253 行，详见[任务书](../../labs/lab7/README.md)。

先通过 CPU 手算 oracle 的检查，再到 GPU 上核对输出；B 的模拟时间与 C 的实测墙钟时间应分别报告。源码借鉴及简化范围见[资源映射](../../labs/lab7/resources.md)，评分依据见[验收表](../../labs/lab7/grade.md)。

## 资料汇总

- [lab7：路由、缓存事件与 CUDA 专家执行](../../labs/lab7/README.md)、[实验规范](../guides/experiments.md)。
- [MoE-Infinity 论文](https://arxiv.org/abs/2401.14361)、[官方代码](https://github.com/EfficientMoE/MoE-Infinity)、[固定入口](../readings/code-map.md)。
- [MegaBlocks](https://github.com/stanford-futuredata/megablocks)、[Nsight Systems 官方文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。
- [外部课程与 lab 对照](../readings/courses-and-labs.md)：沿用 foundations 的 C++/CUDA 编译与时间线阅读能力。

- [Switch Transformers 图 2](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf#page=4)：§1 的路由结构读图，注意 gate 定义差异；全文选读。
- [NVIDIA 传输重叠时间线](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)：§6 的提交顺序与引擎依赖读图，历史设备结果仅作机制说明。

## 自测清单

- [ ] 能从 logits 写出稳定 top-k 及权重，并分别计算总存储、活跃计算量和加载工作集。
- [ ] 能手算 offsets、ordered routes 和 weighted combine，解释重复 route 与空专家的处理。
- [ ] 能画出 `Loaded/Pending/Hit/Blocked` 对应状态，检查 loading 占用、pin 引用和失败准入的原子性。
- [ ] 能用一张 bytes 账解释“miss 降低但流量增加”，区分在线信息与 Belady oracle 的口径。
- [ ] 能画出槽复用的双向依赖，并指出哪些 profiler 轨道可以验证真实重叠。
- [ ] 能在实验报告里区分 B 的模拟时间、C 的实测墙钟，以及尚未接通的缓存策略路径。
- [ ] 能写出一个固定路由与字节预算的消融实验，列出可能推翻预取假设的结果。
