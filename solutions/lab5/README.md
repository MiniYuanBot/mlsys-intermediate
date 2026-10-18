# Lab 5 参考答案与解题导读

[任务书](../../labs/lab5/README.md) · [评分](../../labs/lab5/grade.md) · [公开来源](../../labs/lab5/resources.md)

参考实现在 [cache_student.rs](src/cache_student.rs)（A1–A6、B1–B5）、[cluster_student.rs](src/cluster_student.rs)（C1–C6）、[trace_student.rs](src/trace_student.rs)（R2、R3）和 [real/fit_student.py](real/fit_student.py)（R1）。测试、benchmark、哈希原语、代价模型和 JSON 读取器都在 labs 一侧，`run.py --impl reference` 只把这四个文件覆盖上去。

## A 身份：哪些东西决定了“同一份 KV”

**A1 链式哈希。** 第 $i$ 块的哈希是 $h_i=H(h_{i-1}, x_{iB:(i+1)B}, e_i)$，$h_{-1}$ 是 `NONE_HASH`。链式的好处是"块 $i$ 命中“自动意味着”块 $0..i$ 全部命中"，查找只需从头走到第一个缺失；代价是改动第 0 块会让后面所有块失效，这正是因果 attention 要求的。extra keys 的三条规则各有理由：

- LoRA 放在**每一块**：adapter 从第一层就改变 K/V，每块都依赖它。只靠父链传递也能区分，但 vLLM 选择每块都带，参考实现照做，测试按逐块精确哈希比较。
- 多模态条目只放在与它**重叠**的块上，偏移写相对块起点的值。占位 token 对所有图片都一样，不加标识就会让两张图共享 KV；偏移防止同一张图出现在不同位置时误命中。重叠判定用半开区间，`a1-mm-touching-overlap` 把它改成闭区间，测试中“恰好在块边界结束”的条目会多进一个块。
- `cache_salt` **只放在第 0 块**：父链会把它带到每一块，再放一次是冗余的。把它放错块（`a1-salt-on-last-block`）时，第 0 块在租户之间共享，正是讲义 §9 时序侧信道要防的情况。

**A2 完整祖先键。** 用 `Vec<u32>` 存整条前缀，等价于“永不碰撞的哈希链”。代价是载荷 $4B\sum_{j=1}^{T/B} j = 2B\frac{T}{B}(\frac{T}{B}+1)$ 字节，固定 $B$ 时随 $T$ 二次增长，A benchmark 把它和 8 字节的链式哈希放在一起。

**A3–A6。** 匹配在第一个缺失处停止；缺口后的子块不可达。`victim` 只看合法候选再排序，顺序反过来（先挑最冷的、再看能不能删）会在最冷节点是父节点时出错。`validate` 用枚举而不是字符串报告，测试因此不依赖措辞；检查顺序固定（容量在先，然后按键序逐个节点检查对齐、父节点、pin 单调、时钟），所以“第一个错误”是确定的。父节点必须在**同一 namespace** 中，测试里另一个租户的同名前缀不能充当父节点。

常见错误：把 `pinned_blocks` 按引用数累加；把 `max_depth` 记成 token 数；判断后代时忘了比较 namespace。

## B 所有权：为什么先 clone 再发布

插入要同时满足三件事：路径上的已有块不能被自己淘汰（`protected`），淘汰只能挑合法候选，失败时状态完全不变。在副本上做完全部淘汰和补齐，成功才赋值，失败就丢弃副本，回滚因此一眼可证。`b1-evict-outside-transaction` 把淘汰同时作用在 `self` 上，失败插入的快照就不同了；`b1-path-not-protected` 让插入淘汰自己的祖先再加回来，节点数超过容量。`resize` 和 `invalidate` 也遵守同一条规则：失败不改 clock 与容量，成功才推进 clock。

`acquire` 刷新 `last`，这是 LRU 与 FIFO 的唯一区别；`b2-lru-becomes-fifo` 删掉这一行后，“经 acquire 刷新的块应当留下”的检查失败。`release` 从深到浅减引用，保持 $\mathrm{refs(child)}\le\mathrm{refs(parent)}$ 在每一步都成立；引用归零的块继续留在缓存里，后续请求才能命中它们。下溢必须 panic：把它改成 `saturating_sub` 会掩盖一次重复释放。

## C 准入、全命中规则与路由

**C1。** TwoHit 只准入被观察到至少两次的块，挡住一次性扫描对缓存的污染。历史有界，按最近观察淘汰；如果按首次插入淘汰（FIFO），反复出现的热块会被挤掉而新来的一次性块留下。容量作为参数传入，避免 benchmark 结果依赖一个藏在实现里的倍数。

**C2 全命中规则的推导。** 前向至少要输入一个 token 才能产生 logits，所以复用上限是 $t-1$。vLLM 的块表要求已计算 token 数按块对齐，于是上限再向下取整到块边界：$r=\lfloor\min(m,t-1)/B\rfloor B$。$t=32$、$B=16$、全部命中时 $r=16$，重算整整一块；SGLang（page_size 1）的 $r=31$，只重算 1 个 token。旧版 lab 把完整命中计为零成本，与两个引擎都矛盾；`c2-full-hit-free` 复现了这个错误，被“完整命中永不免费”和暴力枚举同时抓住。

**C3 路由。** 四种策略的差别在于看什么：RoundRobin 不看状态；LeastLoad 只看负载；PrefixAffinity 只看命中，热前缀会把所有请求拉到一个副本上；Hybrid 先看是否失衡（绝对差和相对比**同时**超过阈值），不失衡时才追命中，命中比例还要超过 `cache_threshold`。相对阈值的意义是：负载 6 对 4 的差为 2，绝对值看似大，但比例只有 1.5 倍，不算失衡。

**C4 回放的四步顺序。** 先完成（release 再 insert），再路由到达，再派发，再跳到下一事件。完成必须在派发之前：同一时刻完成的 KV 可以被同一时刻启动的请求复用，而还没完成的不行。路由时的负载包含排队请求：只数运行中的请求（`c4-load-ignores-queue`），同一时刻到达的一批请求会全部落到同一个副本。

**C5/C6。** 命中率按 token 加权；失衡度的分母包含闲置副本，否则一个副本独自承担全部负载、其余全闲时失衡度反而是 1。重复块数衡量“为命中率付出的容量”：Hybrid 在失衡时把请求分到别处，那里要重新算一遍前缀并多存一份，这就是“A Year in LLM Serving”观察到的命中率与重复之间的取舍。

## R 测量、拟合与 trace 回放

**R1。** 每个新 token 的线性层计算量约为 $2N$ FLOPs，对上下文中第 $j$ 个位置做 attention 约为 $4L\,n_q d_h\,j$ FLOPs。$n$ 个新 token 接在 $p$ 个缓存 token 之后，attention 总量是 $4Ln_qd_h\sum_{i=1}^{n}(p+i)\approx 4Ln_qd_h\,n(p+n/2)$，所以回归量取 $n(p+n/2)$。再加上与 $p$ 成正比的访存项和 launch 下限，就得到 $T=\max(T_0, a+bn+cn(p+n/2)+dp)$。下限附近的点由 CPU 发射时间决定，不反映 GPU 计算，放进回归会把所有系数拉偏（`r1-floor-points-fitted`）；丢掉二次项（`r1-no-quadratic`）则在长上下文处系统性低估。$p^*=b/c$ 是每 token 的 attention 开销等于线性层开销的上下文长度：Qwen3-0.6B 的 FLOPs 估计是 $2N/(4Ln_qd_h)=0.88\times10^9/(2.3\times10^5)\approx3.8$k，4060 实测拟合给出 4.1k。

**R2。** 适配器的原则是拒绝无法解释的输入，接受真实 trace 里确实存在的异常：分数、负数、缺字段一律拒绝；`input_length` 与 512 乘以 ID 数不一致、同一请求里 ID 重复，这两种在公开 trace 里确实出现，必须接受。`prompt_tokens` 取 `input_length`，最后一块可以不满。

**R3。** 单个请求复用 $r$ 个、共 $t$ 个 token 时（忽略下限），节省的时间是

$$
T(0,t)-T(r,t-r)=br+\tfrac{c}{2}r^2-dr ,
$$

占比为 $(br+cr^2/2-dr)/(a+bt+ct^2/2)$，一般不等于 $r/t$：复用的前缀仍要被后缀的 attention 读一遍，复用本身还要付 $dp$。下限会把节省吞掉：$T_0=23$ ms 时，几百个 token 的请求无论命中与否都是 23 ms。

## 运行与结果

```bash
python tools/run.py --lab 5 --impl reference --mode test --part all
python tools/run.py --lab 5 --impl reference --mode bench --part all
python tools/run.py --lab 5 --impl reference --mode test --part R --device cuda    # WSL
python tools/fault_inject.py --lab 5 --device cpu
python tools/fault_inject.py --lab 5 --device cuda                                   # WSL
```

2026-10-18 在 WSL（RTX 4060 Laptop，驱动 566.24）上：21 个 Rust 测试与 R1 检查全部通过；`--device cuda` 的短测量 4 行全部 `stable`，正确性闸门 4/4。C benchmark（modeled）中 PrefixAffinity 与 Hybrid 的命中率约 0.80，RoundRobin 约 0.50；TwoHit 在这个没有扫描的负载上全面降低命中率，这是一个要保留的负结果。R benchmark（4 副本、vLLM 规则、4060 拟合）中 Hybrid 的 token 命中率 0.574、省下 49.8% 的 prefill 时间、p95 TTFT 647 ms；纯亲和性命中率 0.564，但失衡度 2.67、p95 TTFT 830 ms。两种全命中规则的差别不到 0.1%，因为合成 trace 里只有 25 个原样重发的请求。

## oracle 与 fault 对照

`solutions/lab5/faults.json` 共 64 个 fault，覆盖全部 20 个 TODO；核心 TODO 至少 2 个。下表按 TODO 汇总，完整列表与注入文本见 JSON；状态来自 2026-10-18 在 WSL 上的 `fault_inject` 运行。

| TODO | 独立 oracle                                          | fault id                                                                                                                                                                        | 注入的错误                                                                       | 在哪个 part 失败 | 状态   |
| ---- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ---------------- | ------ |
| A1   | 用提供的 `hash_block` 逐块重算；链式性质             | `a1-lora-key-dropped`、`a1-salt-on-last-block`、`a1-salt-every-block`、`a1-not-chained`、`a1-mm-touching-overlap`                                                               | 丢 LoRA 键（跨 LoRA 复用）、salt 放错块、salt 每块都放、不链接父哈希、闭区间重叠 | A                | CAUGHT |
| A2   | 逐字段改 namespace；手写前缀                         | `a2-tenant-dropped`、`a2-partial-tail-keyed`                                                                                                                                    | 键里没有租户、尾块也登记                                                         | A                | CAUGHT |
| A3   | 手算 fixture；调用前后快照                           | `a3-skip-gap`、`a3-cumulative-count`                                                                                                                                            | 跳过缺口、按前缀长度累加                                                         | A                | CAUGHT |
| A4   | 按定义筛选再排序的 oracle（300 例随机）              | `a4-evict-non-leaf`、`a4-ignores-pins`、`a4-cross-namespace-child`、`a4-frequency-ignores-hits`                                                                                 | 淘汰非叶子、无视 pin、跨租户算后代、LFU 退化为 LRU                               | A                | CAUGHT |
| A5   | 每种损坏单独构造，按枚举变体比较                     | `a5-orphan-ignored`、`a5-pin-check-reversed`、`a5-future-clock-tolerated`、`a5-capacity-slack`                                                                                  | 不查孤儿、pin 方向反、容忍未来时钟、容量差一                                     | A                | CAUGHT |
| A6   | 手算                                                 | `a6-pinned-counts-refs`、`a6-depth-in-tokens`                                                                                                                                   | pinned 按引用数计、深度按 token 计                                               | A                | CAUGHT |
| B1   | 失败前后完整快照；手算淘汰顺序                       | `b1-evict-outside-transaction`、`b1-path-not-protected`、`b1-clock-moves-on-failure`                                                                                            | 淘汰泄漏到原状态、不保护自身路径、失败时推进时钟                                 | B                | CAUGHT |
| B2   | 手算 refs/hits/last；缺口场景                        | `b2-lru-becomes-fifo`、`b2-pin-past-gap`、`b2-hits-not-counted`                                                                                                                 | 不刷新 last（LRU 变 FIFO）、越过缺口 pin、不计 hits                              | B                | CAUGHT |
| B3   | 引用归零后仍可命中；下溢必须 panic                   | `b3-release-leaf-only`、`b3-remove-on-zero`、`b3-underflow-hidden`                                                                                                              | 只释放叶子、归零即删除、掩盖下溢                                                 | B                | CAUGHT |
| B4   | 另一租户同 token 的节点数                            | `b4-ignores-pins`、`b4-wrong-namespace-match`                                                                                                                                   | 删除被 pin 的块、按模型而不是 namespace 删除                                     | B                | CAUGHT |
| B5   | 失败与扩容前后快照                                   | `b5-capacity-before-commit`、`b5-grow-moves-clock`                                                                                                                              | 失败缩容仍改容量、扩容推进时钟                                                   | B                | CAUGHT |
| C1   | 容量 2 的手算序列（区分 LRU/FIFO）                   | `c1-history-fifo`、`c1-one-hit-admits`、`c1-capacity-ignored`                                                                                                                   | 历史按首次插入淘汰、一次即准入、容量参数被忽略                                   | C                | CAUGHT |
| C2   | 暴力枚举（$t\le40$、$B\le6$）                        | `c2-full-hit-free`、`c2-vllm-no-rounding`、`c2-vllm-ignores-matched`                                                                                                            | 完整命中零成本、vLLM 不取整、上限不取 min                                        | C                | CAUGHT |
| C3   | 9 个手算例子 + 排序式 oracle（2000 例随机）          | `c3-hybrid-ignores-load`、`c3-hybrid-or`、`c3-affinity-tie-by-index`、`c3-round-robin-stuck`、`c3-threshold-inclusive`                                                          | 路由器无视负载阈值、两阈值用“或”、并列按序号、游标不前进、阈值取等               | C                | CAUGHT |
| C4   | 手算时间线（两种规则）、路由时间线、守恒与隔离上界   | `c4-lease-leak`、`c4-publish-in-flight`、`c4-load-ignores-queue`、`c4-rule-ignored`、`c4-block-in-units`                                                                        | lease 泄漏、开始时就发布 KV、负载不含排队、跳过全命中规则、块大小按单元传        | C；最后一个在 R  | CAUGHT |
| C5   | 手算指标                                             | `c5-mean-of-ratios`、`c5-imbalance-busy-only`、`c5-p95-off-by-one`                                                                                                              | 按请求平均命中率、失衡度忽略闲置副本、最近秩差一                                 | C                | CAUGHT |
| C6   | 手算三个缓存的重复数                                 | `c6-count-keys-not-copies`                                                                                                                                                      | 数重复键而不是额外副本                                                           | C                | CAUGHT |
| R1   | 已知系数的精确恢复；残差正交；有限差分；排除点的误差 | `r1-no-quadratic`、`r1-causal-half-missing`、`r1-floor-points-fitted`、`r1-max-rel-err-subset`，以及 `--device cuda` 下的 `r1-no-quadratic-cuda`、`r1-floor-points-fitted-cuda` | 丢掉二次 attention 项、漏掉 $n^2/2$、下限点进回归、误差只在子集上算              | R                | CAUGHT |
| R2   | 手写 JSON 行与逐类错误                               | `r2-time-scale-ignored`、`r2-fraction-accepted`、`r2-prompt-from-ids`、`r2-empty-accepted`                                                                                      | 忽略时间缩放、接受分数、长度按 ID 数算、接受空请求                               | R                | CAUGHT |
| R3   | 三个手算代价模型                                     | `r3-token-hit-as-saved`、`r3-baseline-with-reuse`                                                                                                                               | 把 token 命中率当作省下的时间、基线也扣掉复用                                    | R                | CAUGHT |
