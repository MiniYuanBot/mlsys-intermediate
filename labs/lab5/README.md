# Lab 5：有所有权的前缀缓存与服务调度

[讲义 D5](../../course/modules/05-prefix-cache.md) · [评分](grade.md) · [公开来源与借鉴边界](resources.md) · [参考答案导读](../../solutions/lab5/README.md)

> [!important] 你要实现的系统
> 先用包含完整祖先路径的块键构建缓存，实现事务插入、lease、失效和容量收缩；再加入两次访问准入、带饥饿保护的最长前缀调度，以及并发请求事件回放。A/B/C 都有必填函数和独立测试。完成后，观察缓存命中、请求排队与租户分布如何相互影响。

## 0. 工作量、文件和运行

前置知识：foundations 中的分页 KV、Rust 集合与所有权。需要 Rust≥1.82 和 Python≥3.10，全部使用标准库，可离线运行。Mac 或 WSL 的 CPU 即可完成；实验不分配真实 KV 张量，输出中的 ms 由显式成本模型计算。

| 部分       | 主要实现                                 | 实质参考填空行 |    时间 |
| ---------- | ---------------------------------------- | -------------: | ------: |
| A 基础     | 键、匹配、叶节点、结构审计、字节计数     |             60 |      5h |
| B 核心     | 事务插入、acquire/release、失效、resize  |             59 |      7h |
| C 前沿机制 | two-hit准入、LPM+age调度、并发回放、指标 |            137 |     10h |
| 报告       | 曲线、反例与源码对应                     |       不计代码 |      3h |
| **合计**   | **14个TODO**                             |        **256** | **25h** |

行数只统计答案填空区域中非空、非注释、非纯括号的行，不含已提供的接口、测试和 bench。预计时间包括调试、测量与报告，讲义另计 5h；这些估时尚未经学生试学验证。按每周 36h 安排，可用 30h 完成本讲和 lab，余下 6h 阅读论文、参与讨论。

```text
src/cache_student.rs   A1-A5, B1-B5: exact prefixes and cache ownership
src/replay_student.rs  C1-C4: admission, scheduling, event replay
src/main.rs            provided: A/B/C benchmarks
src/lib.rs             provided: module exports
tests/exercises.rs     provided: 15 tests and hand-calculated oracles
exercise.py            provided: stage runner
```

以下命令统一在 **`mlsys-intermediate/labs/`** 下运行；A 阶段测试不调用 B/C 尚未实现的函数。

```bash
python tools/run.py --lab 5 --impl student --mode test --part A
python tools/run.py --lab 5 --impl student --mode test --part B
python tools/run.py --lab 5 --impl student --mode test --part C
python tools/run.py --lab 5 --impl student --mode bench --part all
python tools/run.py --lab 5 --impl reference --mode test --part all
```

调试单项：`cargo test --offline --release --manifest-path lab5/Cargo.toml --test exercises b_transaction_rollback_and_leaf_order -- --nocapture`。可以新增独立测试，但不要修改现有测试的预期结果来迁就实现。每处 TODO 注释都说明了函数契约、实现步骤和对应测试名，请保留标记。

## A. 基础：先定义“相同前缀”（5h）

### A1 完整祖先键（1h）

实现 `block_keys`。block=2、tokens=[1,2,3,4,5]应生成[1,2]与[1,2,3,4]两个键，不登记不完整的末块。键包含model、adapter、tokenizer、tenant和position_offset；局部token相同而祖先不同不能命中。

画出请求 `[1,2,3,4]` 与 `[9,8,3,4]` 的块键，解释为什么第二块不能共享。这里通过复制完整 prefix 展示前缀依赖关系；生产系统通常使用哈希链或压缩 radix 路径。教学容器的效率较低，不应直接作为目标系统的实现。

### A2 连续命中（0.75h）

`matched_tokens`在首个miss停止且不修改统计。测试故意造一个“父块缺失但子块存在”的坏缓存，匹配必须返回0，而不是跳到子块继续。区分只读query和真正acquire。

### A3 候选淘汰（1.25h）

`victim`只考虑ref=0、非protected、无resident后代的叶节点。LRU按last，Frequency按hits/last，最后用key稳定排序。为什么最老节点是父节点时不能直接删除？画出删除叶节点后其父亲成为新候选的过程。

### A4–A5 不变量与内存账（1.25h）

`validate`检查容量、完整块、每个直接父亲存在、refs(child)≤refs(parent)、时钟不来自未来；错误返回描述字符串。`stats`分别统计pinned blocks和references，同一块被3个lease引用仍只有1个pinned block。

`key_token_bytes`只算完整prefix向量元素；不含Vec/String/BTreeMap/分配器或真实KV。A benchmark输出4个长度×3个block=12行，画key payload增长，推导完整prefix复制的二次增长。这是结构计数，不是运行时内存峰值。

### A 验收（0.75h）

运行 5 个 A 阶段测试，提交 namespace 改变、缺少父块、两次 pin 时的手算状态。说明为什么真实服务还需要模型配置指纹、权重版本与位置语义，不能仅靠模型名称判断能否共享缓存。

## B. 核心：事务与所有权（7h）

### B1 全有或全无插入（2h）

在副本上规划缺块数、保护请求路径、不断淘汰合法叶节点；成功才发布。失败必须保留nodes、refs、last、hits和clock。输入路径超过总容量时快速失败。测试构造“能先淘汰一个叶节点，但之后被pin阻塞”的反例，检查失败前后的完整快照。

这里为了便于理解事务，采用了整表 clone，需要承担相应的复制成本。如果改用 undo log，仍须通过相同的事务测试，并单独记录管理成本。

### B2–B3 Lease 生命周期（1.5h）

acquire只pin连续命中链，每个节点更新last/hits。Lease内部键不可公开构造，release消费它且从叶到根减refs；refs归零不等于删除缓存。测试两个lease共享同一块：只释放一个时禁止淘汰，第二个释放后才允许。

Rust 所有权能避免对同一 Lease 重复 release，但不会自动保证 cache/lease 配对正确。当前 API 要求调用方将 lease 归还给生成它的 cache；开放挑战中可以增加 cache 身份 ID 或 RAII guard。

### B4 namespace失效（1h）

adapter更新或租户清理需要 `invalidate`。任何目标节点被pin则整体拒绝；成功只删除该namespace。测试确保另一个tenant完全不受影响。失效失败也不能移动LRU时钟。

### B5 运行时容量收缩（1h）

实现 `resize`，缩容需要逐个删除合法叶节点；pin阻止达到目标时回滚容量、节点和clock。扩容保留LRU顺序。测试正在运行请求时收缩到1块失败、释放后缩到0成功。

### B 测试与回放（1.5h）

运行 5 个 B 阶段测试和 B benchmark。4 种 capacity×LRU/Frequency 共输出 8 行；结合稳定前缀和变化尾部，解释命中率的差异。保留小容量下两种策略表现相同的负结果，不能仅凭理论预期声称性能提升。

## C. 前沿：准入、调度与缓存组成闭环（10h）

### C1 两次访问准入（2h）

实现有界历史表 `Admission::observe`。每请求每完整块key增加一次访问计数；超过history容量按最近触碰时钟淘汰。Always允许整条完整路径，TwoHit仅允许计数≥2的连续祖先链，遇到未满足条件的父亲必须停止。

这项实现是应对一次性扫描污染的可执行基线，并非 TinyLFU 或某篇论文的完整算法。历史表的容量也有限，history 过小会低估热度，报告中应说明这一代价。测试覆盖 tenant 隔离、扫描污染和 capacity=0。

### C2 最长前缀优先与年龄保护（2h）

`select`返回waiting数组下标。若有人等待超过age_limit，从这些请求中取最早到达者；否则用FIFO或LongestPrefix，平局按arrival/ID。必须只看已经到达的请求。手算一个冷请求被两个热点请求推迟的例子，指出年龄阈值只限制优先级，不保证在持续过载下有固定响应时间上界。

### C3 并发事件回放（4h）

实现带worker slots的事件循环。每个时刻先完成请求：release旧lease，再insert其已经计算好的新prefix；然后接收已到达请求，再给空闲worker派发。启动时查询hit、acquire、冻结准入决定；耗时=(tokens-hit)*prefill_ms+decode_ms。

关键反例：两个同时到达的相同prefix请求，若都miss而并发执行，在任一个完成前都不能“免费复用”尚未计算出的KV。独立手算测试要求t=0同时开始的两个请求都hit0，t=4到达第三个才hit2。

请求处于 active 状态时持续持有 lease，因此其他请求完成后的插入操作可能被 pin 阻止。事件循环应跳到 min(next arrival,next completion)，不能用固定的小时间步近似。保证每个请求只完成一次、所有 lease 最终释放，且容量与拓扑始终合法。这里的 workers 是抽象的并行服务台，不代表多 GPU，也不模拟单 GPU 上真实的 batch 执行。

### C4 全局与租户指标（1h）

汇总按 token 加权的 hit ratio、模拟中实际发生的队列等待 start-arrival、p95 E2E、重算 token 数，以及容量不足导致的插入失败次数。不要平均每请求命中率；2/2与0/8的总体命中率是0.2，不是0.5。TwoHit决定不缓存的路径不是capacity failure；零长度插入成功，报告需要区分两者。

### C 实验矩阵（1h）

默认180请求，热点60→一次性扫描60→热点恢复60；每请求8token，前4可共享、后4带唯一ID。3capacity×2admission×2schedule×3tenant视图=36行。画总命中、按tenant的p95 E2E/mean wait，解释命中更高但尾延迟未改善的组合。

所有策略使用同一个 trace。默认负载是确定性的，重复运行同一输入不能作为独立 seed 的统计样本。报告还应改变 burst/scan 长度，补充一组敏感性实验，并附上完整的原始 CSV。

## 交付与评分

提交2个学生文件、15个测试日志、三阶段CSV、3幅图、4页以内报告，按[grade.md](grade.md)验收。报告必须包含事务快照反例、in-flight miss 反例、按租户分组的负结果和实际工时；验收不要求策略一定优于 LRU。

## 可选挑战

以下挑战另计时间，可选择一项继续：

- 用 radix 压缩替换完整 prefix 复制，保留当前匹配与事务 oracle。
- 接入 SGLang 的真实请求 trace，单独测量服务 TTFT 和 CPU 管理成本。
- 增加取消路径与引擎 KV buffer 归还，检查异常退出时的 lease 生命周期。
- 将 TwoHit 与 TinyLFU 或考虑对象大小的准入方案对照，固定容量与租户共享规则。

## 资料汇总

- [逐任务改编来源](resources.md)：对照 SGLang、vLLM 等上游机制与本实验简化边界。
- [D5 图解与随堂题](../../course/modules/05-prefix-cache.md)：复查块边界、lease 和 in-flight miss 的状态轨迹。
- [评分标准](grade.md)与[参考答案导读](../../solutions/lab5/README.md)：核对事务、不变量和指标口径。
