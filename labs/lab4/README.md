# Lab 4：从接受概率到 SLO 推测服务闭环

[讲义 D4](../../course/modules/04-speculative-slo.md) · [评分](grade.md) · [公开来源与借鉴边界](resources.md) · [参考答案导读](../../solutions/lab4/README.md)

> [!important] 你要交付什么
> 用 Rust 实现概率与校准工具、基于部分观测的控制器，以及包含到达、排队、完成事件的推测服务模拟器。A/B/C 均有必填代码和可执行测试，C 要求实现前沿方案中的机制，阅读论文不能替代实现。输出标为 `synthetic_queue`，不能描述为 GPU serving 实测。

## 0. 前置、工作量与运行入口

前置：foundations 的推测采样正确性和 Rust 所有权。Rust ≥1.82、Python ≥3.10；Cargo 不依赖外部 crate，可离线运行。Mac、Windows WSL 都能完成全部必修；不需要 GPU。本实验假定底层 draft/verify 正确，不实现采样分布修正，重点是其外部的决策和队列。

| 部分       | 代码产物                               | 实质参考填空行 | 预计时间 |
| ---------- | -------------------------------------- | -------------: | -------: |
| A 基础     | 提交分布、矩、分位数、插值、校准表     |             75 |       5h |
| B 核心     | 观测计数、下界、可行 k、漂移、共享预算 |             77 |       7h |
| C 前沿机制 | 状态提交、到达、事件引擎、指标         |            107 |       9h |
| 报告       | 三幅图、反例、公开源码对应表           |       不计代码 |       3h |
| **合计**   | **14 个 TODO**                         |        **259** |  **24h** |

行数只计参考答案 TODO 区域中的有效代码，排除空行、注释、纯括号行和 scaffold/tests/benchmark，不要求你的实现恰好一样长。估时包含调试和实验，讲义 5h 另计；尚未经过学生试学，请记录实际用时。每周36h可安排本讲和实验29h，其余7h用于阅读或组会。

```text
src/policy_student.rs  A1–A5、B1–B5：概率、校准、决策
src/engine_student.rs  C1–C4：状态和离散事件服务
src/lib.rs             提供：模块导出
src/main.rs            提供：A/B/C 三组 benchmark
 tests/exercises.rs    提供：14 个独立测试，不会被答案覆盖
exercise.py            提供：阶段入口
```

以下命令都在 **`mlsys-intermediate/labs/`** 运行。按 A、B、C 的顺序完成；A 阶段测试不会调用 B/C 的 stub。

```bash
python tools/run.py --lab 4 --impl student --mode test --part A
python tools/run.py --lab 4 --impl student --mode test --part B
python tools/run.py --lab 4 --impl student --mode test --part C
python tools/run.py --lab 4 --impl student --mode bench --part all
python tools/run.py --lab 4 --impl reference --mode test --part all
```

直接调试一个测试：`cargo test --offline --release --manifest-path lab4/Cargo.toml --test exercises c_hand_timeline -- --nocapture`。请保留所有 BEGIN/END TODO 标记；每处前面的注释列出契约、边界与对应测试。允许增加 helper、测试和 benchmark，不得把答案导入学生代码。

## A. 基础：把口头的“接受率”变成可检验对象（5h）

### A1–A2 提交分布与矩（1.25h）

实现 `committed_distribution` 与 `committed_moments`。若条件概率是 `[0.5,0.25]`，应得到提交1/2/3个 token 的概率 `[0.5,0.375,0.125]`，均值1.625、方差0.484375。分别处理“首次拒绝位置”和“全部接受”，最后一个桶也要计入纠正/bonus token。

先手算 k=0、全部0、全部1，再解释为什么条件概率相乘。测试 `a_distribution_oracle` 枚举所有二进制向量，用独立事件概率聚合，不调用你的函数计算预期值。

### A3 分位数（0.5h）

实现 nearest-rank 分位数：rank=max(1,ceil(qn))。不要把均值或插值分位数混进来。输入可乱序，不能修改原数组；空输入返回 None。用 `[1,2,3,4]` 的 p75=3 验证定义。

### A4 上下文成本插值（1.25h）

`ProfilePoint.verify[k]` 是 verifier 耗时，draft 是每候选 token 的附加成本。实现 context 维线性插值，区间外夹到边界值，所有行的 verify 宽度相同。报告中必须说明区间外采用夹取值，不能将其写成该区域的实测结果。手算 context100/300之间的200，分别核对 draft 和每个 k。

### A5 从重复测量构造校准表（1.25h）

实现 `calibrate`：按 `(context,k)` 分组，取 A3 定义的中位数，context有序；缺任意 k 应报错。缺失值不能填0，否则会把 verifier 的成本误记为零，算出无限收益。当前测试使用合成 timing sample；若之后替换为 GPU 计时，必须先 warmup、同步、区分 draft 与 verify，保留原始样本。

### A 验收与小实验（0.75h）

运行 A 测试和 A benchmark。输出4个context×5个k共20行；画 E[committed] 和成本对 k 的曲线。解释“概率增长”与“成本增长”为何不能单独决定 k。保存校准缺列的失败用例。

## B. 核心：观测偏差、非线性成本与资源约束（7h）

### B1–B2 可观察数据与保守分数（2h）

`Estimator` 的 success/failure 从0开始。每轮先把所有历史计数乘 decay，再只更新已接受前缀和首次拒绝，未到达的后缀不算失败。`probabilities(0)` 返回 Beta(1,1) 平滑均值；z>0返回 Wilson 形式下界，零证据返回0。

推导下界公式，并比较成功1/1和成功100/100。由于使用折扣计数且分布会变化，这里的结果不再符合经典独立同分布置信区间的解释，应称为保守控制分数。测试覆盖删失、全接受、k=0的衰减，以及小样本/大样本排序。

### B3 局部预算决策（1h）

遍历 k，使用 `draft*k+verify[k]`，过滤 round budget 与 remaining output，再最大化 E[tokens]/cost；并列选小 k。返回 None 表示没有选项满足预算，不能将其处理为“达标的k=0”。独立 oracle 遍历多种概率、remaining和预算验证结果。

### B4 漂移检测（1h）

按注释实现单侧累计偏差检测：下降时累积、超过阈值触发窗口重置。测试稳定高值、下降与单调上升。解释为何不能用“所有候选都算失败”加快检测；那改变了观测含义。

### B5 多请求共享预算（1.5h）

实现 `allocate_round`，先给每请求保留k=0，再按照正的边际期望收益/额外成本选择升级。边界是总预算，目标与B3的单请求比值不同。测试含贪心反例：额外预算1.6时，增益0.6/成本1与增益0.9/成本1.6，贪心选择前者但不是最优。报告需要解释这个反例，也不能将该算法称为最优算法。

这个任务借鉴 AdaServe 对跨请求资源选择的处理思路，但不包含 token tree，也未复现论文的完整约束。它是一项独立的多请求预算练习，C 的单服务台模拟器不会调用它。

### B 对照实验（1.5h）

运行 B benchmark：high/low/recovery 各40步，共120行。每步固定观测 proposed=4，回放数据不随控制器选择改变。画出均值控制与保守控制的 k 和第0位置下界，比较相同证据下的决策；这项开放回放不能证明在线恢复能力。增加一次 decay=1.0 对照和一次 z=0与1 的对照，再在 C 的闭环中讨论“不探索则没有证据”的反例。待实现代码已有明确任务，扫描参数不能替代这些代码任务。

## C. 前沿：从 round 比值到完整请求服务（9h）

### C1–C2 状态机和到达队列（2h）

`State::commit` 在验证完成时一次性提交 `min(accepted+1,remaining)` 个 token，记录每个 token 的可见时刻；同轮 token 可有相同时间。`enqueue` 处理相同到达时刻，cursor只能前进。空队列不代表全体完成，还可能有未来到达。

先画两请求手算：r0在t=0到达需2token，r1在t=0.5到达需1token；k=0每轮1ms，r0提交时刻为1、3，r1为2。新到达先入队、运行请求随后回队，这是指定调度语义。

### C3 离散事件引擎（4h）

实现轮级 round-robin 服务台。维护 sorted arrivals、ready、cursor、finished、now；队列为空时跳到下一个到达。根据context+emitted插值成本，只能用历史估计选择 k；模拟器的 alpha 仅用于环境采样，控制器不得读取。

预算=min(round_bound,absolute_deadline-now)。预算不可行时，明确记录 infeasible，再执行k=0完成工作；不能通过删除晚请求来提高 goodput。真实服务可以选择拒绝或取消请求，本实验则要求排空所有请求。

随机数按 `(seed,request_id,emitted,depth)` 派生，避免策略因为调用次数不同而改变后续请求随机流。这是用于配对比较的合成环境，不代表真实 draft token 的相关性。漂移会触发估计器重置，这可能帮助适应新分布，也可能丢失已学信息。

### C4 指标与结果检查（1.5h）

TTFT=first-arrival，E2E=finish-arrival，达标事件=finish≤absolute_deadline。goodput=达标请求数/(最后完成-首次到达)，输入ms需换秒。它包括排空区间，不等于固定稳态测量窗口。p95采用A3定义；空输入所有指标0。检查 emitted总数=请求output总数，所有请求恰好完成一次，利用率≤1。

这里没有prefill模型、batch并行、网络、实际GPU或用户流式传输；TTFT是这个简化系统的首提交时延。不能把同轮连续token的零间隔当作真实ITL测量。

### C 对照实验（1.5h）

C benchmark 固定90个请求，alpha 按请求 ID 分为高、低、恢复三个阶段；3个arrival gap×5seed×4策略=60行。策略为 k0、k3、mean、conservative_no_probe。最后一种是有意保留的负基线：零证据时下界为0，在当前单调成本下，冷启动后会一直选择 k0。测试检查它与普通解码具有相同的时间线，因此不能将其视为已完成的保守自适应方案。至少画出 p95 E2E、goodput、infeasible 随 arrival gap 的变化，报告5seed的均值和离散程度，不只选择最快的 seed。

附一个过载反例：round bound即使都满足，排队仍可导致deadline miss。再附一个负结果：保守控制并非必然优于mean。说明原因之间的关系，并区分哪些有观测支持、哪些仍是猜测。

## 交付与评分

提交2个学生实现文件、全部测试日志、A/B/C原始CSV、三幅图和不超过4页报告。报告附TODO完成表、实耗时、source-to-lab映射；评分见[grade.md](grade.md)，不设置缺乏依据的“至少加速X倍”要求。

## 可选挑战

以下挑战不计入 24h，可选择一项继续：

- 将真实小模型的 draft/verify 计时接入 A5，单独验证模型输出正确性。
- 增加周期性 probe，把探索轮及其开销计入指标。
- 用动态规划替换 B5 贪心，构造并量化两者的差距。
- 将单服务台扩展为 batch token 预算，明确并发执行语义。

扩展后保留当前合成 oracle，避免性能改动破坏已有契约。

## 资料汇总

- [逐任务改编与上游实现](resources.md)：查找论文机制、借鉴范围和独立测试依据。
- [D4 图解与随堂题](../../course/modules/04-speculative-slo.md)：补做提交量、部分观测和队列指标的手算。
- [评分标准](grade.md)与[参考答案导读](../../solutions/lab4/README.md)：完成学生实现后用于核对，不替代实验报告。
