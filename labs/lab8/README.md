---
title: Lab 8 — 从长度预测到有背压的 rollout/训练闭环
aliases: [Lab8, rollout scheduling lab]
tags: [mlsys, lab, reinforcement-learning, rust, scheduling]
---

# Lab 8 — 从长度预测到有背压的 rollout/训练闭环

[课程讲义](../../course/modules/08-rollout-scheduling.md) · [主源实验与代码](resources.md) · [评分标准](grade.md) · [参考解题导读](../../solutions/lab8/README.md)

本实验从静态排程扩展到三个相连的部分：历史长度预测与采样账本；支持 quantum、暂停恢复和工作窃取的事件调度器；接收完整 group、处理背压与策略滞后的训练队列。最后运行一个两动作 contextual policy，实际更新参数，观察严格的 freshness 要求如何改变样本利用比例和学习曲线。

> [!important] 真实与简化的边界
> policy 的采样、reward、梯度计算和参数更新均实际执行；rollout 的 token 工作量与 worker 耗时则由合成事件模型描述。实验不生成语言文本、不保存真实 KV，也不训练 LLM。C 阶段验证小环境中的语义和学习闭环；B 阶段以 token-unit 表示的 makespan 不能换算成 GPU 加速比。

## 0. 任务量与环境

**13 个 TODO，约284行有效参考实现**：A59、B93、C132行；不含给定结构体、驱动、测试、注释、空行和纯括号。全部使用 Rust 标准库，不依赖在线 crates，不下载模型，Mac/WSL/Linux CPU均可完成。

| 部分   | 必做实现                                          | 编码/调试 | 推导/阅读/实验 | 合计    |
| ------ | ------------------------------------------------- | --------- | -------------- | ------- |
| A 基础 | 有界历史、预测LPT、直方图、组内优势、策略采样     | 3h        | 2h             | 5h      |
| B 核心 | 分段重组、batch完整性、quantum事件与work stealing | 5h        | 2h             | 7h      |
| C 前沿 | 有界group队列、staleness、clipped梯度、闭环       | 5h        | 3h             | 8h      |
| 汇总   | 三张图、错误案例、源码阅读与报告                  | —         | 3h             | 3h      |
| 合计   | 不含开放挑战                                      |           |                | **23h** |

需要了解 RL 的基本概念，不要求熟悉 Rust 异步框架；实验先用确定性的事件循环验证语义。初学 Rust 借用与集合可能增加 3–6h。请记录每个 TODO 的实际用时和失败原因，据此调整自己的时间估计。

产物放 `reports/lab8/`：实现、完整测试与benchmark日志、长度预测图、调度效率图、学习曲线及一张生产者/队列/训练者版本时间线、3–5页报告。

## 1. 运行与源文件

以下命令均在 **`mlsys-intermediate/labs/`** 执行。

```bash
python3 tools/run.py --lab 8 --part A --impl student --mode test
python3 tools/run.py --lab 8 --part B --impl student --mode test
python3 tools/run.py --lab 8 --part C --impl student --mode test
python3 tools/run.py --lab 8 --part all --impl student --mode bench
```

`--impl reference` 在临时副本叠加参考答案。直接调试可用 `python3 lab8/exercise.py --mode test --part B`；正式结果使用公共runner保存源哈希。所有必修TODO在 [rollout_student.rs](src/rollout_student.rs)，契约在 [contracts.rs](tests/contracts.rs)，benchmark在 [main.rs](src/main.rs)。A→B→C 存在前置依赖。分阶段测试便于逐步推进，但 C 仍需要 A/B 的实现才能通过。

本 lab 即使指定 `--device cuda`，也仍在 CPU 上执行，并在输出中明确说明；该结果不作为 GPU 验证。

## 2. A 基础：从信息边界开始

### A0 三个手算任务（45min）

1. 历史窗口3，依次完成长度 `[1,2,3,100]`。下一次预测应使用哪些样本？上中位数是多少？未完成请求长度能否加入？
2. 对预测长度 `[3,3,2,2,2]`，手算两 worker 的 LPT 和最优划分。LPT 用作基线，不保证最优。
3. 两个reward `[0,1]` 的组内优势是什么？将两者都改成1会怎样？假设logit=0，两个uniform `[.1,.9]` 如何决定动作与行为概率？

### A1–A6 编码（3h）

| TODO                | 参考有效行 | 接口与关键语义                           | 测试               |
| ------------------- | ---------: | ---------------------------------------- | ------------------ |
| A1 `observe`        |          4 | 每prompt class的有界完成历史，不混入未来 | `a_length_history` |
| A2 `predict`        |          5 | 历史上中位数；cold start=32              | 同上               |
| A3 `assign`         |         14 | 按预测长度LPT；稳定ID/worker tie-break   | `a_assign`         |
| A4 `advantages`     |          7 | group内人口标准差；常数/空组安全         | `a_advantage`      |
| A5 `histogram`      |         15 | 显式overflow桶；经验分位数rank约定       | `a_histogram`      |
| A6 `Policy::sample` |         14 | 冻结快照采样、reward、行为概率及版本账本 | `a_sample`         |

不要用 A5 统计的真实长度替代 A2 的在线预测。A4 采用总体方差（除 N），而非样本方差（除 N−1）；这是本 lab 的约定，阅读其他框架时仍须检查其实现。A6 的 context 对应 4 种 prompt 类别，动作是 0/1，正确动作=context%2；policy 为每类保存一个可训练的 logit。

### A 对照实验（1h15min）

给定200次完成事件，在100次处发生长度分布变化；window=4/16/64，各输出MAE、p50、p95，共3行。用固定的完成历史比较窗口。较大的窗口不总更准确：它记住更多旧分布。

画一张逐事件预测/实际长度图，说明预测信息何时可用。可在benchmark中增加逐事件输出，不得反过来让预测器读取整条未来轨迹。再提交两个class隔离的例子，说明为何按所有任务混合取中位数会掩盖差异。

## 3. B 核心：允许暂停不等于允许丢状态

### B0 设计事件表（45min）

用6个任务实际长度 `[20,1,3,2,15,2]`、预测都为4、3个worker、quantum=3。每行记录 `time,worker,id,start,count,terminal`。同一job可以多次调度，但版本、group和下一个token偏移不能改变。work stealing只看别人队列长度，不能用actual选择最短/最长任务。

实际长度仅供 executor 判断某个 quantum 是否结束。这个模型没有计入随 context 变化的每 token 时间、KV 搬迁和 kernel 调度成本，因此适合检验调度机制，不能直接预测真实服务时间。

### B1–B3 编码（5h）

B1 先对分段排序，再检查身份、版本、gap、overlap 与 terminal。必须拒绝 terminal 后仍有 token 的输入，不能只检查是否存在 terminal。保存 token 序号只能验证账本，不能证明真实 RNG/KV 的恢复正确。

B2 保留预先选定的整个 batch ID 集合。仅检查各 group 的人数还不够：即使丢掉了整个慢 group，剩下的 group 仍然完整。因此必须检查所有 expected IDs 是否恰好出现一次，且版本一致。

B3 实现以下事件循环：

1. 用A3的预测LPT初始化worker队列。
2. 空闲worker先取本地；为空时从最长非空队列末尾偷一个job。
3. executor发出 `min(quantum,actual-progress)` 个token工作量。
4. 跳到最早完成事件；同一时刻所有完成worker都结算。
5. 记录Segment，未完成job回本地队列；有工作时继续。

| TODO                | 参考有效行 | 测试与不变量                                                      |
| ------------------- | ---------: | ----------------------------------------------------------------- |
| B1 `reassemble`     |         19 | `b_reassemble`，拒绝6类损坏；顺序恢复                             |
| B2 `training_ready` |         13 | `b_batch_gate`，快组偏差、duplicate、stale                        |
| B3 `schedule`       |         61 | `b_scheduler`，总token43守恒、单worker makespan43、全部轨迹可重组 |

### B 实验（1h15min）

64个job，含长尾300，其余10–59，预测都为32；workers=2/4/8，quantum=16/64，steal=false/true，共12行。记录makespan、利用率、segment数与steal数。这里没有oracle预测；实际长度只出现在executor。

画相同quantum下steal开关的配对图；另外画quantum与segment数量。小quantum可以改善任务迁移时机，却可能在真实系统引入更多恢复成本。本模型没有计入这些成本，报告中必须列明这一局限。可以在挑战项中为每个 segment 加入固定开销；仅凭未计开销的结果，不能声称真实系统获得了收益。

## 4. C 前沿：让训练队列和策略更新真正闭环

### C0 阅读三份主源（1h）

按 [resources.md](resources.md) 定位 CS336 的reward/advantage/policy gradient adapter，verl的queue与staleness约定，以及DARTS的分布改变。提交一张对照表：本lab保留的机制、简化的内容、哪种学习目标可能改变。

本 lab 没有实现 DARTS 的冗余轨迹选择，而是先实现研究这类方案所需的完整 group、行为概率、版本与背压约束。截短难样本可能改变数据分布，不能仅作为调度优化看待。

### C1/C2 有界queue（2h）

容量单位是group，不是sample。首次出现新group时占一个槽；同group后续成员进入已保留槽。队列满时返回false，生产者保留样本，在训练者出队后重试。不可在满时覆盖最旧样本；不可返回半个group。

C1验证id/group映射、重复、future/stale、混合版本。C2按group ID选择完整组，先丢弃超龄的整组。已消费group留在consumed集合，阻止重复训练；生产系统还需要有界清理，本lab用有限100波实验。

画容量1、group_size4的时序：第0组只来了一个成员，第1组被拒，剩下第0组成员仍必须可进入。若按sample逐个占满容量，会制造永远无法完成group的死锁。

### C3 带行为概率的真实梯度（1h30min）

令当前动作概率为π，采样时为μ，importance ratio为ρ=π/μ。本lab对组内优势A使用 `min(ρA,clip(ρ,1−ε,1+ε)A)`。实现其对当前logit的导数，注意被截断分支梯度为0；先用旧logit累加整个group梯度，再统一更新。

`c_gradient` 用中心有限差分检查导数，同时覆盖常数 reward 下的零更新，以及触发 clipping 的 off-policy 例子。不要只测试 loss 数值；即使 loss 正确，梯度错误仍会导致训练失败。

### C4 闭环实现（1h30min）

每波从同一策略快照生成4个完整group，每组8条轨迹，保存behavior概率。用B3运行这些轨迹的合成token工作量，按group完成顺序将结果交给有界queue。训练者从完整group更新policy、递增版本；新版本决定剩余样本是否超龄。被背压阻塞的group重试，超龄group整组拒绝。

```mermaid
flowchart LR
  snapshot[冻结策略快照] --> sample[真实随机动作/奖励]
  sample --> execute[B3 token事件执行]
  execute --> queue[完整group有界队列]
  queue --> update[C3梯度更新]
  update --> version[新版本与freshness阈值]
  version --> queue
  version --> snapshot
```

| TODO        | 参考有效行 | 实际检查                                                     |
| ----------- | ---------: | ------------------------------------------------------------ |
| C1 `push`   |         13 | `c_queue_admission`，满队列/duplicate/完整组                 |
| C2 `pop`    |         18 | `c_queue_drain`，整组超龄清理                                |
| C3 `update` |         20 | `c_gradient`，有限差分与截断                                 |
| C4 `learn`  |         81 | `c_learning_closed_loop`，3 seeds真实学习与严格freshness反例 |

### C 实验与解释（2h）

3 seeds × capacity1/2/4 × max_lag0/1/4，共27次训练，每次100波。打印波0/9/49/99的expected reward，共108行；updates/backpressure/stale/token-work/makespan均保存为**截至该波的累计快照**；比较相邻波的增量需要先做差，不能把累计值当作单波计数。

expected reward 根据四类 context 当前的动作概率精确计算，区别于训练 batch 中带噪声的 reward 均值。它用于验证小型 bandit 目标下是否发生了学习，不代表 LLM 的数学能力或泛化能力。测试中，宽松的 lag=4 设置下，3 个 seed 的最终 reward 应超过 0.85；这是固定 toy 环境中的可重复学习验收，不保证任意数据上的性能。

将max_lag=0/4的曲线配对；解释严格freshness为什么可能只利用每波的部分group，并检查哪个context得到更多更新。结合这些结果，讨论系统语义对学习目标的影响。

## 5. 交付报告

提交代码、A3/B12/C108行矩阵、实际学习曲线、三类错误输入（gap/duplicate/stale）、版本与背压时间线、主源阅读卡。报告分别写明固定token预算、固定更新次数、固定墙钟比较有什么差异；目前 C 默认固定波数，因此不能把更多更新带来的收益解释为等计算预算下的优势。

## 可选挑战

以下任务不计入23h。选一项扩展，先验证身份、完整 group 与版本约束，再考虑扩大模型或 GPU 数量。

- 用小 PyTorch 策略替代 Rust 表格策略，保留行为概率与有限差分核对。
- 为每个 segment 加入恢复成本，重新扫描 quantum，检查收益是否仍存在。
- 接入真实小模型长度 trace，对照 token 工作量模型与实际服务时间。
- 统计被保留和被拒绝 group 的 context 分布，检查陈旧度阈值带来的样本选择。
- 实现双缓冲参数广播，明确切换版本的时刻与旧版本样本的归属。

## 资料汇总

- [资源映射](resources.md)、[实验评价规范](../../course/guides/experiments.md)、[评分标准](grade.md)。
- [D8 讲义与读图自测](../../course/modules/08-rollout-scheduling.md)：LPT、segment 恢复、完整组与策略更新的手算题。
- [CS336 Alignment](https://github.com/stanford-cs336/assignment5-alignment)：从小策略实验走向语言模型对齐训练。
- [verl](https://github.com/verl-project/verl)及其[执行结构图](https://verl.readthedocs.io/en/latest/hybrid_flow.html#overall-execution-diagram)：对照控制器、采样与训练 worker 的职责。
- [PPO 原论文图 1](https://arxiv.org/pdf/1707.06347#page=3)：核对 C3 裁剪分支；[DARTS](https://github.com/PKU-DAIR/DARTS)：进一步分析 rollout 分布改变。
